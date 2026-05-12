"""
ml_warmstart_vqe.py
────────────────────────────────────────────────────────────────────────────────
ML Warm-Start for ColdPlasma VQE Waveguide Solver
────────────────────────────────────────────────────────────────────────────────

Drop-in extension for WaveguideModeVQA that trains a neural network to predict
good initial circuit parameters from (nx, ny, mode_type, k, plasma_density),
replacing the random initialisation used in optimize_mode.

Pipeline
────────
  1.  WarmStartCollector   – runs VQE sweeps and persists training data to JSON
  2.  WarmStartPredictor   – trains one MLP per configuration key and predicts θ₀
  3.  WarmStartVQA         – subclass of WaveguideModeVQA with warm-start baked in

Quick-start
────────────
  from src import WarmStartVQA, WarmStartCollector, WarmStartPredictor

  # 1. Generate training data (run once / whenever you have new configs)
  collector = WarmStartCollector(data_path="warmstart_data.json")
  collector.collect(nx=2, ny=2, n_layers=2, mode_type="TM", k=0, n_runs=10)
  collector.collect(nx=2, ny=2, n_layers=2, mode_type="TM", k=1, n_runs=10)

  # 2. Train
  predictor = WarmStartPredictor(data_path="warmstart_data.json")
  predictor.train()

  # 3. Use the warm-started solver exactly like the original class
  solver = WarmStartVQA(nx=2, ny=2, n_layers=2, predictor=predictor,
                        mode_type="TM", plasma_density=1e17)
  eigenvalue, params, history = solver.optimize_mode(k=0)
"""

from __future__ import annotations

import json
import os
import time
import warnings
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# ── Import the original solver ────────────────────────────────────────────────
# Adjust the import to match your project layout.
from .coldplasma_vqe_waveguide import WaveguideModeVQA


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Feature engineering helpers
# ══════════════════════════════════════════════════════════════════════════════

def _config_key(nx: int, ny: int, n_layers: int,
                mode_type: str, k: int,
                plasma_density: float = 0.0,
                density_in_key: bool = False) -> str:
    """Unique string key for a configuration.
 
    When density_in_key is False (default) the key is the same for all
    densities → one shared MLP learns across densities.
 
    When density_in_key is True the density is encoded in the key
    → each density gets its own MLP.
    """
    base = f"nx{nx}_ny{ny}_nl{n_layers}_mt{mode_type}_k{k}"
    if density_in_key:
        tag = "vacuum" if plasma_density == 0.0 else f"ne{plasma_density:.1e}"
        return f"{base}_{tag}"
    return base


def _build_features(nx, ny, n_layers, mode_type, k,
                    plasma_density,
                    target_eigenvalue,
                    density_in_key=False):
    """Encode a solver configuration as a fixed-length feature vector.

    Features (density_in_key=False, default)
    ────────────────────────────────────────
      0   nx                          (int, grid resolution)
      1   ny
      2   n_layers                    (circuit depth)
      3   mode_type_enc               (0 = TM, 1 = TE)
      4   k                           (mode index)
      5   log10(1 + plasma_density)   (log-scale density)
      6   log10(target_eigenvalue)    (classical λ for mode k)

    Features (density_in_key=True)
    ──────────────────────────────
      Same as above without index 5 (density is encoded in the key instead).

    The target eigenvalue is the k-th physical classical eigenvalue obtained
    from np.linalg.eigvalsh(solver.M_dense) — cheap to compute and a strong
    physics-aware fingerprint of which mode the VQE is targeting.
    """

    mode_enc = 0.0 if mode_type == "TM" else 1.0
    log_target = np.log10(target_eigenvalue)

    if density_in_key:
        return np.array([nx, ny, n_layers, mode_enc, k, log_target],
                        dtype=np.float64)
    else:
        log_density = np.log10(1.0 + max(plasma_density, 0.0))
        return np.array([nx, ny, n_layers, mode_enc, k, log_density, log_target],
                        dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Training-data collector
# ══════════════════════════════════════════════════════════════════════════════

class WarmStartCollector:
    """Runs VQE optimisations and appends (features, params) pairs to a JSON store.

    Each successful optimisation produces one training sample:
        X  = feature vector (6-d)
        y  = optimised parameter vector (variable length per config key)

    Multiple runs per configuration increase robustness by covering more of the
    parameter landscape.

    Parameters
    ──────────
    data_path : str
        Path to the JSON file used as the data store (created if absent).
    Lx, Ly    : float
        Waveguide dimensions [m] forwarded to WaveguideModeVQA.
    Ne_func   : optional callable
        Inhomogeneous plasma profile.  If None a uniform density is used.
    """

    def __init__(self, data_path: str = "warmstart_data.json",
                 Lx: float = 0.015, Ly: float = 0.010,
                 Ne_func: Optional[Callable] = None, density_in_key=False):
        self.data_path = data_path
        self.Lx = Lx
        self.Ly = Ly
        self.Ne_func = Ne_func
        self._store: Dict = self._load()
        self.density_in_key = density_in_key

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> Dict:
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                warnings.warn(f"Could not parse {self.data_path}; starting fresh.")
        return {}

    def _save(self) -> None:
        with open(self.data_path, "w") as f:
            json.dump(self._store, f, indent=2)

    # ── Collection ────────────────────────────────────────────────────────────

    def collect(self,
                nx: int, ny: int,
                n_layers: Optional[int] = None,
                mode_type: str = "TM",
                k: int = 0,
                plasma_density: float = 0.0,
                n_runs: int = 5,
                extra_layers_per_mode: int = 1,
                eigenvalue_rtol: float = 0.05,
                verbose: bool = True) -> None:
        """Run `n_runs` independent VQE optimisations and store the results.
    
        Only samples whose eigenvalue is close to the *correct* k-th
        classical eigenvalue (from M_dense) are kept.  This prevents the
        MLP from learning parameters that converge to the wrong mode.
    
        Parameters
        ──────────
        nx, ny          : qubit counts (grid = 2^nx × 2^ny)
        n_layers        : base ansatz depth.  Defaults to nx+ny when None.
        mode_type       : 'TM' or 'TE'
        k               : mode index (0-based)
        plasma_density  : uniform electron density [m⁻³] (0 = vacuum)
        n_runs          : how many random restarts to collect
        extra_layers_per_mode : forwarded to WaveguideModeVQA
        eigenvalue_rtol : float  (NEW)
            Relative tolerance for eigenvalue matching.  A VQE result is
            accepted only if:
                |λ_vqe - λ_classical[k]| / λ_classical[k]  <  eigenvalue_rtol
            Default 0.05 (5%).  Increase for noisy / low-resolution grids.
        """
        from .coldplasma_vqe_waveguide import WaveguideModeVQA
    
        if n_layers is None:
            n_layers = nx + ny
            if verbose:
                print(f"  [collect] n_layers not specified — using nx+ny = {n_layers}")
    
        key = _config_key(nx, ny, n_layers, mode_type, k, plasma_density=plasma_density,
                          density_in_key=self.density_in_key)
        if key not in self._store:
            self._store[key] = {"X": [], "y": []}
    
        solver = WaveguideModeVQA(
            nx=nx, ny=ny, n_layers=n_layers,
            mode_type=mode_type,
            extra_layers_per_mode=extra_layers_per_mode,
            Lx=self.Lx, Ly=self.Ly,
            Ne_func=self.Ne_func,
            plasma_density=plasma_density,
        )
    
        # ── NEW: compute classical eigenvalues from M_dense ───────────────
        classical_eigs = np.sort(np.linalg.eigvalsh(solver.M_dense))
    
        # Filter to physical eigenvalues only:
        #   TM modes: eigenvalues > 0   (Dirichlet BCs, all positive)
        #   TE modes: eigenvalues > 1   (Neumann BCs have a spurious zero mode)
        eig_threshold = 0.0 if mode_type == "TM" else 1.0
        physical_eigs = np.array([e for e in classical_eigs if e > eig_threshold])
    
        if k >= len(physical_eigs):
            raise ValueError(
                f"Mode index k={k} exceeds the number of physical "
                f"{mode_type} eigenvalues ({len(physical_eigs)}) on a "
                f"{2**nx}×{2**ny} grid."
            )
    
        target_eig = physical_eigs[k]
    
        if verbose:
            print(f"  [collect] Classical eigenvalue for {mode_type} k={k}: "
                f"λ_ref = {target_eig:.4f}")
        # ── END NEW ───────────────────────────────────────────────────────

        # ── Bootstrap lower modes on this solver ──────────────────────────
        # optimize_mode(k) for k > 0 needs solver.optimized_states[0..k-1]
        # populated (used by the orthogonality penalty in cost_function).
        # Since we build a fresh solver per collect() call, we must first
        # solve modes 0..k-1 here, rejecting wrong-mode convergences.
        if k > 0:
            bootstrap_max_tries = 10
            for i in range(k):
                target_i = physical_eigs[i]
                tries = 0
                while len(solver.optimized_states) <= i and tries < bootstrap_max_tries:
                    tries += 1
                    try:
                        ev_i, _, _ = solver.optimize_mode(i)
                    except Exception as exc:
                        warnings.warn(
                            f"Bootstrap mode {i} try {tries} failed: {exc}"
                        )
                        continue
                    rel_i = abs(ev_i - target_i) / abs(target_i)
                    if ev_i < 1.0 or rel_i > eigenvalue_rtol:
                        # optimize_mode always appends — drop the bad state
                        if len(solver.optimized_states) > i:
                            solver.optimized_states.pop()
                            solver.eigenvalues.pop()
                        if verbose:
                            print(f"  [bootstrap] mode {i} try {tries}: "
                                  f"λ={ev_i:.4f} REJECTED "
                                  f"(rel_err={rel_i:.2%})")
                    elif verbose:
                        print(f"  [bootstrap] mode {i} locked in "
                              f"λ={ev_i:.4f} after {tries} try(s)")
                if len(solver.optimized_states) <= i:
                    raise RuntimeError(
                        f"Failed to bootstrap mode {i} for key '{key}' "
                        f"after {bootstrap_max_tries} tries; cannot collect k={k}."
                    )
        # ── END bootstrap ─────────────────────────────────────────────────

        n_collected = 0
        n_rejected  = 0                                            # NEW counter
        for run in range(n_runs):
            t0 = time.time()
            try:
                eigenvalue, params, _ = solver.optimize_mode(k)
            except Exception as exc:
                warnings.warn(f"Run {run} failed: {exc}")
                continue
    
            if eigenvalue < 0.0:
                if verbose:
                    print(f"  run {run}: unphysical eigenvalue "
                        f"{eigenvalue:.4f}, skipped.")
                continue
    
            # ── NEW: check that VQE found the correct mode ────────────────
            rel_error = abs(eigenvalue - target_eig) / abs(target_eig)
            if rel_error > eigenvalue_rtol:
                n_rejected += 1
                if verbose:
                    dt = time.time() - t0
                    print(f"  [{key}] run {run+1}/{n_runs} "
                        f"λ={eigenvalue:.4f}  REJECTED "
                        f"(rel_err={rel_error:.2%} vs λ_ref={target_eig:.4f})  "
                        f"({dt:.1f}s)")
                continue
            # ── END NEW ───────────────────────────────────────────────────
    
            features = _build_features(
                nx, ny, n_layers, mode_type, k,
                plasma_density,
                target_eigenvalue=target_eig,
                density_in_key=self.density_in_key
            ).tolist()
    
            self._store[key]["X"].append(features)
            self._store[key]["y"].append(params.tolist())
            n_collected += 1
    
            if verbose:
                dt = time.time() - t0
                print(f"  [{key}] run {run+1}/{n_runs} "
                    f"λ={eigenvalue:.4f}  ({dt:.1f}s)")
    
        self._save()
        print(f"Collected {n_collected} new samples for '{key}' "
            f"(total: {len(self._store[key]['X'])})"
            f"  [rejected {n_rejected} wrong-mode convergences]")    # NEW info

    # ── Inspection ────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a summary of available training data."""
        print(f"\n{'─'*55}")
        print(f"  WarmStart data store: {self.data_path}")
        print(f"{'─'*55}")
        total = 0
        for key, v in self._store.items():
            n = len(v["X"])
            total += n
            print(f"  {key:50s}  {n:4d} samples")
        print(f"{'─'*55}")
        print(f"  Total: {total} samples\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Predictor (one MLP per config key)
# ══════════════════════════════════════════════════════════════════════════════

class WarmStartPredictor:
    """Trains and stores one MLP per (nx, ny, n_layers, mode_type, k) key.

    Architecture
    ────────────
    Input  : 7-d feature vector (6-d when density_in_key=True, see _build_features)
    Hidden : two hidden layers of width max(64, 4·n_params)
    Output : n_params floats in [-π, π]

    The output lives on a torus so we regress both sin and cos of each parameter
    and reconstruct with atan2, keeping predictions in [-π, π] regardless of
    how the MLP extrapolates.

    Parameters
    ──────────
    data_path    : path to the JSON store produced by WarmStartCollector
    hidden_layer_multiplier : hidden-layer width = max(64, m * n_params)
    max_iter     : MLP training iterations
    model_dir    : if given, fitted models are serialised here with joblib
    """

    def __init__(self,
                 data_path: str = "warmstart_data.json",
                 hidden_layer_multiplier: int = 4,
                 max_iter: int = 2000,
                 model_dir: Optional[str] = None,
                 density_in_key=False):
        self.data_path = data_path
        self.hidden_layer_multiplier = hidden_layer_multiplier
        self.max_iter = max_iter
        self.model_dir = model_dir
        self.density_in_key = density_in_key

        # Per-key artefacts
        self._models:   Dict[str, MLPRegressor] = {}
        self._scalers:  Dict[str, StandardScaler] = {}
        self._trained:  Dict[str, bool] = {}

        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            self._try_load_models()

    # ── Persistence of trained models ─────────────────────────────────────────

    def _model_path(self, key: str) -> str:
        assert self.model_dir
        return os.path.join(self.model_dir, f"{key}.joblib")

    def _scaler_path(self, key: str) -> str:
        assert self.model_dir
        return os.path.join(self.model_dir, f"{key}_scaler.joblib")

    def _try_load_models(self) -> None:
        if not self.model_dir:
            return
        for fname in os.listdir(self.model_dir):
            if fname.endswith(".joblib") and "_scaler" not in fname:
                key = fname[:-7]
                try:
                    self._models[key]  = joblib.load(self._model_path(key))
                    self._scalers[key] = joblib.load(self._scaler_path(key))
                    self._trained[key] = True
                    print(f"  Loaded model for '{key}'")
                except Exception as exc:
                    warnings.warn(f"Could not load model for {key}: {exc}")

    def _save_model(self, key: str) -> None:
        if not self.model_dir:
            return
        joblib.dump(self._models[key],  self._model_path(key))
        joblib.dump(self._scalers[key], self._scaler_path(key))

    # ── Circular encoding helpers ─────────────────────────────────────────────

    @staticmethod
    def _encode_circular(params: np.ndarray) -> np.ndarray:
        """Encode each angle θ as [sin θ, cos θ] → shape (2*n,)."""
        return np.concatenate([np.sin(params), np.cos(params)])

    @staticmethod
    def _decode_circular(encoded: np.ndarray) -> np.ndarray:
        """Decode a [sin θ, cos θ] vector back to angles in [-π, π]."""
        n = len(encoded) // 2
        sins, coss = encoded[:n], encoded[n:]
        return np.arctan2(sins, coss)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, verbose: bool = True) -> None:
        """Train one MLP for every key present in the data store.

        Minimum 4 samples are required; keys with fewer are skipped with a
        warning so training never raises even when data is sparse.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Data store not found: {self.data_path}\n"
                "Run WarmStartCollector.collect() first."
            )
        with open(self.data_path) as f:
            store = json.load(f)

        for key, data in store.items():
            # Guard against inhomogeneous feature lengths (e.g. data
            # collected across code versions with different feature sets).
            X_list, Y_list = data["X"], data["y"]
            lengths = [len(x) for x in X_list]
            if not lengths:
                continue
            target_len = max(set(lengths), key=lengths.count)
            mask = [l == target_len for l in lengths]
            if not all(mask):
                n_dropped = sum(not m for m in mask)
                warnings.warn(
                    f"  '{key}': dropped {n_dropped}/{len(mask)} samples "
                    f"with mismatched feature length (expected {target_len})."
                )
            X_raw = np.array([x for x, m in zip(X_list, mask) if m])
            Y_raw = np.array([y for y, m in zip(Y_list, mask) if m])
            N = len(X_raw)

            if N < 4:
                warnings.warn(
                    f"  Skipping '{key}': only {N} samples (need ≥ 4)."
                )
                continue

            # Circular-encode targets → shape (N, 2*n_params)
            Y_enc = np.array([self._encode_circular(y) for y in Y_raw])

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)

            # Determine hidden layer width
            n_params = Y_raw.shape[1]
            width = max(64, self.hidden_layer_multiplier * n_params)
            hidden = (width, width)

            use_early_stop = N >= 10
            mlp = MLPRegressor(
                hidden_layer_sizes=hidden,
                activation="tanh",      # smooth, bounded — good for angles
                solver="adam",
                max_iter=self.max_iter,
                random_state=42,
                early_stopping=use_early_stop,
                validation_fraction=0.2 if use_early_stop else 0.1,
                n_iter_no_change=50,
                learning_rate_init=1e-3,
                tol=1e-5,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlp.fit(X_scaled, Y_enc)

            self._models[key]  = mlp
            self._scalers[key] = scaler
            self._trained[key] = True
            self._save_model(key)

            if verbose:
                score = mlp.score(X_scaled, Y_enc)
                print(f"  Trained '{key}'  N={N}  hidden={hidden}  R²={score:.4f}")

        print(f"\nTraining complete.  {len(self._trained)} model(s) ready.")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self,
                nx: int, ny: int, n_layers: int,
                mode_type: str, k: int,
                plasma_density: float = 0.0,
                target_eigenvalue: Optional[float] = None,
                noise_std: float = 0.05) -> Optional[np.ndarray]:
        """Predict an initial parameter vector θ₀.

        Parameters
        ──────────
        nx, ny, n_layers, mode_type, k, plasma_density
            Solver configuration — must match a trained key.
        target_eigenvalue
            The k-th physical classical eigenvalue (from
            np.linalg.eigvalsh(solver.M_dense)).  Required: it is part of the
            feature vector and gives the MLP a physics-aware fingerprint of
            the targeted mode.
        noise_std
            Small Gaussian noise added to the prediction for exploration.
            Set to 0.0 for a deterministic warm start.

        Returns
        ───────
        θ₀ : np.ndarray of shape (n_params,) or None if no model is available.
        """
        key = _config_key(nx, ny, n_layers, mode_type, k,plasma_density=plasma_density,
                            density_in_key=self.density_in_key)

        if not self._trained.get(key, False):
            return None  # caller falls back to random initialisation

        if target_eigenvalue is None:
            raise ValueError(
                "target_eigenvalue is required.  Compute via "
                "np.sort(np.linalg.eigvalsh(solver.M_dense)) and select the "
                "k-th physical eigenvalue (>0 for TM, >1 for TE)."
            )

        feat = _build_features(
            nx, ny, n_layers, mode_type, k,
            plasma_density,
            target_eigenvalue=target_eigenvalue,
            density_in_key=self.density_in_key
        ).reshape(1, -1)

        X_scaled = self._scalers[key].transform(feat)
        y_enc    = self._models[key].predict(X_scaled)[0]
        params   = self._decode_circular(y_enc)

        if noise_std > 0.0:
            params += np.random.normal(0, noise_std, size=params.shape)

        return params

    def is_trained(self, nx: int, ny: int, n_layers: int,
                   mode_type: str, k: int, plasma_density =0.0) -> bool:
        """Return True if a model is available for this configuration."""
        return self._trained.get(
            _config_key(nx, ny, n_layers, mode_type, k, plasma_density=plasma_density,
                        density_in_key=self.density_in_key), False
        )

    def trained_keys(self) -> List[str]:
        return [k for k, v in self._trained.items() if v]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Warm-started solver subclass
# ══════════════════════════════════════════════════════════════════════════════

class WarmStartVQA(WaveguideModeVQA):
    """Drop-in replacement for WaveguideModeVQA with ML warm-start support.

    All constructor arguments are forwarded unchanged to WaveguideModeVQA.
    The only addition is the `predictor` keyword argument.

    Warm-start logic in optimize_mode
    ──────────────────────────────────
    Attempt 1 : ML prediction (if a trained model exists for this config)
    Attempt 2+: random uniform in [-π, π]

    Parameters
    ──────────
    predictor : WarmStartPredictor or None
        A trained predictor.  Pass None to behave exactly like the base class.
    warm_start_noise : float
        Noise added to the ML prediction (prevents landing on a saddle point).
    """

    def __init__(self, *args,
                 predictor: Optional[WarmStartPredictor] = None,
                 warm_start_noise: float = 0.05,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor       = predictor
        self.warm_start_noise = warm_start_noise

    def optimize_mode(self, k: int) -> Tuple[float, np.ndarray, List[float]]:
        """Override optimize_mode with ML warm-start on the first attempt.

        Everything else — penalty weight β, restart logic, history tracking —
        is preserved from the original implementation.
        """
        from scipy.optimize import minimize

        n_params  = self.n * self._effective_layers(k)
        beta      = 4e5
        eigenvalue = -1.0
        max_attempts = 5
        attempts  = 0
        history   = []

        class _EarlyRestart(Exception):
            pass

        while eigenvalue < 0.0 and attempts < max_attempts:
            attempts += 1
            history   = []

            # ── Initialisation strategy ───────────────────────────────────
            if attempts == 1 and self.predictor is not None:
                # Classical λ for mode k — used as a feature so the MLP knows
                # which point in the spectrum it is targeting.
                classical_eigs = np.sort(np.linalg.eigvalsh(self.M_dense))
                eig_threshold = 0.0 if self.mode_type == "TM" else 1.0
                physical_eigs = classical_eigs[classical_eigs > eig_threshold]
                target_eig = float(physical_eigs[k])

                # Strategy A: ML warm start
                theta0 = self.predictor.predict(
                    nx=self.nx, ny=self.ny,
                    n_layers=self.n_layers,
                    mode_type=self.mode_type,
                    k=k,
                    plasma_density=float(
                        np.mean(self.plasma_potential_flat)
                        * (self.c**2 * self.me * self.eps0 / self.qe**2)
                    ),
                    target_eigenvalue=target_eig,
                    noise_std=self.warm_start_noise,
                )
                if theta0 is not None and len(theta0) == n_params:
                    print(f"[WarmStart] Using ML prediction for mode {k}.")
                else:
                    print(f"[WarmStart] No model for this config — "
                          "falling back to random.")
                    theta0 = None
            else:
                theta0 = None

            if theta0 is None or len(theta0) != n_params:
                # Random fallback
                theta0 = np.random.uniform(-np.pi, np.pi, n_params)

            last_x = [theta0]

            # ── Optimisation (unchanged from base class) ──────────────────
            def callback(xk):
                last_x[0] = xk
                val = self.cost_function(xk, k, beta)
                history.append(val)
                print(f"Attempt {attempts} - Iter {len(history)}: "
                      f"Cost {val:.4f}    ", end="\r")
                # Early restart: if at iteration 50 the cost is already in the
                # unphysical regime (< 1), abort this attempt instead of
                # running all 400 iterations just to discard the result.
                if len(history) == 50 and val < 1.0:
                    raise _EarlyRestart()

            try:
                result = minimize(
                    lambda p: self.cost_function(p, k, beta),
                    theta0,
                    method="L-BFGS-B",
                    callback=callback,
                    options={"maxiter": 400},
                )
                eigenvalue       = result.fun
                optimized_params = result.x
            except _EarlyRestart:
                eigenvalue       = history[-1]
                optimized_params = last_x[0]
                print(f"\n[EarlyRestart] Attempt {attempts} cost "
                      f"{eigenvalue:.4f} at iter 50 (< 1). Restarting...")
                continue

            if eigenvalue < 1.0:
                print(f"\n[Warning] Attempt {attempts} found eigenvalue "
                      f"{eigenvalue:.4f} (< 0). Restarting...")

        self.optimized_states.append(
            __import__("qiskit.quantum_info", fromlist=["Statevector"])
            .Statevector(self.ansatz(optimized_params, k)).data
        )
        self.eigenvalues.append(eigenvalue)

        return eigenvalue, optimized_params, history


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Utility: plot warm-start quality
# ══════════════════════════════════════════════════════════════════════════════

def plot_warmstart_quality(data_path: str,
                           predictor: WarmStartPredictor,
                           key: Optional[str] = None) -> None:
    """Compare ML-predicted vs. true optimised parameters for visual diagnostics.

    For each training sample it plots  θ_predicted  vs  θ_true, coloured by
    parameter index.  A perfect predictor sits on the diagonal y = x.

    Parameters
    ──────────
    data_path  : path to the JSON data store
    predictor  : a trained WarmStartPredictor
    key        : if None, all available keys are plotted sequentially
    """
    import matplotlib.pyplot as plt

    with open(data_path) as f:
        store = json.load(f)

    keys = [key] if key else list(store.keys())

    for k in keys:
        if k not in store:
            print(f"Key '{k}' not found in store.")
            continue

        X_list, Y_list = store[k]["X"], store[k]["y"]
        lengths = [len(x) for x in X_list]
        if not lengths:
            print(f"Key '{k}' has no samples.")
            continue
        target_len = max(set(lengths), key=lengths.count)
        mask = [l == target_len for l in lengths]
        X_raw = np.array([x for x, m in zip(X_list, mask) if m])
        Y_raw = np.array([y for y, m in zip(Y_list, mask) if m])
        N, n_params = Y_raw.shape

        Y_pred = np.zeros_like(Y_raw)
        for i, feat in enumerate(X_raw):
            nx      = int(feat[0])
            ny      = int(feat[1])
            n_layers= int(feat[2])
            mt      = "TM" if feat[3] < 0.5 else "TE"
            ki      = int(feat[4])
            # Feature layout:
            #   density_in_key=False → [..., k, log_density, log_target] (7-d)
            #   density_in_key=True  → [..., k, log_target] (6-d)
            if len(feat) >= 7:
                density    = 10**feat[5] - 1
                target_eig = 10**feat[6]
            else:
                density    = 0.0  # density encoded in the key, not features
                target_eig = 10**feat[5]

            p = predictor.predict(
                nx, ny, n_layers, mt, ki, density,
                target_eigenvalue=target_eig,
                noise_std=0.0,
            )
            Y_pred[i] = p if p is not None else np.zeros(n_params)

        fig, ax = plt.subplots(figsize=(5, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, n_params))
        for j in range(n_params):
            ax.scatter(Y_raw[:, j], Y_pred[:, j],
                       color=colors[j], s=18, alpha=0.6)

        lim = (-np.pi - 0.1, np.pi + 0.1)
        ax.plot(lim, lim, "k--", lw=0.8, label="y = x")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("True θ (optimised)")
        ax.set_ylabel("Predicted θ (warm start)")
        ax.set_title(f"Warm-start quality — {k}")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Utility: benchmark warm start vs random
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_warmstart(predictor: WarmStartPredictor,
                        nx: int, ny: int, n_layers: int,
                        mode_type: str = "TM", k: int = 0,
                        plasma_density: float = 0.0,
                        n_trials: int = 5,
                        Lx: float = 0.015, Ly: float = 0.010) -> Dict:
    """Compare convergence speed and final cost for warm vs. random starts.

    Returns a dict with keys 'warm' and 'random', each containing lists of
    (final_eigenvalue, n_iterations) pairs.
    """
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    results = {"warm": [], "random": []}

    for strategy in ("warm", "random"):
        print(f"\n── {strategy.upper()} START  ({n_trials} trials) ──")

        solver = WarmStartVQA(
            nx=nx, ny=ny, n_layers=n_layers,
            mode_type=mode_type,
            plasma_density=plasma_density,
            Lx=Lx, Ly=Ly,
            predictor=predictor if strategy == "warm" else None,
        )

        n_params = solver.n * solver._effective_layers(k)
        beta = 4e5

        # Classical λ for mode k — required feature for the predictor.
        classical_eigs = np.sort(np.linalg.eigvalsh(solver.M_dense))
        eig_threshold = 0.0 if mode_type == "TM" else 1.0
        physical_eigs = classical_eigs[classical_eigs > eig_threshold]
        target_eig = float(physical_eigs[k])

        for trial in range(n_trials):
            if strategy == "warm":
                theta0 = predictor.predict(
                    nx, ny, n_layers, mode_type, k, plasma_density,
                    target_eigenvalue=target_eig,
                    noise_std=0.05,
                )
                if theta0 is None:
                    theta0 = np.random.uniform(-np.pi, np.pi, n_params)
            else:
                theta0 = np.random.uniform(-np.pi, np.pi, n_params)

            iters = [0]
            def cb(xk):
                iters[0] += 1

            res = minimize(
                lambda p: solver.cost_function(p, k, beta),
                theta0, method="L-BFGS-B",
                callback=cb,
                options={"maxiter": 400},
            )
            results[strategy].append((res.fun, iters[0]))
            print(f"  trial {trial+1}: λ={res.fun:.4f}  iters={iters[0]}")

    # Quick summary plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in zip(
        axes,
        [0, 1],
        ["Final eigenvalue (cost)", "Iterations to converge"]
    ):
        for strat, color in [("warm", "steelblue"), ("random", "coral")]:
            vals = [r[metric] for r in results[strat]]
            ax.bar(
                np.arange(n_trials) + (0 if strat == "warm" else 0.4),
                vals, width=0.35, color=color, alpha=0.8, label=strat
            )
        ax.set_xlabel("Trial")
        ax.set_ylabel(label)
        ax.legend()
        ax.set_title(label)
    plt.suptitle(f"Warm vs. Random start  |  {mode_type} k={k}  nx={nx} ny={ny}")
    plt.tight_layout()
    plt.show()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Example entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    DATA_PATH  = "warmstart_data.json"
    MODEL_DIR  = "warmstart_models"
    NX, NY, NL = 2, 2, 2
    DENSITY    = 1e17          # [m⁻³]

    print("=" * 60)
    print("  STEP 1 — Generate training data")
    print("=" * 60)
    collector = WarmStartCollector(data_path=DATA_PATH)

    for mode in ("TM", "TE"):
        for k in (0, 1):
            collector.collect(
                nx=NX, ny=NY, n_layers=NL,
                mode_type=mode, k=k,
                plasma_density=DENSITY,
                n_runs=8,
            )

    collector.summary()

    print("=" * 60)
    print("  STEP 2 — Train predictors")
    print("=" * 60)
    predictor = WarmStartPredictor(data_path=DATA_PATH, model_dir=MODEL_DIR)
    predictor.train()

    print("=" * 60)
    print("  STEP 3 — Benchmark warm vs random start")
    print("=" * 60)
    benchmark_warmstart(
        predictor,
        nx=NX, ny=NY, n_layers=NL,
        mode_type="TM", k=0,
        plasma_density=DENSITY,
        n_trials=5,
    )

    print("=" * 60)
    print("  STEP 4 — Solve with warm start")
    print("=" * 60)
    solver = WarmStartVQA(
        nx=NX, ny=NY, n_layers=NL,
        mode_type="TM",
        plasma_density=DENSITY,
        predictor=predictor,
        warm_start_noise=0.05,
    )
    eigenvalue, params, history = solver.optimize_mode(k=0)
    solver.print_plot_parameters(0, eigenvalue, params)