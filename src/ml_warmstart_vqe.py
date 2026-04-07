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
                mode_type: str, k: int) -> str:
    """Unique string key for a (grid, depth, mode) configuration.

    Parameters for this key share the **same parameter-vector dimensionality**
    n_params = (nx+ny) * (n_layers + k)   (simplified; see _effective_layers)
    and are therefore trained with a single MLP.
    """
    return f"nx{nx}_ny{ny}_nl{n_layers}_mt{mode_type}_k{k}"


def _build_features(nx: int, ny: int, n_layers: int,
                    mode_type: str, k: int,
                    plasma_density: float,
                    target_eigenvalue: Optional[float] = None) -> np.ndarray:
    """Encode a solver configuration as a fixed-length feature vector.

    Features
    ────────
      0   nx                          (int, grid resolution)
      1   ny
      2   n_layers                    (circuit depth)
      3   mode_type_enc               (0 = TM, 1 = TE)
      4   k                           (mode index)
      5   log10(1 + plasma_density)   (log-scale density)
      6   target_eigenvalue           (0.0 if unknown)

    A log transform on the density prevents the large range of Ne (1e14–1e20)
    from swamping the other features.
    """
    mode_enc = 0.0 if mode_type == "TM" else 1.0
    log_density = np.log10(1.0 + max(plasma_density, 0.0))
    eig = float(target_eigenvalue) if target_eigenvalue is not None else 0.0
    return np.array([nx, ny, n_layers, mode_enc, k, log_density, eig],
                    dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Training-data collector
# ══════════════════════════════════════════════════════════════════════════════

class WarmStartCollector:
    """Runs VQE optimisations and appends (features, params) pairs to a JSON store.

    Each successful optimisation produces one training sample:
        X  = feature vector (7-d)
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
                 Ne_func: Optional[Callable] = None):
        self.data_path = data_path
        self.Lx = Lx
        self.Ly = Ly
        self.Ne_func = Ne_func
        self._store: Dict = self._load()

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
                verbose: bool = True) -> None:
        """Run `n_runs` independent VQE optimisations and store the results.

        Parameters
        ──────────
        nx, ny          : qubit counts (grid = 2^nx × 2^ny)
        n_layers        : base ansatz depth.  Defaults to nx+ny when None,
                          which is the recommended rule-of-thumb: the circuit
                          depth grows with the grid so that the ansatz always
                          has enough expressivity to represent the eigenfunction.
        mode_type       : 'TM' or 'TE'
        k               : mode index (0-based)
        plasma_density  : uniform electron density [m⁻³] (0 = vacuum)
        n_runs          : how many random restarts to collect
        extra_layers_per_mode : forwarded to WaveguideModeVQA
        """
        if n_layers is None:
            n_layers = nx + ny
            if verbose:
                print(f"  [collect] n_layers not specified — using nx+ny = {n_layers}")

        key = _config_key(nx, ny, n_layers, mode_type, k)
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

        n_collected = 0
        for run in range(n_runs):
            t0 = time.time()
            try:
                eigenvalue, params, _ = solver.optimize_mode(k)
            except Exception as exc:
                warnings.warn(f"Run {run} failed: {exc}")
                continue

            if eigenvalue < 0.0:
                if verbose:
                    print(f"  run {run}: unphysical eigenvalue {eigenvalue:.4f}, skipped.")
                continue

            features = _build_features(
                nx, ny, n_layers, mode_type, k,
                plasma_density, eigenvalue
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
              f"(total: {len(self._store[key]['X'])})")

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
    Input  : 7-d feature vector (see _build_features)
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
                 model_dir: Optional[str] = None):
        self.data_path = data_path
        self.hidden_layer_multiplier = hidden_layer_multiplier
        self.max_iter = max_iter
        self.model_dir = model_dir

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
            X_raw = np.array(data["X"])   # (N, 7)
            Y_raw = np.array(data["y"])   # (N, n_params)
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

            mlp = MLPRegressor(
                hidden_layer_sizes=hidden,
                activation="tanh",      # smooth, bounded — good for angles
                solver="adam",
                max_iter=self.max_iter,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2 if N >= 10 else 0.0,
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
            If known from a coarse estimate, including it improves accuracy.
        noise_std
            Small Gaussian noise added to the prediction for exploration.
            Set to 0.0 for a deterministic warm start.

        Returns
        ───────
        θ₀ : np.ndarray of shape (n_params,) or None if no model is available.
        """
        key = _config_key(nx, ny, n_layers, mode_type, k)

        if not self._trained.get(key, False):
            return None  # caller falls back to random initialisation

        feat = _build_features(
            nx, ny, n_layers, mode_type, k,
            plasma_density, target_eigenvalue
        ).reshape(1, -1)

        X_scaled = self._scalers[key].transform(feat)
        y_enc    = self._models[key].predict(X_scaled)[0]
        params   = self._decode_circular(y_enc)

        if noise_std > 0.0:
            params += np.random.normal(0, noise_std, size=params.shape)

        return params

    def is_trained(self, nx: int, ny: int, n_layers: int,
                   mode_type: str, k: int) -> bool:
        """Return True if a model is available for this configuration."""
        return self._trained.get(
            _config_key(nx, ny, n_layers, mode_type, k), False
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
    Attempt 2 : saved JSON parameters from a previous run  (unchanged)
    Attempt 3+: random uniform in [-π, π]                  (unchanged)

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

        while eigenvalue < 0.0 and attempts < max_attempts:
            attempts += 1
            history   = []

            # ── Initialisation strategy ───────────────────────────────────
            if attempts == 1 and self.predictor is not None:
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
                    noise_std=self.warm_start_noise,
                )
                if theta0 is not None and len(theta0) == n_params:
                    print(f"[WarmStart] Using ML prediction for mode {k}.")
                else:
                    print(f"[WarmStart] No model for this config — "
                          "falling back to saved params / random.")
                    theta0 = None

            elif attempts == 2 and self.use_params_file:
                # Strategy B: load from JSON (original logic)
                theta0 = self.load_params(k)
                if theta0 is not None:
                    theta0 += np.abs(np.random.normal(0, 0.1, n_params))

            else:
                theta0 = None

            if theta0 is None or len(theta0) != n_params:
                # Strategy C: random
                theta0 = np.random.uniform(-np.pi, np.pi, n_params)

            # ── Optimisation (unchanged from base class) ──────────────────
            def callback(xk):
                val = self.cost_function(xk, k, beta)
                history.append(val)
                print(f"Attempt {attempts} - Iter {len(history)}: "
                      f"Cost {val:.4f}    ", end="\r")

            result = minimize(
                lambda p: self.cost_function(p, k, beta),
                theta0,
                method="L-BFGS-B",
                callback=callback,
                options={"maxiter": 400},
            )

            eigenvalue       = result.fun
            optimized_params = result.x

            if eigenvalue < 1.0:
                print(f"\n[Warning] Attempt {attempts} found eigenvalue "
                      f"{eigenvalue:.4f} (< 0). Restarting...")

        self.optimized_states.append(
            __import__("qiskit.quantum_info", fromlist=["Statevector"])
            .Statevector(self.ansatz(optimized_params, k)).data
        )
        self.eigenvalues.append(eigenvalue)
        self.optimized_params_list = [optimized_params]

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

        X_raw = np.array(store[k]["X"])
        Y_raw = np.array(store[k]["y"])
        N, n_params = Y_raw.shape

        Y_pred = np.zeros_like(Y_raw)
        for i, feat in enumerate(X_raw):
            nx      = int(feat[0])
            ny      = int(feat[1])
            n_layers= int(feat[2])
            mt      = "TM" if feat[3] < 0.5 else "TE"
            ki      = int(feat[4])
            density = 10**feat[5] - 1
            eigen   = feat[6]

            p = predictor.predict(
                nx, ny, n_layers, mt, ki, density, eigen, noise_std=0.0
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

        for trial in range(n_trials):
            if strategy == "warm":
                theta0 = predictor.predict(
                    nx, ny, n_layers, mode_type, k, plasma_density,
                    noise_std=0.05
                ) or np.random.uniform(-np.pi, np.pi, n_params)
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