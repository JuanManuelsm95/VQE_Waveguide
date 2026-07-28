"""
transfer_warmstart.py
────────────────────────────────────────────────────────────────────────────────
Size-Transferable ML Warm-Start (cross-grid parameter prediction)
────────────────────────────────────────────────────────────────────────────────

Motivation
──────────
WarmStartPredictor trains one MLP *per config key*, so its output dimension is
tied to one grid size: samples collected on a 2x2-qubit problem can never
inform a 3x3 prediction.  This module reformulates the regression so that data
from *all* grid sizes pools into a single model.

Reformulation — predict one angle, not one vector
─────────────────────────────────────────────────
The HEA ansatz parameter θ[L·n + q] is the RY angle of qubit q in layer L
(n = nx + ny qubits, layer-major ordering — see WaveguideModeVQA.ansatz).
Instead of mapping  config → full θ-vector,  a single shared MLP maps

    (global config features, structural coordinates of one parameter) → θ_{L,q}

Global features (size-comparable)
    nx, ny                       grid resolution (qubits per dimension)
    mode_enc                     0 = TM, 1 = TE
    k                            mode index
    log10(1 + plasma_density)
    log10(target_eigenvalue)     ≈ grid-independent continuum fingerprint

Structural coordinates of θ_{L,q}  (all normalised to [0, 1] so they are
comparable across sizes)
    layer_frac   L / (n_layers_eff − 1)      depth position in the circuit
    qubit_frac   q / (n − 1)                 position along the linear CX chain
    in_y_reg     0 if q < nx (x register), else 1
    sig_frac     bit significance inside its register / (register size − 1)
                 (qubit 0 of each register is the least-significant bit of the
                  corresponding spatial index → spatial-frequency scale)

The target is the circular encoding [sin θ, cos θ] (2 outputs), decoded with
atan2 so predictions always land in [-π, π].

Each collected VQE run thus contributes n_params training rows instead of one
sample, and predicting for a new size just evaluates the learnt "parameter
field" at that size's (L, q) positions.

Basin canonicalisation is applied per config key *before* exploding vectors
into rows — without it, rows coming from runs in different basins contradict
each other at identical features and the regression collapses to a meaningless
mean (same failure mode as in the per-key model, see BasinCanonicalizer).

Usage
─────
    from src.transfer_warmstart import TransferWarmStartPredictor

    pred = TransferWarmStartPredictor(
        data_paths=["data/warmstart_data_nx3_ny3_test.json"],
        include=lambda key: key.startswith("nx2_"),   # e.g. train on 2x2 only
    )
    pred.train()

    theta0 = pred.predict(nx=3, ny=3, n_layers=6, mode_type="TM", k=0,
                          plasma_density=0.0, target_eigenvalue=141158.0,
                          noise_std=0.0)

`predict` has the same signature as WarmStartPredictor.predict, so a trained
TransferWarmStartPredictor drops straight into WarmStartVQA(predictor=...).
"""

from __future__ import annotations

import json
import os
import re
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

KEY_RX = re.compile(r"nx(\d+)_ny(\d+)_nl(\d+)_mt(TM|TE)_k(\d+)")

# Feature layout of one training row (see module docstring):
#   0 nx   1 ny   2 mode_enc   3 k   4 log_density   5 log_target
#   6 layer_frac   7 qubit_frac   8 in_y_reg   9 sig_frac
N_GLOBAL = 6
N_STRUCT = 4


def _struct_coords(nx: int, ny: int, n_layers_eff: int,
                   L: int, q: int) -> List[float]:
    """Normalised structural coordinates of parameter θ[L·n + q]."""
    n = nx + ny
    layer_frac = L / max(n_layers_eff - 1, 1)
    qubit_frac = q / max(n - 1, 1)
    if q < nx:                       # x register: qubits 0..nx-1 (LSB first)
        in_y, pos, reg = 0.0, q, nx
    else:                            # y register: qubits nx..n-1
        in_y, pos, reg = 1.0, q - nx, ny
    sig_frac = pos / max(reg - 1, 1)
    return [layer_frac, qubit_frac, in_y, sig_frac]


def _rows_for_config(global_feats: Sequence[float],
                     nx: int, ny: int, n_layers_eff: int) -> np.ndarray:
    """Feature rows for every parameter position of one configuration,
    ordered exactly like the ansatz parameter vector (layer-major)."""
    n = nx + ny
    rows = [list(global_feats) + _struct_coords(nx, ny, n_layers_eff, L, q)
            for L in range(n_layers_eff) for q in range(n)]
    return np.array(rows, dtype=np.float64)


class TransferWarmStartPredictor:
    """One shared MLP over per-parameter structural coordinates.

    Pools every config key of every supplied data store into a single
    regression problem, so small-grid samples improve large-grid predictions
    (and vice versa).

    Parameters
    ──────────
    data_paths : list of JSON stores produced by WarmStartCollector
                 (7-feature layout, density_in_key=False).
    include    : optional predicate on the config key — return False to leave
                 a key out of training (e.g. to test pure cross-size transfer).
    canonicalize : apply BasinCanonicalizer per key before building rows
                 (strongly recommended; see module docstring).
    hidden     : MLP hidden-layer sizes.
    max_iter   : MLP training iterations.
    model_path : if given, the fitted (model, scaler) pair is persisted there
                 with joblib and re-loaded on construction when present.
    """

    def __init__(self,
                 data_paths: Sequence[str],
                 include: Optional[Callable[[str], bool]] = None,
                 canonicalize: bool = True,
                 hidden: Tuple[int, ...] = (128, 128),
                 max_iter: int = 4000,
                 model_path: Optional[str] = None,
                 random_state: int = 42):
        self.data_paths = list(data_paths)
        self.include = include
        self.canonicalize = canonicalize
        self.hidden = tuple(hidden)
        self.max_iter = max_iter
        self.model_path = model_path
        self.random_state = random_state

        self._model: Optional[MLPRegressor] = None
        self._scaler: Optional[StandardScaler] = None
        self.training_keys_: List[str] = []
        self.n_rows_: int = 0

        if model_path and os.path.exists(model_path):
            try:
                self._model, self._scaler, self.training_keys_ = \
                    joblib.load(model_path)
                print(f"  Loaded transfer model from '{model_path}' "
                      f"(trained on {len(self.training_keys_)} keys).")
            except Exception as exc:
                warnings.warn(f"Could not load '{model_path}': {exc}")

    # ── Dataset construction ──────────────────────────────────────────────────

    def _load_keys(self) -> Dict[str, Dict]:
        merged: Dict[str, Dict] = {}
        for path in self.data_paths:
            with open(path) as f:
                store = json.load(f)
            for key, data in store.items():
                if self.include is not None and not self.include(key):
                    continue
                if key not in merged:
                    merged[key] = {"X": [], "y": []}
                merged[key]["X"].extend(data.get("X", []))
                merged[key]["y"].extend(data.get("y", []))
        return merged

    def build_dataset(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Explode every stored (config, θ-vector) sample into per-parameter
        rows.  Returns (rows, targets) with targets = [sin θ, cos θ]."""
        rows_all: List[np.ndarray] = []
        tgts_all: List[np.ndarray] = []
        self.training_keys_ = []

        for key, data in self._load_keys().items():
            m = KEY_RX.match(key)
            if m is None:
                warnings.warn(f"  Skipping unparseable key '{key}'.")
                continue
            nx, ny, _, mt, k = (int(m[1]), int(m[2]), int(m[3]),
                                m[4], int(m[5]))
            n = nx + ny

            # Keep only the dominant (feature-length, param-length) shape.
            pairs = [(x, y) for x, y in zip(data["X"], data["y"])
                     if len(x) == 7 and len(y) % n == 0]
            if len(pairs) < 2:
                if verbose and data["X"]:
                    print(f"  '{key}': skipped ({len(pairs)} usable samples).")
                continue
            y_len = max({len(y) for _, y in pairs},
                        key=[len(y) for _, y in pairs].count)
            pairs = [(x, y) for x, y in pairs if len(y) == y_len]
            X_raw = np.array([x for x, _ in pairs])
            Y_raw = np.array([y for _, y in pairs])
            n_layers_eff = y_len // n

            # Collapse multimodal targets onto one basin per density.
            if self.canonicalize:
                # Imported lazily: the class definition is currently missing
                # from ml_warmstart_vqe (never committed), so importing it at
                # module level would break every consumer of this module even
                # when canonicalisation is not used.
                from .ml_warmstart_vqe import BasinCanonicalizer
                Y_raw = BasinCanonicalizer(density_col=5).fit(
                    X_raw, Y_raw).transform_dataset(X_raw, Y_raw)

            # After canonicalisation all runs at one feature point share one
            # target — keep a single copy per unique feature row.
            seen = {}
            for x, y in zip(X_raw, Y_raw):
                seen[tuple(np.round(x, 9))] = (x, y)

            for x, y in seen.values():
                # Stored layout: [nx, ny, n_layers, mode_enc, k, log_ne, log_λ]
                glob = [x[0], x[1], x[3], x[4], x[5], x[6]]
                rows_all.append(_rows_for_config(glob, nx, ny, n_layers_eff))
                tgts_all.append(np.column_stack([np.sin(y), np.cos(y)]))

            self.training_keys_.append(key)
            if verbose:
                print(f"  '{key}': {len(seen)} feature point(s) × "
                      f"{y_len} params → {len(seen) * y_len} rows.")

        if not rows_all:
            raise RuntimeError("No usable training data found.")
        X = np.vstack(rows_all)
        Y = np.vstack(tgts_all)
        self.n_rows_ = len(X)
        return X, Y

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, verbose: bool = True) -> "TransferWarmStartPredictor":
        X, Y = self.build_dataset(verbose=verbose)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            activation="tanh",
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=len(X) >= 500,
            validation_fraction=0.15,
            n_iter_no_change=100,
            learning_rate_init=1e-3,
            tol=1e-6,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(X_scaled, Y)

        self._model, self._scaler = mlp, scaler
        if self.model_path:
            joblib.dump((mlp, scaler, self.training_keys_), self.model_path)

        if verbose:
            print(f"\n  Trained shared model on {self.n_rows_} rows from "
                  f"{len(self.training_keys_)} key(s)  "
                  f"hidden={self.hidden}  R²={mlp.score(X_scaled, Y):.4f}")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self,
                nx: int, ny: int, n_layers: int,
                mode_type: str, k: int,
                plasma_density: float = 0.0,
                target_eigenvalue: Optional[float] = None,
                noise_std: float = 0.05,
                extra_layers_per_mode: int = 1) -> Optional[np.ndarray]:
        """Predict θ₀ for *any* grid size (same signature as
        WarmStartPredictor.predict, so WarmStartVQA accepts this class).

        extra_layers_per_mode must match the solver's setting so the predicted
        vector has length n · (n_layers + extra_layers_per_mode · max(0, k)).
        """
        if self._model is None:
            return None
        if target_eigenvalue is None:
            raise ValueError(
                "target_eigenvalue is required (k-th physical eigenvalue of "
                "np.linalg.eigvalsh(solver.M_dense)).")

        n_layers_eff = n_layers + extra_layers_per_mode * max(0, k)
        mode_enc = 0.0 if mode_type == "TM" else 1.0
        log_ne = np.log10(1.0 + max(plasma_density, 0.0))
        glob = [nx, ny, mode_enc, k, log_ne, np.log10(target_eigenvalue)]

        rows = _rows_for_config(glob, nx, ny, n_layers_eff)
        enc = self._model.predict(self._scaler.transform(rows))
        params = np.arctan2(enc[:, 0], enc[:, 1])

        if noise_std > 0.0:
            params = params + np.random.normal(0, noise_std, size=params.shape)
        return params

    def is_trained(self, *args, **kwargs) -> bool:
        """A single shared model serves every configuration."""
        return self._model is not None


# ══════════════════════════════════════════════════════════════════════════════
# State-space transfer: multigrid prolongation + fidelity fit
# ══════════════════════════════════════════════════════════════════════════════

class ProlongationWarmStart:
    """Cross-size warm start through *state space* instead of parameter space.

    Raw HEA angles do not transfer across qubit counts (angles compose through
    a size-specific CX chain), but the physical mode field does: the 8×8
    TM/TE mode is a finer sampling of the same continuum eigenfunction as the
    4×4 one.  This class therefore

      1. takes a small-grid solution state (e.g. a stored 2x2 optimum),
      2. prolongs its real-space field to the large grid by bilinear
         interpolation on the cell-centred mesh (multigrid prolongation),
      3. fits the large-grid ansatz parameters θ by maximising the statevector
         fidelity |⟨φ_target | ψ(θ)⟩|² — a purely classical optimisation that
         costs one small statevector simulation per evaluation (no Estimator
         pubs, no orthogonality penalty),

    and returns the fitted θ as the warm start for the actual VQE.

    Parameters
    ──────────
    maxiter    : L-BFGS-B iterations per fidelity fit.
    n_restarts : random restarts of the fidelity fit (it is multimodal too,
                 but each restart costs only seconds).
    seed       : RNG seed for the restarts.
    """

    def __init__(self, maxiter: int = 300, n_restarts: int = 4, seed: int = 0):
        self.maxiter = maxiter
        self.n_restarts = n_restarts
        self.seed = seed

    # ── Prolongation ──────────────────────────────────────────────────────────

    @staticmethod
    def prolong_field(field_small: np.ndarray,
                      Ny_big: int, Nx_big: int) -> np.ndarray:
        """Bilinear prolongation between cell-centred grids on [0,1]²."""
        from scipy.interpolate import RegularGridInterpolator

        Ny_s, Nx_s = field_small.shape
        ys = (np.arange(Ny_s) + 0.5) / Ny_s
        xs = (np.arange(Nx_s) + 0.5) / Nx_s
        interp = RegularGridInterpolator(
            (ys, xs), field_small, method="linear",
            bounds_error=False, fill_value=None)   # nearest extrapolation at edges

        yb = (np.arange(Ny_big) + 0.5) / Ny_big
        xb = (np.arange(Nx_big) + 0.5) / Nx_big
        YY, XX = np.meshgrid(yb, xb, indexing="ij")
        return interp(np.stack([YY.ravel(), XX.ravel()], axis=-1)
                      ).reshape(Ny_big, Nx_big)

    # ── Fidelity fit ──────────────────────────────────────────────────────────

    def fit_params_to_state(self, solver, k: int,
                            target_state: np.ndarray,
                            verbose: bool = False
                            ) -> Tuple[np.ndarray, float]:
        """Return (θ, fidelity) maximising |⟨target | ψ(θ)⟩|² for solver's
        ansatz at mode k."""
        from qiskit.quantum_info import Statevector
        from scipy.optimize import minimize

        phi = np.asarray(target_state, dtype=complex)
        phi = phi / np.linalg.norm(phi)
        n_params = solver.n * solver._effective_layers(k)
        rng = np.random.default_rng(self.seed)

        def neg_fid(theta):
            psi = Statevector(solver.ansatz(theta, k)).data
            return -abs(np.vdot(phi, psi)) ** 2

        best_theta, best_fid = None, -1.0
        for _ in range(self.n_restarts):
            theta0 = rng.uniform(-np.pi, np.pi, n_params)
            res = minimize(neg_fid, theta0, method="L-BFGS-B",
                           options={"maxiter": self.maxiter, "ftol": 1e-12})
            if -res.fun > best_fid:
                best_fid, best_theta = -res.fun, res.x
            if best_fid > 0.999:
                break
        if verbose:
            print(f"    fidelity fit: |⟨φ|ψ(θ)⟩|² = {best_fid:.4f} "
                  f"({n_params} params)")
        return best_theta, best_fid

    # ── End-to-end prediction ─────────────────────────────────────────────────

    def predict(self, solver_big, k: int,
                params_small: np.ndarray, solver_small,
                noise_std: float = 0.0,
                verbose: bool = False) -> Tuple[np.ndarray, float]:
        """Warm start for solver_big from a small-grid optimised parameter
        vector.  Returns (θ₀, fidelity of the fit to the prolonged target).
        """
        from qiskit.quantum_info import Statevector

        state_s = Statevector(solver_small.ansatz(params_small, k)).data
        field_s = np.real(state_s).reshape(2 ** solver_small.ny,
                                           2 ** solver_small.nx)
        field_b = self.prolong_field(field_s,
                                     2 ** solver_big.ny, 2 ** solver_big.nx)
        target = field_b.ravel().astype(complex)

        theta, fid = self.fit_params_to_state(solver_big, k, target,
                                              verbose=verbose)
        if noise_std > 0.0:
            theta = theta + np.random.normal(0, noise_std, size=theta.shape)
        return theta, fid
