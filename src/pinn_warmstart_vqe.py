"""
pinn_warmstart_vqe.py
────────────────────────────────────────────────────────────────────────────────
Physics-Informed Neural-Network (PINN) Warm-Start for the ColdPlasma VQE
────────────────────────────────────────────────────────────────────────────────

A *third* warm-start strategy for ``WaveguideModeVQA``, sitting alongside the
supervised MLP (``ml_warmstart_vqe.py``) and the cross-grid transfer model
(``transfer_warmstart.py``).

What makes it "physics-informed"
────────────────────────────────
``WarmStartPredictor`` is **supervised**: it first runs VQE many times to harvest
``(features → optimised θ)`` label pairs, then regresses on them.  The labels are
expensive (every one is a full VQE solve) and the network never sees the physics
directly — it only mimics whatever the optimiser happened to land on.

This module trains the network with **no labels at all**.  The loss *is* the
variational eigenvalue functional (the Rayleigh quotient):

        θ₀ = NN(features) ,   features = (nx, ny, n_layers, mode, k,
                                          log₁₀ Nₑ, log₁₀ λ_target)

        L  =  mean over a batch of plasma densities of
                  ⟨ψ(θ₀)|M_dense|ψ(θ₀)⟩ / λ_target            (mode k = 0)
              + β · Σ_{i: λ_i < λ_target} |⟨v_i|ψ(θ₀)⟩|²       (excited modes)

where ``M_dense`` and the lower eigenvectors ``v_i`` are the *same* single source
of truth used everywhere else in the repo (``np.linalg.eigvalsh(M_dense)``).  The
network is therefore literally taught to minimise the physical energy.

Because the ansatz is a single-``Ry``-per-parameter ``Ry``+``CX`` circuit, the cost
is an exact sinusoid in every angle, so ``∂L/∂θ`` is obtained from the *exact*
parameter-shift rule (no finite differences, no autodiff framework needed).  The
gradient is then back-propagated through a small hand-written NumPy MLP.

Trained across a *range* of densities on one grid, the PINN becomes an **amortised
optimiser**: a single forward pass predicts a good θ₀ for a brand-new density,
with no VQE run required.

Drop-in compatibility
─────────────────────
``PINNWarmStartPredictor`` mirrors the public interface of
``ml_warmstart_vqe.WarmStartPredictor`` (``.predict`` / ``.is_trained`` with the
same signature), so it plugs straight into the existing helpers::

    from src import WaveguideModeVQA
    from src.ml_warmstart_vqe import WarmStartVQA, benchmark_warmstart
    from src.pinn_warmstart_vqe import PINNWarmStartPredictor

    pinn = PINNWarmStartPredictor(model_dir="pinn_models")
    pinn.train(specs=[(2, 2, 2, "TM", 0, ne) for ne in densities])

    # compare against random init with the *existing* benchmark
    benchmark_warmstart(pinn, nx=2, ny=2, n_layers=2, mode_type="TM", k=0,
                        plasma_density=1e17)

    # or solve with the warm start baked in
    solver = WarmStartVQA(nx=2, ny=2, n_layers=2, mode_type="TM",
                          plasma_density=1e17, predictor=pinn)
    solver.optimize_mode(k=0)

Author : Juan Manuel (+ Claude)
License: MIT
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from .coldplasma_vqe_waveguide import WaveguideModeVQA
# Reuse the *exact* feature engineering of the supervised warm start so the two
# methods are directly comparable and share one model-key convention.
from .ml_warmstart_vqe import _build_features, _config_key


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Differentiable statevector simulator for the Ry + CX ansatz
# ══════════════════════════════════════════════════════════════════════════════
#
# A tiny NumPy reimplementation of WaveguideModeVQA.ansatz that returns the same
# statevector as qiskit's Statevector (verified numerically — see
# tests/test_pinn_warmstart.py).  It exists so the physics energy ⟨ψ|M|ψ⟩ and its
# parameter-shift gradient can be evaluated thousands of times during training
# without paying qiskit's circuit-build overhead.
#
# Convention: the flat amplitude index i encodes qubit q in bit q (qubit 0 = LSB),
# exactly as qiskit.  Reshaping the flat vector to (2,)*n therefore maps axis a to
# qubit (n-1-a); we keep the work on the flat vector and reshape locally.

@lru_cache(maxsize=None)
def _cx_perm(n: int, control: int, target: int) -> Tuple[int, ...]:
    """Permutation σ such that (CX·ψ)_j = ψ_{σ(j)} for CX(control→target).

    CX is its own inverse, so the permutation matrix P = Pᵀ = P⁻¹ and the action
    is a pure gather on the flat statevector.
    """
    N = 1 << n
    perm = np.arange(N)
    cbit = 1 << control
    tbit = 1 << target
    flip = (perm & cbit) != 0          # where the control qubit is set
    perm = np.where(flip, perm ^ tbit, perm)
    return tuple(int(x) for x in perm)


def _apply_ry(psi: np.ndarray, n: int, qubit: int, theta: float) -> np.ndarray:
    """Apply Ry(theta) on `qubit` to the flat statevector `psi` (length 2ⁿ)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    # Ry = [[c, -s], [s, c]]; reshape so the rotated qubit is the middle axis.
    hi = 1 << (n - 1 - qubit)
    lo = 1 << qubit
    v = psi.reshape(hi, 2, lo)
    a = v[:, 0, :]
    b = v[:, 1, :]
    out = np.empty_like(v)
    out[:, 0, :] = c * a - s * b
    out[:, 1, :] = s * a + c * b
    return out.reshape(-1)


def ansatz_statevector(params: np.ndarray, n: int, n_layers: int,
                       ansatz_type: str = "HEA") -> np.ndarray:
    """Statevector of the HEA/ALT ansatz, identical to WaveguideModeVQA.ansatz.

    Parameters are layer-major: ``params[L*n + q]`` is the Ry angle on qubit q in
    layer L.  Returns a complex flat array of length 2ⁿ in qiskit ordering.
    """
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0
    idx = 0
    for _ in range(n_layers):
        for q in range(n):
            psi = _apply_ry(psi, n, q, params[idx])
            idx += 1
        if ansatz_type == "HEA":
            for q in range(n - 1):
                psi = psi[np.asarray(_cx_perm(n, q, q + 1))]
        else:  # 'ALT' — alternating even/odd entanglement
            for q in range(0, n - 1, 2):
                psi = psi[np.asarray(_cx_perm(n, q, q + 1))]
            for q in range(1, n - 1, 2):
                psi = psi[np.asarray(_cx_perm(n, q, q + 1))]
    return psi


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Physics energy and its exact parameter-shift gradient
# ══════════════════════════════════════════════════════════════════════════════

class _Problem:
    """A single training instance: everything needed to score a θ on one config.

    Bundles the dense operator (source of truth), the lower eigenvectors used for
    the orthogonality penalty, the normalising eigenvalue and the ansatz shape.
    """

    __slots__ = ("M", "pen_vecs", "lam", "n", "n_layers", "ansatz_type",
                 "n_params", "beta", "feat")

    def __init__(self, M, pen_vecs, lam, n, n_layers, ansatz_type, beta, feat):
        self.M = M
        self.pen_vecs = pen_vecs        # (m, N) real, eigenvectors below lam
        self.lam = lam                  # normalising target eigenvalue
        self.n = n
        self.n_layers = n_layers
        self.ansatz_type = ansatz_type
        self.n_params = n * n_layers
        self.beta = beta
        self.feat = feat                # cached feature vector

    def energy(self, theta: np.ndarray) -> float:
        """Normalised variational energy L(θ) = ⟨ψ|M|ψ⟩/λ + penalty/λ."""
        psi = ansatz_statevector(theta, self.n, self.n_layers, self.ansatz_type)
        e = np.real(np.vdot(psi, self.M @ psi))
        if self.pen_vecs is not None and len(self.pen_vecs):
            overlaps = self.pen_vecs @ psi          # complex (m,)
            e += self.beta * np.sum(np.abs(overlaps) ** 2)
        return e / self.lam

    def grad(self, theta: np.ndarray) -> Tuple[np.ndarray, float]:
        """Exact ∂L/∂θ via the parameter-shift rule; also returns L(θ).

        Each angle is a single Ry generator (eigenvalues ±1/2), so
            ∂L/∂θ_i = ½[L(θ + (π/2)e_i) − L(θ − (π/2)e_i)]
        is exact (no truncation error).
        """
        g = np.empty_like(theta)
        shift = np.pi / 2.0
        for i in range(len(theta)):
            tp = theta.copy(); tp[i] += shift
            tm = theta.copy(); tm[i] -= shift
            g[i] = 0.5 * (self.energy(tp) - self.energy(tm))
        return g, self.energy(theta)


def build_problem(nx: int, ny: int, n_layers: int, mode_type: str, k: int,
                  plasma_density: float, ansatz_type: str, beta: float,
                  Lx: float = 0.015, Ly: float = 0.010
                  ) -> Tuple[_Problem, float]:
    """Instantiate a solver and bundle a ``_Problem`` for one configuration.

    Harvests ``M_dense`` (the repo-wide source of truth), applies the physical
    -mode filter (eigs > 0 for TM, > 1 for TE) to locate the k-th target
    eigenvalue, and collects every eigenvector strictly below it for the
    orthogonality penalty.  Returns ``(problem, target_eigenvalue)``.

    ``n_layers`` here is the *effective* depth of the ansatz being trained
    (i.e. already including any ``extra_layers_per_mode`` contribution).
    """
    solver = WaveguideModeVQA(
        nx=nx, ny=ny, n_layers=n_layers, mode_type=mode_type,
        ansatz_type=ansatz_type, Lx=Lx, Ly=Ly,
        plasma_density=plasma_density,
    )
    M = solver.M_dense
    eigvals, eigvecs = np.linalg.eigh(M)

    # Physical-mode filter, identical to the rest of the repo.
    eig_threshold = 0.0 if mode_type == "TM" else 1.0
    physical = eigvals[eigvals > eig_threshold]
    if k >= len(physical):
        raise ValueError(
            f"k={k} exceeds the {len(physical)} physical {mode_type} modes "
            f"on a {2**nx}×{2**ny} grid."
        )
    target_eig = float(physical[k])

    # Orthogonality penalty: every eigenvector strictly below the target,
    # which automatically includes the spurious sub-threshold TE mode.
    below = eigvals < target_eig - 1e-9 * abs(target_eig)
    pen_vecs = eigvecs[:, below].T.conj().copy() if below.any() else None

    prob = _Problem(M, pen_vecs, target_eig, nx + ny, n_layers,
                    ansatz_type, beta, None)
    return prob, target_eig


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Minimal NumPy MLP with manual back-prop and Adam
# ══════════════════════════════════════════════════════════════════════════════

class _MLP:
    """Fully-connected tanh network  features → θ ∈ (−π, π).

    The output layer is squashed by ``π·tanh`` so predictions always lie on the
    angle torus regardless of how the net extrapolates (same idea as the circular
    encoding in the supervised predictor, here enforced directly on the output).
    """

    def __init__(self, n_in: int, n_out: int,
                 hidden: Sequence[int] = (64, 64), seed: int = 0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            # Xavier/Glorot init — appropriate for tanh activations.
            scale = np.sqrt(6.0 / (a + b))
            self.W.append(rng.uniform(-scale, scale, size=(a, b)))
            self.b.append(np.zeros(b))
        # Adam moments
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    # ── forward / backward ────────────────────────────────────────────────────

    def forward(self, X: np.ndarray):
        """X: (B, n_in) → θ: (B, n_out). Caches activations for backward()."""
        acts = [X]
        zs = []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            zs.append(z)
            h = np.tanh(z)          # hidden layers and (pre-scale) output both tanh
            acts.append(h)
        theta = np.pi * h           # squash final tanh output into (−π, π)
        self._cache = (acts, zs)
        return theta

    def backward(self, dtheta: np.ndarray):
        """Back-prop dL/dθ (B, n_out) → per-parameter gradients."""
        acts, zs = self._cache
        B = dtheta.shape[0]
        # d/dz of last layer:  θ = π·tanh(z_last) ⇒ dθ/dz = π(1 − tanh²)
        delta = dtheta * np.pi * (1.0 - np.tanh(zs[-1]) ** 2)
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta / B
            gb[i] = delta.mean(axis=0)
            if i > 0:
                # propagate through tanh of the previous hidden layer
                delta = (delta @ self.W[i].T) * (1.0 - np.tanh(zs[i - 1]) ** 2)
        return gW, gb

    def adam_step(self, gW, gb, lr: float,
                  b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self._t += 1
        t = self._t
        for i in range(len(self.W)):
            for m, v, g, P in ((self._mW, self._vW, gW, self.W),
                               (self._mb, self._vb, gb, self.b)):
                m[i] = b1 * m[i] + (1 - b1) * g[i]
                v[i] = b2 * v[i] + (1 - b2) * (g[i] ** 2)
                mhat = m[i] / (1 - b1 ** t)
                vhat = v[i] / (1 - b2 ** t)
                P[i] -= lr * mhat / (np.sqrt(vhat) + eps)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  The predictor — one PINN per configuration key
# ══════════════════════════════════════════════════════════════════════════════

class PINNWarmStartPredictor:
    """Trains and serves one physics-informed network per config key.

    A "config key" is ``nx{}_ny{}_nl{}_mt{}_k{}`` (density excluded by default), so
    a single network is shared across plasma densities on a given grid — that is
    exactly the axis the PINN learns to generalise over.

    Public interface mirrors ``ml_warmstart_vqe.WarmStartPredictor`` so the two are
    interchangeable in ``benchmark_warmstart`` and ``WarmStartVQA``.

    Parameters
    ──────────
    hidden        : hidden-layer widths of the MLP.
    lr            : Adam learning rate.
    epochs        : full-batch training epochs per key.
    beta          : orthogonality-penalty weight for excited modes (matches VQE).
    model_dir     : if given, trained nets are serialised here with joblib.
    density_in_key: if True, every density gets its own network (defeats the
                    amortisation; off by default).
    seed          : RNG seed for weight init and prediction noise.
    """

    def __init__(self,
                 hidden: Sequence[int] = (64, 64),
                 lr: float = 5e-3,
                 epochs: int = 600,
                 beta: float = 4e5,
                 Lx: float = 0.015, Ly: float = 0.010,
                 model_dir: Optional[str] = None,
                 density_in_key: bool = False,
                 seed: int = 0):
        self.hidden = tuple(hidden)
        self.lr = lr
        self.epochs = epochs
        self.beta = beta
        self.Lx = Lx
        self.Ly = Ly
        self.model_dir = model_dir
        self.density_in_key = density_in_key
        self.seed = seed

        self._models: Dict[str, _MLP] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self._trained: Dict[str, bool] = {}
        self._n_params: Dict[str, int] = {}
        # Per-key training-loss history (mean normalised energy per epoch).
        self.loss_history: Dict[str, List[float]] = {}

        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            self._try_load_models()

    # ── persistence ───────────────────────────────────────────────────────────

    def _bundle_path(self, key: str) -> str:
        assert self.model_dir
        return os.path.join(self.model_dir, f"{key}_pinn.joblib")

    def _save_model(self, key: str) -> None:
        if not self.model_dir:
            return
        joblib.dump(
            {"mlp": self._models[key],
             "scaler": self._scalers[key],
             "n_params": self._n_params[key],
             "loss_history": self.loss_history.get(key, [])},
            self._bundle_path(key),
        )

    def _try_load_models(self) -> None:
        if not self.model_dir or not os.path.isdir(self.model_dir):
            return
        for fname in os.listdir(self.model_dir):
            if not fname.endswith("_pinn.joblib"):
                continue
            key = fname[: -len("_pinn.joblib")]
            try:
                bundle = joblib.load(os.path.join(self.model_dir, fname))
                self._models[key] = bundle["mlp"]
                self._scalers[key] = bundle["scaler"]
                self._n_params[key] = bundle["n_params"]
                self.loss_history[key] = bundle.get("loss_history", [])
                self._trained[key] = True
                print(f"  Loaded PINN for '{key}'")
            except Exception as exc:                       # pragma: no cover
                warnings.warn(f"Could not load PINN for {key}: {exc}")

    # ── build a training instance from a solver config ─────────────────────────

    def _make_problem(self, nx, ny, n_layers, mode_type, k,
                      plasma_density, ansatz_type) -> Tuple[str, _Problem]:
        """Instantiate a solver, harvest M_dense and the physical target, and
        bundle a ``_Problem`` plus its feature vector."""
        prob, target_eig = build_problem(nx, ny, n_layers, mode_type, k,
                                         plasma_density, ansatz_type,
                                         self.beta, Lx=self.Lx, Ly=self.Ly)
        prob.feat = _build_features(nx, ny, n_layers, mode_type, k,
                                    plasma_density,
                                    target_eigenvalue=target_eig,
                                    density_in_key=self.density_in_key)
        key = _config_key(nx, ny, n_layers, mode_type, k,
                          plasma_density=plasma_density,
                          density_in_key=self.density_in_key)
        return key, prob

    # ── training ───────────────────────────────────────────────────────────────

    def train(self,
              specs: Sequence[Tuple],
              ansatz_type: str = "HEA",
              verbose: bool = True) -> None:
        """Train one PINN per config key from a list of problem specs.

        Parameters
        ──────────
        specs : sequence of tuples
            Each tuple is ``(nx, ny, n_layers, mode_type, k, plasma_density)``.
            Specs that share a config key (same grid/mode/k, different density)
            are pooled into one network — that is the intended usage.
        ansatz_type : 'HEA' (default) or 'ALT'; must match the solver you will
            warm-start.
        """
        # Group problems by config key.
        groups: Dict[str, List[_Problem]] = {}
        for spec in specs:
            nx, ny, n_layers, mode_type, k, plasma_density = spec
            key, prob = self._make_problem(nx, ny, n_layers, mode_type, k,
                                           plasma_density, ansatz_type)
            groups.setdefault(key, []).append(prob)

        for key, probs in groups.items():
            n_params = probs[0].n_params
            n_in = len(probs[0].feat)

            # Standardise the input features across this key's training set.
            X_raw = np.array([p.feat for p in probs], dtype=np.float64)
            scaler = StandardScaler().fit(X_raw)
            X = scaler.transform(X_raw)

            mlp = _MLP(n_in, n_params, hidden=self.hidden, seed=self.seed)
            history: List[float] = []

            for epoch in range(self.epochs):
                theta = mlp.forward(X)                      # (B, n_params)
                # Per-sample physics gradient via exact parameter shift.
                dtheta = np.empty_like(theta)
                batch_loss = 0.0
                for i, prob in enumerate(probs):
                    g, e = prob.grad(theta[i])
                    dtheta[i] = g
                    batch_loss += e
                batch_loss /= len(probs)
                history.append(batch_loss)

                gW, gb = mlp.backward(dtheta)
                mlp.adam_step(gW, gb, self.lr)

                if verbose and (epoch % max(1, self.epochs // 10) == 0
                                or epoch == self.epochs - 1):
                    print(f"  [{key}] epoch {epoch+1:4d}/{self.epochs}  "
                          f"mean L = {batch_loss:.5f}")

            self._models[key] = mlp
            self._scalers[key] = scaler
            self._n_params[key] = n_params
            self._trained[key] = True
            self.loss_history[key] = history
            self._save_model(key)

            if verbose:
                print(f"  Trained PINN '{key}'  N={len(probs)}  "
                      f"n_params={n_params}  final L = {history[-1]:.5f}")

        print(f"\nPINN training complete.  {len(self._trained)} network(s) ready.")

    # ── prediction (signature-compatible with WarmStartPredictor) ──────────────

    def predict(self,
                nx: int, ny: int, n_layers: int,
                mode_type: str, k: int,
                plasma_density: float = 0.0,
                target_eigenvalue: Optional[float] = None,
                noise_std: float = 0.0) -> Optional[np.ndarray]:
        """Predict an initial parameter vector θ₀ for one configuration.

        Returns ``None`` (so the caller falls back to random init) when no network
        has been trained for this config key.  ``target_eigenvalue`` is required —
        it is part of the feature vector — and is obtained from
        ``np.linalg.eigvalsh(solver.M_dense)`` exactly as for the supervised model.
        """
        key = _config_key(nx, ny, n_layers, mode_type, k,
                          plasma_density=plasma_density,
                          density_in_key=self.density_in_key)
        if not self._trained.get(key, False):
            return None
        if target_eigenvalue is None:
            raise ValueError(
                "target_eigenvalue is required.  Compute via "
                "np.sort(np.linalg.eigvalsh(solver.M_dense)) and select the k-th "
                "physical eigenvalue (>0 for TM, >1 for TE)."
            )

        feat = _build_features(nx, ny, n_layers, mode_type, k, plasma_density,
                               target_eigenvalue=target_eigenvalue,
                               density_in_key=self.density_in_key).reshape(1, -1)
        X = self._scalers[key].transform(feat)
        theta = self._models[key].forward(X)[0]

        if noise_std > 0.0:
            rng = np.random.default_rng()
            theta = theta + rng.normal(0.0, noise_std, size=theta.shape)
        return theta

    def is_trained(self, nx: int, ny: int, n_layers: int,
                   mode_type: str, k: int, plasma_density: float = 0.0) -> bool:
        """True if a network is available for this configuration."""
        return self._trained.get(
            _config_key(nx, ny, n_layers, mode_type, k,
                        plasma_density=plasma_density,
                        density_in_key=self.density_in_key),
            False,
        )

    def trained_keys(self) -> List[str]:
        return [k for k, v in self._trained.items() if v]
