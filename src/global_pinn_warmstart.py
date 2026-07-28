"""
global_pinn_warmstart.py
────────────────────────────────────────────────────────────────────────────────
Global (size-transferable) Physics-Informed Warm-Start
────────────────────────────────────────────────────────────────────────────────

One network for *every* qubit configuration.  This module merges the two ideas
that each solved half of the problem:

* ``transfer_warmstart.TransferWarmStartPredictor`` made the network shape
  independent of grid size by predicting **one angle at a time** from
  ``(global config features, normalised structural coordinates of that
  parameter)`` — but it was *supervised*, regressing on harvested VQE labels,
  and zero-shot parameter transfer across qubit counts failed.

* ``pinn_warmstart_vqe.PINNWarmStartPredictor`` needed **no labels** — the loss
  is the physical Rayleigh quotient ⟨ψ(θ)|M|ψ(θ)⟩/λ_target, differentiated with
  the exact parameter-shift rule — but trained one net per config key, so its
  output dimension was locked to one grid size.

``GlobalPINNPredictor`` uses the per-angle row representation of the former and
the label-free physics loss of the latter: a single shared MLP maps

    (nx, ny, mode_enc, k, log₁₀ Nₑ, log₁₀ λ_target,          ← global features
     layer_frac, qubit_frac, in_y_reg, sig_frac)             ← per-angle coords
        →  θ_{L,q}

and is trained jointly on problems spanning *several grid sizes* by
back-propagating each problem's exact physics gradient ∂L/∂θ into its parameter
rows.  A trained instance therefore predicts a θ₀ of the right length for ANY
``(nx, ny, n_layers, mode, k, density)`` — including configurations never seen
in training.

The training loss is the Rayleigh quotient plus a small *basin guide*
``guide_weight·(1 − |⟨v_target|ψ⟩|²)`` on the training grids (see
``_GuidedProblem``): the pure energy loss has zero-gradient excited-state traps
that the shared network was observed to fall into for some grids.  The guide
uses only train-time classical data that the pipeline computes anyway;
prediction for an unseen grid needs nothing beyond the scalar λ_target.

Three prediction strategies for unseen configurations, weakest to strongest:

1. ``predict(...)``                 — direct forward pass (parameter-space
   extrapolation; cheap but historically the fragile axis across qubit counts).
2. ``predict(..., polish_steps=m)`` — direct prediction refined by ``m``
   L-BFGS-B iterations on the exact physics energy (still no VQE run, no
   labels; costs ~``2·m·n_params`` classical energy evaluations).
3. ``predict_prolonged(...)``       — the known-good state-space route: predict
   on the largest *trained* grid, prolong the real-space field to the target
   grid (multigrid bilinear interpolation, reusing
   ``ProlongationWarmStart.prolong_field``), then fit the target ansatz to the
   prolonged state by fidelity maximisation.

``predict`` / ``is_trained`` keep the ``WarmStartPredictor`` signature, so a
trained ``GlobalPINNPredictor`` drops straight into ``WarmStartVQA`` and
``benchmark_warmstart``.

Usage
─────
    from src.global_pinn_warmstart import GlobalPINNPredictor

    densities = [0.0, 1e16, 1e17, 1e18, 1e19]
    specs = [(nx, ny, 3, "TM", 0, ne)
             for (nx, ny) in [(2, 2), (2, 3), (3, 2), (3, 3)]
             for ne in densities]

    gp = GlobalPINNPredictor(model_path="pinn_models/global_pinn.joblib")
    gp.train(specs)

    # zero-shot: (3, 4) was never trained
    theta0 = gp.predict(nx=3, ny=4, n_layers=3, mode_type="TM", k=0,
                        plasma_density=5e17, target_eigenvalue=lam_34)

Author : Juan Manuel (+ Claude)
License: MIT
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from .pinn_warmstart_vqe import _MLP, _Problem, ansatz_statevector, build_problem
from .transfer_warmstart import _rows_for_config, ProlongationWarmStart


class _GuidedProblem(_Problem):
    """``_Problem`` plus a train-time basin guide.

    The pure Rayleigh-quotient loss has excited-state basins where the energy
    gradient is (near-)zero and the state is exactly orthogonal to the target
    mode — a symmetry trap.  When one shared network serves many grids, it can
    settle into such a basin for some of them (observed for 3x2/3x3 in the
    first zero-shot evaluation).  The guide term

        guide_w · (1 − |⟨v_target|ψ(θ)⟩|²)

    disambiguates the basin.  ``v_target`` comes from the same
    ``np.linalg.eigh(M_dense)`` that already supplies ``λ_target`` and the
    orthogonality ``pen_vecs``, and is used at TRAINING time on the (small,
    classically cheap) training grids only — prediction for an unseen grid
    never touches that grid's eigenvectors, only the scalar λ.  The loss stays
    an expectation of a Hermitian operator, so the parameter-shift gradient
    remains exact.
    """

    __slots__ = ("guide_vec", "guide_w")

    def __init__(self, prob: _Problem, guide_vec: np.ndarray, guide_w: float):
        _Problem.__init__(self, prob.M, prob.pen_vecs, prob.lam, prob.n,
                          prob.n_layers, prob.ansatz_type, prob.beta, None)
        self.guide_vec = guide_vec
        self.guide_w = guide_w

    def energy(self, theta: np.ndarray) -> float:
        psi = ansatz_statevector(theta, self.n, self.n_layers,
                                 self.ansatz_type)
        e = np.real(np.vdot(psi, self.M @ psi))
        if self.pen_vecs is not None and len(self.pen_vecs):
            e += self.beta * np.sum(np.abs(self.pen_vecs @ psi) ** 2)
        e = e / self.lam
        if self.guide_w > 0.0:
            e += self.guide_w * (1.0 - np.abs(np.vdot(self.guide_vec, psi)) ** 2)
        return e

    def energy_pure(self, theta: np.ndarray) -> float:
        """Un-guided normalised energy (the metric everything is judged on)."""
        return _Problem.energy(self, theta)


def _global_feats(nx: int, ny: int, mode_type: str, k: int,
                  plasma_density: float, target_eigenvalue: float
                  ) -> List[float]:
    """Size-comparable global features — same layout as transfer_warmstart
    (N_GLOBAL = 6): nx, ny, mode_enc, k, log₁₀(1+Nₑ), log₁₀ λ_target."""
    mode_enc = 0.0 if mode_type == "TM" else 1.0
    log_ne = np.log10(1.0 + max(plasma_density, 0.0))
    return [float(nx), float(ny), mode_enc, float(k),
            log_ne, np.log10(target_eigenvalue)]


class GlobalPINNPredictor:
    """A single physics-informed network shared across all grid sizes.

    Parameters
    ──────────
    hidden       : hidden-layer widths of the shared MLP (default (128, 128) —
                   larger than the per-key PINN because one net serves every
                   configuration).
    lr           : Adam learning rate.
    epochs       : full-batch training epochs.
    beta         : orthogonality-penalty weight for excited modes (matches VQE).
    guide_weight : weight of the train-time basin guide (see ``_GuidedProblem``;
                   0 disables it and recovers the pure Rayleigh-quotient loss,
                   which was observed to strand some grids in excited-state
                   basins).
    extra_layers_per_mode : must match the solver's setting so predicted
                   vectors have length ``n·(n_layers + extra·max(0,k))``.
    model_path   : if given, the trained bundle is persisted there with joblib
                   and re-loaded on construction when present.
    seed         : RNG seed for weight init.
    """

    def __init__(self,
                 hidden: Sequence[int] = (128, 128),
                 lr: float = 3e-3,
                 epochs: int = 800,
                 beta: float = 4e5,
                 guide_weight: float = 2.0,
                 extra_layers_per_mode: int = 1,
                 Lx: float = 0.015, Ly: float = 0.010,
                 model_path: Optional[str] = None,
                 seed: int = 0):
        self.hidden = tuple(hidden)
        self.lr = lr
        self.epochs = epochs
        self.beta = beta
        self.guide_weight = guide_weight
        self.extra_layers_per_mode = extra_layers_per_mode
        self.Lx = Lx
        self.Ly = Ly
        self.model_path = model_path
        self.seed = seed

        self._mlp: Optional[_MLP] = None
        self._scaler: Optional[StandardScaler] = None
        self.trained_specs_: List[Tuple] = []     # (nx, ny, nl, mt, k, ne)
        self.loss_history: List[float] = []
        # Per-problem loss at the final epoch, keyed by spec — useful to check
        # that no single grid dominated or was left behind.
        self.final_losses_: Dict[Tuple, float] = {}

        if model_path and os.path.exists(model_path):
            try:
                bundle = joblib.load(model_path)
                self._mlp = bundle["mlp"]
                self._scaler = bundle["scaler"]
                self.trained_specs_ = bundle["trained_specs"]
                self.loss_history = bundle.get("loss_history", [])
                self.final_losses_ = bundle.get("final_losses", {})
                print(f"  Loaded global PINN from '{model_path}' "
                      f"({len(self.trained_specs_)} training specs).")
            except Exception as exc:               # pragma: no cover
                warnings.warn(f"Could not load '{model_path}': {exc}")

    # ── persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        if not self.model_path:
            return
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        joblib.dump(
            {"mlp": self._mlp, "scaler": self._scaler,
             "trained_specs": self.trained_specs_,
             "loss_history": self.loss_history,
             "final_losses": self.final_losses_},
            self.model_path,
        )

    # ── training ──────────────────────────────────────────────────────────────

    def _effective_layers(self, n_layers: int, k: int) -> int:
        return n_layers + self.extra_layers_per_mode * max(0, k)

    def train(self,
              specs: Sequence[Tuple],
              ansatz_type: str = "HEA",
              verbose: bool = True) -> "GlobalPINNPredictor":
        """Train the shared network on a pool of problems.

        Parameters
        ──────────
        specs : sequence of ``(nx, ny, n_layers, mode_type, k, plasma_density)``
            tuples.  Unlike the per-key PINN there is no grouping — every spec
            contributes its parameter rows to ONE joint regression, so specs
            should span several grid sizes for the model to learn (and be
            tested on) size transfer.
        """
        probs: List[_GuidedProblem] = []
        row_blocks: List[np.ndarray] = []
        for spec in specs:
            nx, ny, n_layers, mode_type, k, ne = spec
            nl_eff = self._effective_layers(n_layers, k)
            prob, lam = build_problem(nx, ny, nl_eff, mode_type, k, ne,
                                      ansatz_type, self.beta,
                                      Lx=self.Lx, Ly=self.Ly)
            # Target eigenvector for the train-time basin guide — same
            # classical diagonalisation that supplied λ_target and pen_vecs.
            eigvals, eigvecs = np.linalg.eigh(prob.M)
            v_target = eigvecs[:, int(np.argmin(np.abs(eigvals - lam)))]
            probs.append(_GuidedProblem(prob, v_target, self.guide_weight))
            row_blocks.append(_rows_for_config(
                _global_feats(nx, ny, mode_type, k, ne, lam),
                nx, ny, nl_eff))

        X_raw = np.vstack(row_blocks)
        scaler = StandardScaler().fit(X_raw)
        X = scaler.transform(X_raw)

        # Row slice of each problem inside the stacked design matrix.
        slices: List[slice] = []
        start = 0
        for block in row_blocks:
            slices.append(slice(start, start + len(block)))
            start += len(block)

        B, P = len(X), len(probs)
        mlp = _MLP(X.shape[1], 1, hidden=self.hidden, seed=self.seed)
        history: List[float] = []

        for epoch in range(self.epochs):
            theta_all = mlp.forward(X)[:, 0]          # (B,)
            dtheta = np.zeros((B, 1))
            loss = 0.0
            for prob, sl in zip(probs, slices):
                g, e = prob.grad(theta_all[sl])
                # _MLP.backward divides by B; scale so the update is the exact
                # gradient of the *mean-over-problems* loss.  Weighting per
                # problem (not per row) keeps large grids — which contribute
                # more rows — from dominating the batch.
                dtheta[sl, 0] = g * (B / P)
                loss += e
            loss /= P
            history.append(loss)

            gW, gb = mlp.backward(dtheta)
            mlp.adam_step(gW, gb, self.lr)

            if verbose and (epoch % max(1, self.epochs // 10) == 0
                            or epoch == self.epochs - 1):
                print(f"  epoch {epoch+1:4d}/{self.epochs}  "
                      f"mean L = {loss:.5f}")

        self._mlp = mlp
        self._scaler = scaler
        self.trained_specs_ = [tuple(s) for s in specs]
        self.loss_history = history

        # Judge the final network on the pure (un-guided) physics energy.
        theta_all = mlp.forward(X)[:, 0]
        self.final_losses_ = {
            tuple(spec): float(prob.energy_pure(theta_all[sl]))
            for spec, prob, sl in zip(specs, probs, slices)
        }
        self._save()

        if verbose:
            grids = sorted({(s[0], s[1]) for s in self.trained_specs_})
            print(f"\nGlobal PINN trained on {P} problems / {B} rows "
                  f"spanning grids {grids}.  Final mean L = {history[-1]:.5f}")
        return self

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(self,
                nx: int, ny: int, n_layers: int,
                mode_type: str, k: int,
                plasma_density: float = 0.0,
                target_eigenvalue: Optional[float] = None,
                noise_std: float = 0.0,
                polish_steps: int = 0,
                ansatz_type: str = "HEA") -> Optional[np.ndarray]:
        """Predict θ₀ for *any* configuration, trained or not.

        Signature-compatible with ``WarmStartPredictor.predict`` (the extra
        keyword arguments have defaults).  ``target_eigenvalue`` is required —
        it is a feature — and is the k-th physical eigenvalue of
        ``np.linalg.eigvalsh(solver.M_dense)``.

        polish_steps : if > 0, refine the raw prediction with that many
            L-BFGS-B iterations on the exact physics energy of the *target*
            configuration (label-free, no VQE run; costs ~2·steps·n_params
            classical energy evaluations).  This is the recommended mode for
            configurations outside the training envelope.
        """
        if self._mlp is None:
            return None
        if target_eigenvalue is None:
            raise ValueError(
                "target_eigenvalue is required.  Compute via "
                "np.sort(np.linalg.eigvalsh(solver.M_dense)) and select the "
                "k-th physical eigenvalue (>0 for TM, >1 for TE).")

        nl_eff = self._effective_layers(n_layers, k)
        rows = _rows_for_config(
            _global_feats(nx, ny, mode_type, k, plasma_density,
                          target_eigenvalue),
            nx, ny, nl_eff)
        theta = self._mlp.forward(self._scaler.transform(rows))[:, 0]

        if polish_steps > 0:
            theta = self._polish(theta, nx, ny, nl_eff, mode_type, k,
                                 plasma_density, ansatz_type, polish_steps)
        if noise_std > 0.0:
            theta = theta + np.random.default_rng().normal(
                0.0, noise_std, size=theta.shape)
        return theta

    def _polish(self, theta, nx, ny, nl_eff, mode_type, k,
                plasma_density, ansatz_type, maxiter) -> np.ndarray:
        """A few L-BFGS-B steps on the exact physics energy of the target."""
        from scipy.optimize import minimize

        prob, _ = build_problem(nx, ny, nl_eff, mode_type, k, plasma_density,
                                ansatz_type, self.beta, Lx=self.Lx, Ly=self.Ly)
        res = minimize(prob.energy, theta, method="L-BFGS-B",
                       jac=lambda t: prob.grad(t)[0],
                       options={"maxiter": maxiter})
        return res.x

    def is_trained(self, *args, **kwargs) -> bool:
        """A single shared model serves every configuration."""
        return self._mlp is not None

    # ── state-space route for unseen grids ────────────────────────────────────

    def _pick_source_grid(self, nx: int, ny: int) -> Tuple[int, int, int]:
        """Largest trained grid that fits inside the target (falls back to the
        largest trained grid overall).  Returns (nx_s, ny_s, n_layers_s)."""
        if not self.trained_specs_:
            raise RuntimeError("Model has not been trained.")
        cands = [(s[0], s[1], s[2]) for s in self.trained_specs_]
        fitting = [c for c in cands if c[0] <= nx and c[1] <= ny]
        pool = fitting or cands
        return max(pool, key=lambda c: (c[0] + c[1], c[2]))

    def predict_prolonged(self,
                          nx: int, ny: int, n_layers: int,
                          mode_type: str, k: int,
                          plasma_density: float = 0.0,
                          target_eigenvalue: Optional[float] = None,
                          source_grid: Optional[Tuple[int, int]] = None,
                          n_restarts: int = 4,
                          maxiter: int = 300,
                          polish_source_steps: int = 10,
                          polish_steps: int = 0,
                          noise_std: float = 0.0,
                          ansatz_type: str = "HEA",
                          verbose: bool = False
                          ) -> Tuple[np.ndarray, float]:
        """Warm start for an unseen grid through *state space*.

        Pipeline (all classical, no VQE run, no labels):
          1. predict θ on the largest trained source grid and refine it with
             ``polish_source_steps`` L-BFGS-B iterations on the (small, cheap)
             source physics energy — the prolonged state can only be as good
             as the source state, so this is where polish pays off most,
          2. simulate its statevector and prolong the real-space field to the
             target grid (``ProlongationWarmStart.prolong_field``),
          3. fit the target-grid ansatz to the prolonged state by maximising
             the fidelity |⟨φ|ψ(θ)⟩|² (multi-restart L-BFGS-B on the NumPy
             simulator).  If ``target_eigenvalue`` is given, the direct global
             -PINN prediction for the target seeds the first restart,
          4. optionally descend ``polish_steps`` iterations on the *target*
             physics energy from the fitted θ.

        Returns ``(θ₀, fidelity of the fit to the prolonged target)``.
        """
        from scipy.optimize import minimize

        if self._mlp is None:
            raise RuntimeError("Model has not been trained.")

        # 1. Source-grid prediction.  Its λ_target is a feature, so build the
        #    (small, cheap) source problem to read it off M_dense.
        nx_s, ny_s, nl_s = (source_grid + (n_layers,) if source_grid
                            else self._pick_source_grid(nx, ny))
        nl_s_eff = self._effective_layers(nl_s, k)
        prob_s, lam_s = build_problem(nx_s, ny_s, nl_s_eff, mode_type, k,
                                      plasma_density, ansatz_type, self.beta,
                                      Lx=self.Lx, Ly=self.Ly)
        theta_s = self.predict(nx_s, ny_s, nl_s, mode_type, k,
                               plasma_density, target_eigenvalue=lam_s,
                               polish_steps=polish_source_steps,
                               ansatz_type=ansatz_type)

        # 2. Prolong the real-space field (x register = low bits → x fastest).
        psi_s = ansatz_statevector(theta_s, nx_s + ny_s, nl_s_eff, ansatz_type)
        field_s = np.real(psi_s).reshape(2 ** ny_s, 2 ** nx_s)
        field_b = ProlongationWarmStart.prolong_field(field_s, 2 ** ny, 2 ** nx)
        phi = field_b.ravel()
        phi = phi / np.linalg.norm(phi)

        # 3. Fidelity fit on the target ansatz (real states — Ry+CX only).
        n = nx + ny
        nl_eff = self._effective_layers(n_layers, k)
        n_params = n * nl_eff

        def neg_fid(theta):
            psi = np.real(ansatz_statevector(theta, n, nl_eff, ansatz_type))
            return -float(phi @ psi) ** 2

        starts: List[np.ndarray] = []
        if target_eigenvalue is not None:
            starts.append(self.predict(nx, ny, n_layers, mode_type, k,
                                       plasma_density,
                                       target_eigenvalue=target_eigenvalue))
        rng = np.random.default_rng(self.seed)
        while len(starts) < n_restarts:
            starts.append(rng.uniform(-np.pi, np.pi, n_params))

        best_theta, best_fid = None, -1.0
        for theta0 in starts:
            res = minimize(neg_fid, theta0, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12})
            if -res.fun > best_fid:
                best_fid, best_theta = -res.fun, res.x
            if best_fid > 0.999:
                break
        if verbose:
            print(f"    prolongation {nx_s}x{ny_s} → {nx}x{ny}: "
                  f"|⟨φ|ψ(θ)⟩|² = {best_fid:.4f}  ({n_params} params)")

        if polish_steps > 0:
            best_theta = self._polish(best_theta, nx, ny, nl_eff, mode_type,
                                      k, plasma_density, ansatz_type,
                                      polish_steps)
        if noise_std > 0.0:
            best_theta = best_theta + np.random.default_rng().normal(
                0.0, noise_std, size=best_theta.shape)
        return best_theta, best_fid
