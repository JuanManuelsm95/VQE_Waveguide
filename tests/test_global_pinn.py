"""Fast structural tests for the size-transferable global PINN.

These deliberately use tiny networks / few epochs — they check wiring
(shapes, ordering, persistence contracts), not warm-start quality.
Quality is measured by scripts/eval_global_pinn.py.
"""

import numpy as np
import pytest

from src.global_pinn_warmstart import GlobalPINNPredictor
from src.pinn_warmstart_vqe import build_problem


@pytest.fixture(scope="module")
def tiny_model():
    specs = [(2, 2, 2, "TM", 0, 0.0),
             (2, 2, 2, "TM", 0, 1e17),
             (2, 3, 2, "TM", 0, 0.0),
             (2, 3, 2, "TM", 0, 1e17)]
    gp = GlobalPINNPredictor(hidden=(16, 16), epochs=40, seed=0)
    gp.train(specs, verbose=False)
    return gp


def test_training_reduces_physics_loss(tiny_model):
    hist = tiny_model.loss_history
    assert len(hist) == 40
    assert hist[-1] < hist[0]


def test_predict_unseen_grid_has_solver_matched_length(tiny_model):
    # (3, 2) was never trained; predicted vector must still match the solver's
    # layer-major parameter count n * n_layers.
    prob, lam = build_problem(3, 2, 2, "TM", 0, 0.0, "HEA", 4e5)
    theta = tiny_model.predict(3, 2, 2, "TM", 0, 0.0, target_eigenvalue=lam)
    assert theta is not None
    assert theta.shape == (prob.n_params,) == (10,)
    assert np.all(np.abs(theta) <= np.pi + 1e-9)
    assert np.all(np.isfinite(theta))


def test_predict_requires_target_eigenvalue(tiny_model):
    with pytest.raises(ValueError):
        tiny_model.predict(2, 2, 2, "TM", 0, 0.0)


def test_untrained_model_returns_none():
    gp = GlobalPINNPredictor()
    assert not gp.is_trained()
    assert gp.predict(2, 2, 2, "TM", 0, 0.0, target_eigenvalue=1e5) is None


def test_polish_does_not_increase_energy(tiny_model):
    prob, lam = build_problem(3, 2, 2, "TM", 0, 0.0, "HEA", 4e5)
    raw = tiny_model.predict(3, 2, 2, "TM", 0, 0.0, target_eigenvalue=lam)
    pol = tiny_model.predict(3, 2, 2, "TM", 0, 0.0, target_eigenvalue=lam,
                             polish_steps=5)
    assert prob.energy(pol) <= prob.energy(raw) + 1e-12


def test_prolonged_prediction_shape_and_fidelity_range(tiny_model):
    prob, lam = build_problem(3, 2, 2, "TM", 0, 0.0, "HEA", 4e5)
    theta, fid = tiny_model.predict_prolonged(
        3, 2, 2, "TM", 0, 0.0, target_eigenvalue=lam,
        n_restarts=2, maxiter=60)
    assert theta.shape == (prob.n_params,)
    assert 0.0 <= fid <= 1.0 + 1e-9


def test_extra_layers_per_mode_lengths(tiny_model):
    # For k > 0 WarmStartVQA expects n * (n_layers + extra * k) parameters.
    prob, lam = build_problem(2, 2, 2 + 1, "TM", 1, 0.0, "HEA", 4e5)
    theta = tiny_model.predict(2, 2, 2, "TM", 1, 0.0, target_eigenvalue=lam)
    assert theta.shape == (prob.n_params,) == (12,)
