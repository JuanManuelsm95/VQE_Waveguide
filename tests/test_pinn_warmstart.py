"""
Tests for the physics-informed (PINN) warm start.

Ground truth is, as everywhere in this repo, the dense operator ``M_dense`` via
exact diagonalisation.  We check that:

  1. the NumPy ansatz simulator reproduces qiskit's Statevector exactly,
  2. the simulated energy ⟨ψ|M|ψ⟩ equals ``cost_function`` (k=0), and
  3. a short PINN training run lowers the physics loss and predicts, on a
     held-out density, a θ₀ whose energy is well below random initialisation.

The training run uses the smallest 4-qubit grid so it finishes in a few seconds.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qiskit.quantum_info import Statevector  # noqa: E402

from src import WaveguideModeVQA  # noqa: E402
from src.pinn_warmstart_vqe import (  # noqa: E402
    ansatz_statevector,
    PINNWarmStartPredictor,
)


@pytest.mark.parametrize("nx,ny,nl,at", [(2, 2, 2, "HEA"), (3, 2, 2, "HEA"),
                                         (2, 2, 2, "ALT")])
def test_simulator_matches_qiskit(nx, ny, nl, at):
    solver = WaveguideModeVQA(nx=nx, ny=ny, n_layers=nl, mode_type="TM",
                              ansatz_type=at)
    rng = np.random.default_rng(0)
    for k in (0, 1):
        n_params = solver.n * solver._effective_layers(k)
        theta = rng.uniform(-np.pi, np.pi, n_params)
        ref = Statevector(solver.ansatz(theta, k)).data
        got = ansatz_statevector(theta, solver.n, solver._effective_layers(k), at)
        assert np.max(np.abs(ref - got)) < 1e-10


@pytest.mark.parametrize("ne", [0.0, 1e17])
def test_energy_matches_cost_function(ne):
    solver = WaveguideModeVQA(nx=2, ny=2, n_layers=2, mode_type="TM",
                              plasma_density=ne)
    rng = np.random.default_rng(1)
    theta = rng.uniform(-np.pi, np.pi, solver.n * solver.n_layers)
    psi = ansatz_statevector(theta, solver.n, solver.n_layers, "HEA")
    e_sim = np.real(np.vdot(psi, solver.M_dense @ psi))
    e_cost = solver.cost_function(theta, 0, 0.0)
    assert abs(e_sim - e_cost) / abs(e_cost) < 1e-9


def test_pinn_warmstart_beats_random():
    densities = np.linspace(0.0, 5e17, 6)
    specs = [(2, 2, 2, "TM", 0, float(ne)) for ne in densities]

    pinn = PINNWarmStartPredictor(hidden=(32, 32), lr=5e-3, epochs=150,
                                  model_dir=None, seed=0)
    pinn.train(specs, verbose=False)

    # Physics loss must have decreased.
    history = next(iter(pinn.loss_history.values()))
    assert history[-1] < history[0]

    # Held-out density: PINN θ0 energy beats the mean random θ0 energy.
    ne_test = 3.3e17
    solver = WaveguideModeVQA(nx=2, ny=2, n_layers=2, mode_type="TM",
                              plasma_density=ne_test)
    eigs = np.sort(np.linalg.eigvalsh(solver.M_dense))
    target = float(eigs[eigs > 0][0])

    theta0 = pinn.predict(2, 2, 2, "TM", 0, ne_test,
                          target_eigenvalue=target, noise_std=0.0)
    assert theta0 is not None and len(theta0) == 8

    e_pinn = solver.cost_function(theta0, 0, 0.0)
    rng = np.random.default_rng(2)
    e_rand = np.mean([solver.cost_function(rng.uniform(-np.pi, np.pi, 8), 0, 0.0)
                      for _ in range(40)])
    assert e_pinn < e_rand
    # And it should be close to the true ground eigenvalue.
    assert e_pinn / target < 1.2


def test_predict_returns_none_for_unknown_config():
    pinn = PINNWarmStartPredictor(model_dir=None)
    assert pinn.predict(2, 2, 2, "TM", 0, 1e17, target_eigenvalue=1.0) is None
    assert not pinn.is_trained(2, 2, 2, "TM", 0, 1e17)
