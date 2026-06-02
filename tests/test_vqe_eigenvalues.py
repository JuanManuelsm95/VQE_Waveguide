"""
Regression tests for the VQE waveguide solver.

The strategy is to use the dense finite-difference operator (``M_dense``) as
ground truth via exact diagonalisation, and check that:

  1. the operator is well formed (shape + symmetry),
  2. adding plasma shifts the spectrum upward (physics sanity check), and
  3. the VQE ground-mode eigenvalue matches the classical one.

The VQE test seeds NumPy for reproducibility and uses the smallest grid
(4 qubits) so it runs in a few seconds. If it ever proves flaky in your
environment, widen ``REL_TOL`` slightly or raise ``n_layers``.

Run with:  pytest         (after `pip install -e ".[dev]"`)
"""

import os
import sys

import numpy as np
import pytest

# Allow running from a bare checkout without `pip install -e .`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import WaveguideModeVQA  # noqa: E402


# Relative tolerance for VQE vs. classical eigenvalue. Tighten once you have
# confirmed convergence behaviour in CI.
REL_TOL = 0.05


def physical_eigs(solver):
    """Sorted classical eigenvalues above the boundary-condition threshold."""
    eigs = np.sort(np.linalg.eigvalsh(solver.M_dense))
    threshold = 1e-6 if solver.mode_type == "TM" else 1.0
    return eigs[eigs > threshold]


def test_operator_shape_and_symmetry():
    solver = WaveguideModeVQA(nx=2, ny=2, n_layers=2, mode_type="TM")
    dim = 2 ** solver.n
    assert solver.M_dense.shape == (dim, dim)
    assert np.allclose(solver.M_dense, solver.M_dense.T), "operator must be symmetric"


def test_classical_eigs_real_positive_sorted():
    solver = WaveguideModeVQA(nx=2, ny=2, n_layers=2, mode_type="TM")
    eigs = physical_eigs(solver)
    assert len(eigs) >= 2
    assert np.all(eigs > 0)
    assert np.all(np.diff(eigs) >= 0), "eigenvalues should be sorted ascending"


def test_plasma_shifts_spectrum_upward():
    """The +omega_p^2/c^2 term must raise the lowest eigenvalue."""
    vac = WaveguideModeVQA(nx=2, ny=2, n_layers=2, mode_type="TM")
    plasma = WaveguideModeVQA(
        nx=2, ny=2, n_layers=2, mode_type="TM", plasma_density=1e18
    )
    assert physical_eigs(plasma)[0] > physical_eigs(vac)[0]


@pytest.mark.parametrize("mode_type", ["TM", "TE"])
def test_vqe_ground_mode_matches_classical(mode_type):
    np.random.seed(0)
    solver = WaveguideModeVQA(nx=2, ny=2, n_layers=2, mode_type=mode_type)
    classical_lambda = physical_eigs(solver)[0]

    eigenvalue, params, history = solver.optimize_mode(k=0)

    assert eigenvalue > 0
    assert len(history) > 0
    rel_err = abs(eigenvalue - classical_lambda) / classical_lambda
    assert rel_err < REL_TOL, (
        f"{mode_type} mode 0: VQE={eigenvalue:.4f}, "
        f"classical={classical_lambda:.4f}, rel_err={rel_err:.4%}"
    )