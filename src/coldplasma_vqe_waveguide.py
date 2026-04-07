"""
coldplasma_vqe_waveguide.py
────────────────────────────────────────────────────────────────────────────────
Variational Quantum Eigensolver for Rectangular Waveguide Modes
────────────────────────────────────────────────────────────────────────────────

Solves waveguide eigenvalue problems on a 2^nx × 2^ny finite-difference grid
using a variational quantum algorithm (VQA) with IBM Qiskit primitives.

Supports two physical regimes:

  • **Vacuum waveguide** (default):
        (-∂²/∂x² - ∂²/∂y²) Ψ = λ Ψ
    where λ = ω²_cutoff / c².

  • **Cold-plasma-filled waveguide (O-mode)**:
        (-∂²/∂x² - ∂²/∂y² + ωp²(x,y)/c²) E₃ = (ω²/c²) E₃
    where ωp²(x,y) = qe² Ne(x,y) / (me ε₀).

The boundary conditions are Dirichlet (TM modes) or Neumann (TE modes).

Reference
─────────
  Eq. 4.9 in the accompanying thesis / report.

Author : Juan Manuel
License: MIT
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, Operator
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
from typing import List, Tuple, Callable, Optional
import matplotlib.pyplot as plt
import os
import json
import sys


class WaveguideModeVQA:
    """
    Variational Quantum Algorithm for solving waveguide modes
    using IBM's Estimator primitive with efficient SparsePauliOp construction.

    Supports both vacuum and plasma-filled (O-mode) waveguides.

    For a plasma-filled waveguide, solves the O-mode eigenvalue problem
    derived from Maxwell's equations in cold magnetised plasma (Eq. 4.9):

        (-∂xx - ∂yy + ωp²(x,y)/c²) E3 = (ω²/c²) E3

    where ωp²(x,y) = qe² Ne(x,y) / (me ε0) is the local plasma frequency
    squared and the eigenvalue λ = ω²/c² gives the wave frequency.

    For the vacuum case simply omit Ne_func and plasma_density (both default
    to zero), recovering the standard Helmholtz equation.

    NOTE: The O-mode always uses Dirichlet boundary conditions (mode_type='TM').

    Parameters
    ----------
    nx, ny : int
        Number of qubits per spatial dimension (grid is 2^nx × 2^ny).
    n_layers : int
        Base ansatz circuit depth.
    mode_type : str
        'TM' (Dirichlet) or 'TE' (Neumann) boundary conditions.
    ansatz_type : str
        'HEA' for hardware-efficient ansatz (linear entanglement),
        or 'ALT' for alternating even/odd entanglement.
    extra_layers_per_mode : int
        Additional circuit layers per mode index (k > 0).
    use_params_file : bool
        If True, attempt to load initial parameters from a JSON file.
    params_dir : str
        Directory for saving/loading optimised parameter files.
    Lx, Ly : float
        Physical waveguide dimensions in metres.
    Ne_func : callable or None
        Electron density profile Ne(x, y) → float [m⁻³].
    plasma_density : float
        Uniform plasma density [m⁻³] (used when Ne_func is None).
    """

    def __init__(self, nx: int, ny: int, n_layers: int, mode_type: str = 'TM',
                 ansatz_type: str = 'HEA',
                 extra_layers_per_mode: int = 1,
                 use_params_file: bool = False,
                 params_dir: str = 'saved_parameters',
                 Lx: float = 0.015, Ly: float = 0.010,
                 Ne_func: Optional[Callable[[float, float], float]] = None,
                 plasma_density: float = 0.0):

        self.nx = nx
        self.ny = ny
        self.n = nx + ny
        self.mode_type = mode_type
        self.ansatz_type = ansatz_type
        self.use_params_file = use_params_file
        self.params_dir = params_dir
        self.params_file = os.path.join(params_dir, 'coldplasma_optimized_params.json')
        self.n_layers = n_layers
        self.extra_layers_per_mode = extra_layers_per_mode
        self.estimator = StatevectorEstimator()
        self.optimized_params_list = []

        self.Lx = Lx
        self.Ly = Ly
        self.dx = self.Lx / (2**self.nx)
        self.dy = self.Ly / (2**self.ny)

        # ── Physical constants (SI) ──────────────────────────────────────────
        self.c    = 299792458              # Speed of light [m/s]
        self.qe   = 1.602176634e-19       # Electron charge [C]
        self.me   = 9.1093837015e-31      # Electron mass [kg]
        self.eps0 = 8.8541878128e-12      # Vacuum permittivity [F/m]
        self.B0   = 1.0                   # Background magnetic field [T]

        # ── Plasma density profile ───────────────────────────────────────────
        if Ne_func is not None:
            self.Ne_func = Ne_func
        elif plasma_density > 0.0:
            self.Ne_func = lambda x, y: plasma_density   # uniform
        else:
            self.Ne_func = lambda x, y: 0.0              # vacuum

        # Pre-compute Vp(x,y) = ωp²(x,y)/c² on the grid (flat, row-major).
        self.plasma_potential_flat = self._compute_plasma_potential()
        self.plasma_enabled = np.any(self.plasma_potential_flat > 0.0)

        self.M_dense = self._build_mode_matrix_dense()
        self.H_ops, self.V_dense, self.W_dense = self._build_operators()

        self.optimized_states = []
        self.eigenvalues = []

    # ──────────────────────────────────────────────────────────────────────────
    # Pauli operator helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_id_op(self, n: int) -> SparsePauliOp:
        """Returns Identity operator on n qubits: I^n"""
        if n == 0:
            return SparsePauliOp(["I"], coeffs=[1.0])
        return SparsePauliOp("I" * n)

    def _get_x_op(self) -> SparsePauliOp:
        """Returns Pauli X operator"""
        return SparsePauliOp("X")

    def _get_i0_op(self, n: int) -> SparsePauliOp:
        """Returns Projector |0><0| on n qubits: I0 = 0.5 * (I + Z)"""
        if n <= 0:
            return self._get_id_op(0)
        op_proj = 0.5 * (SparsePauliOp("I") + SparsePauliOp("Z"))
        full_op = op_proj
        for _ in range(n - 1):
            full_op = full_op.tensor(op_proj)
        return full_op

    # ──────────────────────────────────────────────────────────────────────────
    # Plasma potential
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_plasma_potential(self) -> np.ndarray:
        """
        Evaluates the O-mode plasma potential Vp(x,y) = ωp²(x,y)/c² on every
        grid point and returns a flat (row-major) array of length Ny*Nx.

        Grid points are at cell centres: x_i = (i+0.5)*dx, y_j = (j+0.5)*dy.
        """
        Nx = 2**self.nx
        Ny = 2**self.ny
        prefactor = self.qe**2 / (self.me * self.eps0 * self.c**2)

        Vp = np.zeros((Ny, Nx))
        for j in range(Ny):
            y = (j + 0.5) * self.dy
            for i in range(Nx):
                x = (i + 0.5) * self.dx
                Vp[j, i] = prefactor * self.Ne_func(x, y)

        return Vp.flatten()

    # ──────────────────────────────────────────────────────────────────────────
    # Hamiltonian construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_operators(self) -> Tuple[List[SparsePauliOp], np.ndarray, np.ndarray]:
        """
        Constructs the Hamiltonian terms using SparsePauliOp logic.
        Returns (H_ops, V_dense, W_dense) where V and W are cyclic shift unitaries.
        """
        nx, ny = self.nx, self.ny
        op_X = self._get_x_op()
        op_I2 = self._get_id_op(1)

        b = 1 if self.mode_type == 'TM' else -1
        scale_factor_x = 1 / (self.dx**2)
        scale_factor_y = 1 / (self.dy**2)
        H_ops = []

        # Term 0: -I^(N-1) ⊗ X
        h0 = self._get_id_op(ny + nx - 1).tensor(op_X) * -1
        h0 = scale_factor_x * h0
        H_ops.append(h0)

        # Term 1: -I^(ny-1) ⊗ X ⊗ I^(nx)
        h1 = self._get_id_op(ny - 1).tensor(op_X).tensor(self._get_id_op(nx)) * -1
        h1 = scale_factor_y * h1
        H_ops.append(h1)

        # Term 2: -I^(N-1) ⊗ X
        h2 = self._get_id_op(ny + nx - 1).tensor(op_X) * -1
        h2 = scale_factor_x * h2
        H_ops.append(h2)

        # Term 3: Iy ⊗ I0^(nx-1) ⊗ X
        h3 = self._get_id_op(ny).tensor(self._get_i0_op(nx - 1)).tensor(op_X)
        h3 = scale_factor_x * h3
        H_ops.append(h3)

        # Term 4: b*Iy ⊗ I0^(nx-1) ⊗ I
        h4 = self._get_id_op(ny).tensor(self._get_i0_op(nx - 1)).tensor(op_I2) * b
        h4 = scale_factor_x * h4
        H_ops.append(h4)

        # Term 5: -I^(ny-1) ⊗ X ⊗ I^(nx)
        h5 = self._get_id_op(ny - 1).tensor(op_X).tensor(self._get_id_op(nx)) * -1
        h5 = scale_factor_y * h5
        H_ops.append(h5)

        # Term 6: I0^(ny-1) ⊗ X ⊗ Ix
        h6 = self._get_i0_op(ny - 1).tensor(op_X).tensor(self._get_id_op(nx))
        h6 = scale_factor_y * h6
        H_ops.append(h6)

        # Term 7: b*I0^(ny-1) ⊗ I^(nx+1)
        h7 = self._get_i0_op(ny - 1).tensor(self._get_id_op(nx + 1)) * b
        h7 = scale_factor_y * h7
        H_ops.append(h7)

        Px = self._cyclic_shift_matrix(self.nx)
        Py = self._cyclic_shift_matrix(self.ny)
        Iy_dense = np.eye(2**self.ny)
        Ix_dense = np.eye(2**self.nx)

        V_dense = np.kron(Iy_dense, Px)
        W_dense = np.kron(Py, Ix_dense)

        V_dense = Operator(V_dense).to_instruction()
        W_dense = Operator(W_dense).to_instruction()

        return H_ops, V_dense, W_dense

    def _cyclic_shift_matrix(self, n_qubits: int) -> np.ndarray:
        """Constructs the cyclic shift matrix on n_qubits."""
        N = 2**n_qubits
        P = np.zeros((N, N))
        for i in range(N):
            P[(i + 1) % N, i] = 1
        return P

    def _build_1d_matrix(self, n_qubits: int, boundary: str, step_size: float) -> np.ndarray:
        """Builds 1D finite-difference matrix with specified boundary conditions."""
        N = 2**n_qubits
        M = np.zeros((N, N))
        if boundary == 'D':
            M[0, 0] = 3; M[0, 1] = -1; M[-1, -1] = 3; M[-1, -2] = -1
        else:
            M[0, 0] = 1; M[0, 1] = -1; M[-1, -1] = 1; M[-1, -2] = -1
        for i in range(1, N - 1):
            M[i, i - 1] = -1; M[i, i] = 2; M[i, i + 1] = -1
        return M / (step_size**2)

    def _build_mode_matrix_dense(self) -> np.ndarray:
        """
        Constructs the dense operator matrix for eigenvalue checking.

        Vacuum case  : M = -∂xx - ∂yy                (standard Helmholtz)
        Plasma O-mode: M = -∂xx - ∂yy + diag(Vp)     (Eq. 4.9)
        """
        if self.mode_type == 'TM':
            Mx = self._build_1d_matrix(self.nx, 'D', self.dx)
            My = self._build_1d_matrix(self.ny, 'D', self.dy)
        else:
            Mx = self._build_1d_matrix(self.nx, 'N', self.dx)
            My = self._build_1d_matrix(self.ny, 'N', self.dy)
        Iy = np.eye(2**self.ny)
        Ix = np.eye(2**self.nx)
        M = np.kron(Iy, Mx) + np.kron(My, Ix)

        if self.plasma_enabled:
            M += np.diag(self.plasma_potential_flat)

        return M

    # ──────────────────────────────────────────────────────────────────────────
    # Ansatz
    # ──────────────────────────────────────────────────────────────────────────

    def _effective_layers(self, k: int) -> int:
        """Returns ansatz depth for mode k; grows only for k > 0."""
        return self.n_layers + self.extra_layers_per_mode * max(0, k)

    def ansatz(self, params: np.ndarray, k: int = 0) -> QuantumCircuit:
        """Constructs the variational ansatz circuit."""
        qc = QuantumCircuit(self.n)
        param_idx = 0
        n_layers = self._effective_layers(k)

        for _ in range(n_layers):
            for qubit in range(self.n):
                qc.ry(params[param_idx], qubit)
                param_idx += 1

            if self.ansatz_type == 'HEA':
                for qubit in range(self.n - 1):
                    qc.cx(qubit, qubit + 1)
            else:
                for qubit in range(0, self.n - 1, 2):
                    qc.cx(qubit, qubit + 1)
                for qubit in range(1, self.n - 1, 2):
                    qc.cx(qubit, qubit + 1)

        return qc

    def apply_transform(self, qc: QuantumCircuit, transform: str) -> QuantumCircuit:
        """Apply V or W unitary transform to circuit."""
        if transform == 'V':
            qc.append(self.V_dense, qc.qubits)
        elif transform == 'W':
            qc.append(self.W_dense, qc.qubits)
        return qc

    # ──────────────────────────────────────────────────────────────────────────
    # Cost function
    # ──────────────────────────────────────────────────────────────────────────

    def cost_function(self, params: np.ndarray, k: int, beta: float) -> float:
        """
        Calculates the cost function for given parameters and mode index k.

        H = -∂²/∂x² - ∂²/∂y² + Vp(x,y)

        The differential part is evaluated via Pauli operators with cyclic-shift
        unitaries. The plasma potential ⟨ψ|Vp|ψ⟩ is computed directly from the
        statevector (diagonal in position space).
        """
        qc = self.ansatz(params, k)

        cost = 4.0 * (self.dx**2 + self.dy**2) / (2 * (self.dx**2 * self.dy**2))

        # Differential operator terms (0-1: no transform, 2-4: V, 5-7: W)
        circuits = []
        for i in range(len(self.H_ops)):
            qc_transformed = qc.copy()
            if 2 <= i <= 4:
                qc_transformed = self.apply_transform(qc_transformed, 'V')
            elif i >= 5:
                qc_transformed = self.apply_transform(qc_transformed, 'W')
            circuits.append(qc_transformed)

        pubs = [(circuits[i], self.H_ops[i]) for i in range(len(self.H_ops))]
        job = self.estimator.run(pubs)
        result = job.result()

        for i in range(len(result)):
            cost += float(result[i].data.evs)

        # Plasma potential term: ⟨ψ|Vp|ψ⟩
        if self.plasma_enabled:
            state = Statevector(qc).data
            plasma_ev = np.real(np.vdot(state, self.plasma_potential_flat * state))
            cost += plasma_ev

        # Orthogonality penalty for excited states
        if k > 0:
            state = Statevector(qc).data
            if len(self.optimized_states) < k:
                previous_params = self.load_params(k - 1)
                if previous_params is not None:
                    previous_state = Statevector(self.ansatz(previous_params, k - 1)).data
                    overlap = np.abs(np.vdot(previous_state, state))**2
                    cost += beta * overlap
            else:
                for i in range(k):
                    overlap = np.abs(np.vdot(self.optimized_states[i], state))**2
                    cost += beta * overlap

        return cost

    # ──────────────────────────────────────────────────────────────────────────
    # Optimisation
    # ──────────────────────────────────────────────────────────────────────────

    def optimize_mode(self, k: int) -> Tuple[float, np.ndarray, List[float]]:
        """Optimise the k-th mode with automatic restarts for unphysical eigenvalues."""
        n_params = self.n * self._effective_layers(k)
        beta = 4e5
        eigenvalue = -1.0
        max_attempts = 5
        attempts = 0

        while eigenvalue < 1.0 and attempts < max_attempts: # This condition is aplied because the first TE mode converges to a non physical one
            attempts += 1
            history = []

            if self.use_params_file is True and attempts == 1:
                current_initial_params = self.load_params(k)
                if current_initial_params is not None:
                    current_initial_params += np.abs(np.random.normal(0, 0.1, n_params))
                else:
                    current_initial_params = np.random.uniform(-np.pi, np.pi, n_params)
            else:
                current_initial_params = np.random.uniform(-np.pi, np.pi, n_params)

            def callback(xk):
                val = self.cost_function(xk, k, beta)
                history.append(val)
                print(f"Attempt {attempts} - Iter {len(history)}: Cost {val:.4f}    ", end='\r')

            result = minimize(
                lambda p: self.cost_function(p, k, beta),
                current_initial_params,
                method='L-BFGS-B',
                callback=callback,
                options={'maxiter': 400}
            )

            eigenvalue = result.fun
            optimized_params = result.x

            if eigenvalue < 1.0:
                print(f"\n[Warning] Attempt {attempts} found eigenvalue "
                      f"{eigenvalue:.4f} (< 1). Restarting...")

        self.optimized_states.append(Statevector(self.ansatz(optimized_params, k)).data)
        self.eigenvalues.append(eigenvalue)
        self.optimized_params_list = []
        self.optimized_params_list.append(optimized_params)

        return eigenvalue, optimized_params, history

    # ──────────────────────────────────────────────────────────────────────────
    # Parameter persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save_all_results(self):
        """Save all optimised modes stored in memory to the JSON file."""
        if not hasattr(self, 'optimized_params_list'):
            print("No parameters found to save. Run optimize_mode first.")
            return
        os.makedirs(self.params_dir, exist_ok=True)
        for k, params in enumerate(self.optimized_params_list):
            self.save_params(params, k, self.eigenvalues[k])
        print(f"Successfully saved {len(self.optimized_params_list)} modes to {self.params_file}")

    def save_params(self, params, k, eigenvalue):
        """Save optimised parameters to a JSON file."""
        os.makedirs(self.params_dir, exist_ok=True)
        data = {}
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file, 'r') as f:
                    data = json.load(f)
            except Exception:
                pass

        plasma_tag = "plasma" if self.plasma_enabled else "vacuum"
        key = f"{self.mode_type}_{plasma_tag}_nx{self.nx}_ny{self.ny}_nlayer_{self.n_layers}"
        if key not in data:
            data[key] = {}
        data[key][f'mode_{k}'] = {'params': params.tolist(), 'eigenvalue': eigenvalue}

        with open(self.params_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_params(self, k):
        """Load optimised parameters from JSON file."""
        if not os.path.exists(self.params_file):
            return None
        try:
            with open(self.params_file, 'r') as f:
                data = json.load(f)
            plasma_tag = "plasma" if self.plasma_enabled else "vacuum"
            key = f"{self.mode_type}_{plasma_tag}_nx{self.nx}_ny{self.ny}_nlayer_{self.n_layers}"
            return np.array(data[key][f'mode_{k}']['params'])
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Post-processing and visualisation
    # ──────────────────────────────────────────────────────────────────────────

    def reconstruct_field(self, params: np.ndarray, k: int) -> np.ndarray:
        """Reconstructs the 2D field distribution from optimised parameters."""
        qc = self.ansatz(params, k)
        state = Statevector(qc).data
        field_2d = -np.real(state.reshape(2**self.ny, 2**self.nx))
        return field_2d

    def compute_cutoff_frequency(self, eigenvalue: float) -> float:
        """Computes the cutoff frequency [Hz] from the eigenvalue."""
        return (self.c * np.sqrt(abs(eigenvalue))) / (2 * np.pi)

    def calculate_fidelity(self, k: int, state: np.ndarray) -> float:
        """Calculates the fidelity (overlap squared) with saved state for mode k."""
        optimized_params = self.load_params(k)
        if optimized_params is None:
            print(f"No saved parameters found for mode {k}. Cannot compute fidelity.")
            return -10.0
        eigenvector = Statevector(self.ansatz(optimized_params, k)).data
        overlap = np.abs(np.vdot(eigenvector, state))**2
        return overlap

    def print_plot_parameters(self, k, eigenvalue, params):
        """Prints summary and plots the field distribution for mode k."""
        print(f"\nMode {k} Result:")
        print(f"Eigenvalue λ = ω²/c² : {eigenvalue:.9f} m⁻²")

        if self.plasma_enabled:
            f_wave = self.compute_cutoff_frequency(eigenvalue)
            print(f"Wave frequency  ω/(2π)  : {f_wave / 1e9:.6f} GHz")
            print(f"(Plasma mode: eigenvalue includes ωp²/c² shift)")
        else:
            f_cutoff = self.compute_cutoff_frequency(eigenvalue) / 1e9
            print(f"Cutoff Freq : {f_cutoff:.9f} GHz")

        # Classical reference
        classical_eigs = np.sort(np.linalg.eigvalsh(self.M_dense))
        classical_eigs = [x for x in classical_eigs if x > 1] #This is because a wrong convergence of a non-physical TE mode
        exact_eig = classical_eigs[k]
        f_nume = self.compute_cutoff_frequency(exact_eig) / 1e9
        error_nume = abs(self.compute_cutoff_frequency(eigenvalue) / 1e9 - f_nume) / abs(f_nume) * 100
        print(f"\nNumerical Eigenvalue (dense) : {exact_eig:.9f} m⁻²")
        print(f"Cutoff / wave freq (numerical): {f_nume:.9f} GHz")
        print(f"Relative Error vs numerical   : {error_nume:.9f}%")

        if not self.plasma_enabled:
            f_cutoff = self.compute_cutoff_frequency(eigenvalue) / 1e9
            if self.Lx != 0.008 or self.Ly != 0.008:
                if self.mode_type == 'TE':
                    theo = [9.9931, 14.9896]
                else:
                    theo = [18.0153, 24.9827]
            else:
                if self.mode_type == 'TE':
                    theo = [0.0000001, 18.616864966978948]
                else:
                    theo = [18.616864966978948, 36.51835726525253]
            if k < len(theo):
                error_theo = abs(f_cutoff - theo[k]) / abs(theo[k]) * 100
                print(f"Relative Error vs theoretical : {error_theo:.9f}%")

        # Plot field distribution
        plt.style.use('seaborn-v0_8-muted')
        field = self.reconstruct_field(params, k)

        fig1, ax1 = plt.subplots(figsize=(6, 5))
        im1 = ax1.imshow(field, cmap='jet', interpolation='antialiased',
                         extent=[0, self.Lx * 1e3, 0, self.Ly * 1e3], origin='lower')
        ax1.set_xlabel('x [mm]')
        ax1.set_ylabel('y [mm]')
        plt.colorbar(im1, ax=ax1, label='Field amplitude', orientation='horizontal')

        if self.plasma_enabled:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            plasma_2d = self.plasma_potential_flat.reshape(2**self.ny, 2**self.nx)
            Ne_2d = plasma_2d * self.c**2 * self.me * self.eps0 / self.qe**2
            im2 = ax2.imshow(Ne_2d, cmap='hot', interpolation='antialiased',
                             extent=[0, self.Lx * 1e3, 0, self.Ly * 1e3])
            ax2.set_title('Plasma density Nₑ(x,y)')
            ax2.set_xlabel('x [mm]')
            ax2.set_ylabel('y [mm]')
            plt.colorbar(im2, ax=ax2, label='Nₑ [m⁻³]', orientation='horizontal')

        plt.tight_layout()
        plt.show()

    def print_plasma_info(self, eigenvalue):
        """Prints a summary of the plasma configuration and grid parameters."""
        print("=" * 55)
        print("  Plasma-filled waveguide – O-mode solver")
        print("=" * 55)
        print(f"  Grid          : {2**self.nx} x {2**self.ny}  "
              f"({self.nx}+{self.ny} qubits)")
        print(f"  Waveguide     : Lx={self.Lx * 1e3:.1f} mm, "
              f"Ly={self.Ly * 1e3:.1f} mm")
        print(f"  Cell size     : dx={self.dx * 1e6:.2f} µm, "
              f"dy={self.dy * 1e6:.2f} µm")
        print(f"  Plasma        : {'enabled' if self.plasma_enabled else 'vacuum (disabled)'}")
        if self.plasma_enabled:
            Vp = self.plasma_potential_flat
            Ne_factor = self.c**2 * self.me * self.eps0 / self.qe**2
            Ne_vals = Vp * Ne_factor
            print(f"  Ne range      : [{Ne_vals.min():.3e}, "
                  f"{Ne_vals.max():.3e}] m⁻³")
            omega_p_max = self.compute_cutoff_frequency(Vp.max())
            omega_c = self.qe * self.B0 / (self.me * 2 * np.pi)
            print(f"  max ωp/(2π)   : {omega_p_max / 1e9:.3f} GHz")
            print(f"  ωc/(2π)       : {omega_c / 1e9:.3f} GHz")
            print(f"  Refraction idx: "
                  f"{1 - omega_p_max / self.compute_cutoff_frequency(eigenvalue):.3f}")
            print(f"  Eigenvalue = ω²/c²  (O-mode, Eq. 4.9)")
        else:
            print(f"  Eigenvalue = ω_cutoff²/c²  (vacuum Helmholtz)")
        print("=" * 55)

    def plot_converge(self, history, k):
        """Plot convergence history."""
        plt.figure(figsize=(10, 6))
        plt.plot(history, marker='.', linestyle='-', color='b')
        value = np.linalg.eigvalsh(self.M_dense)
        value = [x for x in value if x >= 1]
        plt.plot(range(len(history)), value[k] * np.ones(len(history)),
                 '--', label='Classical Energy')
        plt.xlabel('Iterations')
        plt.ylabel('Cost Function Value')
        plt.title(f'Optimisation Progress (Mode {k})\n'
                  f'Grid: {2**self.nx}x{2**self.ny}, Ansatz: {self.ansatz_type}')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
