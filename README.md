# Quantum VQE Waveguide Solver

Variational Quantum Eigensolver for computing electromagnetic modes of vacuum
and cold-plasma-filled rectangular waveguides.

<p align="center">
  <img src="figures/repo_overview.png" alt="Waveguide VQE overview" width="700">
</p>

## Overview

This repository implements a **Variational Quantum Algorithm (VQA)** to solve
eigenvalue problems arising in rectangular waveguide theory:

| Regime | Equation | Eigenvalue |
|--------|----------|------------|
| **Vacuum** | $(-\partial_{xx} - \partial_{yy})\,\Psi = \lambda\,\Psi$ | $\lambda = \omega_{\text{cutoff}}^2 / c^2$ |
| **Cold Plasma (O-mode)** | $(-\partial_{xx} - \partial_{yy} + \omega_p^2/c^2)\,E_3 = (\omega^2/c^2)\,E_3$ | $\lambda = \omega^2 / c^2$ |

The spatial domain is discretised on a $2^{n_x} \times 2^{n_y}$ grid and
encoded into $n_x + n_y$ qubits. A hardware-efficient ansatz (HEA) with
Ry gates and CNOT entanglers is optimised with L-BFGS-B.

### ML Warm-Start

An **MLP neural network** is trained to predict good initial circuit parameters
$\theta_0$ from the problem configuration (grid size, mode index, plasma density),
replacing random initialisation and typically reducing VQE iterations by 30–60%.

## Repository Structure

```
├── src/
│   ├── __init__.py
│   ├── coldplasma_vqe_waveguide.py   # Core VQE solver
│   └── ml_warmstart_vqe.py           # ML warm-start extension
│
├── notebooks/
│   ├── 01_vacuum_waveguide.ipynb      # Vacuum TM/TE modes
│   ├── 02_cold_plasma_waveguide.ipynb # Plasma modes (no ML)
│   ├── 03_train_ml_warmstart.ipynb    # Train the ML predictor
│   └── 04_cold_plasma_with_ml.ipynb   # Plasma modes with ML warm-start
│
├── saved_parameters/                  # Pre-optimised VQE parameters (JSON)
├── warmstart_models/                  # Trained MLP models (joblib)
├── data/                              # Training data for ML warm-start
├── figures/                           # Generated plots
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run a vacuum waveguide simulation

```python
from src import WaveguideModeVQA

solver = WaveguideModeVQA(
    nx=2, ny=2, n_layers=2,
    mode_type='TM',
    Lx=0.015, Ly=0.010,
)

eigenvalue, params, history = solver.optimize_mode(k=0)
solver.print_plot_parameters(0, eigenvalue, params)
```

### 3. Add a plasma fill

```python
solver = WaveguideModeVQA(
    nx=2, ny=2, n_layers=2,
    mode_type='TM',
    Lx=0.015, Ly=0.010,
    plasma_density=1e17,   # Uniform Ne = 10^17 m^-3
)

eigenvalue, params, history = solver.optimize_mode(k=0)
solver.print_plasma_info(eigenvalue)
```

### 4. Use ML warm-start

```python
from src import WarmStartPredictor, WarmStartVQA

# Load pre-trained predictor
predictor = WarmStartPredictor(
    data_path='data/warmstart_data.json',
    model_dir='warmstart_models',
)

# Solve with warm-start
solver = WarmStartVQA(
    nx=2, ny=2, n_layers=2,
    mode_type='TM',
    plasma_density=1e17,
    predictor=predictor,
)

eigenvalue, params, history = solver.optimize_mode(k=0)
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| [01_vacuum_waveguide.ipynb](notebooks/01_vacuum_waveguide.ipynb) | Compute TM and TE cutoff modes of a vacuum waveguide. Compares VQE results against classical diagonalisation. |
| [02_cold_plasma_waveguide.ipynb](notebooks/02_cold_plasma_waveguide.ipynb) | Solve O-mode eigenvalues with uniform and Gaussian plasma profiles. Includes a density sweep. |
| [03_train_ml_warmstart.ipynb](notebooks/03_train_ml_warmstart.ipynb) | Full ML training pipeline: data collection → MLP training → diagnostics → benchmarking. |
| [04_cold_plasma_with_ml.ipynb](notebooks/04_cold_plasma_with_ml.ipynb) | Side-by-side comparison of ML warm-start vs random initialisation convergence. |

## Pre-Trained Models

Pre-trained warm-start models and training data are included in
`warmstart_models/` and `data/`. To retrain from scratch, run notebook 03.

Supported configurations (default training):
- Grid: 4×4 (`nx=2, ny=2`), 2 ansatz layers
- Modes: TM k=0,1 and TE k=0,1
- Densities: vacuum and $N_e = 10^{17}$ m$^{-3}$

## Dependencies

- Python ≥ 3.9
- Qiskit ≥ 1.0
- qiskit-algorithms
- NumPy, SciPy, Matplotlib
- scikit-learn, joblib (for ML warm-start)

See [requirements.txt](requirements.txt) for exact versions.

## Physics Background

### Vacuum Waveguide

For a rectangular waveguide of dimensions $L_x \times L_y$ with perfectly
conducting walls, the transverse eigenvalue problem reduces to the 2D
Helmholtz equation. The analytical cutoff frequencies for TM modes are:

$$f_{mn} = \frac{c}{2}\sqrt{\left(\frac{m}{L_x}\right)^2 + \left(\frac{n}{L_y}\right)^2}$$

### Cold Plasma O-Mode

When the waveguide is filled with a cold magnetised plasma, the O-mode
(ordinary mode) propagation is governed by a modified Helmholtz equation
where the plasma frequency $\omega_p$ shifts the eigenvalue spectrum upward.
The plasma frequency depends on the local electron density:

$$\omega_p^2(x,y) = \frac{q_e^2\, N_e(x,y)}{m_e\, \varepsilon_0}$$

## Citation

If you use this code in your research, please cite:

```bibtex
@software{quantum_vqe_waveguide,
  title  = {Quantum VQE Waveguide Solver with ML Warm-Start},
  author = {Juan Manuel},
  year   = {2025},
  url    = {https://github.com/YOUR_USERNAME/quantum-vqe-waveguide}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
