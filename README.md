# les2
Linear algebra helpers built on NumPy: LU decomposition (full pivoting), QR via Householder reflections, iterative solvers (Jacobi and Gauss-Seidel), determinants, inverses, and condition numbers. Everything operates on `numpy.ndarray` with simple procedural functions (no OOP wrapper).

## Project layout
- `main.py` — implementations you can import or extend.
- `pyproject.toml` — project metadata and dependency (`numpy`).
- `uv.lock` — locked dependencies (optional).

## Requirements
- Python 3.14+
- NumPy (pulled in on install)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .            # or: pip install numpy  |  uv sync
```

## Usage
Import and call the functions directly:
```python
import numpy as np
from main import (
    generate_matrix, generate_vector, QR_slae_solve,
    slae_sol, determinant, invert_matrix, find_conditionality,
    Jacobi, Seidel,
)

A = generate_matrix(4)
b = generate_vector(A)

# Solve with QR
x_qr = QR_slae_solve(A, b)

# Solve with LU (full pivoting)
x_lu = slae_sol(A, b)

detA = determinant(A)
A_inv = invert_matrix(A)
cond = find_conditionality(A)

# Iterative methods on a diagonally dominant system
Ad = np.array([[4., 1., 0.],
               [1., 3., 1.],
               [0., 1., 2.]])
bd = np.array([1., 2., 3.])
x_jacobi = Jacobi(Ad, bd)
x_seidel = Seidel(Ad, bd)
```

