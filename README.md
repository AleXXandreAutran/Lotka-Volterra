
# Lotka--Volterra optimal control
````markdown
This repository contains the Python scripts used for the numerical experiments in the study of controlled Lotka--Volterra reaction--diffusion systems.

The code includes:

- one-dimensional Lotka--Volterra simulations;
- two-dimensional reaction--diffusion simulations with Neumann and Dirichlet boundary conditions;
- numerical experiments for constant control regimes;
- benchmark comparisons between optimization methods for the optimal control problem.
````

## Requirements
````markdown
The scripts require Python 3 and the following packages:
````

````bash
pip install numpy scipy pandas matplotlib
````

The file `optimal_control_core.py` contains the common routines used for the optimal control benchmark, including the state solver, the discrete adjoint solver, the cost functional, the reduced gradient, and the optimization methods.

## Basic simulations

To run the one-dimensional Lotka--Volterra simulation:

```bash
python lotka_volterra/01_lotka_volterra_1d.py
```

To run the two-dimensional simulation with homogeneous Neumann boundary conditions:

```bash
python lotka_volterra/02_lotka_volterra_2d_neumann.py
```

To run the two-dimensional simulation with homogeneous Dirichlet boundary conditions:

```bash
python lotka_volterra/03_lotka_volterra_2d_dirichlet.py
```

## Constant control regimes

The script

```bash
python lotka_volterra/07_constant_control_regimes.py
```

illustrates the effect of constant harvesting controls. It compares three regimes depending on the sign of

```math
r-q\bar u_2.
```

This quantity measures the balance between the natural prey growth rate and the harvesting mortality in the absence of predators.

## Optimal control benchmark

The optimal control benchmark compares three numerical optimization strategies:

1. forward--backward sweep with relaxation;
2. projected gradient with Armijo line search;
3. L-BFGS-B.

The benchmark is run with:

```bash
python lotka_volterra/04_run_baseline_comparison.py
```

The tested problem is one-dimensional, with

```math
\Omega=(0,1), \qquad T=4.
```

The space interval is discretized with $M=80$ grid points and the time interval with $N=120$ time steps. The model parameters are

```math
m=1, \qquad r=1.2, \qquad \alpha=\beta=q=1,
\qquad \mu_1=\mu_2=0.002.
```

The control is constrained by

```math
0 \leq u_2(x,t) \leq U_2^{\max},
\qquad U_2^{\max}=2.
```

The cost parameters are

```math
\gamma_1=\gamma_2=0.5,
\qquad
\lambda_2=0.05,
\qquad
\eta=0.8.
```

All optimization methods are initialized with the same admissible control $u_2^{(0)}\equiv 0$. The maximum number of iterations is set to 100.

The benchmark reports the final discrete cost, the number of iterations, the CPU time, and the projected-gradient residual.

## Influence of the relaxation parameter

The script

```bash
python lotka_volterra/06_fbs_theta_comparison.py
```

studies the influence of the relaxation parameter (\theta) in the forward--backward sweep method.

The tested values are

```math
\theta \in \{1,0.75,0.5,0.25,0.1,0.05\}.
```

This experiment shows how relaxation affects the stability and convergence of the fixed-point iteration.
