# Lotka-Volterra — Python scripts extracted from the notebook

These `.py` files were generated from `Lotka-Volterra.ipynb`.

## Files

- `01_lotka_volterra_1d.py`: 1D Lotka-Volterra simulation with animation.
- `02_lotka_volterra_2d_neumann.py`: 2D simulation with Neumann boundary conditions.
- `03_lotka_volterra_2d_dirichlet.py`: 2D simulation with Dirichlet boundary conditions.
- `optimal_control_core.py`: core optimal-control problem, solvers, cost functional, and gradient.
- `04_run_baseline_comparison.py`: baseline method comparison and numerical summary export.
- `05_comparison_plots.py`: improved comparative plotting functions.
- `06_fbs_theta_comparison.py`: comparison of Forward--Backward Sweep relaxation parameters.
- `07_report_figures_pretty.py`: final publication/report-ready figures.

## Cleanup performed

- Empty notebook cells were ignored.
- Repeated plotting blocks from the last notebook cells were omitted.
- Comparison scripts import the shared core from `optimal_control_core.py`, which avoids duplicating the same simulation and optimization functions.
- File names are ordered and suitable for a GitHub repository.

## Main dependencies

```bash
pip install numpy scipy pandas matplotlib
```

Some scripts may require `ffmpeg` to save animations as MP4 files.
