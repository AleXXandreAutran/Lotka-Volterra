import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix


# ============================================================
# Parameters
# ============================================================

m = 0.6
r = 1.2
alpha = 1.0
beta = 1.0
q = 1.0

mu1 = 1e-3   # diffusion coefficient for predators V_1
mu2 = 1e-3   # diffusion coefficient for prey V_2


# ============================================================
# Space discretization
# ============================================================

Nx = 120
x = np.linspace(0.0, 1.0, Nx)
h = x[1] - x[0]


# ============================================================
# Spatially heterogeneous stationary control U_2(x)
# ============================================================

U2 = 0.4 + 0.9 * np.exp(-((x - 0.5) ** 2) / (2 * 0.12 ** 2))


# ============================================================
# Neumann Laplacian in 1D
# ============================================================

def laplacian_neumann(w):
    lap = np.zeros_like(w)

    lap[1:-1] = (w[2:] - 2.0 * w[1:-1] + w[:-2]) / h**2

    # Ghost-point Neumann treatment
    lap[0] = 2.0 * (w[1] - w[0]) / h**2
    lap[-1] = 2.0 * (w[-2] - w[-1]) / h**2

    return lap


# ============================================================
# Stationary residual
# ============================================================
# We solve:
#
# 0 = mu1 Delta V1 + V1(-m + alpha V2)
# 0 = mu2 Delta V2 + V2(r - beta V1) - q U2(x) V2
#
# with homogeneous Neumann boundary conditions.

def stationary_residual(y):
    V1 = y[:Nx]
    V2 = y[Nx:]

    R1 = mu1 * laplacian_neumann(V1) + V1 * (-m + alpha * V2)
    R2 = mu2 * laplacian_neumann(V2) + V2 * (r - beta * V1) - q * U2 * V2

    return np.concatenate([R1, R2])


# ============================================================
# Sparsity pattern for the nonlinear solver
# ============================================================

rows = []
cols = []

for i in range(Nx):
    # First equation: R1_i depends on V1_{i-1}, V1_i, V1_{i+1}, V2_i
    for j in [i - 1, i, i + 1]:
        if 0 <= j < Nx:
            rows.append(i)
            cols.append(j)

    rows.append(i)
    cols.append(Nx + i)

    # Second equation: R2_i depends on V2_{i-1}, V2_i, V2_{i+1}, V1_i
    for j in [i - 1, i, i + 1]:
        if 0 <= j < Nx:
            rows.append(Nx + i)
            cols.append(Nx + j)

    rows.append(Nx + i)
    cols.append(i)

jac_sparsity = coo_matrix(
    (np.ones(len(rows)), (rows, cols)),
    shape=(2 * Nx, 2 * Nx)
).tocsr()


# ============================================================
# Initial guess
# ============================================================

V1_guess = 0.8 - 0.7 * np.exp(-((x - 0.5) ** 2) / (2 * 0.18 ** 2))
V2_guess = 0.6 - 0.35 * np.exp(-((x - 0.5) ** 2) / (2 * 0.14 ** 2))

V1_guess = np.maximum(V1_guess, 0.05)
V2_guess = np.maximum(V2_guess, 0.05)

y0 = np.concatenate([V1_guess, V2_guess])


# ============================================================
# Solve the nonlinear stationary system
# ============================================================

solution = least_squares(
    stationary_residual,
    y0,
    jac_sparsity=jac_sparsity,
    bounds=(1e-8, 5.0),
    max_nfev=5000,
    xtol=1e-11,
    ftol=1e-11,
    gtol=1e-11,
    verbose=1
)

V1 = solution.x[:Nx]
V2 = solution.x[Nx:]

print("Infinity norm of the residual:")
print(np.linalg.norm(stationary_residual(solution.x), ord=np.inf))


# ============================================================
# Plot
# ============================================================

fig, ax1 = plt.subplots(figsize=(8, 4.5))

ax1.plot(x, V1, color="black", lw=2.5, label=r"Predators $V_1(x)$")
ax1.plot(x, V2, color="red", lw=2.5, label=r"Prey $V_2(x)$")

ax1.set_xlabel(r"$x$")
ax1.set_ylabel("Population density")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(x, U2, color="gray", lw=2.0, ls="--", label=r"Control $U_2(x)$")
ax2.set_ylabel(r"Fishing effort $U_2(x)$")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="center left",
    frameon=True,
    facecolor="white",
    framealpha=0.9
)

plt.title("Nonconstant stationary profiles with heterogeneous control")
plt.tight_layout()

filename = "nontrivial_stationary_profile.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")

print(f"Figure saved as: {filename}")

plt.show()
