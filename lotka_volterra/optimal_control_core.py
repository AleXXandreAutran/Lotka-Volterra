import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import diags, eye, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import minimize


# ============================================================
# 1. Parameters
# ============================================================

class Params:
    def __init__(self):
        # Space-time discretization
        self.M = 80                 # number of spatial points
        self.N = 120                # number of time steps
        self.T = 4.0                # final time
        self.Lx = 1.0               # spatial interval length

        # Model parameters
        self.mu1 = 0.002            # predator diffusion
        self.mu2 = 0.002            # prey diffusion
        self.m = 1.0                # predator mortality
        self.alpha = 1.0            # predator gain from prey
        self.r = 1.2                # prey growth rate
        self.beta = 1.0             # predation intensity
        self.q = 1.0                # fishing efficiency

        # Cost parameters
        self.gamma1 = 0.5
        self.gamma2 = 0.5
        self.lam = 0.05             # lambda_2
        self.eta = 0.8              # fishing reward

        # Control bounds
        self.Umax = 2.0

        # Algorithmic parameters
        self.max_iter_fbs = 100
        self.max_iter_pg = 100
        self.max_iter_lbfgsb = 100
        self.tol = 1e-6

        # Relaxation for FBS
        self.theta = 0.5

        # Projected gradient / Armijo
        self.rho0 = 1.0
        self.armijo_c = 1e-4
        self.armijo_sigma = 0.5
        self.armijo_max_backtracks = 20

        # L-BFGS-B
        self.lbfgsb_memory = 10


P = Params()


# ============================================================
# 2. Mesh and discrete operators
# ============================================================

def build_mesh(P):
    x = np.linspace(0.0, P.Lx, P.M)
    dx = x[1] - x[0]
    dt = P.T / P.N
    t = np.linspace(0.0, P.T, P.N + 1)
    return x, t, dx, dt


def build_neumann_laplacian(P, dx):
    """
    Symmetric finite-difference Laplacian with homogeneous Neumann boundary conditions.

    Interior:
        (v_{i+1} - 2 v_i + v_{i-1}) / dx^2

    Boundary:
        (v_1 - v_0) / dx^2
        (v_{M-2} - v_{M-1}) / dx^2

    This matrix is symmetric and satisfies L * 1 = 0.
    """
    M = P.M

    main = -2.0 * np.ones(M)
    upper = np.ones(M - 1)
    lower = np.ones(M - 1)

    main[0] = -1.0
    main[-1] = -1.0

    L = diags(
        diagonals=[lower, main, upper],
        offsets=[-1, 0, 1],
        shape=(M, M),
        format="csc"
    ) / dx**2

    return L


x, tgrid, dx, dt = build_mesh(P)
Lh = build_neumann_laplacian(P, dx)

Ih = eye(P.M, format="csc")
A1 = csc_matrix(Ih - dt * P.mu1 * Lh)
A2 = csc_matrix(Ih - dt * P.mu2 * Lh)

LU1 = splu(A1)
LU2 = splu(A2)

# Since Lh is symmetric, A_i^T = A_i.
# But we keep separate notation for clarity.
LU1T = splu(A1.T)
LU2T = splu(A2.T)


# ============================================================
# 3. Initial data and targets
# ============================================================

def initial_conditions(P, x):
    """
    Smooth positive initial predator and prey profiles.
    """
    v10 = 1.3 * np.exp(-((x - 0.30) ** 2) / (2 * 0.10**2)) + 0.05
    v20 = 1.4 * np.exp(-((x - 0.70) ** 2) / (2 * 0.10**2)) + 0.05
    return v10, v20


def target_profiles(P, x, tgrid):
    """
    Time-independent target profiles.

    We choose targets close to the positive coexistence equilibrium
        v1* = r / beta,
        v2* = m / alpha,
    with a small spatial modulation.
    """
    v1_star = P.r / P.beta
    v2_star = P.m / P.alpha

    spatial_bump = np.cos(2 * np.pi * x)

    v1_target_space = v1_star + 0.15 * spatial_bump
    v2_target_space = v2_star - 0.15 * spatial_bump

    v1_tar = np.repeat(v1_target_space[None, :], P.N, axis=0)
    v2_tar = np.repeat(v2_target_space[None, :], P.N, axis=0)

    return v1_tar, v2_tar


v10, v20 = initial_conditions(P, x)
v1_tar, v2_tar = target_profiles(P, x, tgrid)


# ============================================================
# 4. Projection and cost functional
# ============================================================

def project_control(u, P):
    return np.minimum(P.Umax, np.maximum(0.0, u))


def compute_cost(P, u, v1, v2, v1_tar, v2_tar, dx, dt):
    """
    Discrete cost with rectangle rule on n = 0,...,N-1.

    J = sum dt dx [
        1/2 gamma1 (v1 - v1tar)^2
      + 1/2 gamma2 (v2 - v2tar)^2
      + 1/2 lambda u^2
      - eta q u v2
    ].
    """
    v1n = v1[:-1, :]
    v2n = v2[:-1, :]

    integrand = (
        0.5 * P.gamma1 * (v1n - v1_tar) ** 2
        + 0.5 * P.gamma2 * (v2n - v2_tar) ** 2
        + 0.5 * P.lam * u ** 2
        - P.eta * P.q * u * v2n
    )

    return dt * dx * np.sum(integrand)


# ============================================================
# 5. State solver
# ============================================================

def solve_state(P, u):
    """
    Semi-implicit state solver.

    A1 v1_{n+1} = v1_n + dt v1_n(-m + alpha v2_n)
    A2 v2_{n+1} = v2_n + dt [v2_n(r - beta v1_n) - q u_n v2_n]
    """
    v1 = np.zeros((P.N + 1, P.M))
    v2 = np.zeros((P.N + 1, P.M))

    v1[0, :] = v10
    v2[0, :] = v20

    for n in range(P.N):
        rhs1 = v1[n, :] + dt * v1[n, :] * (-P.m + P.alpha * v2[n, :])
        rhs2 = v2[n, :] + dt * (
            v2[n, :] * (P.r - P.beta * v1[n, :])
            - P.q * u[n, :] * v2[n, :]
        )

        v1[n + 1, :] = LU1.solve(rhs1)
        v2[n + 1, :] = LU2.solve(rhs2)

    return v1, v2


# ============================================================
# 6. Discrete adjoint solver
# ============================================================

def solve_adjoint(P, u, v1, v2, v1_tar, v2_tar):
    """
    Discrete adjoint compatible with the semi-implicit state scheme.

    A1^T p1_n = p1_{n+1}
       + dt [ gamma1(v1_n - v1tar_n)
              + (-m + alpha v2_n) p1_{n+1}
              - beta v2_n p2_{n+1} ]

    A2^T p2_n = p2_{n+1}
       + dt [ gamma2(v2_n - v2tar_n)
              - eta q u_n
              + alpha v1_n p1_{n+1}
              + (r - beta v1_n - q u_n) p2_{n+1} ]
    """
    p1 = np.zeros((P.N + 1, P.M))
    p2 = np.zeros((P.N + 1, P.M))

    # No terminal cost
    p1[P.N, :] = 0.0
    p2[P.N, :] = 0.0

    for n in reversed(range(P.N)):
        rhs1 = p1[n + 1, :] + dt * (
            P.gamma1 * (v1[n, :] - v1_tar[n, :])
            + (-P.m + P.alpha * v2[n, :]) * p1[n + 1, :]
            - P.beta * v2[n, :] * p2[n + 1, :]
        )

        rhs2 = p2[n + 1, :] + dt * (
            P.gamma2 * (v2[n, :] - v2_tar[n, :])
            - P.eta * P.q * u[n, :]
            + P.alpha * v1[n, :] * p1[n + 1, :]
            + (P.r - P.beta * v1[n, :] - P.q * u[n, :]) * p2[n + 1, :]
        )

        p1[n, :] = LU1T.solve(rhs1)
        p2[n, :] = LU2T.solve(rhs2)

    return p1, p2


# ============================================================
# 7. Reduced objective and gradient
# ============================================================

def reduced_cost_and_gradient(P, u, return_state=False):
    """
    Computes j_h(u) and its gradient using the discrete adjoint.

    Gradient:
        grad_n = lambda u_n - eta q v2_n - q v2_n p2_{n+1}

    The dx*dt factor is included in the gradient so that it is the Euclidean gradient
    associated with the flattened vector.
    """
    u = project_control(u.reshape(P.N, P.M), P)

    v1, v2 = solve_state(P, u)
    J = compute_cost(P, u, v1, v2, v1_tar, v2_tar, dx, dt)

    p1, p2 = solve_adjoint(P, u, v1, v2, v1_tar, v2_tar)

    grad = (
        P.lam * u
        - P.eta * P.q * v2[:-1, :]
        - P.q * v2[:-1, :] * p2[1:, :]
    )

    # Euclidean gradient of the quadrature-weighted cost
    grad = dx * dt * grad

    if return_state:
        return J, grad, v1, v2, p1, p2

    return J, grad


def projected_gradient_residual(P, u, grad):
    """
    Residual for bound-constrained first-order optimality:
        R(u) = u - P_U(u - grad).
    """
    u_proj = project_control(u - grad.reshape(P.N, P.M), P)
    R = u - u_proj
    return np.linalg.norm(R.ravel(), ord=np.inf)


# ============================================================
# 8. Method 1: Forward--Backward Sweep
# ============================================================

def run_forward_backward_sweep(P, theta=None):
    if theta is None:
        theta = P.theta

    start = time.time()

    u = np.zeros((P.N, P.M))

    history = {
        "cost": [],
        "control_change": [],
        "pg_residual": [],
        "time": [],
    }

    for k in range(P.max_iter_fbs):
        J, grad, v1, v2, p1, p2 = reduced_cost_and_gradient(P, u, return_state=True)

        u_raw = (
            P.q * v2[:-1, :] * (P.eta + p2[1:, :]) / P.lam
        )

        u_new = project_control(theta * u_raw + (1.0 - theta) * u, P)

        control_change = np.linalg.norm((u_new - u).ravel(), ord=np.inf)
        pg_res = projected_gradient_residual(P, u, grad)

        history["cost"].append(J)
        history["control_change"].append(control_change)
        history["pg_residual"].append(pg_res)
        history["time"].append(time.time() - start)

        u = u_new

        if control_change < P.tol:
            break

    J, grad, v1, v2, p1, p2 = reduced_cost_and_gradient(P, u, return_state=True)

    result = {
        "method": f"FBS theta={theta}",
        "u": u,
        "v1": v1,
        "v2": v2,
        "p1": p1,
        "p2": p2,
        "cost": J,
        "grad": grad,
        "iterations": len(history["cost"]),
        "runtime": time.time() - start,
        "history": history,
    }

    return result


# ============================================================
# 9. Method 2: Projected Gradient with Armijo
# ============================================================

def run_projected_gradient_armijo(P):
    start = time.time()

    u = np.zeros((P.N, P.M))

    history = {
        "cost": [],
        "control_change": [],
        "pg_residual": [],
        "step_size": [],
        "time": [],
    }

    for k in range(P.max_iter_pg):
        J, grad, v1, v2, p1, p2 = reduced_cost_and_gradient(P, u, return_state=True)

        pg_res = projected_gradient_residual(P, u, grad)
        history["cost"].append(J)
        history["pg_residual"].append(pg_res)
        history["time"].append(time.time() - start)

        if pg_res < P.tol:
            history["control_change"].append(0.0)
            history["step_size"].append(0.0)
            break

        rho = P.rho0

        # Projected descent direction
        u_trial = project_control(u - rho * grad.reshape(P.N, P.M), P)
        d = u_trial - u

        # Armijo backtracking
        accepted = False
        for _ in range(P.armijo_max_backtracks):
            u_trial = project_control(u - rho * grad.reshape(P.N, P.M), P)
            d = u_trial - u

            J_trial, _ = reduced_cost_and_gradient(P, u_trial, return_state=False)

            # Sufficient decrease condition
            # Use ||d||^2 as a robust projected-gradient decrease measure
            decrease = P.armijo_c * np.sum(d.ravel() ** 2)

            if J_trial <= J - decrease:
                accepted = True
                break

            rho *= P.armijo_sigma

        if not accepted:
            # If no Armijo step is found, take the smallest trial step.
            u_trial = project_control(u - rho * grad.reshape(P.N, P.M), P)

        control_change = np.linalg.norm((u_trial - u).ravel(), ord=np.inf)

        history["control_change"].append(control_change)
        history["step_size"].append(rho)

        u = u_trial

        if control_change < P.tol:
            break

    J, grad, v1, v2, p1, p2 = reduced_cost_and_gradient(P, u, return_state=True)

    result = {
        "method": "Projected Gradient + Armijo",
        "u": u,
        "v1": v1,
        "v2": v2,
        "p1": p1,
        "p2": p2,
        "cost": J,
        "grad": grad,
        "iterations": len(history["cost"]),
        "runtime": time.time() - start,
        "history": history,
    }

    return result


# ============================================================
# 10. Method 3: L-BFGS-B
# ============================================================

def run_lbfgsb(P):
    start = time.time()

    u0 = np.zeros((P.N, P.M))
    bounds = [(0.0, P.Umax)] * (P.N * P.M)

    history = {
        "cost": [],
        "pg_residual": [],
        "time": [],
    }

    def fun_and_jac(u_flat):
        u = u_flat.reshape(P.N, P.M)
        J, grad = reduced_cost_and_gradient(P, u, return_state=False)
        return J, grad.ravel()

    def callback(u_flat):
        u = u_flat.reshape(P.N, P.M)
        J, grad = reduced_cost_and_gradient(P, u, return_state=False)
        pg_res = projected_gradient_residual(P, u, grad)
        history["cost"].append(J)
        history["pg_residual"].append(pg_res)
        history["time"].append(time.time() - start)

    opt = minimize(
        fun=lambda z: fun_and_jac(z),
        x0=u0.ravel(),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": P.max_iter_lbfgsb,
            "maxcor": P.lbfgsb_memory,
            "ftol": 1e-12,
            "gtol": P.tol,
            "disp": False,
        },
    )

    u = opt.x.reshape(P.N, P.M)
    J, grad, v1, v2, p1, p2 = reduced_cost_and_gradient(P, u, return_state=True)

    if len(history["cost"]) == 0:
        history["cost"].append(J)
        history["pg_residual"].append(projected_gradient_residual(P, u, grad))
        history["time"].append(time.time() - start)

    result = {
        "method": "L-BFGS-B",
        "u": u,
        "v1": v1,
        "v2": v2,
        "p1": p1,
        "p2": p2,
        "cost": J,
        "grad": grad,
        "iterations": opt.nit,
        "runtime": time.time() - start,
        "history": history,
        "scipy_message": opt.message,
        "success": opt.success,
    }

    return result
