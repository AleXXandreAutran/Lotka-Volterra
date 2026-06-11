import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters of the model
a = 0.6
b = 1.2
nu1 = 2e-4   # diffusion coefficient used for the prey u = v_2
nu2 = 1e-4   # diffusion coefficient used for the predators v = v_1

# Space discretization
L = 1.0
Nx = 300
x = np.linspace(0, L, Nx)
h = x[1] - x[0]

# Time discretization
T = 10.0
dt = 1e-3
Nt = int(T / dt)

save_every = 20

# Initial conditions
def gauss(x, mu, sigma, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# u is the prey density v_2
# v is the predator density v_1
u = gauss(x, 0.7, 0.08, 1.4) + 0.02
v = gauss(x, 0.3, 0.12, 1.5) + 0.02

# Homogeneous Dirichlet boundary conditions at t = 0
u[0] = 0.0
u[-1] = 0.0
v[0] = 0.0
v[-1] = 0.0

# Laplacian with homogeneous Dirichlet boundary conditions
def laplacian_dirichlet(w):
    lap = np.zeros_like(w)

    # Centered finite differences at interior nodes only
    lap[1:-1] = (w[2:] - 2*w[1:-1] + w[:-2]) / h**2

    # Boundary values are fixed by Dirichlet conditions,
    # so the Laplacian is not used at the boundary nodes.
    lap[0] = 0.0
    lap[-1] = 0.0

    return lap

# Simulation
U_hist = []
V_hist = []
t_hist = []

for n in range(Nt):
    Lu = laplacian_dirichlet(u)
    Lv = laplacian_dirichlet(v)

    u_new = u.copy()
    v_new = v.copy()

    # Explicit Euler update at interior nodes only
    u_new[1:-1] = u[1:-1] + dt * (
        u[1:-1] * (b - v[1:-1]) + nu1 * Lu[1:-1]
    )

    v_new[1:-1] = v[1:-1] + dt * (
        v[1:-1] * (u[1:-1] - a) + nu2 * Lv[1:-1]
    )

    # Positivity projection
    u_new = np.maximum(u_new, 0.0)
    v_new = np.maximum(v_new, 0.0)

    # Homogeneous Dirichlet boundary conditions
    u_new[0] = 0.0
    u_new[-1] = 0.0
    v_new[0] = 0.0
    v_new[-1] = 0.0

    u = u_new
    v = v_new

    if n % save_every == 0:
        U_hist.append(u.copy())
        V_hist.append(v.copy())
        t_hist.append(n * dt)

U_hist = np.array(U_hist)
V_hist = np.array(V_hist)
t_hist = np.array(t_hist)

# Animation
fig, ax = plt.subplots(figsize=(10, 5))

line_u, = ax.plot([], [], 'r', lw=3, label='Prey v_2(x,t)')
line_v, = ax.plot([], [], 'k', lw=3, label='Predators v_1(x,t)')
title = ax.set_title("")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.2 * max(U_hist.max(), V_hist.max()))
ax.set_xlabel("x")
ax.set_ylabel("Population")
ax.legend()
ax.grid()

def init():
    line_u.set_data([], [])
    line_v.set_data([], [])
    return line_u, line_v

def update(i):
    line_u.set_data(x, U_hist[i])
    line_v.set_data(x, V_hist[i])
    title.set_text(f"t = {t_hist[i]:.3f}")
    return line_u, line_v, title

ani = FuncAnimation(
    fig,
    update,
    frames=len(t_hist),
    init_func=init,
    interval=40,
    blit=True
)

ani.save(
    "simulation_lv_1D_dirichlet.mp4",
    writer="ffmpeg",
    fps=25,
    dpi=200
)

plt.show()
