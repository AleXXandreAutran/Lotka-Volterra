# Generated from Lotka-Volterra.ipynb. Empty cells and repeated plotting blocks were omitted.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Model parameters
a = 0.6
b = 1.2
nu1 = 2e-4
nu2 = 1e-4

# Discretization
L = 1.0
Nx = 300
x = np.linspace(0, L, Nx)
h = x[1] - x[0]

T = 10.0
dt = 1e-3
Nt = int(T/dt)

save_every = 20

# Initial conditions

def gauss(x, mu, sigma, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

u = gauss(x, 0.7, 0.08, 1.4) + 0.02
v = gauss(x, 0.3, 0.12, 1.5) + 0.02


# Laplacian
def laplacian(w):
    lap = np.zeros_like(w)
    lap[1:-1] = (w[2:] - 2*w[1:-1] + w[:-2]) / h**2
    lap[0]  = 2*(w[1]  - w[0])  / h**2
    lap[-1] = 2*(w[-2] - w[-1]) / h**2
    return lap


# Simulation
U_hist = []
V_hist = []
t_hist = []

for n in range(Nt):
    Lu = laplacian(u)
    Lv = laplacian(v)

    u_new = u + dt * (u*(b - v) + nu1*Lu)
    v_new = v + dt * (v*(u - a) + nu2*Lv)

    u = np.maximum(u_new, 0)
    v = np.maximum(v_new, 0)

    if n % save_every == 0:
        U_hist.append(u.copy())
        V_hist.append(v.copy())
        t_hist.append(n*dt)

U_hist = np.array(U_hist)
V_hist = np.array(V_hist)
t_hist = np.array(t_hist)


# Animation
fig, ax = plt.subplots(figsize=(10,5))

line_u, = ax.plot([], [], 'r', lw=3, label='Prey v_2(x,t)')
line_v, = ax.plot([], [], 'k', lw=3, label='Predators v_1(x,t)')
title = ax.set_title("")

ax.set_xlim(0,1)
ax.set_ylim(0, 1.2*max(U_hist.max(), V_hist.max()))
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

ani = FuncAnimation(fig, update, frames=len(t_hist),
                    init_func=init, interval=40, blit=True)



ani.save(
    "simulation_lv_1D.mp4",
    writer="ffmpeg",
    fps=25,
    dpi=200
)

plt.show()
