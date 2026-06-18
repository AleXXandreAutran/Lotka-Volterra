# Code 2D avec rotation
# Conditions initiales gaussiennes
# Conditions de Dirichlet homogènes au bord

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os


# Parameters of the model

a = 0.6
b = 1.2

# u denotes the prey density v_2
# v denotes the predator density v_1
nu_u = 8e-5   # diffusion coefficient for the prey u = v_2
nu_v = 5e-5   # diffusion coefficient for the predators v = v_1


# Spatial discretization

Lx, Ly = 1.0, 1.0
Nx, Ny = 80, 80

x = np.linspace(0.0, Lx, Nx)
y = np.linspace(0.0, Ly, Ny)

hx = x[1] - x[0]
hy = y[1] - y[0]

X, Y = np.meshgrid(x, y, indexing="ij")


# Time discretization

T = 4.0
dt = 5e-4
Nt = int(T / dt)

save_every = 20


# Simple explicit diffusion stability check
stability_u = dt * nu_u * (1.0 / hx**2 + 1.0 / hy**2)
stability_v = dt * nu_v * (1.0 / hx**2 + 1.0 / hy**2)

print("Diffusion stability quantity for u:", stability_u)
print("Diffusion stability quantity for v:", stability_v)

if max(stability_u, stability_v) > 0.5:
    print("Warning: the explicit diffusion stability condition may not be satisfied.")



# Dirichlet boundary values

u_bc = 0.0
v_bc = 0.0


def impose_dirichlet(u, v):
    """
    Imposes homogeneous Dirichlet boundary conditions:
    u = 0 and v = 0 on the boundary of the square domain.
    """
    u[0, :] = u_bc
    u[-1, :] = u_bc
    u[:, 0] = u_bc
    u[:, -1] = u_bc

    v[0, :] = v_bc
    v[-1, :] = v_bc
    v[:, 0] = v_bc
    v[:, -1] = v_bc

    return u, v


# Initial conditions: 2D Gaussian functions

def gauss2d(X, Y, x0, y0, sx, sy, A):
    return A * np.exp(
        -(
            (X - x0)**2 / (2 * sx**2)
            +
            (Y - y0)**2 / (2 * sy**2)
        )
    )


# u is the prey density v_2
u = gauss2d(X, Y, 0.68, 0.60, 0.10, 0.12, 1.4) + 0.02

# v is the predator density v_1
v = gauss2d(X, Y, 0.35, 0.42, 0.12, 0.10, 1.3) + 0.02


# Alternative initial conditions: superposition of two Gaussian functions
# u = (
#     gauss2d(X, Y, 0.3, 0.3, 0.08, 0.08, 1.2)
#     +
#     gauss2d(X, Y, 0.7, 0.7, 0.08, 0.08, 1.2)
#     + 0.02
# )
#
# v = (
#     gauss2d(X, Y, 0.7, 0.3, 0.10, 0.10, 1.2)
#     +
#     gauss2d(X, Y, 0.3, 0.7, 0.10, 0.10, 1.2)
#     + 0.02
# )


# Imposition of Dirichlet boundary conditions at t = 0
u, v = impose_dirichlet(u, v)


# 2D Laplacian with homogeneous Dirichlet boundary conditions
# The centered finite-difference Laplacian is computed only at
# interior nodes. Boundary nodes are fixed by the Dirichlet conditions.

def laplacian2d_dirichlet(W, hx, hy):
    lap = np.zeros_like(W)

    lap[1:-1, 1:-1] = (
        (W[2:, 1:-1] - 2.0 * W[1:-1, 1:-1] + W[:-2, 1:-1]) / hx**2
        +
        (W[1:-1, 2:] - 2.0 * W[1:-1, 1:-1] + W[1:-1, :-2]) / hy**2
    )

    return lap


# Simulation with explicit Euler time discretization

U_hist = [u.copy()]
V_hist = [v.copy()]
t_hist = [0.0]

for n in range(Nt):
    Lu = laplacian2d_dirichlet(u, hx, hy)
    Lv = laplacian2d_dirichlet(v, hx, hy)

    u_new = u.copy()
    v_new = v.copy()

    # Explicit Euler update at interior nodes only
    u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (
        u[1:-1, 1:-1] * (b - v[1:-1, 1:-1])
        +
        nu_u * Lu[1:-1, 1:-1]
    )

    v_new[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (
        v[1:-1, 1:-1] * (u[1:-1, 1:-1] - a)
        +
        nu_v * Lv[1:-1, 1:-1]
    )

    # Positivity projection
    u_new = np.maximum(u_new, 0.0)
    v_new = np.maximum(v_new, 0.0)

    # Reimpose homogeneous Dirichlet boundary conditions
    u, v = impose_dirichlet(u_new, v_new)

    if (n + 1) % save_every == 0:
        U_hist.append(u.copy())
        V_hist.append(v.copy())
        t_hist.append((n + 1) * dt)

U_hist = np.array(U_hist)
V_hist = np.array(V_hist)
t_hist = np.array(t_hist)

print("Number of frames:", len(t_hist))


# Output directory

output_dir = os.path.expanduser("~/Desktop")
os.makedirs(output_dir, exist_ok=True)


# Static final figure

zmax = 1.1 * max(U_hist.max(), V_hist.max())

fig = plt.figure(figsize=(14, 6))
fig.subplots_adjust(top=0.88, wspace=0.25)

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(
    X, Y, U_hist[-1],
    cmap="Reds",
    rstride=1,
    cstride=1,
    linewidth=0,
    edgecolor="none",
    antialiased=True,
    shade=True
)
ax1.set_title(f"Prey $v_2(x,y,t)$\nt = {t_hist[-1]:.3f}", pad=15)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("v_2")
ax1.set_zlim(0, zmax)
ax1.view_init(elev=28, azim=-60)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(
    X, Y, V_hist[-1],
    cmap="Greys",
    rstride=1,
    cstride=1,
    linewidth=0,
    edgecolor="none",
    antialiased=True,
    shade=True
)
ax2.set_title(f"Predators $v_1(x,y,t)$\nt = {t_hist[-1]:.3f}", pad=15)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("v_1")
ax2.set_zlim(0, zmax)
ax2.view_init(elev=28, azim=-60)

static_filename = os.path.join(output_dir, "Lotka_volterra_2D_Dirichlet_final.png")
fig.savefig(static_filename, dpi=300, bbox_inches="tight")

print("Static figure saved here:")
print(static_filename)

plt.show()


# 3D animation with rotation

fig_anim = plt.figure(figsize=(14, 6))
fig_anim.subplots_adjust(top=0.88, wspace=0.25)

axu = fig_anim.add_subplot(1, 2, 1, projection="3d")
axv = fig_anim.add_subplot(1, 2, 2, projection="3d")

azim0 = -60


def draw_frame(k):
    axu.clear()
    axv.clear()

    # Progressive rotation
    azim = azim0 + 280.0 * k / len(t_hist)

    # Prey surface
    axu.plot_surface(
        X, Y, U_hist[k],
        cmap="Reds",
        rstride=1,
        cstride=1,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=True
    )
    axu.set_title(f"Prey $v_2(x,y,t)$\nt = {t_hist[k]:.3f}", pad=20)
    axu.set_xlabel("x")
    axu.set_ylabel("y")
    axu.set_zlabel("v_2")
    axu.set_zlim(0, zmax)
    axu.view_init(elev=28, azim=azim)

    # Predator surface
    axv.plot_surface(
        X, Y, V_hist[k],
        cmap="Greys",
        rstride=1,
        cstride=1,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=True
    )
    axv.set_title(f"Predators $v_1(x,y,t)$\nt = {t_hist[k]:.3f}", pad=20)
    axv.set_xlabel("x")
    axv.set_ylabel("y")
    axv.set_zlabel("v_1")
    axv.set_zlim(0, zmax)
    axv.view_init(elev=28, azim=azim)

    return []


ani = FuncAnimation(
    fig_anim,
    draw_frame,
    frames=len(t_hist),
    interval=60,
    blit=False
)

# Save animation

animation_filename = os.path.join(output_dir, "Lotka_volterra_2D_Dirichlet_rotation.mp4")

writer = FFMpegWriter(
    fps=30,
    codec="libx264",
    bitrate=1800
)

print("Saving animation...")
ani.save(animation_filename, writer=writer, dpi=180)

print("Animation saved here:")
print(animation_filename)

plt.show()


# Open the saved animation on macOS
os.system(f'open "{animation_filename}"')
