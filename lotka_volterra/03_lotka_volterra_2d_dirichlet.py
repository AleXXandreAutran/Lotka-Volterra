# Generated from Lotka-Volterra.ipynb. Empty cells and repeated plotting blocks were omitted.

# 2D code with rotation, Gaussian initial conditions, and Dirichlet boundary conditions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import os


# Model parameters
a = 0.6
b = 1.2
nu1 = 8e-5
nu2 = 5e-5

# 2D spatial domain
Lx, Ly = 1.0, 1.0
Nx, Ny = 80, 80

x = np.linspace(0.0, Lx, Nx)
y = np.linspace(0.0, Ly, Ny)
hx = x[1] - x[0]
hy = y[1] - y[0]

X, Y = np.meshgrid(x, y, indexing="ij")


# Time
T = 4.0
dt = 5e-4
Nt = int(T / dt)

save_every = 20

# Boundary Dirichlet values
u_bc = 0.0
v_bc = 0.0


# Initial conditions (2D Gaussian profiles)
def gauss2d(X, Y, x0, y0, sx, sy, A):
    return A * np.exp(-((X - x0)**2 / (2*sx**2) + (Y - y0)**2 / (2*sy**2)))

u = gauss2d(X, Y, 0.68, 0.60, 0.10, 0.12, 1.4) + 0.02
v = gauss2d(X, Y, 0.35, 0.42, 0.12, 0.10, 1.3) + 0.02

# Alternative initial conditions: superposition of two Gaussian profiles
# u = (gauss2d(X,Y,0.3,0.3,0.08,0.08,1.2) +
#      gauss2d(X,Y,0.7,0.7,0.08,0.08,1.2) + 0.02)

# v = (gauss2d(X,Y,0.7,0.3,0.10,0.10,1.2) +
#      gauss2d(X,Y,0.3,0.7,0.10,0.10,1.2) + 0.02)


# Apply Dirichlet boundary conditions at t=0
u[0, :] = u_bc
u[-1, :] = u_bc
u[:, 0] = u_bc
u[:, -1] = u_bc

v[0, :] = v_bc
v[-1, :] = v_bc
v[:, 0] = v_bc
v[:, -1] = v_bc


# Laplacian2D (Dirichlet)
def laplacian2d_dirichlet(W, hx, hy):
    lap = np.zeros_like(W)
    lap[1:-1, 1:-1] = (
        (W[2:, 1:-1] - 2.0 * W[1:-1, 1:-1] + W[:-2, 1:-1]) / hx**2
        +
        (W[1:-1, 2:] - 2.0 * W[1:-1, 1:-1] + W[1:-1, :-2]) / hy**2
    )
    return lap


# Simulation
U_hist = [u.copy()]
V_hist = [v.copy()]
t_hist = [0.0]

for n in range(Nt):
    Lu = laplacian2d_dirichlet(u, hx, hy)
    Lv = laplacian2d_dirichlet(v, hx, hy)

    u_new = u + dt * (u * (b - v) + nu1 * Lu)
    v_new = v + dt * (v * (u - a) + nu2 * Lv)

    u = np.maximum(u_new, 0.0)
    v = np.maximum(v_new, 0.0)

    # Reapply Dirichlet boundary conditions on the boundary
    u[0, :] = u_bc
    u[-1, :] = u_bc
    u[:, 0] = u_bc
    u[:, -1] = u_bc

    v[0, :] = v_bc
    v[-1, :] = v_bc
    v[:, 0] = v_bc
    v[:, -1] = v_bc

    if n % save_every == 0:
        U_hist.append(u.copy())
        V_hist.append(v.copy())
        t_hist.append((n + 1) * dt)

U_hist = np.array(U_hist)
V_hist = np.array(V_hist)
t_hist = np.array(t_hist)

print("Nombre de frames :", len(t_hist))

# Pour l'affichage statique d'une frame finale
fig = plt.figure(figsize=(14, 6))
fig.subplots_adjust(top=0.88)

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, U_hist[-1], cmap='Reds', linewidth=0)
ax1.set_title("Prey v_2(x,y,t)", pad=15)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("v_2")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, V_hist[-1], cmap='Greys', linewidth=0)
ax2.set_title("Predators v_1(x,y,t)", pad=15)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("v_1")

plt.subplots_adjust(wspace=0.25)
plt.show()

# 3D animation: use one figure with two surfaces
zmax = 1.1 * max(U_hist.max(), V_hist.max())

fig_anim = plt.figure(figsize=(14, 6))
fig_anim.subplots_adjust(top=0.88)

axu = fig_anim.add_subplot(1, 2, 1, projection='3d')
axv = fig_anim.add_subplot(1, 2, 2, projection='3d')

azim0 = -60

def draw_frame(k):
    axu.clear()
    axv.clear()

    azim = azim0 + 280.0 * k / len(t_hist)

    # Prey
    axu.plot_surface(X, Y, U_hist[k], cmap='Reds', linewidth=0)
    axu.set_title(f"Prey v_2(x,y,t)\nt = {t_hist[k]:.3f}", pad=20)
    axu.set_xlabel("x")
    axu.set_ylabel("y")
    axu.set_zlabel("v_2")
    axu.set_zlim(0, zmax)
    axu.view_init(elev=28, azim=azim)

    # Predators
    axv.plot_surface(X, Y, V_hist[k], cmap='Greys', linewidth=0)
    axv.set_title(f"Predators v_1(x,y,t)\nt = {t_hist[k]:.3f}", pad=20)
    axv.set_xlabel("x")
    axv.set_ylabel("y")
    axv.set_zlabel("v_1")
    axv.set_zlim(0, zmax)
    axv.view_init(elev=28, azim=azim)

    return []

ani = FuncAnimation(fig_anim, draw_frame, frames=len(t_hist), interval=60)

plt.subplots_adjust(wspace=0.25)
plt.show()

# Save as MP4
filename = os.path.expanduser("~/Desktop/LLLotka_volterra_2D_dirichlet.mp4")

writer = FFMpegWriter(fps=30, codec="libx264", bitrate=1800)

print("Sauvegarde en cours...")
ani.save(filename, writer=writer, dpi=180)

print("File saved at:")
print(filename)

os.system(f'open "{filename}"')
