import os
import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 0.6
r = 1.2
alpha = 1.0
beta = 1.0
q = 1.0

# Time discretization
T = 40.0
dt = 1e-3
Nt = int(T / dt)
t = np.linspace(0.0, T, Nt + 1)

# Initial condition
v1_0 = 0.8   # predators
v2_0 = 1.0   # prey

# Three constant controls
controls = [
    (1.6, r"$r-q\bar u_2<0$"),
    (1.2, r"$r-q\bar u_2=0$"),
    (0.6, r"$r-q\bar u_2>0$")
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (ubar, title) in zip(axes, controls):
    v1 = np.zeros(Nt + 1)
    v2 = np.zeros(Nt + 1)

    v1[0] = v1_0
    v2[0] = v2_0

    for n in range(Nt):
        dv1 = v1[n] * (-m + alpha * v2[n])
        dv2 = v2[n] * (r - beta * v1[n]) - q * ubar * v2[n]

        v1[n + 1] = max(v1[n] + dt * dv1, 0.0)
        v2[n + 1] = max(v2[n] + dt * dv2, 0.0)

    ax.plot(t, v1, label=r"Predators $v_1$")
    ax.plot(t, v2, label=r"Prey $v_2$")
    ax.set_title(title + "\n" + rf"$\bar u_2={ubar}$")
    ax.set_xlabel("t")
    ax.set_ylabel("Population")
    ax.grid(True)
    ax.legend()

plt.tight_layout()

# Save figure
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, "constant_control_regimes.png")
plt.savefig(filename, dpi=300, bbox_inches="tight")

print(f"Figure saved as: {filename}")

plt.show()
