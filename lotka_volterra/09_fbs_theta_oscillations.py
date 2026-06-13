from optimal_control_core import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_fbs_with_iterates(theta, max_iter=80):
    u = np.zeros((P.N, P.M))

    rows = []
    controls = [u.copy()]

    for k in range(max_iter):
        J, grad, v1, v2, p1, p2 = reduced_cost_and_gradient(P, u, return_state=True)

        u_raw = (P.q / P.lam) * v2[:-1, :] * (P.eta + p2[1:, :])
        u_new = project_control(theta * u_raw + (1.0 - theta) * u, P)

        du = u_new - u

        control_change_rms = np.linalg.norm(du.ravel()) / np.sqrt(du.size)
        pg_res = projected_gradient_residual(P, u, grad)

        rows.append({
            "iteration": k,
            "cost": J,
            "control_change_rms": control_change_rms,
            "pg_residual": pg_res,
            "u_mean": np.mean(u),
            "u_min": np.min(u),
            "u_max": np.max(u),
        })

        u = u_new
        controls.append(u.copy())

    return pd.DataFrame(rows), np.stack(controls, axis=0)


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
thetas = [1.0, 0.75, 0.5, 0.25, 0.1]
max_iter = 80

histories = {}
controls = {}

for theta in thetas:
    df, Uhist = run_fbs_with_iterates(theta=theta, max_iter=max_iter)
    histories[theta] = df
    controls[theta] = Uhist


# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


# ------------------------------------------------------------
# Single figure with 3 panels
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.5))


# ------------------------------------------------------------
# Panel 1: RMS control update
# ------------------------------------------------------------
ax = axes[0]

for theta in thetas:
    df = histories[theta]
    y = df["control_change_rms"].to_numpy()
    y = np.maximum(y, 1e-15)

    ax.semilogy(
        df["iteration"],
        y,
        linewidth=2,
        label=fr"$\theta={theta}$"
    )

ax.set_xlabel(r"FBS iteration $k$")
ax.set_ylabel(r"$\|u^{k+1}-u^k\|_2 / \sqrt{N_tN_x}$")
ax.set_title("Mean size of the control update")
ax.grid(True, which="both", alpha=0.35)
ax.legend(ncol=3)


# ------------------------------------------------------------
# Panel 2: discrete cost
# ------------------------------------------------------------
ax = axes[1]

for theta in thetas:
    df = histories[theta]

    ax.plot(
        df["iteration"],
        df["cost"],
        linewidth=2,
        label=fr"$\theta={theta}$"
    )

ax.set_xlabel(r"FBS iteration $k$")
ax.set_ylabel(r"$J_h(u^k)$")
ax.set_title("Discrete cost along the FBS iterations")
ax.grid(True, alpha=0.35)
ax.legend(ncol=3)


# ------------------------------------------------------------
# Panel 3: mean control value
# ------------------------------------------------------------
ax = axes[2]

for theta in thetas:
    Uhist = controls[theta]
    ubar = np.mean(Uhist, axis=(1, 2))

    ax.plot(
        np.arange(len(ubar)),
        ubar,
        linewidth=2,
        label=fr"$\theta={theta}$"
    )

ax.set_xlabel(r"FBS iteration $k$")
ax.set_ylabel(r"$\overline{u}^{\,k}$")
ax.set_title("Mean control value along the FBS iterations")
ax.grid(True, alpha=0.35)
ax.legend(ncol=3)


plt.tight_layout()
plt.savefig("fbs_theta_oscillations_summary_readable.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved figures:")
print("  fbs_theta_oscillations_summary_readable.png")
