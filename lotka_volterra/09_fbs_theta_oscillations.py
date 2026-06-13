# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
thetas = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]
max_iter = 100

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
# Single stretched figure with 3 panels
# ------------------------------------------------------------
fig, axes = plt.subplots(
    3, 1,
    figsize=(13, 12),
    sharex=True
)

xticks = np.arange(0, max_iter + 1, 20)


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
ax.set_xlim(0, max_iter)
ax.set_xticks(xticks)
ax.tick_params(axis="x", labelbottom=True)   # force l'affichage des labels
ax.grid(True, which="both", alpha=0.35)
ax.legend(ncol=3)


# ------------------------------------------------------------
# Panel 2: reduced discrete cost
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
ax.set_ylabel(r"$j_h(u_h^k)$")
ax.set_title("Reduced discrete cost along the FBS iterations")
ax.set_xlim(0, max_iter)
ax.set_xticks(xticks)
ax.tick_params(axis="x", labelbottom=True)   # force l'affichage des labels
ax.grid(True, alpha=0.35)
ax.legend(ncol=3)


# ------------------------------------------------------------
# Panel 3: mean control value
# ------------------------------------------------------------
ax = axes[2]

for theta in thetas:
    df = histories[theta]

    ax.plot(
        df["iteration"],
        df["u_mean"],
        linewidth=2,
        label=fr"$\theta={theta}$"
    )

ax.set_xlabel(r"FBS iteration $k$")
ax.set_ylabel(r"$\bar u_h^k$")
ax.set_title("Mean control value along the FBS iterations")
ax.set_xlim(0, max_iter)
ax.set_xticks(xticks)
ax.tick_params(axis="x", labelbottom=True)
ax.grid(True, alpha=0.35)
ax.legend(ncol=3)


plt.tight_layout()

plt.savefig("fbs_theta_oscillations.png", dpi=300, bbox_inches="tight")

plt.show()

print("Saved figure:")
print("fbs_theta_oscillations.png")
