# Generated from Lotka-Volterra.ipynb. Empty cells and repeated plotting blocks were omitted.

import os
import numpy as np
import matplotlib.pyplot as plt

from optimal_control_core import projected_gradient_residual

# ============================================================
# 13. Comparative plots -- improved version
# ============================================================

FIGDIR = "figures_comparison"
os.makedirs(FIGDIR, exist_ok=True)


def safe_name(name):
    return (
        name.replace(" ", "_")
        .replace("+", "plus")
        .replace("=", "")
        .replace(".", "p")
        .replace("-", "_")
    )


def plot_cost_histories(results):
    plt.figure(figsize=(8, 5))

    for res in results:
        hist = res["history"]
        if "cost" in hist and len(hist["cost"]) > 0:
            plt.plot(
                np.arange(len(hist["cost"])),
                hist["cost"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=res["method"],
            )

    plt.xlabel("Iteration")
    plt.ylabel(r"Discrete cost $J_h$")
    plt.title("Cost history")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "01_cost_history.png"), dpi=250)
    plt.show()


def plot_cost_histories_semilogy_shifted(results):
    """
    Semilog plot of J_k - J_min + eps.
    Useful when costs become negative.
    """
    all_costs = []
    for res in results:
        hist = res["history"]
        if "cost" in hist and len(hist["cost"]) > 0:
            all_costs.extend(hist["cost"])

    Jmin = np.min(all_costs)
    eps = 1e-12

    plt.figure(figsize=(8, 5))

    for res in results:
        hist = res["history"]
        if "cost" in hist and len(hist["cost"]) > 0:
            costs = np.array(hist["cost"])
            shifted = costs - Jmin + eps
            plt.semilogy(
                np.arange(len(costs)),
                shifted,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=res["method"],
            )

    plt.xlabel("Iteration")
    plt.ylabel(r"$J_h(u^k)-\min(J_h)+10^{-12}$")
    plt.title("Shifted cost history")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "02_shifted_cost_history.png"), dpi=250)
    plt.show()


def plot_residual_histories(results):
    plt.figure(figsize=(8, 5))

    for res in results:
        hist = res["history"]
        if "pg_residual" in hist and len(hist["pg_residual"]) > 0:
            residuals = np.maximum(np.array(hist["pg_residual"]), 1e-16)
            plt.semilogy(
                np.arange(len(residuals)),
                residuals,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=res["method"],
            )

    plt.xlabel("Iteration")
    plt.ylabel("Projected-gradient residual")
    plt.title("Optimality residual history")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "03_projected_gradient_residual.png"), dpi=250)
    plt.show()


def plot_control_change_histories(results):
    plt.figure(figsize=(8, 5))

    for res in results:
        hist = res["history"]
        if "control_change" in hist and len(hist["control_change"]) > 0:
            values = np.maximum(np.array(hist["control_change"]), 1e-16)
            plt.semilogy(
                np.arange(len(values)),
                values,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=res["method"],
            )

    plt.xlabel("Iteration")
    plt.ylabel(r"$\|u^{k+1}-u^k\|_\infty$")
    plt.title("Control update size")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "04_control_change.png"), dpi=250)
    plt.show()


def plot_final_cost_bar(results):
    methods = [res["method"] for res in results]
    costs = [res["cost"] for res in results]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, costs)
    plt.ylabel(r"Final cost $J_h$")
    plt.title("Final cost comparison")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "05_final_cost_bar.png"), dpi=250)
    plt.show()


def plot_final_residual_bar(P, results):
    methods = []
    residuals = []

    for res in results:
        methods.append(res["method"])
        residuals.append(projected_gradient_residual(P, res["u"], res["grad"]))

    residuals = np.maximum(np.array(residuals), 1e-16)

    plt.figure(figsize=(8, 5))
    plt.bar(methods, residuals)
    plt.yscale("log")
    plt.ylabel("Final projected-gradient residual")
    plt.title("Final optimality residual comparison")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "06_final_residual_bar.png"), dpi=250)
    plt.show()


def plot_runtime_bar(results):
    methods = [res["method"] for res in results]
    runtimes = [res["runtime"] for res in results]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, runtimes)
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime comparison")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "07_runtime_bar.png"), dpi=250)
    plt.show()


def plot_iterations_bar(results):
    methods = [res["method"] for res in results]
    iterations = [res["iterations"] for res in results]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, iterations)
    plt.ylabel("Number of iterations")
    plt.title("Iteration count comparison")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "08_iterations_bar.png"), dpi=250)
    plt.show()


def plot_final_state_profiles(results, x):
    plt.figure(figsize=(8, 5))

    for res in results:
        plt.plot(
            x,
            res["v1"][-1, :],
            linewidth=1.8,
            label=res["method"],
        )

    plt.xlabel("x")
    plt.ylabel(r"$v_1(x,T)$")
    plt.title("Final predator profiles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "09_final_predator_profiles.png"), dpi=250)
    plt.show()

    plt.figure(figsize=(8, 5))

    for res in results:
        plt.plot(
            x,
            res["v2"][-1, :],
            linewidth=1.8,
            label=res["method"],
        )

    plt.xlabel("x")
    plt.ylabel(r"$v_2(x,T)$")
    plt.title("Final prey profiles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "10_final_prey_profiles.png"), dpi=250)
    plt.show()


def plot_final_states_against_targets(results, x, v1_tar, v2_tar):
    plt.figure(figsize=(8, 5))

    plt.plot(x, v1_tar[-1, :], "k--", linewidth=2.0, label="Target predator")

    for res in results:
        plt.plot(
            x,
            res["v1"][-1, :],
            linewidth=1.5,
            label=res["method"],
        )

    plt.xlabel("x")
    plt.ylabel(r"$v_1(x,T)$")
    plt.title("Final predator profiles vs target")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "11_predator_vs_target.png"), dpi=250)
    plt.show()

    plt.figure(figsize=(8, 5))

    plt.plot(x, v2_tar[-1, :], "k--", linewidth=2.0, label="Target prey")

    for res in results:
        plt.plot(
            x,
            res["v2"][-1, :],
            linewidth=1.5,
            label=res["method"],
        )

    plt.xlabel("x")
    plt.ylabel(r"$v_2(x,T)$")
    plt.title("Final prey profiles vs target")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "12_prey_vs_target.png"), dpi=250)
    plt.show()


def plot_final_controls(results, x):
    plt.figure(figsize=(8, 5))

    for res in results:
        plt.plot(
            x,
            res["u"][-1, :],
            linewidth=1.8,
            label=res["method"],
        )

    plt.xlabel("x")
    plt.ylabel(r"$u_2(x,T-\Delta t)$")
    plt.title("Final-time control profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "13_final_control_profiles.png"), dpi=250)
    plt.show()


def plot_mean_control_in_time(results, tgrid):
    plt.figure(figsize=(8, 5))

    for res in results:
        u_mean_t = np.mean(res["u"], axis=1)
        plt.plot(
            tgrid[:-1],
            u_mean_t,
            linewidth=1.8,
            label=res["method"],
        )

    plt.xlabel("t")
    plt.ylabel(r"Spatial mean of $u_2$")
    plt.title("Mean control over space as a function of time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "14_mean_control_time.png"), dpi=250)
    plt.show()


def plot_mean_control_in_space(results, x):
    plt.figure(figsize=(8, 5))

    for res in results:
        u_mean_x = np.mean(res["u"], axis=0)
        plt.plot(
            x,
            u_mean_x,
            linewidth=1.8,
            label=res["method"],
        )

    plt.xlabel("x")
    plt.ylabel(r"Time mean of $u_2$")
    plt.title("Mean control over time as a function of space")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "15_mean_control_space.png"), dpi=250)
    plt.show()


def plot_control_heatmaps(results, x, tgrid):
    for res in results:
        plt.figure(figsize=(8, 5))
        plt.imshow(
            res["u"],
            aspect="auto",
            origin="lower",
            extent=[x[0], x[-1], tgrid[0], tgrid[-2]],
        )
        plt.colorbar(label=r"$u_2(x,t)$")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.title(f"Control heatmap: {res['method']}")
        plt.tight_layout()

        fname = "16_control_heatmap_" + safe_name(res["method"]) + ".png"
        plt.savefig(os.path.join(FIGDIR, fname), dpi=250)
        plt.show()


def plot_state_heatmaps_for_best_method(results, x, tgrid):
    """
    Plot state heatmaps for the method with the smallest final cost.
    """
    best = min(results, key=lambda r: r["cost"])

    plt.figure(figsize=(8, 5))
    plt.imshow(
        best["v1"],
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], tgrid[0], tgrid[-1]],
    )
    plt.colorbar(label=r"$v_1(x,t)$")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(f"Predator heatmap for best method: {best['method']}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "17_best_predator_heatmap.png"), dpi=250)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.imshow(
        best["v2"],
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], tgrid[0], tgrid[-1]],
    )
    plt.colorbar(label=r"$v_2(x,t)$")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(f"Prey heatmap for best method: {best['method']}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "18_best_prey_heatmap.png"), dpi=250)
    plt.show()


def plot_constraint_activity(P, results):
    methods = []
    active_lower = []
    inactive = []
    active_upper = []

    eps = 1e-10

    for res in results:
        u = res["u"]
        methods.append(res["method"])
        lower = np.mean(u <= eps)
        upper = np.mean(u >= P.Umax - eps)
        mid = 1.0 - lower - upper

        active_lower.append(lower)
        inactive.append(mid)
        active_upper.append(upper)

    ind = np.arange(len(methods))

    plt.figure(figsize=(9, 5))
    plt.bar(ind, active_lower, label=r"$u=0$")
    plt.bar(ind, inactive, bottom=active_lower, label=r"$0<u<U_{\max}$")
    plt.bar(
        ind,
        active_upper,
        bottom=np.array(active_lower) + np.array(inactive),
        label=r"$u=U_{\max}$",
    )

    plt.xticks(ind, methods, rotation=20, ha="right")
    plt.ylabel("Fraction of space-time control nodes")
    plt.title("Constraint activity of the control")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "19_constraint_activity.png"), dpi=250)
    plt.show()


def make_all_comparison_plots(P, results, x, tgrid, v1_tar, v2_tar):
    plot_cost_histories(results)
    plot_cost_histories_semilogy_shifted(results)
    plot_residual_histories(results)
    plot_control_change_histories(results)

    plot_final_cost_bar(results)
    plot_final_residual_bar(P, results)
    plot_runtime_bar(results)
    plot_iterations_bar(results)

    plot_final_state_profiles(results, x)
    plot_final_states_against_targets(results, x, v1_tar, v2_tar)
    plot_final_controls(results, x)

    plot_mean_control_in_time(results, tgrid)
    plot_mean_control_in_space(results, x)
    plot_control_heatmaps(results, x, tgrid)

    plot_state_heatmaps_for_best_method(results, x, tgrid)
    plot_constraint_activity(P, results)

    print(f"\nAll comparison figures saved in: {FIGDIR}/")


make_all_comparison_plots(P, results, x, tgrid, v1_tar, v2_tar)
