import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from optimal_control_core import *

# Run comparison with several FBS relaxation parameters

def run_all_methods_with_fbs_thetas(P):
    results = []

    fbs_thetas = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]

    for theta in fbs_thetas:
        print(f"Running Forward--Backward Sweep with theta={theta}...")
        results.append(run_forward_backward_sweep(P, theta=theta))

    print("Running Projected Gradient + Armijo...")
    results.append(run_projected_gradient_armijo(P))

    print("Running L-BFGS-B...")
    results.append(run_lbfgsb(P))

    return results


results = run_all_methods_with_fbs_thetas(P)

print("\nMethods actually computed:")
for res in results:
    print(res["method"])

# Extra plots: comparison of FBS relaxation parameters

import os

FIGDIR_THETA = "figures_fbs_theta_comparison"
os.makedirs(FIGDIR_THETA, exist_ok=True)


def get_fbs_results(results):
    return [res for res in results if res["method"].startswith("FBS")]


def plot_fbs_theta_costs(results):
    fbs_results = get_fbs_results(results)

    plt.figure(figsize=(8, 5))

    for res in fbs_results:
        costs = np.array(res["history"]["cost"])
        plt.plot(
            np.arange(len(costs)),
            costs,
            linewidth=1.6,
            label=res["method"],
        )

    plt.xlabel("Iteration")
    plt.ylabel(r"Discrete cost $J_h$")
    plt.title(r"FBS cost history for different relaxation parameters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR_THETA, "01_fbs_theta_cost_history.png"), dpi=250)
    plt.show()


def plot_fbs_theta_shifted_costs(results):
    fbs_results = get_fbs_results(results)

    all_costs = []
    for res in fbs_results:
        all_costs.extend(res["history"]["cost"])

    Jmin = np.min(all_costs)
    eps = 1e-12

    plt.figure(figsize=(8, 5))

    for res in fbs_results:
        costs = np.array(res["history"]["cost"])
        shifted = costs - Jmin + eps

        plt.semilogy(
            np.arange(len(costs)),
            shifted,
            linewidth=1.6,
            label=res["method"],
        )

    plt.xlabel("Iteration")
    plt.ylabel(r"$J_h(u^k)-\min J_h+10^{-12}$")
    plt.title(r"Shifted FBS cost history for different $\theta$")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR_THETA, "02_fbs_theta_shifted_cost_history.png"), dpi=250)
    plt.show()


def plot_fbs_theta_residuals(results):
    fbs_results = get_fbs_results(results)

    plt.figure(figsize=(8, 5))

    for res in fbs_results:
        residuals = np.maximum(np.array(res["history"]["pg_residual"]), 1e-16)
        plt.semilogy(
            np.arange(len(residuals)),
            residuals,
            linewidth=1.6,
            label=res["method"],
        )

    plt.xlabel("Iteration")
    plt.ylabel("Projected-gradient residual")
    plt.title(r"FBS projected-gradient residual for different $\theta$")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR_THETA, "03_fbs_theta_residual_history.png"), dpi=250)
    plt.show()


def plot_fbs_theta_control_changes(results):
    fbs_results = get_fbs_results(results)

    plt.figure(figsize=(8, 5))

    for res in fbs_results:
        changes = np.maximum(np.array(res["history"]["control_change"]), 1e-16)
        plt.semilogy(
            np.arange(len(changes)),
            changes,
            linewidth=1.6,
            label=res["method"],
        )

    plt.xlabel("Iteration")
    plt.ylabel(r"$\|u^{k+1}-u^k\|_\infty$")
    plt.title(r"FBS control update size for different $\theta$")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR_THETA, "04_fbs_theta_control_change.png"), dpi=250)
    plt.show()


def plot_fbs_theta_final_cost_bar(results):
    fbs_results = get_fbs_results(results)

    labels = [res["method"].replace("FBS theta=", r"$\theta=$") for res in fbs_results]
    costs = [res["cost"] for res in fbs_results]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, costs)
    plt.ylabel(r"Final cost $J_h$")
    plt.title(r"Final FBS cost as a function of $\theta$")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR_THETA, "05_fbs_theta_final_cost_bar.png"), dpi=250)
    plt.show()


def plot_fbs_theta_final_residual_bar(P, results):
    fbs_results = get_fbs_results(results)

    labels = [res["method"].replace("FBS theta=", r"$\theta=$") for res in fbs_results]
    residuals = [
        projected_gradient_residual(P, res["u"], res["grad"])
        for res in fbs_results
    ]
    residuals = np.maximum(np.array(residuals), 1e-16)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, residuals)
    plt.yscale("log")
    plt.ylabel("Final projected-gradient residual")
    plt.title(r"Final FBS residual as a function of $\theta$")
    plt.grid(True, axis="y", which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR_THETA, "06_fbs_theta_final_residual_bar.png"), dpi=250)
    plt.show()


def plot_fbs_theta_iterations_bar(results):
    fbs_results = get_fbs_results(results)

    labels = [res["method"].replace("FBS theta=", r"$\theta=$") for res in fbs_results]
    iterations = [res["iterations"] for res in fbs_results]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, iterations)
    plt.ylabel("Number of iterations")
    plt.title(r"FBS iteration count as a function of $\theta$")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR_THETA, "07_fbs_theta_iterations_bar.png"), dpi=250)
    plt.show()


def plot_fbs_theta_final_controls(results, x):
    fbs_results = get_fbs_results(results)

    plt.figure(figsize=(8, 5))

    for res in fbs_results:
        plt.plot(
            x,
            res["u"][-1, :],
            linewidth=1.6,
            label=res["method"],
        )

    plt.xlabel("x")
    plt.ylabel(r"$u_2(x,T-\Delta t)$")
    plt.title(r"Final-time FBS controls for different $\theta$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR_THETA, "08_fbs_theta_final_controls.png"), dpi=250)
    plt.show()


def plot_fbs_theta_mean_control_time(results, tgrid):
    fbs_results = get_fbs_results(results)

    plt.figure(figsize=(8, 5))

    for res in fbs_results:
        u_mean_t = np.mean(res["u"], axis=1)
        plt.plot(
            tgrid[:-1],
            u_mean_t,
            linewidth=1.6,
            label=res["method"],
        )

    plt.xlabel("t")
    plt.ylabel(r"Spatial mean of $u_2$")
    plt.title(r"FBS mean control over space for different $\theta$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR_THETA, "09_fbs_theta_mean_control_time.png"), dpi=250)
    plt.show()


def plot_fbs_theta_summary_table(P, results):
    fbs_results = get_fbs_results(results)

    rows = []
    for res in fbs_results:
        theta_str = res["method"].replace("FBS theta=", "")
        theta_val = float(theta_str)

        rows.append({
            "theta": theta_val,
            "final_cost": res["cost"],
            "iterations": res["iterations"],
            "runtime_seconds": res["runtime"],
            "final_projected_gradient_residual": projected_gradient_residual(P, res["u"], res["grad"]),
            "u_mean": np.mean(res["u"]),
            "u_min": np.min(res["u"]),
            "u_max": np.max(res["u"]),
        })

    df_theta = pd.DataFrame(rows)
    df_theta = df_theta.sort_values("theta", ascending=False)

    print("FBS THETA COMPARISON")
    print(df_theta.to_string(index=False))

    df_theta.to_csv("fbs_theta_comparison_summary.csv", index=False)

    return df_theta


def make_fbs_theta_plots(P, results, x, tgrid):
    plot_fbs_theta_costs(results)
    plot_fbs_theta_shifted_costs(results)
    plot_fbs_theta_residuals(results)
    plot_fbs_theta_control_changes(results)

    plot_fbs_theta_final_cost_bar(results)
    plot_fbs_theta_final_residual_bar(P, results)
    plot_fbs_theta_iterations_bar(results)

    plot_fbs_theta_final_controls(results, x)
    plot_fbs_theta_mean_control_time(results, tgrid)

    df_theta = plot_fbs_theta_summary_table(P, results)

    print(f"\nAll FBS theta comparison figures saved in: {FIGDIR_THETA}/")

    return df_theta


df_theta = make_fbs_theta_plots(P, results, x, tgrid)
