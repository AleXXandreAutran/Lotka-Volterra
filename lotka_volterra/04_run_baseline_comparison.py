from optimal_control_core import *

# ============================================================
# 11. Run comparison
# ============================================================

def run_all_methods(P):
    results = []

    print("Running Forward--Backward Sweep...")
    results.append(run_forward_backward_sweep(P, theta=P.theta))

    print("Running Projected Gradient + Armijo...")
    results.append(run_projected_gradient_armijo(P))

    print("Running L-BFGS-B...")
    results.append(run_lbfgsb(P))

    return results


results = run_all_methods(P)


# ============================================================
# 12. Numerical summary table
# ============================================================

def summarize_results(P, results):
    rows = []

    for res in results:
        grad = res["grad"]
        u = res["u"]

        pg_res = projected_gradient_residual(P, u, grad)

        rows.append({
            "method": res["method"],
            "final_cost": res["cost"],
            "iterations": res["iterations"],
            "runtime_seconds": res["runtime"],
            "projected_gradient_residual": pg_res,
            "u_min": np.min(u),
            "u_max": np.max(u),
            "u_mean": np.mean(u),
            "v1_final_min": np.min(res["v1"][-1, :]),
            "v1_final_max": np.max(res["v1"][-1, :]),
            "v2_final_min": np.min(res["v2"][-1, :]),
            "v2_final_max": np.max(res["v2"][-1, :]),
        })

    df = pd.DataFrame(rows)
    return df


summary = summarize_results(P, results)

print("\n==================== COMPARISON TABLE ====================")
print(summary.to_string(index=False))

summary.to_csv("comparison_summary.csv", index=False)
print("\nSaved numerical summary to comparison_summary.csv")




# ============================================================
# 14. Extra diagnostic: constraint activity
# ============================================================

def activity_statistics(P, results):
    rows = []
    eps = 1e-10

    for res in results:
        u = res["u"]
        active_lower = np.mean(u <= eps)
        active_upper = np.mean(u >= P.Umax - eps)
        inactive = 1.0 - active_lower - active_upper

        rows.append({
            "method": res["method"],
            "active_lower_fraction": active_lower,
            "active_upper_fraction": active_upper,
            "inactive_fraction": inactive,
        })

    return pd.DataFrame(rows)


activity = activity_statistics(P, results)

print("\n==================== ACTIVITY TABLE ====================")
print(activity.to_string(index=False))

activity.to_csv("activity_summary.csv", index=False)
print("\nSaved activity summary to activity_summary.csv")


# ============================================================
# 15. Save full histories
# ============================================================

def save_histories(results):
    all_rows = []

    for res in results:
        hist = res["history"]
        max_len = max(len(v) for v in hist.values())

        for k in range(max_len):
            row = {"method": res["method"], "iteration": k}

            for key, values in hist.items():
                if k < len(values):
                    row[key] = values[k]
                else:
                    row[key] = np.nan

            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv("comparison_histories.csv", index=False)
    return df


histories_df = save_histories(results)
print("\nSaved histories to comparison_histories.csv")
