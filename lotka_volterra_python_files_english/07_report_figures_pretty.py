# Generated from Lotka-Volterra.ipynb. Empty cells and repeated plotting blocks were omitted.

"""
Pretty figures for the numerical comparison section of the report.

This file assumes that the optimization results have already been computed and
are available through a Python variable named `results`. It also uses `P.Umax`
when available. If `P` is not available, set UMAX manually below.

Typical use in your notebook/script, after computing `P` and `results`:

    import report_figures_pretty as figs
    figs.make_report_figures(P, results)

or, if you want to run this file directly in an environment where `P` and
`results` already exist:

    exec(open("report_figures_pretty.py").read())

The script generates the four figures used in the report:
  1. final discrete cost comparison,
  2. final projected-gradient residual comparison,
  3. final FBS cost as a function of theta,
  4. activity of the box constraints.

Files are saved in both PDF and PNG formats in FIGDIR.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, ScalarFormatter


# ============================================================
# User settings
# ============================================================

FIGDIR = "figures_report_pretty"
UMAX = None  # If P is not available, replace by your numerical value, e.g. UMAX = 1.0
DPI = 450

# Figure size adapted to LaTeX reports.
FIGSIZE = (6.2, 3.55)
FIGSIZE_WIDE = (6.7, 3.55)


# ============================================================
# Plot style
# ============================================================

@dataclass(frozen=True)
class Style:
    blue: str = "#4C78A8"
    orange: str = "#F58518"
    green: str = "#54A24B"
    red: str = "#E45756"
    purple: str = "#B279A2"
    gray: str = "#7F7F7F"
    dark: str = "#222222"
    grid: str = "#D9D9D9"


S = Style()


def set_report_style() -> None:
    """Set a clean style suitable for a LaTeX report."""
    plt.rcParams.update({
        "figure.figsize": FIGSIZE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": S.dark,
        "axes.titleweight": "bold",
        "axes.labelcolor": S.dark,
        "xtick.color": S.dark,
        "ytick.color": S.dark,
        "grid.color": S.grid,
        "grid.linewidth": 0.75,
        "grid.alpha": 0.85,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "cm",
    })


def clean_axes(ax: plt.Axes, *, ygrid: bool = True) -> None:
    """Apply final axis cosmetics."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"), dpi=DPI)
    plt.close(fig)


# ============================================================
# Helpers
# ============================================================

def method_label(method: str) -> str:
    """Short labels used in the figures."""
    if method.startswith("FBS theta="):
        theta = method.replace("FBS theta=", "")
        return rf"FBS, $\theta={theta}$"
    if "Projected" in method:
        return "Projected gradient"
    if "L-BFGS" in method or "LBFGS" in method:
        return "L-BFGS-B"
    return method


def get_theta(method: str) -> Optional[float]:
    match = re.search(r"theta=([0-9.]+)", method)
    return float(match.group(1)) if match else None


def get_fbs_results(results: Iterable[dict]) -> list[dict]:
    fbs = [r for r in results if str(r.get("method", "")).startswith("FBS")]
    return sorted(
        fbs,
        key=lambda r: get_theta(r["method"]) if get_theta(r["method"]) is not None else -1,
        reverse=True,
    )


def select_main_results(results: Iterable[dict]) -> list[dict]:
    """Select FBS theta=0.5, projected gradient, and L-BFGS-B when available."""
    results = list(results)
    selected = []

    # Prefer FBS theta=0.5 for the comparison figures.
    fbs = get_fbs_results(results)
    fbs_theta_05 = [
        r for r in fbs
        if abs((get_theta(r["method"]) or -999) - 0.5) < 1e-12
    ]

    if fbs_theta_05:
        selected.append(fbs_theta_05[0])
    elif fbs:
        selected.append(fbs[0])

    pg = [r for r in results if "Projected" in str(r.get("method", ""))]
    if pg:
        selected.append(pg[0])

    lbfgsb = [
        r for r in results
        if "L-BFGS" in str(r.get("method", "")) or "LBFGS" in str(r.get("method", ""))
    ]
    if lbfgsb:
        selected.append(lbfgsb[0])

    # Remove duplicates while preserving order.
    out = []
    seen = set()
    for r in selected:
        key = id(r)
        if key not in seen:
            out.append(r)
            seen.add(key)

    return out


def get_umax(P=None) -> float:
    if P is not None and hasattr(P, "Umax"):
        return float(P.Umax)
    if UMAX is not None:
        return float(UMAX)
    raise ValueError(
        "Cannot determine Umax. Pass P with attribute P.Umax, or set UMAX in this file."
    )


def projected_gradient_residual_local(P, u: np.ndarray, grad: np.ndarray) -> float:
    """Compute ||u - Pi(u-grad)||_infty for box constraints [0,Umax]."""
    umax = get_umax(P)
    projected = np.clip(u - grad, 0.0, umax)
    return float(np.max(np.abs(u - projected)))


def residual_value(P, res: dict) -> float:
    """Return the final projected-gradient residual, computing it if needed."""
    if "pg_residual" in res:
        return float(res["pg_residual"])

    hist = res.get("history", {})
    if "pg_residual" in hist and len(hist["pg_residual"]) > 0:
        return float(hist["pg_residual"][-1])

    if "u" in res and "grad" in res:
        return projected_gradient_residual_local(
            P,
            np.asarray(res["u"]),
            np.asarray(res["grad"]),
        )

    raise KeyError(f"Cannot compute residual for method {res.get('method', '<unknown>')}.")


def annotate_bars(
    ax: plt.Axes,
    bars,
    values,
    *,
    log_scale: bool = False,
    fmt: str = "{:.2g}",
) -> None:
    """Add small value labels above bars."""
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()

        if log_scale:
            ax.text(
                x,
                y * 1.25,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        else:
            offset = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            if value >= 0:
                ax.text(
                    x,
                    y + offset,
                    fmt.format(value),
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )
            else:
                ax.text(
                    x,
                    y - offset,
                    fmt.format(value),
                    ha="center",
                    va="top",
                    fontsize=8.5,
                )


# ============================================================
# Figure 6 -- final cost comparison
# ============================================================

def plot_final_cost_comparison(results: list[dict]) -> None:
    main = select_main_results(results)
    labels = [method_label(r["method"]) for r in main]
    values = np.array([r["cost"] for r in main], dtype=float)
    colors = [S.blue, S.orange, S.green][:len(values)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(
        labels,
        values,
        color=colors,
        edgecolor=S.dark,
        linewidth=0.6,
        width=0.62,
    )

    ax.axhline(0, color=S.dark, linewidth=0.9)
    ax.set_ylabel(r"Final cost $J_h(u_h)$")
    ax.set_title("Final discrete cost")

    clean_axes(ax)
    annotate_bars(ax, bars, values, fmt="{:.3f}")

    fig.tight_layout()
    save_figure(fig, "figure_06_final_cost_comparison")


# ============================================================
# Figure 7 -- final residual comparison
# ============================================================

def plot_final_residual_comparison(P, results: list[dict]) -> None:
    main = select_main_results(results)
    labels = [method_label(r["method"]) for r in main]
    values = np.maximum(
        np.array([residual_value(P, r) for r in main], dtype=float),
        1e-16,
    )
    colors = [S.blue, S.orange, S.green][:len(values)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(
        labels,
        values,
        color=colors,
        edgecolor=S.dark,
        linewidth=0.6,
        width=0.62,
    )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_ylabel("Projected-gradient residual")
    ax.set_title("Final optimality residual")

    clean_axes(ax)
    annotate_bars(ax, bars, values, log_scale=True, fmt="{:.1e}")

    fig.tight_layout()
    save_figure(fig, "figure_07_final_projected_gradient_residual")


# ============================================================
# Figure 8 -- final FBS cost as a function of theta
# ============================================================

def plot_fbs_theta_final_cost(results: list[dict]) -> None:
    fbs = get_fbs_results(results)

    if not fbs:
        raise ValueError("No FBS results found. Method names should start with 'FBS'.")

    theta = np.array([get_theta(r["method"]) for r in fbs], dtype=float)
    costs = np.array([r["cost"] for r in fbs], dtype=float)
    labels = [rf"$\theta={x:g}$" for x in theta]

    colors = plt.cm.Blues(np.linspace(0.45, 0.85, len(costs)))

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.bar(
        labels,
        costs,
        color=colors,
        edgecolor=S.dark,
        linewidth=0.6,
        width=0.62,
    )

    ax.axhline(0, color=S.dark, linewidth=0.9)
    ax.set_ylabel(r"Final cost $J_h(u_h)$")
    ax.set_title(r"Influence of the relaxation parameter $\theta$")

    clean_axes(ax)
    annotate_bars(ax, bars, costs, fmt="{:.2f}")

    fig.tight_layout()
    save_figure(fig, "figure_08_fbs_theta_final_cost")


# ============================================================
# Figure 9 -- activity of the box constraints
# ============================================================

def plot_constraint_activity(P, results: list[dict]) -> None:
    main = select_main_results(results)
    labels = [method_label(r["method"]) for r in main]
    umax = get_umax(P)
    eps = 1e-10

    active_lower = []
    inactive = []
    active_upper = []

    for res in main:
        u = np.asarray(res["u"])

        lower = float(np.mean(u <= eps))
        upper = float(np.mean(u >= umax - eps))
        mid = max(0.0, 1.0 - lower - upper)

        active_lower.append(lower)
        inactive.append(mid)
        active_upper.append(upper)

    active_lower = np.array(active_lower)
    inactive = np.array(inactive)
    active_upper = np.array(active_upper)

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    ax.bar(
        x,
        active_lower,
        color=S.blue,
        edgecolor="white",
        linewidth=0.8,
        label=r"$u_h=0$",
    )
    ax.bar(
        x,
        inactive,
        bottom=active_lower,
        color=S.orange,
        edgecolor="white",
        linewidth=0.8,
        label=r"$0<u_h<U_2^{\max}$",
    )
    ax.bar(
        x,
        active_upper,
        bottom=active_lower + inactive,
        color=S.green,
        edgecolor="white",
        linewidth=0.8,
        label=r"$u_h=U_2^{\max}$",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of space-time control nodes")
    ax.set_title("Activity of the box constraints")
    ax.yaxis.set_major_formatter(ScalarFormatter())

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
    )

    clean_axes(ax)

    fig.tight_layout()
    save_figure(fig, "figure_09_constraint_activity")


# ============================================================
# Main entry point
# ============================================================

def make_report_figures(P, results: list[dict]) -> None:
    """Generate the four report figures."""
    set_report_style()
    os.makedirs(FIGDIR, exist_ok=True)

    plot_final_cost_comparison(results)
    plot_final_residual_comparison(P, results)
    plot_fbs_theta_final_cost(results)
    plot_constraint_activity(P, results)

    print(f"Saved the four report figures in: {FIGDIR}/")
    print("Both PDF and PNG versions were created.")


# If the file is executed in a namespace where P and results already exist,
# generate the figures automatically. Otherwise, simply define the functions.
if __name__ == "__main__":
    if "results" not in globals():
        raise RuntimeError(
            "Variable 'results' was not found. Run your optimization code first, "
            "then call make_report_figures(P, results)."
        )

    P_obj = globals().get("P", None)
    make_report_figures(P_obj, globals()["results"])
