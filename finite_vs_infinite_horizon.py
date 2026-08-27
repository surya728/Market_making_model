

from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from simulator.engine import SimulationEngine
from strategy.reservation_infinite import InfiniteHorizonReservationPrice

# --------------------------------------------------------------------------
# Shared / common parameters -- copied verbatim from main.py's horizon
# section. Do not change any of these values.
# --------------------------------------------------------------------------
GAMMA = 0.1
N_RUNS = 300
SEED = 42

COMMON_PARAMS = dict(
    s0=100.0,
    dt=0.005,
    sigma=2.0,
    gamma=GAMMA,
    k=1.5,
    A=140.0,
    initial_quantity=0,
    initial_cash=0.0,
)

FINITE_T = 1.0
INFINITE_T = FINITE_T
INFINITE_Q_MAX = 10
INFINITE_OMEGA = InfiniteHorizonReservationPrice.omega_for_q_max(
    gamma=COMMON_PARAMS["gamma"], sigma=COMMON_PARAMS["sigma"], q_max=INFINITE_Q_MAX,
)

OUTPUT_DIR = os.path.join("results", "horizon_comparison")


# --------------------------------------------------------------------------
# Run the experiment
# --------------------------------------------------------------------------

def run_experiment():
    print(
        f"\nCommon parameters: s0={COMMON_PARAMS['s0']}, sigma={COMMON_PARAMS['sigma']}, "
        f"gamma={COMMON_PARAMS['gamma']}, k={COMMON_PARAMS['k']}, A={COMMON_PARAMS['A']}, "
        f"dt={COMMON_PARAMS['dt']}, q0={COMMON_PARAMS['initial_quantity']}, "
        f"n_runs={N_RUNS}, seed={SEED}"
    )
    print(f"Finite-horizon:   T={FINITE_T}")
    print(f"Infinite-horizon: omega={INFINITE_OMEGA:.4f} (derived from q_max={INFINITE_Q_MAX}), "
          f"T={INFINITE_T} (duration only, not used in pricing)")

    finite_engine = SimulationEngine(
        T=FINITE_T,
        model="finite_horizon",
        seed=SEED,
        **COMMON_PARAMS,
    )
    infinite_engine = SimulationEngine(
        T=INFINITE_T,
        model="infinite_horizon",
        omega=INFINITE_OMEGA,
        q_max=INFINITE_Q_MAX,
        seed=SEED,
        **COMMON_PARAMS,
    )

    finite_mc = finite_engine.run_monte_carlo(n_runs=N_RUNS, strategy="inventory")
    infinite_mc = infinite_engine.run_monte_carlo(n_runs=N_RUNS, strategy="inventory")

    finite_stats = SimulationEngine.summarise(finite_mc)
    infinite_stats = SimulationEngine.summarise(infinite_mc)

    # Common-random-number check: both engines were constructed from the
    # same seed, so the underlying mid-price path for run 0 must match.
    finite_single = finite_engine.run(strategy="inventory")
    infinite_single = infinite_engine.run(strategy="inventory")
    finite_mid = [r.mid_price for r in finite_single.path]
    infinite_mid = [r.mid_price for r in infinite_single.path]
    crn_ok = len(finite_mid) == len(infinite_mid) and np.allclose(finite_mid, infinite_mid)

    return finite_mc, infinite_mc, finite_stats, infinite_stats, crn_ok


def _print_comparison_table(finite_stats: dict, infinite_stats: dict) -> None:
    sep = "-" * 74
    rows = [
        ("N simulations", "n_runs", "d"),
        ("Mean final P&L", "mean_profit", ".4f"),
        ("Std final P&L", "std_profit", ".4f"),
        ("Mean final inventory", "mean_final_q", ".4f"),
        ("Std final inventory", "std_final_q", ".4f"),
        ("Mean |inventory|", "mean_abs_inventory", ".4f"),
        ("Max |inventory|", "max_abs_inventory", ".4f"),
        ("Average spread", "average_spread", ".4f"),
        ("Mean buy fills", "mean_buy_fills", ".2f"),
        ("Mean sell fills", "mean_sell_fills", ".2f"),
        ("Mean total fills", "mean_total_fills", ".2f"),
    ]
    print()
    print(sep)
    print(f"  {'Metric':<24s}{'Finite Horizon':>22s}{'Infinite Horizon':>22s}")
    print(sep)
    for label, key, fmt in rows:
        fv = format(finite_stats[key], fmt)
        iv = format(infinite_stats[key], fmt)
        print(f"  {label:<24s}{fv:>22s}{iv:>22s}")
    print(sep)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

COLOR_FINITE = "#1f77b4"
COLOR_INFINITE = "#d62728"


def _annotate_stats(ax, label, values, color, y):
    ax.text(
        0.98, y,
        f"{label}: mean={np.mean(values):.2f}, std={np.std(values):.2f}, n={len(values)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color=color,
    )


def plot_pnl_distribution(finite_pnl, infinite_pnl, save_path):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    bins = np.histogram_bin_edges(np.concatenate([finite_pnl, infinite_pnl]), bins=30)
    ax.hist(finite_pnl, bins=bins, alpha=0.55, label="Finite Horizon", color=COLOR_FINITE, edgecolor="black")
    ax.hist(infinite_pnl, bins=bins, alpha=0.55, label="Infinite Horizon", color=COLOR_INFINITE, edgecolor="black")
    ax.axvline(np.mean(finite_pnl), color=COLOR_FINITE, linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(infinite_pnl), color=COLOR_INFINITE, linestyle="--", linewidth=1.5)
    ax.set_title("Final P&L: Finite vs Infinite Horizon", fontsize=13, fontweight="bold")
    ax.set_xlabel("Final P&L")
    ax.set_ylabel("Frequency")
    _annotate_stats(ax, "Finite", finite_pnl, COLOR_FINITE, 0.97)
    _annotate_stats(ax, "Infinite", infinite_pnl, COLOR_INFINITE, 0.90)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_inventory_distribution(finite_inv, infinite_inv, save_path):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    lo = int(min(finite_inv.min(), infinite_inv.min()))
    hi = int(max(finite_inv.max(), infinite_inv.max()))
    bins = np.arange(lo - 0.5, hi + 1.5, 1)
    ax.hist(finite_inv, bins=bins, alpha=0.55, label="Finite Horizon", color=COLOR_FINITE, edgecolor="black")
    ax.hist(infinite_inv, bins=bins, alpha=0.55, label="Infinite Horizon", color=COLOR_INFINITE, edgecolor="black")
    ax.axvline(np.mean(finite_inv), color=COLOR_FINITE, linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(infinite_inv), color=COLOR_INFINITE, linestyle="--", linewidth=1.5)
    ax.set_title("Final Inventory: Finite vs Infinite Horizon", fontsize=13, fontweight="bold")
    ax.set_xlabel("Final Inventory (shares)")
    ax.set_ylabel("Frequency")
    _annotate_stats(ax, "Finite", finite_inv, COLOR_FINITE, 0.97)
    _annotate_stats(ax, "Infinite", infinite_inv, COLOR_INFINITE, 0.90)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_average_spread(finite_stats, infinite_stats, save_path):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    labels = ["Finite Horizon", "Infinite Horizon"]
    values = [finite_stats["average_spread"], infinite_stats["average_spread"]]
    bars = ax.bar(labels, values, color=[COLOR_FINITE, COLOR_INFINITE], edgecolor="black")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=11)
    ax.set_title("Average Spread: Finite vs Infinite Horizon", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Spread")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_fill_comparison(finite_stats, infinite_stats, save_path):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    categories = ["Mean buy fills", "Mean sell fills", "Mean total fills"]
    finite_vals = [finite_stats["mean_buy_fills"], finite_stats["mean_sell_fills"], finite_stats["mean_total_fills"]]
    infinite_vals = [infinite_stats["mean_buy_fills"], infinite_stats["mean_sell_fills"], infinite_stats["mean_total_fills"]]
    x = np.arange(len(categories))
    width = 0.35
    b1 = ax.bar(x - width / 2, finite_vals, width, label="Finite Horizon", color=COLOR_FINITE, edgecolor="black")
    b2 = ax.bar(x + width / 2, infinite_vals, width, label="Infinite Horizon", color=COLOR_INFINITE, edgecolor="black")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title("Order Fills: Finite vs Infinite Horizon", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_inventory_risk(finite_stats, infinite_stats, save_path):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    categories = ["Std final inventory", "Mean |inventory|", "Max |inventory|"]
    finite_vals = [finite_stats["std_final_q"], finite_stats["mean_abs_inventory"], finite_stats["max_abs_inventory"]]
    infinite_vals = [infinite_stats["std_final_q"], infinite_stats["mean_abs_inventory"], infinite_stats["max_abs_inventory"]]
    x = np.arange(len(categories))
    width = 0.35
    b1 = ax.bar(x - width / 2, finite_vals, width, label="Finite Horizon", color=COLOR_FINITE, edgecolor="black")
    b2 = ax.bar(x + width / 2, infinite_vals, width, label="Infinite Horizon", color=COLOR_INFINITE, edgecolor="black")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title("Inventory Risk: Finite vs Infinite Horizon", fontsize=13, fontweight="bold")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# CSV summary
# --------------------------------------------------------------------------

def save_csv_summary(finite_stats, infinite_stats, save_path):
    rows = [
        ("mean_final_pnl", finite_stats["mean_profit"], infinite_stats["mean_profit"]),
        ("std_final_pnl", finite_stats["std_profit"], infinite_stats["std_profit"]),
        ("mean_final_inventory", finite_stats["mean_final_q"], infinite_stats["mean_final_q"]),
        ("std_final_inventory", finite_stats["std_final_q"], infinite_stats["std_final_q"]),
        ("mean_abs_inventory", finite_stats["mean_abs_inventory"], infinite_stats["mean_abs_inventory"]),
        ("max_abs_inventory", finite_stats["max_abs_inventory"], infinite_stats["max_abs_inventory"]),
        ("average_spread", finite_stats["average_spread"], infinite_stats["average_spread"]),
        ("mean_buy_fills", finite_stats["mean_buy_fills"], infinite_stats["mean_buy_fills"]),
        ("mean_sell_fills", finite_stats["mean_sell_fills"], infinite_stats["mean_sell_fills"]),
        ("mean_total_fills", finite_stats["mean_total_fills"], infinite_stats["mean_total_fills"]),
    ]
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "finite_horizon", "infinite_horizon"])
        for metric, fv, iv in rows:
            writer.writerow([metric, fv, iv])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    finite_mc, infinite_mc, finite_stats, infinite_stats, crn_ok = run_experiment()

    # Pull the 300 individual observations straight out of the Monte Carlo
    # result pools -- no re-simulation, no aggregate-only shortcuts.
    finite_pnl = np.array([r.final_pnl for r in finite_mc])
    infinite_pnl = np.array([r.final_pnl for r in infinite_mc])
    finite_inventory = np.array([r.final_inventory for r in finite_mc])
    infinite_inventory = np.array([r.final_inventory for r in infinite_mc])

    assert len(finite_pnl) == N_RUNS
    assert len(infinite_pnl) == N_RUNS
    assert len(finite_inventory) == N_RUNS
    assert len(infinite_inventory) == N_RUNS

    print("\n" + "=" * 60)
    print("300-RUN FINITE VS INFINITE HORIZON RESULTS")
    print("=" * 60)
    _print_comparison_table(finite_stats, infinite_stats)
    print(f"\nCommon-random-number mid-price-path check: {'PASS' if crn_ok else 'FAIL'}")

    plot_pnl_distribution(finite_pnl, infinite_pnl, os.path.join(OUTPUT_DIR, "pnl_distribution.png"))
    plot_inventory_distribution(finite_inventory, infinite_inventory, os.path.join(OUTPUT_DIR, "inventory_distribution.png"))
    plot_average_spread(finite_stats, infinite_stats, os.path.join(OUTPUT_DIR, "average_spread.png"))
    plot_fill_comparison(finite_stats, infinite_stats, os.path.join(OUTPUT_DIR, "fill_comparison.png"))
    plot_inventory_risk(finite_stats, infinite_stats, os.path.join(OUTPUT_DIR, "inventory_risk.png"))

    csv_path = os.path.join(OUTPUT_DIR, "horizon_comparison_summary.csv")
    save_csv_summary(finite_stats, infinite_stats, csv_path)

    print("\nPlots saved to:")
    print(f"{OUTPUT_DIR}/")
    print("\nSummary saved to:")
    print(csv_path)
    plt.show()

if __name__ == "__main__":
    main()
