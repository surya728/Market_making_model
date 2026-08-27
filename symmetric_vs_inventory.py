from simulator.runner import SimulationRunner
from visualization.dashboard import Dashboard
import numpy as np


# --------------------------------------------------------
# Simulation parameters
# --------------------------------------------------------

GAMMA = 0.1
N_RUNS = 300
SEED = 42


# --------------------------------------------------------
# Run experiments
# --------------------------------------------------------

runner = SimulationRunner()

print("=" * 70)
print("   Avellaneda-Stoikov — Symmetric vs Inventory Strategy")
print("=" * 70)

print("\nRunning Monte Carlo simulations...")

report = runner.run_paper_tables(
    n_runs=N_RUNS,
    seed=SEED
)

print("\nSimulation finished.\n")


# --------------------------------------------------------
# Representative single-run paths
# --------------------------------------------------------

engine = runner._engine

inventory_result, symmetric_result = engine.run_both()


# --------------------------------------------------------
# Monte Carlo simulations — Inventory Strategy
# --------------------------------------------------------

print("Running Inventory Strategy Monte Carlo simulations...")

inventory_mc = engine.run_monte_carlo(
    n_runs=N_RUNS,
    strategy="inventory"
)


# --------------------------------------------------------
# Monte Carlo simulations — Symmetric Strategy
# --------------------------------------------------------

print("Running Symmetric Strategy Monte Carlo simulations...")

symmetric_mc = engine.run_monte_carlo(
    n_runs=N_RUNS,
    strategy="symmetric",
    fixed_half_spread=inventory_mc[0].average_spread / 2.0,
)


# --------------------------------------------------------
# Extract final P&L values
# --------------------------------------------------------

inventory_pnl = np.array(
    [result.final_pnl for result in inventory_mc]
)

symmetric_pnl = np.array(
    [result.final_pnl for result in symmetric_mc]
)


# --------------------------------------------------------
# Calculate statistics
# --------------------------------------------------------

inventory_mean = np.mean(inventory_pnl)
inventory_std = np.std(inventory_pnl)

symmetric_mean = np.mean(symmetric_pnl)
symmetric_std = np.std(symmetric_pnl)


# --------------------------------------------------------
# Print Monte Carlo results
# --------------------------------------------------------

print("\n")
print("=" * 70)
print("              FINAL P&L — 300 MONTE CARLO RUNS")
print("=" * 70)

print("\nInventory Strategy:")
print(f"  Mean final P&L       = {inventory_mean:.4f}")
print(f"  Std final P&L        = {inventory_std:.4f}")
print(f"  Number of runs       = {len(inventory_pnl)}")

print("\nSymmetric Strategy:")
print(f"  Mean final P&L       = {symmetric_mean:.4f}")
print(f"  Std final P&L        = {symmetric_std:.4f}")
print(f"  Number of runs       = {len(symmetric_pnl)}")

print("\n" + "-" * 70)

print("Comparison:")
print(
    f"  Inventory mean P&L  : {inventory_mean:.4f}"
)

print(
    f"  Symmetric mean P&L  : {symmetric_mean:.4f}"
)

print(
    f"  Inventory std       : {inventory_std:.4f}"
)

print(
    f"  Symmetric std       : {symmetric_std:.4f}"
)

print("=" * 70)


# --------------------------------------------------------
# Visualization
# --------------------------------------------------------

print("\nGenerating plots...")

dashboard = Dashboard()

dashboard.show_all(
    inventory_result,
    symmetric_result,
    inventory_mc,
    symmetric_mc,
    gamma=GAMMA,
    save_prefix="results/simulation"
)

print("\nPlots generated successfully.")