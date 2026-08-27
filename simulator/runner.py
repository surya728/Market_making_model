"""
simulator/runner.py

SimulationRunner for the Avellaneda-Stoikov (2008) market-making framework.

Orchestrates the full 1 000-run Monte Carlo experiment from section 3.3 of
the paper, comparing the inventory-aware strategy against the symmetric
benchmark across the three risk-aversion levels examined in Tables 1–3:

    gamma = 0.1  (Table 1) — moderately risk-averse
    gamma = 0.01 (Table 2) — nearly risk-neutral
    gamma = 1.0  (Table 3) — highly risk-averse

The runner follows the exact comparison protocol described in the paper:

  1. Run N inventory-strategy simulations to obtain a pool of results.
  2. Compute the mean spread from those runs.
  3. Run N symmetric-strategy simulations using that same mean spread,
     so the only difference between strategies is *where* quotes are
     centred (reservation price vs raw mid-price).
  4. Report mean profit, profit std, mean final inventory, final inventory
     std — exactly the five columns of Tables 1–3.

Paper: Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a
       limit order book. Quantitative Finance 8(3), 217–224.

Usage
-----
    # Reproduce Table 1 (gamma = 0.1):
    python -m simulator.runner

    # Programmatic:
    from simulator.runner import SimulationRunner
    runner = SimulationRunner(gamma=0.1, n_runs=1000, seed=42)
    report = runner.run()
    runner.print_report(report)
"""

from __future__ import annotations

import logging
import math
import time
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulator.engine import SimulationEngine, SimulationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StrategyStats:
    """
    Summary statistics for one strategy over N Monte Carlo runs.

    Mirrors the column layout of Tables 1–3 in the paper.

    Attributes
    ----------
    strategy : str
        'inventory' or 'symmetric'.
    n_runs : int
        Number of completed simulation paths.
    average_spread : float
        Mean bid-ask spread averaged over time and over all runs.
    mean_profit : float
        Mean terminal P&L  E[X_T + q_T * S_T].
    std_profit : float
        Standard deviation of terminal P&L.
    mean_final_q : float
        Mean residual inventory at time T.
    std_final_q : float
        Standard deviation of residual inventory at time T.
    mean_buy_fills : float
        Mean number of buy fills per run.
    mean_sell_fills : float
        Mean number of sell fills per run.
    elapsed_seconds : float
        Wall-clock time to complete all runs.
    """
    strategy: str
    n_runs: int
    average_spread: float
    mean_profit: float
    std_profit: float
    mean_final_q: float
    std_final_q: float
    mean_buy_fills: float
    mean_sell_fills: float
    elapsed_seconds: float


@dataclass
class ComparisonReport:
    """
    Full comparison report produced by SimulationRunner.run().

    Attributes
    ----------
    inventory : StrategyStats
        Results for the inventory-aware strategy.
    symmetric : StrategyStats
        Results for the symmetric benchmark.
    gamma : float
        Risk-aversion parameter used in this experiment.
    params : dict
        All engine parameters for reproducibility.
    profit_reduction_pct : float
        How much lower mean profit is for the inventory strategy vs symmetric
        (negative = inventory strategy earned less, as expected from the paper).
    std_reduction_pct : float
        How much lower profit std is for the inventory strategy vs symmetric
        (positive = inventory strategy has less risk, the key result).
    inv_std_reduction_pct : float
        Reduction in final-inventory std (inventory vs symmetric).
    """
    inventory: StrategyStats
    symmetric: StrategyStats
    gamma: float
    params: dict
    profit_reduction_pct: float
    std_reduction_pct: float
    inv_std_reduction_pct: float


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _compute_stats(
    results: List[SimulationResult],
    strategy_label: str,
    elapsed: float,
) -> StrategyStats:
    """
    Aggregate a list of SimulationResult objects into a StrategyStats record.

    Parameters
    ----------
    results : list[SimulationResult]
    strategy_label : str
    elapsed : float
        Wall-clock seconds taken to generate the results.

    Returns
    -------
    StrategyStats
    """
    pnls   = np.array([r.final_pnl       for r in results], dtype=float)
    inv    = np.array([r.final_inventory  for r in results], dtype=float)
    sprd   = np.array([r.average_spread   for r in results], dtype=float)
    buys   = np.array([r.total_buy_fills  for r in results], dtype=float)
    sells  = np.array([r.total_sell_fills for r in results], dtype=float)

    return StrategyStats(
        strategy=strategy_label,
        n_runs=len(results),
        average_spread=float(np.mean(sprd)),
        mean_profit=float(np.mean(pnls)),
        std_profit=float(np.std(pnls)),
        mean_final_q=float(np.mean(inv)),
        std_final_q=float(np.std(inv)),
        mean_buy_fills=float(np.mean(buys)),
        mean_sell_fills=float(np.mean(sells)),
        elapsed_seconds=elapsed,
    )


def _pct_change(new: float, old: float) -> float:
    """Return 100 * (new - old) / |old|, safe against zero denominator."""
    if abs(old) < 1e-12:
        return 0.0
    return 100.0 * (new - old) / abs(old)


# ---------------------------------------------------------------------------
# SimulationRunner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """
    Orchestrates the full 1 000-run Monte Carlo comparison from section 3.3.

    Parameters
    ----------
    s0 : float
        Initial mid-price (paper: 100).
    T : float
        Terminal time (paper: 1).
    dt : float
        Time step (paper: 0.005).
    sigma : float
        Mid-price volatility σ (paper: 2).
    gamma : float
        Risk-aversion coefficient γ.  Paper studies 0.1, 0.01, 1.0.
    k : float
        Order-arrival decay (paper: 1.5).
    A : float
        Baseline arrival intensity (paper: 140).
    n_runs : int
        Number of Monte Carlo paths per strategy (paper: 1000).
    seed : int, optional
        Master random seed for full reproducibility.
    log_level : str
        Python logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    """

    # Default parameters from the paper (section 3.3)
    _PAPER_DEFAULTS: dict = dict(
        s0=100.0, T=1.0, dt=0.005,
        sigma=2.0, k=1.5, A=140.0,
    )

    def __init__(
        self,
        s0: float = 100.0,
        T: float = 1.0,
        dt: float = 0.005,
        sigma: float = 2.0,
        gamma: float = 0.1,
        k: float = 1.5,
        A: float = 140.0,
        n_runs: int = 1000,
        seed: Optional[int] = 42,
        log_level: str = "INFO",
    ) -> None:
        # ---- configure logging ----------------------------------------------
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            level=getattr(logging, log_level.upper(), logging.INFO),
        )

        # ---- validate -------------------------------------------------------
        for name, val in [("s0", s0), ("T", T), ("dt", dt),
                          ("sigma", sigma), ("gamma", gamma),
                          ("k", k), ("A", A), ("n_runs", n_runs)]:
            if val <= 0:
                raise ValueError(f"'{name}' must be strictly positive, got {val}.")
        if dt >= T:
            raise ValueError("dt must be smaller than T.")

        self.s0 = s0
        self.T = T
        self.dt = dt
        self.sigma = sigma
        self.gamma = gamma
        self.k = k
        self.A = A
        self.n_runs = n_runs
        self.seed = seed

        self._params: dict = dict(
            s0=s0, T=T, dt=dt, sigma=sigma,
            gamma=gamma, k=k, A=A,
            n_runs=n_runs, seed=seed,
        )

        # ---- build engine ---------------------------------------------------
        self._engine = SimulationEngine(
            s0=s0, T=T, dt=dt,
            sigma=sigma, gamma=gamma,
            k=k, A=A,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_inventory_mc(self) -> Tuple[List[SimulationResult], float]:
        """
        Run n_runs inventory-strategy paths.

        Returns
        -------
        (results, elapsed_seconds)
        """
        logger.info(
            "=== INVENTORY strategy: running %d simulations (gamma=%.3f) ===",
            self.n_runs, self.gamma,
        )
        t0 = time.perf_counter()
        results = self._engine.run_monte_carlo(
            n_runs=self.n_runs,
            strategy="inventory",
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "Inventory MC done in %.2fs", elapsed
        )
        return results, elapsed

    def _run_symmetric_mc(
        self,
        fixed_half_spread: float,
    ) -> Tuple[List[SimulationResult], float]:
        """
        Run n_runs symmetric-strategy paths with a fixed half-spread.

        Parameters
        ----------
        fixed_half_spread : float
            Half of the mean spread from the inventory runs.

        Returns
        -------
        (results, elapsed_seconds)
        """
        logger.info(
            "=== SYMMETRIC strategy: running %d simulations "
            "(half_spread=%.4f) ===",
            self.n_runs, fixed_half_spread,
        )
        t0 = time.perf_counter()
        results = self._engine.run_monte_carlo(
            n_runs=self.n_runs,
            strategy="symmetric",
            fixed_half_spread=fixed_half_spread,
        )
        elapsed = time.perf_counter() - t0
        logger.info("Symmetric MC done in %.2fs", elapsed)
        return results, elapsed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ComparisonReport:
        """
        Execute the full comparison experiment and return a ComparisonReport.

        Protocol (mirrors section 3.3 of the paper):
          1. Run n_runs inventory-strategy simulations.
          2. Compute mean spread from those runs.
          3. Run n_runs symmetric-strategy simulations with that spread.
          4. Compute and return all comparison statistics.

        Returns
        -------
        ComparisonReport
        """
        logger.info(
            "SimulationRunner starting: gamma=%.4f  n_runs=%d  seed=%s",
            self.gamma, self.n_runs, self.seed,
        )

        # ---- Step 1: inventory runs ----------------------------------------
        inv_results, inv_elapsed = self._run_inventory_mc()

        # ---- Step 2: derive mean spread ------------------------------------
        all_spreads = [
            r.average_spread for r in inv_results
            if not math.isnan(r.average_spread)
        ]
        mean_spread = float(np.mean(all_spreads)) if all_spreads else 0.0
        half_spread = mean_spread / 2.0

        logger.info(
            "Mean inventory spread = %.4f  =>  fixed half-spread = %.4f",
            mean_spread, half_spread,
        )

        # ---- Step 3: symmetric runs ----------------------------------------
        sym_results, sym_elapsed = self._run_symmetric_mc(half_spread)

        # ---- Step 4: aggregate stats ---------------------------------------
        inv_stats = _compute_stats(inv_results, "inventory", inv_elapsed)
        sym_stats = _compute_stats(sym_results, "symmetric", sym_elapsed)

        # Comparison percentages (inventory relative to symmetric)
        profit_reduction = _pct_change(
            inv_stats.mean_profit, sym_stats.mean_profit
        )
        std_reduction = _pct_change(
            inv_stats.std_profit, sym_stats.std_profit
        )
        inv_std_reduction = _pct_change(
            inv_stats.std_final_q, sym_stats.std_final_q
        )

        report = ComparisonReport(
            inventory=inv_stats,
            symmetric=sym_stats,
            gamma=self.gamma,
            params=dict(self._params),
            profit_reduction_pct=profit_reduction,
            std_reduction_pct=std_reduction,
            inv_std_reduction_pct=inv_std_reduction,
        )

        logger.info("Comparison report ready.")
        return report

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: ComparisonReport) -> None:
        """
        Print a formatted comparison table matching the layout of
        Tables 1–3 in the Avellaneda-Stoikov paper.

        Parameters
        ----------
        report : ComparisonReport
        """
        sep  = "─" * 72
        sep2 = "═" * 72

        def _row(label: str, inv_val, sym_val, fmt: str = ".4f") -> str:
            return (
                f"  {label:<28s}"
                f"{format(inv_val, fmt):>18s}"
                f"{format(sym_val, fmt):>18s}"
            )

        print()
        print(sep2)
        print(
            f"  Avellaneda-Stoikov (2008) — Monte Carlo Results"
            f"  [gamma = {report.gamma}]"
        )
        print(sep2)
        print(
            f"  {'Metric':<28s}"
            f"{'Inventory':>18s}"
            f"{'Symmetric':>18s}"
        )
        print(sep)

        i = report.inventory
        s = report.symmetric

        print(_row("N simulations",       i.n_runs,          s.n_runs,          "d"))
        print(_row("Average spread",      i.average_spread,  s.average_spread))
        print(sep)
        print(_row("Mean profit",         i.mean_profit,     s.mean_profit))
        print(_row("Std(profit)",         i.std_profit,      s.std_profit))
        print(sep)
        print(_row("Mean final inventory",i.mean_final_q,    s.mean_final_q))
        print(_row("Std(final inventory)",i.std_final_q,     s.std_final_q))
        print(sep)
        print(_row("Mean buy fills",      i.mean_buy_fills,  s.mean_buy_fills,  ".1f"))
        print(_row("Mean sell fills",     i.mean_sell_fills, s.mean_sell_fills, ".1f"))
        print(sep)
        print(_row("Elapsed (s)",         i.elapsed_seconds, s.elapsed_seconds, ".2f"))
        print(sep2)

        # Interpretation summary
        print()
        print("  INTERPRETATION")
        print(sep)
        _sign  = lambda v: ("+" if v >= 0 else "") + f"{v:.2f}%"
        print(
            f"  Profit change (inv vs sym):         {_sign(report.profit_reduction_pct)}"
        )
        print(
            f"  Profit std change (inv vs sym):     {_sign(report.std_reduction_pct)}"
            + ("  ← lower risk" if report.std_reduction_pct < 0 else "")
        )
        print(
            f"  Inv. std change  (inv vs sym):      {_sign(report.inv_std_reduction_pct)}"
            + ("  ← tighter inventory control" if report.inv_std_reduction_pct < 0 else "")
        )
        print(sep2)
        print()

    @staticmethod
    def to_json(report: ComparisonReport, indent: int = 2) -> str:
        """
        Serialise the report to a JSON string.

        Parameters
        ----------
        report : ComparisonReport
        indent : int

        Returns
        -------
        str
        """
        def _serial(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            raise TypeError(f"Not serialisable: {type(obj)}")

        return json.dumps(asdict(report), indent=indent, default=_serial)

    @staticmethod
    def save_json(report: ComparisonReport, path: str) -> None:
        """
        Write the report to a JSON file.

        Parameters
        ----------
        report : ComparisonReport
        path : str
            File path, e.g. 'results/gamma_0.1.json'.
        """
        payload = SimulationRunner.to_json(report)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(payload)
        logger.info("Report saved to %s", path)

    # ------------------------------------------------------------------
    # Batch runner — reproduces all three paper tables in one call
    # ------------------------------------------------------------------

    @classmethod
    def run_paper_tables(
        cls,
        n_runs: int = 1000,
        seed: Optional[int] = 42,
        print_results: bool = True,
    ) -> Dict[float, ComparisonReport]:
        """
        Run the full experiment for all three gamma values studied in the
        paper (Tables 1, 2, 3) and return a dict keyed by gamma.

        Parameters
        ----------
        n_runs : int
            Simulations per strategy per gamma (paper uses 1000).
        seed : int, optional
            Master seed; each gamma gets a deterministic child seed.
        print_results : bool
            If True, print each comparison table as it completes.

        Returns
        -------
        dict mapping gamma -> ComparisonReport
        """
        gammas = [0.1, 0.01, 1.0]   # Tables 1, 2, 3 respectively
        reports: Dict[float, ComparisonReport] = {}

        # Derive per-gamma seeds deterministically from the master seed
        seed_seq = np.random.SeedSequence(
            seed if seed is not None else 42
        )
        child_ints = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(len(gammas))]

        for gamma, child_seed in zip(gammas, child_ints):
            logger.info(
                "──────────────────────────────────────────────"
            )
            logger.info(
                "Running paper table for gamma = %.2f  (seed=%d)",
                gamma, child_seed,
            )
            runner = cls(
                gamma=gamma,
                n_runs=n_runs,
                seed=child_seed,
            )
            report = runner.run()
            reports[gamma] = report
            if print_results:
                cls.print_report(report)

        return reports

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SimulationRunner("
            f"gamma={self.gamma}, n_runs={self.n_runs}, "
            f"sigma={self.sigma}, T={self.T}, dt={self.dt}, "
            f"k={self.k}, A={self.A}, seed={self.seed})"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the Avellaneda-Stoikov (2008) Monte Carlo comparison. "
            "Defaults reproduce Table 1 (gamma=0.1, 1000 runs)."
        )
    )
    p.add_argument("--gamma",   type=float, default=0.1,
                   help="Risk-aversion coefficient (default: 0.1)")
    p.add_argument("--n-runs",  type=int,   default=1000,
                   help="Monte Carlo paths per strategy (default: 1000)")
    p.add_argument("--s0",      type=float, default=100.0,
                   help="Initial mid-price (default: 100)")
    p.add_argument("--T",       type=float, default=1.0,
                   help="Terminal time (default: 1)")
    p.add_argument("--dt",      type=float, default=0.005,
                   help="Time step (default: 0.005)")
    p.add_argument("--sigma",   type=float, default=2.0,
                   help="Volatility (default: 2)")
    p.add_argument("--k",       type=float, default=1.5,
                   help="Arrival decay (default: 1.5)")
    p.add_argument("--A",       type=float, default=140.0,
                   help="Arrival intensity (default: 140)")
    p.add_argument("--seed",    type=int,   default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--all-tables", action="store_true",
                   help="Reproduce all three paper tables (gamma=0.1/0.01/1.0)")
    p.add_argument("--save-json", type=str, default=None, metavar="PATH",
                   help="Save the JSON report to this file path")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (default: INFO)")
    return p


def main() -> None:
    """CLI entry point.  Run as:  python -m simulator.runner  [--options]"""
    parser = _build_parser()
    args = parser.parse_args()

    if args.all_tables:
        reports = SimulationRunner.run_paper_tables(
            n_runs=args.n_runs,
            seed=args.seed,
            print_results=True,
        )
        if args.save_json:
            # Serialise all three reports into one file
            combined = {
                str(g): json.loads(SimulationRunner.to_json(r))
                for g, r in reports.items()
            }
            with open(args.save_json, "w", encoding="utf-8") as fh:
                json.dump(combined, fh, indent=2)
            print(f"All reports saved to {args.save_json}")
    else:
        runner = SimulationRunner(
            s0=args.s0,
            T=args.T,
            dt=args.dt,
            sigma=args.sigma,
            gamma=args.gamma,
            k=args.k,
            A=args.A,
            n_runs=args.n_runs,
            seed=args.seed,
            log_level=args.log_level,
        )
        report = runner.run()
        SimulationRunner.print_report(report)

        if args.save_json:
            SimulationRunner.save_json(report, args.save_json)


if __name__ == "__main__":
    main()