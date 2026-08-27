"""
visualization/dashboard.py

Matplotlib dashboard for the Avellaneda-Stoikov (2008) market-making simulation.

Produces four publication-quality plots that together reproduce the visual
analysis from section 3.3 of the paper:

  Panel 1 — Mid-price vs Bid/Ask quotes (Figure 1 in the paper)
             Shows how the optimal quotes track the mid-price and shift
             asymmetrically around the reservation price as inventory changes.

  Panel 2 — Inventory over time
             Reveals how the inventory-aware strategy reverts inventory to
             zero more aggressively than the symmetric strategy.

  Panel 3 — P&L over time (mark-to-market)
             Illustrates the smoother P&L profile of the inventory strategy
             vs the higher variance of the symmetric benchmark.

  Panel 4 — Final P&L histogram (Figure 2 / 3 / 4 in the paper)
             Compares the terminal P&L distributions across 1 000 runs,
             showing the tighter spread of the inventory strategy.

All functions accept plain lists / numpy arrays so they work without pandas.

Usage
-----
    # Quick 4-panel dashboard for a single simulation run:
    from visualization.dashboard import Dashboard
    dash = Dashboard()
    fig = dash.plot_single_run(inv_result, sym_result)
    fig.savefig("dashboard.png", dpi=150)

    # Full Monte Carlo histogram comparison:
    fig2 = dash.plot_monte_carlo(inv_results, sym_results, gamma=0.1)
    fig2.savefig("histogram.png", dpi=150)

    # All plots in one call:
    dash.show_all(inv_result, sym_result, inv_results, sym_results, gamma=0.1)
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, TYPE_CHECKING, Any

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Lazy import guard — engine types are used only for type hints / extraction
# ---------------------------------------------------------------------------
#from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from simulator.engine import SimulationResult, StepRecord
else:
    SimulationResult = Any
    StepRecord = Any


# ---------------------------------------------------------------------------
# Colour palette  (colour-blind safe, matches paper's monochrome intent)
# ---------------------------------------------------------------------------
_C = dict(
    mid        = "#1f1f1f",   # near-black    — mid-price
    bid        = "#2166ac",   # blue          — bid quote
    ask        = "#d6604d",   # red-orange    — ask quote
    reservation= "#4dac26",   # green         — reservation price
    inventory  = "#7b2d8b",   # purple        — inventory (inventory strategy)
    inventory_s= "#b2abd2",   # light purple  — inventory (symmetric)
    pnl_inv    = "#2166ac",   # blue          — P&L inventory
    pnl_sym    = "#d6604d",   # red-orange    — P&L symmetric
    fill_buy   = "#2166ac",   # blue dot      — buy fill marker
    fill_sell  = "#d6604d",   # red dot       — sell fill marker
    zero       = "#aaaaaa",   # grey          — zero reference line
    shading    = "#f0f0f0",   # light grey    — spread shading
)

_ALPHA_FILL  = 0.18
_ALPHA_SHADE = 0.25
_LW_MAIN     = 1.4
_LW_THIN     = 0.9
_MS_FILL     = 5          # marker size for fill events


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------

def _extract_path(
    result: "SimulationResult"
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Unpack a SimulationResult path into numpy arrays.

    Returns
    -------
    time, mid, bid, ask, reservation, inventory, pnl,
    risk, buy_fill_mask, sell_fill_mask
    """
    path = result.path
    time        = np.array([r.time        for r in path], dtype=float)
    mid         = np.array([r.mid_price   for r in path], dtype=float)
    bid         = np.array([r.bid         for r in path], dtype=float)
    ask         = np.array([r.ask         for r in path], dtype=float)
    reservation = np.array([r.reservation for r in path], dtype=float)
    inventory   = np.array([r.inventory   for r in path], dtype=float)
    pnl         = np.array([r.pnl         for r in path], dtype=float)
    risk        = np.array([r.risk_exposure for r in path], dtype=float)
    buy_mask    = np.array([r.buy_filled  for r in path], dtype=bool)
    sell_mask   = np.array([r.sell_filled for r in path], dtype=bool)
    return time, mid, bid, ask, reservation, inventory, pnl, risk, buy_mask, sell_mask


def _safe_nan(arr: np.ndarray) -> np.ndarray:
    """Replace inf with nan so matplotlib skips them gracefully."""
    out = arr.copy()
    out[~np.isfinite(out)] = np.nan
    return out


# ---------------------------------------------------------------------------
# Dashboard class
# ---------------------------------------------------------------------------

class Dashboard:
    """
    Matplotlib dashboard for Avellaneda-Stoikov simulation results.

    Parameters
    ----------
    style : str, optional
        Matplotlib style name (default: 'seaborn-v0_8-whitegrid' with a
        fallback to 'ggplot').
    figsize_single : tuple
        Figure size for the 4-panel single-run dashboard.
    figsize_hist : tuple
        Figure size for the Monte Carlo histogram comparison.
    dpi : int
        Resolution for rendered figures.
    """

    def __init__(
        self,
        style: Optional[str] = None,
        figsize_single: tuple[float, float] = (16, 12),
        figsize_hist: tuple[float, float] = (12, 5),
        dpi: int = 120,
    ) -> None:
        self.figsize_single = figsize_single
        self.figsize_hist   = figsize_hist
        self.dpi            = dpi
        self._apply_style(style)

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_style(style: Optional[str]) -> None:
        candidates = (
            [style] if style else
            ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot", "default"]
        )
        for s in candidates:
            try:
                plt.style.use(s)
                return
            except OSError:
                continue

    # ------------------------------------------------------------------
    # Panel 1 — Mid-price vs Bid/Ask quotes
    # ------------------------------------------------------------------

    def plot_quotes(
        self,
        result: "SimulationResult",
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        show_reservation: bool = True,
        show_fills: bool = True,
        show_spread_shading: bool = True,
    ) -> Axes:
        """
        Plot mid-price, bid quote, ask quote, and optionally the reservation
        price for a single simulation run.

        Reproduces Figure 1 of the paper.

        Parameters
        ----------
        result : SimulationResult
        ax : Axes, optional
            Axes to draw on.  A new figure is created if None.
        title : str, optional
        show_reservation : bool
            Overlay the reservation price r(s,q,t).
        show_fills : bool
            Mark buy/sell fill events with coloured dots.
        show_spread_shading : bool
            Shade the region between bid and ask.

        Returns
        -------
        Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4), dpi=self.dpi)

        (time, mid, bid, ask, reservation,
         inventory, pnl, risk, buy_mask, sell_mask) = _extract_path(result)

        bid = _safe_nan(bid)
        ask = _safe_nan(ask)

        # Spread shading
        if show_spread_shading:
            ax.fill_between(
                time, bid, ask,
                alpha=_ALPHA_SHADE,
                color=_C["shading"],
                label="_spread_shade",
                zorder=1,
            )

        # Core price lines
        ax.plot(time, mid, color=_C["mid"],  lw=_LW_MAIN,
                label="Mid-price $S_t$",    zorder=3)
        ax.plot(time, bid, color=_C["bid"],  lw=_LW_THIN,
                linestyle="--", label="Bid $p^b$",  zorder=3)
        ax.plot(time, ask, color=_C["ask"],  lw=_LW_THIN,
                linestyle="--", label="Ask $p^a$",  zorder=3)

        if show_reservation:
            ax.plot(time, reservation, color=_C["reservation"], lw=_LW_THIN,
                    linestyle=":", alpha=0.85,
                    label="Reservation $r(s,q,t)$", zorder=2)

        # Fill event markers
        if show_fills:
            if buy_mask.any():
                ax.scatter(
                    time[buy_mask], bid[buy_mask],
                    color=_C["fill_buy"], s=_MS_FILL ** 2,
                    zorder=5, label="Buy fill", marker="^",
                )
            if sell_mask.any():
                ax.scatter(
                    time[sell_mask], ask[sell_mask],
                    color=_C["fill_sell"], s=_MS_FILL ** 2,
                    zorder=5, label="Sell fill", marker="v",
                )

        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Price ($)")
        ax.set_title(
            title or f"Mid-price vs Bid/Ask Quotes  "
                     f"[γ={result.params.get('gamma','?')}]"
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        _despine(ax)
        return ax

    # ------------------------------------------------------------------
    # Panel 2 — Inventory over time
    # ------------------------------------------------------------------

    def plot_inventory(
        self,
        inv_result: "SimulationResult",
        sym_result: Optional["SimulationResult"] = None,
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
    ) -> Axes:
        """
        Plot inventory q_t over time for one or two strategies.

        Parameters
        ----------
        inv_result : SimulationResult
            Inventory-aware strategy result.
        sym_result : SimulationResult, optional
            Symmetric benchmark result (overlaid if provided).
        ax : Axes, optional
        title : str, optional

        Returns
        -------
        Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 3), dpi=self.dpi)

        time_i, *_, inventory_i, _, _, _, _ = _extract_path(inv_result)

        ax.axhline(0, color=_C["zero"], lw=0.8, linestyle="--", zorder=1)

        if sym_result is not None:
            time_s, *_, inventory_s, _, _, _, _ = _extract_path(sym_result)
            ax.step(
                time_s, inventory_s,
                where="post",
                color=_C["inventory_s"],
                lw=_LW_THIN,
                alpha=0.75,
                label="Symmetric",
                zorder=2,
            )

        ax.step(
            time_i, inventory_i,
            where="post",
            color=_C["inventory"],
            lw=_LW_MAIN,
            label="Inventory strategy",
            zorder=3,
        )

        # Shade positive (long) and negative (short) regions
        ax.fill_between(
            time_i, inventory_i, 0,
            where=(inventory_i > 0),
            step="post",
            alpha=_ALPHA_FILL,
            color=_C["bid"],
            label="_long_shade",
        )
        ax.fill_between(
            time_i, inventory_i, 0,
            where=(inventory_i < 0),
            step="post",
            alpha=_ALPHA_FILL,
            color=_C["ask"],
            label="_short_shade",
        )

        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Inventory $q_t$ (shares)")
        ax.set_title(title or "Inventory over Time")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        handles = [
            Line2D([0], [0], color=_C["inventory"],   lw=_LW_MAIN, label="Inventory strategy"),
        ]
        if sym_result is not None:
            handles.append(
                Line2D([0], [0], color=_C["inventory_s"], lw=_LW_THIN,
                       alpha=0.75, label="Symmetric")
            )
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)
        _despine(ax)
        return ax

    # ------------------------------------------------------------------
    # Panel 3 — P&L over time
    # ------------------------------------------------------------------

    def plot_pnl_path(
        self,
        inv_result: "SimulationResult",
        sym_result: Optional["SimulationResult"] = None,
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        show_risk: bool = False,
    ) -> Axes:
        """
        Plot mark-to-market P&L  (X_t + q_t * S_t)  over the simulation.

        Parameters
        ----------
        inv_result : SimulationResult
        sym_result : SimulationResult, optional
        ax : Axes, optional
        title : str, optional
        show_risk : bool
            Overlay the inventory risk exposure on a twin y-axis.

        Returns
        -------
        Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 3), dpi=self.dpi)

        time_i, _, _, _, _, _, pnl_i, risk_i, _, _ = _extract_path(inv_result)

        ax.axhline(0, color=_C["zero"], lw=0.8, linestyle="--", zorder=1)

        if sym_result is not None:
            time_s, _, _, _, _, _, pnl_s, _, _, _ = _extract_path(sym_result)
            ax.plot(
                time_s, pnl_s,
                color=_C["pnl_sym"],
                lw=_LW_THIN,
                alpha=0.7,
                label="Symmetric",
                zorder=2,
            )

        ax.plot(
            time_i, pnl_i,
            color=_C["pnl_inv"],
            lw=_LW_MAIN,
            label="Inventory strategy",
            zorder=3,
        )

        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Mark-to-Market P&L ($)")
        ax.set_title(title or "P&L over Time  ($X_t + q_t S_t$)")

        handles = [
            Line2D([0], [0], color=_C["pnl_inv"], lw=_LW_MAIN,
                   label="Inventory strategy"),
        ]
        if sym_result is not None:
            handles.append(
                Line2D([0], [0], color=_C["pnl_sym"], lw=_LW_THIN,
                       alpha=0.7, label="Symmetric")
            )

        if show_risk:
            ax2 = ax.twinx()
            ax2.plot(
                time_i, risk_i,
                color=_C["reservation"],
                lw=_LW_THIN,
                linestyle=":",
                alpha=0.75,
                label="Risk exposure",
            )
            ax2.set_ylabel("Risk exposure", color=_C["reservation"], fontsize=8)
            ax2.tick_params(axis="y", labelcolor=_C["reservation"], labelsize=7)
            handles.append(
                Line2D([0], [0], color=_C["reservation"], lw=_LW_THIN,
                       linestyle=":", alpha=0.75, label="Risk exposure")
            )

        ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.8)
        _despine(ax)
        return ax

    # ------------------------------------------------------------------
    # Panel 4 — Final P&L histogram (Monte Carlo)
    # ------------------------------------------------------------------

    def plot_pnl_histogram(
        self,
        inv_results: List["SimulationResult"],
        sym_results: List["SimulationResult"],
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        gamma: Optional[float] = None,
        n_bins: int = 40,
        show_stats: bool = True,
    ) -> Axes:
        """
        Plot overlapping histograms of terminal P&L for both strategies.

        Reproduces Figures 2, 3, 4 of the paper.

        Parameters
        ----------
        inv_results : list[SimulationResult]
            Monte Carlo results for the inventory strategy.
        sym_results : list[SimulationResult]
            Monte Carlo results for the symmetric strategy.
        ax : Axes, optional
        title : str, optional
        gamma : float, optional
            Annotated on the plot if provided.
        n_bins : int
            Number of histogram bins.
        show_stats : bool
            Annotate mean ± std for each strategy.

        Returns
        -------
        Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5), dpi=self.dpi)

        pnl_inv = np.array([r.final_pnl for r in inv_results], dtype=float)
        pnl_sym = np.array([r.final_pnl for r in sym_results], dtype=float)

        # Shared bin edges so the histograms are directly comparable
        all_vals = np.concatenate([pnl_inv, pnl_sym])
        finite   = all_vals[np.isfinite(all_vals)]
        lo, hi   = finite.min(), finite.max()
        margin   = (hi - lo) * 0.05
        bins     = np.linspace(lo - margin, hi + margin, n_bins + 1)

        ax.hist(
            pnl_inv, bins=bins,
            color=_C["pnl_inv"], alpha=0.60,
            label="Inventory strategy",
            edgecolor="white", linewidth=0.4,
            zorder=3,
        )
        ax.hist(
            pnl_sym, bins=bins,
            color=_C["pnl_sym"], alpha=0.55,
            label="Symmetric strategy",
            edgecolor="white", linewidth=0.4,
            zorder=2,
        )

        # Vertical mean lines
        ax.axvline(
            np.mean(pnl_inv), color=_C["pnl_inv"],
            lw=1.6, linestyle="--", zorder=4,
        )
        ax.axvline(
            np.mean(pnl_sym), color=_C["pnl_sym"],
            lw=1.6, linestyle="--", zorder=4,
        )

        if show_stats:
            _annotate_stats(ax, pnl_inv, _C["pnl_inv"], side="left")
            _annotate_stats(ax, pnl_sym, _C["pnl_sym"], side="right")

        g_str = f"γ = {gamma}" if gamma is not None else ""
        ax.set_xlabel("Terminal P&L  ($X_T + q_T S_T$)  [$]")
        ax.set_ylabel("Frequency")
        ax.set_title(
            title or f"Final P&L Distribution — {len(inv_results)} runs"
                     + (f"  [{g_str}]" if g_str else "")
        )

        legend_handles = [
            mpatches.Patch(color=_C["pnl_inv"], alpha=0.7,
                           label=f"Inventory  (n={len(inv_results)})"),
            mpatches.Patch(color=_C["pnl_sym"], alpha=0.7,
                           label=f"Symmetric  (n={len(sym_results)})"),
        ]
        ax.legend(handles=legend_handles, fontsize=8, framealpha=0.85)
        _despine(ax)
        return ax

    # ------------------------------------------------------------------
    # Composite dashboards
    # ------------------------------------------------------------------

    def plot_single_run(
        self,
        inv_result: "SimulationResult",
        sym_result: Optional["SimulationResult"] = None,
        suptitle: Optional[str] = None,
        show_reservation: bool = True,
        show_fills: bool = True,
        show_risk: bool = False,
    ) -> Figure:
        """
        4-panel dashboard for a single simulation run.

        Layout
        ------
        ┌───────────────────────────────────────┐
        │  Panel 1: Mid-price vs Bid/Ask         │
        ├─────────────────┬─────────────────────┤
        │  Panel 2: Inventory over time          │
        ├─────────────────────────────────────── ┤
        │  Panel 3: P&L over time                │
        └───────────────────────────────────────┘

        (No histogram here — that requires Monte Carlo results.)

        Parameters
        ----------
        inv_result : SimulationResult
            Primary result (inventory strategy or any single run).
        sym_result : SimulationResult, optional
            If provided, overlaid on panels 2 and 3.
        suptitle : str, optional
        show_reservation : bool
        show_fills : bool
        show_risk : bool

        Returns
        -------
        Figure
        """
        fig = plt.figure(figsize=self.figsize_single, dpi=self.dpi)
        gs = gridspec.GridSpec(
            3, 1,
            figure=fig,
            hspace=0.42,
            height_ratios=[2.5, 1.2, 1.2],
        )

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        params = inv_result.params
        gamma  = params.get("gamma", "?")
        sigma  = params.get("sigma", "?")
        k      = params.get("k",     "?")

        self.plot_quotes(
            inv_result, ax=ax1,
            title=f"Mid-price vs Bid/Ask Quotes  "
                  f"[γ={gamma},  σ={sigma},  k={k}]",
            show_reservation=show_reservation,
            show_fills=show_fills,
        )

        self.plot_inventory(
            inv_result, sym_result=sym_result, ax=ax2,
            title="Inventory $q_t$",
        )

        self.plot_pnl_path(
            inv_result, sym_result=sym_result, ax=ax3,
            title="Mark-to-Market P&L",
            show_risk=show_risk,
        )

        if suptitle is None:
            suptitle = (
                f"Avellaneda-Stoikov Market Maker  —  Single Run  "
                f"[γ={gamma}]"
            )
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)
        return fig

    def plot_monte_carlo(
        self,
        inv_results: List["SimulationResult"],
        sym_results: List["SimulationResult"],
        gamma: Optional[float] = None,
        suptitle: Optional[str] = None,
        n_bins: int = 40,
    ) -> Figure:
        """
        Single-panel Monte Carlo histogram figure.

        Parameters
        ----------
        inv_results : list[SimulationResult]
        sym_results : list[SimulationResult]
        gamma : float, optional
        suptitle : str, optional
        n_bins : int

        Returns
        -------
        Figure
        """
        fig, ax = plt.subplots(
            figsize=self.figsize_hist, dpi=self.dpi
        )
        self.plot_pnl_histogram(
            inv_results, sym_results,
            ax=ax, gamma=gamma, n_bins=n_bins,
        )
        g_str = f"  [γ = {gamma}]" if gamma is not None else ""
        fig.suptitle(
            suptitle or f"Final P&L Distribution — {len(inv_results)} MC runs{g_str}",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        return fig

    def show_all(
        self,
        inv_result: "SimulationResult",
        sym_result: "SimulationResult",
        inv_results: List["SimulationResult"],
        sym_results: List["SimulationResult"],
        gamma: Optional[float] = None,
        save_prefix: Optional[str] = None,
        show: bool = True,
    ) -> tuple[Figure, Figure]:
        """
        Render the full 4-panel single-run dashboard AND the Monte Carlo
        histogram comparison in two separate figures.

        Parameters
        ----------
        inv_result : SimulationResult
            Representative single inventory run (e.g. first MC path).
        sym_result : SimulationResult
            Representative single symmetric run.
        inv_results : list[SimulationResult]
            All Monte Carlo inventory runs (for the histogram).
        sym_results : list[SimulationResult]
            All Monte Carlo symmetric runs.
        gamma : float, optional
            Annotated on titles.
        save_prefix : str, optional
            If provided, figures are saved as
            ``{save_prefix}_single_run.png`` and
            ``{save_prefix}_histogram.png``.
        show : bool
            Call plt.show() after rendering.

        Returns
        -------
        (fig_single, fig_hist) : tuple[Figure, Figure]
        """
        g_str = f"γ={gamma}" if gamma is not None else ""

        fig_single = self.plot_single_run(
            inv_result, sym_result,
            suptitle=(
                f"Avellaneda-Stoikov Market Maker  —  Single Run  [{g_str}]"
                if g_str else None
            ),
        )

        fig_hist = self.plot_monte_carlo(
            inv_results, sym_results,
            gamma=gamma,
        )

        if save_prefix:
            print("Saving plots...")
            
            fig_single.savefig(
                f"{save_prefix}_single_run.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )
            
            fig_hist.savefig(
                f"{save_prefix}_histogram.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )

            print("Plots saved")
        if show:
            plt.show()

        return fig_single, fig_hist

    # ------------------------------------------------------------------
    # Standalone multi-gamma comparison (all three paper tables)
    # ------------------------------------------------------------------

    def plot_gamma_comparison(
        self,
        results_by_gamma: dict,
        n_bins: int = 35,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """
        3-column histogram grid comparing inventory vs symmetric across
        all three gamma values studied in the paper (Tables 1, 2, 3).

        Parameters
        ----------
        results_by_gamma : dict
            Keys are gamma values (float).  Each value is a dict:
            ``{"inventory": list[SimulationResult],
               "symmetric": list[SimulationResult]}``.
        n_bins : int
        save_path : str, optional
        show : bool

        Returns
        -------
        Figure
        """
        gammas = sorted(results_by_gamma.keys())
        n_cols = len(gammas)

        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(6 * n_cols, 5),
            dpi=self.dpi,
            sharey=False,
        )
        if n_cols == 1:
            axes = [axes]

        for ax, g in zip(axes, gammas):
            bucket = results_by_gamma[g]
            self.plot_pnl_histogram(
                inv_results=bucket["inventory"],
                sym_results=bucket["symmetric"],
                ax=ax,
                gamma=g,
                n_bins=n_bins,
                show_stats=True,
                title=f"γ = {g}",
            )

        fig.suptitle(
            "Avellaneda-Stoikov (2008) — Final P&L Distributions\n"
            "Inventory strategy vs Symmetric benchmark",
            fontsize=12, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Spread and risk diagnostics
    # ------------------------------------------------------------------

    def plot_spread_and_risk(
        self,
        result: "SimulationResult",
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
    ) -> Axes:
        """
        Dual-axis plot: bid-ask spread (left) and inventory risk exposure
        0.5·γ·q²·σ²·(T−t) (right) over time.

        Parameters
        ----------
        result : SimulationResult
        ax : Axes, optional
        title : str, optional

        Returns
        -------
        Axes (primary)
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 3), dpi=self.dpi)

        (time, _, bid, ask, _, _, _, risk, _, _) = _extract_path(result)

        spread = _safe_nan(ask - bid)

        ax.plot(
            time, spread,
            color=_C["mid"], lw=_LW_MAIN,
            label="Bid-ask spread",
        )
        ax.fill_between(
            time, spread, alpha=_ALPHA_FILL, color=_C["mid"],
        )
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Spread ($)", color=_C["mid"])
        ax.tick_params(axis="y", labelcolor=_C["mid"])

        ax2 = ax.twinx()
        ax2.plot(
            time, risk,
            color=_C["reservation"], lw=_LW_THIN,
            linestyle=":", alpha=0.85,
            label="Risk exposure",
        )
        ax2.set_ylabel(
            r"Risk  $\frac{1}{2}\gamma q^2\sigma^2(T-t)$",
            color=_C["reservation"], fontsize=8,
        )
        ax2.tick_params(axis="y", labelcolor=_C["reservation"], labelsize=7)

        lines = [
            Line2D([0], [0], color=_C["mid"],         lw=_LW_MAIN,
                   label="Bid-ask spread"),
            Line2D([0], [0], color=_C["reservation"], lw=_LW_THIN,
                   linestyle=":", alpha=0.85, label="Risk exposure"),
        ]
        ax.legend(handles=lines, fontsize=8, loc="upper right", framealpha=0.8)
        ax.set_title(title or "Spread & Inventory Risk over Time")
        _despine(ax)
        return ax

    # ------------------------------------------------------------------
    # Generic two-model comparison panels (finite-horizon vs
    # infinite-horizon, or any other pair of SimulationResult sets).
    #
    # These mirror plot_inventory / plot_pnl_histogram / plot_spread_and_risk
    # above (same helpers, same colour palette, same axis conventions) but
    # take explicit, generic labels instead of the hardcoded
    # "Inventory strategy" / "Symmetric" text those methods use -- so they
    # can be reused for any two-way comparison (e.g. horizon formulation)
    # without mislabeling the legend. No new plotting framework: same
    # _extract_path / _despine / _C helpers as the rest of this class.
    # ------------------------------------------------------------------

    def plot_inventory_comparison(
        self,
        result_a: "SimulationResult",
        result_b: "SimulationResult",
        label_a: str = "Model A",
        label_b: str = "Model B",
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
    ) -> Axes:
        """
        Overlay inventory q_t over time for two arbitrary simulation runs.

        Generic counterpart of ``plot_inventory`` (which hardcodes
        "Inventory strategy" / "Symmetric" labels for the paper's Section
        3.3 comparison) -- used for the finite-vs-infinite horizon
        comparison in main.py.

        Parameters
        ----------
        result_a, result_b : SimulationResult
        label_a, label_b : str
            Legend labels, e.g. "Finite Horizon" / "Infinite Horizon".
        ax : Axes, optional
        title : str, optional

        Returns
        -------
        Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 3), dpi=self.dpi)

        time_a, *_, inventory_a, _, _, _, _ = _extract_path(result_a)
        time_b, *_, inventory_b, _, _, _, _ = _extract_path(result_b)

        ax.axhline(0, color=_C["zero"], lw=0.8, linestyle="--", zorder=1)

        ax.step(
            time_a, inventory_a,
            where="post",
            color=_C["pnl_inv"],
            lw=_LW_MAIN,
            label=label_a,
            zorder=3,
        )
        ax.step(
            time_b, inventory_b,
            where="post",
            color=_C["pnl_sym"],
            lw=_LW_MAIN,
            alpha=0.85,
            label=label_b,
            zorder=2,
        )

        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Inventory $q_t$ (shares)")
        ax.set_title(title or f"Inventory over Time — {label_a} vs {label_b}")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
        _despine(ax)
        return ax

    def plot_pnl_distribution_comparison(
        self,
        results_a: List["SimulationResult"],
        results_b: List["SimulationResult"],
        label_a: str = "Model A",
        label_b: str = "Model B",
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        n_bins: int = 30,
        show_stats: bool = True,
    ) -> Axes:
        """
        Overlapping histograms of terminal P&L for two Monte Carlo pools.

        Generic counterpart of ``plot_pnl_histogram``.

        Parameters
        ----------
        results_a, results_b : list[SimulationResult]
        label_a, label_b : str
        ax : Axes, optional
        title : str, optional
        n_bins : int
        show_stats : bool
            Annotate mean +/- std for each pool.

        Returns
        -------
        Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5), dpi=self.dpi)

        pnl_a = np.array([r.final_pnl for r in results_a], dtype=float)
        pnl_b = np.array([r.final_pnl for r in results_b], dtype=float)

        all_vals = np.concatenate([pnl_a, pnl_b])
        finite   = all_vals[np.isfinite(all_vals)]
        lo, hi   = finite.min(), finite.max()
        margin   = (hi - lo) * 0.05 if hi > lo else 1.0
        bins     = np.linspace(lo - margin, hi + margin, n_bins + 1)

        ax.hist(
            pnl_a, bins=bins,
            color=_C["pnl_inv"], alpha=0.60,
            label=f"{label_a}  (n={len(results_a)})",
            edgecolor="white", linewidth=0.4,
            zorder=3,
        )
        ax.hist(
            pnl_b, bins=bins,
            color=_C["pnl_sym"], alpha=0.55,
            label=f"{label_b}  (n={len(results_b)})",
            edgecolor="white", linewidth=0.4,
            zorder=2,
        )
        ax.axvline(np.mean(pnl_a), color=_C["pnl_inv"], lw=1.6, linestyle="--", zorder=4)
        ax.axvline(np.mean(pnl_b), color=_C["pnl_sym"], lw=1.6, linestyle="--", zorder=4)

        if show_stats:
            _annotate_stats(ax, pnl_a, _C["pnl_inv"], side="left")
            _annotate_stats(ax, pnl_b, _C["pnl_sym"], side="right")

        ax.set_xlabel("Terminal P&L  ($X_T + q_T S_T$)  [$]")
        ax.set_ylabel("Frequency")
        ax.set_title(
            title or f"Final P&L Distribution — {label_a} vs {label_b} "
                     f"({len(results_a)} runs each)"
        )
        ax.legend(fontsize=8, framealpha=0.85)
        _despine(ax)
        return ax

    def plot_average_spread_comparison(
        self,
        results_a: List["SimulationResult"],
        results_b: List["SimulationResult"],
        label_a: str = "Model A",
        label_b: str = "Model B",
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
    ) -> Axes:
        """
        Bar chart comparing the average bid-ask spread (mean +/- std across
        the Monte Carlo pool) between two models.

        Parameters
        ----------
        results_a, results_b : list[SimulationResult]
        label_a, label_b : str
        ax : Axes, optional
        title : str, optional

        Returns
        -------
        Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 4), dpi=self.dpi)

        sprd_a = np.array([r.average_spread for r in results_a], dtype=float)
        sprd_b = np.array([r.average_spread for r in results_b], dtype=float)

        means = [float(np.mean(sprd_a)), float(np.mean(sprd_b))]
        stds  = [float(np.std(sprd_a)),  float(np.std(sprd_b))]
        colors = [_C["pnl_inv"], _C["pnl_sym"]]
        labels = [label_a, label_b]

        bars = ax.bar(
            labels, means, yerr=stds, capsize=6,
            color=colors, alpha=0.75, edgecolor="white", linewidth=0.6,
        )
        for bar, m in zip(bars, means):
            ax.annotate(
                f"{m:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, m),
                xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=8, fontweight="bold",
            )

        ax.set_ylabel("Average spread ($)")
        ax.set_title(title or f"Average Spread — {label_a} vs {label_b}")
        _despine(ax)
        return ax

    def plot_inventory_risk_comparison(
        self,
        result_a: "SimulationResult",
        result_b: "SimulationResult",
        label_a: str = "Model A",
        label_b: str = "Model B",
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
    ) -> Axes:
        """
        Overlay the inventory risk exposure
        (0.5*gamma*q_t^2*sigma^2*(T-t), StepRecord.risk_exposure) over time
        for two simulation runs.

        Note: this risk metric is computed identically by the engine for
        BOTH the finite-horizon and infinite-horizon models (it lives in
        InventoryManager.compute_risk(), untouched by the horizon
        extension), so it is a fair, common yardstick for comparing
        inventory risk actually carried by each strategy over the same
        simulated duration -- it is a diagnostic using the classical
        formula for comparability, not a claim that the infinite-horizon
        agent optimises this particular quantity.

        Parameters
        ----------
        result_a, result_b : SimulationResult
        label_a, label_b : str
        ax : Axes, optional
        title : str, optional

        Returns
        -------
        Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 3), dpi=self.dpi)

        time_a, _, _, _, _, _, _, risk_a, _, _ = _extract_path(result_a)
        time_b, _, _, _, _, _, _, risk_b, _, _ = _extract_path(result_b)

        ax.plot(
            time_a, risk_a,
            color=_C["pnl_inv"], lw=_LW_MAIN,
            label=label_a, zorder=3,
        )
        ax.plot(
            time_b, risk_b,
            color=_C["pnl_sym"], lw=_LW_MAIN,
            alpha=0.85, label=label_b, zorder=2,
        )

        ax.set_xlabel("Time $t$")
        ax.set_ylabel(r"Risk  $\frac{1}{2}\gamma q^2\sigma^2(T-t)$")
        ax.set_title(title or f"Inventory Risk over Time — {label_a} vs {label_b}")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
        _despine(ax)
        return ax

    def plot_horizon_comparison(
        self,
        finite_result: "SimulationResult",
        infinite_result: "SimulationResult",
        finite_results: List["SimulationResult"],
        infinite_results: List["SimulationResult"],
        label_a: str = "Finite Horizon",
        label_b: str = "Infinite Horizon",
        suptitle: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """
        2x2 composite dashboard comparing the finite-horizon and
        infinite-horizon models: inventory trajectory, P&L distribution,
        average spread, and inventory-risk over time.

        Built the same way as ``plot_single_run`` / ``plot_gamma_comparison``
        above (GridSpec + the four panel methods) -- reuses this class's
        existing helpers rather than introducing a new plotting framework.

        Parameters
        ----------
        finite_result, infinite_result : SimulationResult
            Representative single runs (e.g. the first Monte Carlo path)
            for each model, used for the time-series panels.
        finite_results, infinite_results : list[SimulationResult]
            Full Monte Carlo pools for each model, used for the P&L
            distribution and average-spread panels.
        label_a, label_b : str
        suptitle : str, optional
        save_path : str, optional
            If provided, the figure is saved to this path.
        show : bool
            Call plt.show() after rendering.

        Returns
        -------
        Figure
        """
        fig = plt.figure(figsize=self.figsize_single, dpi=self.dpi)
        gs = gridspec.GridSpec(
            3, 2,
            figure=fig,
            hspace=0.55,
            wspace=0.3,
            height_ratios=[1.2, 1.2, 1.2],
        )

        ax_inv    = fig.add_subplot(gs[0, :])
        ax_risk   = fig.add_subplot(gs[1, :])
        ax_pnl    = fig.add_subplot(gs[2, 0])
        ax_spread = fig.add_subplot(gs[2, 1])

        self.plot_inventory_comparison(
            finite_result, infinite_result, label_a, label_b, ax=ax_inv,
        )
        self.plot_inventory_risk_comparison(
            finite_result, infinite_result, label_a, label_b, ax=ax_risk,
        )
        self.plot_pnl_distribution_comparison(
            finite_results, infinite_results, label_a, label_b, ax=ax_pnl,
        )
        self.plot_average_spread_comparison(
            finite_results, infinite_results, label_a, label_b, ax=ax_spread,
        )

        fig.suptitle(
            suptitle or f"Avellaneda-Stoikov — {label_a} vs {label_b}",
            fontsize=13, fontweight="bold", y=1.01,
        )

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def plot_single_run(
    inv_result: "SimulationResult",
    sym_result: Optional["SimulationResult"] = None,
    **kwargs,
) -> Figure:
    """Shortcut: ``Dashboard().plot_single_run(...)``."""
    return Dashboard().plot_single_run(inv_result, sym_result, **kwargs)


def plot_histogram(
    inv_results: List["SimulationResult"],
    sym_results: List["SimulationResult"],
    gamma: Optional[float] = None,
    **kwargs,
) -> Figure:
    """Shortcut: ``Dashboard().plot_monte_carlo(...)``."""
    return Dashboard().plot_monte_carlo(inv_results, sym_results,
                                        gamma=gamma, **kwargs)


def show_all(
    inv_result: "SimulationResult",
    sym_result: "SimulationResult",
    inv_results: List["SimulationResult"],
    sym_results: List["SimulationResult"],
    gamma: Optional[float] = None,
    save_prefix: Optional[str] = None,
) -> tuple[Figure, Figure]:
    """Shortcut: ``Dashboard().show_all(...)``."""
    return Dashboard().show_all(
        inv_result, sym_result,
        inv_results, sym_results,
        gamma=gamma,
        save_prefix=save_prefix,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _despine(ax: Axes) -> None:
    """Remove top and right spines for a cleaner look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _annotate_stats(
    ax: Axes,
    values: np.ndarray,
    color: str,
    side: str = "left",
) -> None:
    """
    Add a μ ± σ text box inside the axes for a given data array.

    Parameters
    ----------
    ax : Axes
    values : np.ndarray
    color : str
    side : str
        'left' or 'right' — horizontal placement of the annotation.
    """
    mu  = np.mean(values)
    std = np.std(values)
    x_pos = 0.03 if side == "left" else 0.97
    ha    = "left" if side == "left" else "right"
    ax.text(
        x_pos, 0.96,
        f"μ = {mu:.1f}\nσ = {std:.1f}",
        transform=ax.transAxes,
        fontsize=7.5,
        color=color,
        va="top", ha=ha,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=color,
            alpha=0.75,
            linewidth=0.8,
        ),
    )


# ---------------------------------------------------------------------------
# __main__ — quick smoke test with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Smoke-test: generate synthetic StepRecord-like data and render the
    dashboard without needing a running simulation engine.
    """
    from types import SimpleNamespace

    rng = np.random.default_rng(42)
    N = 200
    T = 1.0
    dt = T / N
    s0 = 100.0
    gamma = 0.1
    sigma = 2.0
    k = 1.5

    # --- synthetic mid-price path ---
    mid = np.cumsum(rng.normal(0, sigma * math.sqrt(dt), N)) + s0

    # --- synthetic inventory (mean-reverting random walk) ---
    inv_q = np.zeros(N, dtype=int)
    for i in range(1, N):
        inv_q[i] = int(np.clip(inv_q[i - 1] + rng.integers(-1, 2), -5, 5))

    # --- derive quotes and P&L ---
    times = np.arange(N) * dt
    tau   = T - times
    r_price = mid - inv_q * gamma * sigma ** 2 * tau
    spread  = gamma * sigma ** 2 * tau + (2 / gamma) * math.log(1 + gamma / k)
    half    = spread / 2.0
    bid     = r_price - half
    ask     = r_price + half
    cash    = np.cumsum(rng.choice([-1, 0, 1], size=N) * 0.5)
    pnl     = cash + inv_q * mid
    risk    = 0.5 * gamma * inv_q ** 2 * sigma ** 2 * tau
    buy_m   = rng.random(N) < 0.05
    sell_m  = rng.random(N) < 0.05

    def _make_path(mid_, bid_, ask_, r_, inv_, pnl_, risk_, bm, sm):
        path = []
        for i in range(N):
            path.append(SimpleNamespace(
                step=i, time=times[i],
                mid_price=mid_[i], bid=bid_[i], ask=ask_[i],
                reservation=r_[i], inventory=inv_[i],
                pnl=pnl_[i], risk_exposure=risk_[i],
                buy_filled=bool(bm[i]), sell_filled=bool(sm[i]),
                cash=cash[i], lambda_bid=0.0, lambda_ask=0.0,
                spread=ask_[i] - bid_[i],
            ))
        return path

    params = dict(gamma=gamma, sigma=sigma, k=k, A=140, T=T, dt=dt, s0=s0)

    inv_result = SimpleNamespace(
        path=_make_path(mid, bid, ask, r_price, inv_q, pnl, risk, buy_m, sell_m),
        final_pnl=float(pnl[-1]), final_inventory=int(inv_q[-1]),
        total_buy_fills=int(buy_m.sum()), total_sell_fills=int(sell_m.sum()),
        average_spread=float(spread.mean()), strategy="inventory", params=params,
    )

    # symmetric: centred on mid, same spread
    bid_s = mid - half
    ask_s = mid + half
    pnl_s = cash * 1.1 + inv_q * mid
    sym_result = SimpleNamespace(
        path=_make_path(mid, bid_s, ask_s, mid, inv_q, pnl_s, risk, buy_m, sell_m),
        final_pnl=float(pnl_s[-1]), final_inventory=int(inv_q[-1]),
        total_buy_fills=int(buy_m.sum()), total_sell_fills=int(sell_m.sum()),
        average_spread=float(spread.mean()), strategy="symmetric", params=params,
    )

    # Monte Carlo pool (300 synthetic runs)
    def _mc_pool(n=300, mu=65, std=7, q_std=3):
        results = []
        for _ in range(n):
            pnl_val = float(rng.normal(mu, std))
            q_val   = int(rng.integers(-q_std, q_std + 1))
            results.append(SimpleNamespace(
                final_pnl=pnl_val, final_inventory=q_val,
                average_spread=float(rng.normal(1.49, 0.05)),
                total_buy_fills=0, total_sell_fills=0,
                path=[], strategy="", params=params,
            ))
        return results

    inv_mc  = _mc_pool(mu=65.0, std=6.6,  q_std=3)
    sym_mc  = _mc_pool(mu=68.4, std=12.7, q_std=8)

    dash = Dashboard(dpi=100)
    fig1 = dash.plot_single_run(inv_result, sym_result, show_reservation=True)
    fig2 = dash.plot_monte_carlo(inv_mc, sym_mc, gamma=gamma)

    plt.show()