"""
simulator/engine.py

SimulationEngine for the Avellaneda-Stoikov (2008) market-making framework.

Implements the complete discrete-time simulation loop described in section 3.3
of the paper.  At each time step dt the engine:

  1. Updates the mid-price via arithmetic Brownian motion:
         dS_t = sigma * dW_t   =>   S_{t+dt} = S_t + sigma * sqrt(dt) * Z
                                              Z ~ N(0,1)              [eq. 1]

  2. Asks the agent to compute the reservation price:
         r(s,q,t) = s - q * gamma * sigma^2 * (T-t)                  [eq. 8]

  3. Asks the agent to compute the optimal spread:
         delta^a + delta^b = gamma*sigma^2*(T-t)
                             + (2/gamma) * ln(1 + gamma/k)            [eq. 30]

  4. Generates bid/ask quotes centred on the reservation price:
         p^b = r - half_spread
         p^a = r + half_spread

  5. Simulates executions by drawing the number of Poisson arrivals on
     each side within the step (not merely whether >=1 arrival occurred):
         N^b_step ~ Poisson(lambda^b(delta^b) * dt)
         N^a_step ~ Poisson(lambda^a(delta^a) * dt)
         where lambda(delta) = A * exp(-k * delta)                    [eq. 12]
     Each arrival is filled sequentially at the step's standing quote.
     (See SimulationEngine._simulate_fills / _poisson_count_from_uniform
     for why exact-inversion sampling is used to preserve common random
     numbers between the finite- and infinite-horizon engines.)

  6. Updates cash:   dX_t = p^a * dN^a_t - p^b * dN^b_t
  7. Updates inventory: q_t = N^b_t - N^a_t
  8. Computes mark-to-market P&L: X_t + q_t * S_t

The engine also supports a *symmetric* benchmark strategy (section 3.3)
that quotes the same average spread around the raw mid-price instead of
the reservation price, enabling a like-for-like comparison.

Paper: Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a
       limit order book. Quantitative Finance 8(3), 217–224.
"""

from __future__ import annotations

import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from simulator.agent import MarketMakerAgent, Quote
from strategy.inventory import InventoryLimits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """
    Complete record of one simulation time step.

    Stores every quantity produced during the 8-stage loop so that the
    metrics and visualisation layers can reconstruct the full path without
    re-running the simulation.
    """
    step: int               # integer step index
    time: float             # simulation time t
    mid_price: float        # S_t  after the price update
    reservation: float      # r(s,q,t)
    spread: float           # total bid-ask spread (delta^a + delta^b)
    bid: float              # p^b
    ask: float              # p^a
    buy_filled: bool        # True if >=1 buy  fill occurred this step
    sell_filled: bool       # True if >=1 sell fill occurred this step
    cash: float             # X_t  after fills
    inventory: int          # q_t  after fills
    pnl: float              # X_t + q_t * S_t  (mark-to-market)
    risk_exposure: float    # 0.5 * gamma * q^2 * sigma^2 * (T-t)
    lambda_bid: float       # fill intensity on the bid side
    lambda_ask: float       # fill intensity on the ask side
    buy_fill_count: int = 0   # number of buy  fills this step (Poisson count, >=0)
    sell_fill_count: int = 0  # number of sell fills this step (Poisson count, >=0)


@dataclass
class SimulationResult:
    """
    Output of a completed simulation run.

    Attributes
    ----------
    path : list[StepRecord]
        Tick-by-tick record of the simulation.
    final_pnl : float
        Terminal mark-to-market P&L  X_T + q_T * S_T.
    final_inventory : int
        Residual share position at time T.
    total_buy_fills : int
    total_sell_fills : int
    average_spread : float
        Time-averaged bid-ask spread over the run.
    strategy : str
        'inventory' or 'symmetric'.
    params : dict
        Copy of the parameters used in this run.
    """
    path: List[StepRecord]
    final_pnl: float
    final_inventory: int
    total_buy_fills: int
    total_sell_fills: int
    average_spread: float
    strategy: str
    params: dict

    # ---- quick pandas conversion (optional) --------------------------------
    def to_dataframe(self):
        """Convert the path to a pandas DataFrame (requires pandas)."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for to_dataframe().") from exc
        return pd.DataFrame([vars(r) for r in self.path])


# ---------------------------------------------------------------------------
# SimulationEngine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Discrete-time simulation engine for the Avellaneda-Stoikov market maker.

    Parameters
    ----------
    s0 : float
        Initial mid-price S_0  (paper uses s=100).
    T : float
        Terminal time of the trading session (paper uses T=1).
    dt : float
        Time step size.  Fills are drawn from the exact Poisson arrival
        count each step, so correctness does not require dt < 1/(2*A);
        keeping dt small mainly controls how often quotes are refreshed
        relative to the arrival rate.  Paper uses dt=0.005.
    sigma : float
        Mid-price volatility σ (paper uses σ=2).
    gamma : float
        Risk-aversion coefficient γ (paper uses γ=0.1 / 0.01 / 1).
    k : float
        Exponential order-arrival decay (paper uses k=1.5).
    A : float
        Baseline order-arrival intensity Λ (paper uses A=140).
    limits : InventoryLimits, optional
        Hard/soft inventory limits.
    initial_quantity : int
        Starting inventory q_0 (paper starts at 0).
    initial_cash : float
        Starting cash X_0 (paper starts at 0).
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    agent : MarketMakerAgent
        The inventory-aware market-making agent.
    n_steps : int
        Total number of time steps  round(T / dt).
    """

    def __init__(
        self,
        s0: float = 100.0,
        T: float = 1.0,
        dt: float = 0.005,
        sigma: float = 2.0,
        gamma: float = 0.1,
        k: float = 1.5,
        A: float = 140.0,
        model: str = "finite_horizon",
        omega: Optional[float] = None,
        q_max: Optional[int] = None,
        limits: Optional[InventoryLimits] = None,
        initial_quantity: int = 0,
        initial_cash: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        # ---- validate -------------------------------------------------------
        for name, val in [("s0", s0), ("T", T), ("dt", dt),
                          ("sigma", sigma), ("gamma", gamma),
                          ("k", k), ("A", A)]:
            if val <= 0:
                raise ValueError(f"Parameter '{name}' must be strictly positive.")
        if dt >= T:
            raise ValueError("dt must be smaller than T.")

        # Warn if the Poisson mean in one step can exceed 0.5. This is no
        # longer a correctness problem (fills are sampled from the exact
        # Poisson count via _simulate_fills / _poisson_count_from_uniform,
        # so multiple arrivals per step are handled correctly), but it is
        # still useful operationally: it means the discretisation is coarse
        # relative to the order-arrival rate, so quotes are only updated
        # once per several arrivals instead of continuously.
        max_fill_prob = A * dt
        if max_fill_prob > 0.5:
            logger.warning(
                "A*dt = %.3f > 0.5: multiple fills per step are likely. "
                "This is handled correctly (fills are drawn from the exact "
                "Poisson count, not a single Bernoulli trial), but quotes "
                "are still only refreshed once per step. Consider reducing "
                "dt or A if you want quotes to react to each individual "
                "arrival.", max_fill_prob
            )

        # ---- store parameters -----------------------------------------------
        self.s0 = s0
        self.T = T
        self.dt = dt
        self.sigma = sigma
        self.gamma = gamma
        self.k = k
        self.A = A
        self.model = model
        self.omega = omega
        self.q_max = q_max
        self.initial_quantity = initial_quantity
        self.initial_cash = initial_cash
        self.seed = seed

        # NOTE: n_steps is always driven by T/dt regardless of 'model'. For
        # model='infinite_horizon', T is simulation *duration* only -- it is
        # never used inside the infinite-horizon pricing formulas (those use
        # 'omega' instead). See MarketMakerAgent for the same note.
        self.n_steps: int = round(T / dt)
        self._params: dict = dict(
            s0=s0, T=T, dt=dt, sigma=sigma, gamma=gamma,
            k=k, A=A, initial_quantity=initial_quantity,
            initial_cash=initial_cash, seed=seed,
            model=model, omega=omega, q_max=q_max,
        )

        # ---- build agent ----------------------------------------------------
        self.agent = MarketMakerAgent(
            gamma=gamma,
            sigma=sigma,
            T=T,
            k=k,
            A=A,
            model=model,
            omega=omega,
            q_max=q_max,
            limits=limits,
            initial_quantity=initial_quantity,
            initial_cash=initial_cash,
        )

        # ---- RNG ------------------------------------------------------------
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Intensity function  lambda(delta) = A * exp(-k * delta)  [eq. 12]
    # ------------------------------------------------------------------

    def _intensity(self, delta: float) -> float:
        """
        Poisson arrival intensity as a function of quote distance δ.

        λ(δ) = A * exp(−k * δ)                                    [eq. 12]

        Parameters
        ----------
        delta : float
            Distance of the quote from the mid-price (always ≥ 0).

        Returns
        -------
        float
            Non-negative arrival intensity.
        """
        return self.A * math.exp(-self.k * max(delta, 0.0))

    # ------------------------------------------------------------------
    # Execution simulation
    # ------------------------------------------------------------------

    @staticmethod
    def _poisson_count_from_uniform(mu: float, u: float, max_iter: int = 10_000) -> int:
        """
        Exact-inversion Poisson sampler consuming a SINGLE uniform variate.

        Given u ~ Uniform(0,1), returns the count k such that
            F(k-1) <= u < F(k)
        where F is the Poisson(mu) CDF.  This is Knuth's classic inversion
        algorithm, built incrementally so it never needs the full CDF table:

            p_0 = exp(-mu),  F_0 = p_0
            p_k = p_{k-1} * mu / k,  F_k = F_{k-1} + p_k
            return smallest k with u < F_k

        We use this (rather than ``rng.poisson(mu)``) specifically so that
        the arrival draw for a given side consumes *exactly one* float from
        the RNG stream, regardless of mu.  That preserves the engine's
        existing common-random-numbers (CRN) property: the finite-horizon
        and infinite-horizon engines call ``self._rng.random(2)`` exactly
        once per step no matter what mu turns out to be on either side, so
        their mid-price paths (and the RNG's overall stream position)
        stay byte-identical even though the two strategies quote different
        spreads (and hence different mu). ``rng.poisson()`` cannot offer
        this guarantee because NumPy's generator selects a different
        internal algorithm (and burns a different number of underlying
        random draws) depending on mu.

        Parameters
        ----------
        mu : float
            Poisson mean, i.e. lambda(delta) * dt. Must be >= 0.
        u : float
            Uniform(0,1) variate driving this draw.
        max_iter : int
            Safety cap on the number of terms evaluated. With mu <= A*dt
            (0.7 in the paper's parameterisation) convergence takes only a
            handful of iterations; this cap only guards against
            pathological inputs (e.g. mu blowing up) and is never expected
            to bind in practice.

        Returns
        -------
        int
            Number of arrivals in [t, t+dt], >= 0.
        """
        if mu <= 0.0:
            return 0
        p = math.exp(-mu)
        cdf = p
        k = 0
        while u >= cdf and k < max_iter:
            k += 1
            p *= mu / k
            cdf += p
        return k

    def _simulate_fills(
        self,
        quote: Quote,
    ) -> Tuple[int, int]:
        """
        Sample the number of buy and sell fills that occur in this time step.

        The paper models buy/sell order arrivals as independent Poisson
        processes with intensities λ^b(δ^b) and λ^a(δ^a)  [eq. 12]. Over a
        step of size dt, the *number* of arrivals on each side is Poisson
        distributed with mean mu = λ * dt:

            N^b_step ~ Poisson(λ^b(δ^b) * dt)
            N^a_step ~ Poisson(λ^a(δ^a) * dt)

        Earlier versions of this engine modelled only whether *at least
        one* fill occurred, i.e. Bernoulli(1 - exp(-mu)). That collapses
        the two-or-more-arrivals branch of the Poisson distribution into
        "one fill", which under-counts fills whenever mu is not small.
        With the paper's parameters (A=140, dt=0.005), mu can reach
        A*dt = 0.7 when quotes sit right on the mid-price (delta ~ 0,
        which happens for the infinite-horizon strategy's tight average
        spread). At mu=0.7, P(N>=2 | N>=1) ~= 30%, i.e. roughly three in
        ten "at least one fill" steps are actually undercounted by the
        old Bernoulli model. This method draws the exact Poisson count
        instead, so multiple arrivals within a single dt are handled
        consistently with the paper's underlying continuous-time model.

        Parameters
        ----------
        quote : Quote
            The current bid/ask quote.

        Returns
        -------
        (n_buy, n_sell) : tuple[int, int]
            Number of buy-side and sell-side fills this step, each >= 0.
            (n_buy > 1) is possible whenever mu = lambda*dt is not small.
        """
        u_bid, u_ask = self._rng.random(2)

        # bid side  (market sell orders hit our bid)
        if math.isnan(quote.bid):
            n_buy = 0
        else:
            lam_b = self._intensity(quote.bid_distance)
            n_buy = self._poisson_count_from_uniform(lam_b * self.dt, u_bid)

        # ask side  (market buy orders lift our ask)
        if math.isnan(quote.ask):
            n_sell = 0
        else:
            lam_a = self._intensity(quote.ask_distance)
            n_sell = self._poisson_count_from_uniform(lam_a * self.dt, u_ask)

        return n_buy, n_sell

    # ------------------------------------------------------------------
    # Single-step logic  (all 8 stages)
    # ------------------------------------------------------------------

    def _step(
        self,
        step_idx: int,
        current_time: float,
        mid_price: float,
        strategy: str,
        fixed_half_spread: Optional[float],
    ) -> Tuple[StepRecord, float]:
        """
        Execute one complete simulation step and return the record plus
        the updated mid-price.

        Parameters
        ----------
        step_idx : int
        current_time : float
        mid_price : float
            S_t *before* the price update.
        strategy : str
            'inventory' or 'symmetric'.
        fixed_half_spread : float or None
            For the symmetric strategy the half-spread is fixed at this
            value and quotes are centred on the raw mid-price.

        Returns
        -------
        (StepRecord, new_mid_price)
        """
        agent = self.agent

        # ----------------------------------------------------------------
        # Stage 1 — update mid-price  dS = sigma * sqrt(dt) * Z
        # ----------------------------------------------------------------
        dW = self._rng.standard_normal() * self.sigma * math.sqrt(self.dt)
        new_mid = mid_price + dW

        # ----------------------------------------------------------------
        # Stage 2 — reservation price  r = s - q*gamma*sigma^2*(T-t)
        # Stage 3 — optimal spread
        # Stage 4 — generate bid/ask quotes
        # ----------------------------------------------------------------
        if strategy == "inventory":
            # Full optimal strategy: quotes centred on reservation price
            quote = agent.quote(mid_price=new_mid, current_time=current_time)

        else:  # symmetric benchmark
            # Same spread, centred on the *mid-price* instead of r(s,q,t)
            half = fixed_half_spread  # set by caller from inventory run avg
            raw_bid = new_mid - half
            raw_ask = new_mid + half
            # Wrap in a Quote object so _simulate_fills can consume it
            from simulator.agent import Quote as _Quote
            quote = _Quote(
                bid=max(raw_bid, 1e-6),
                ask=max(raw_ask, raw_bid + 1e-6),
                reservation=new_mid,          # no skew
                half_spread=half,
                mid_price=new_mid,
                time=current_time,
            )
            # Still push through the agent so fill history is recorded
            agent._last_quote = quote

        # ----------------------------------------------------------------
        # Stage 5 — simulate executions (Poisson arrival counts)
        # ----------------------------------------------------------------
        n_buy, n_sell = self._simulate_fills(quote)

        # ----------------------------------------------------------------
        # Stage 6 & 7 — update cash and inventory
        #
        # Quotes are held fixed within [t, t+dt) (this is the discretisation
        # already used throughout the engine, e.g. the mid-price only moves
        # once per step). Consequently, if n_buy (or n_sell) arrivals land
        # in the same step, each is filled sequentially at the SAME quoted
        # bid (or ask) price -- consistent with the paper's continuous-time
        # model, where every arrival within the interval hits the quote
        # that was standing throughout that interval.
        # ----------------------------------------------------------------
        buy_fill_count = 0
        for _ in range(n_buy):
            try:
                agent.on_buy_fill(mid_price=new_mid, current_time=current_time)
                buy_fill_count += 1
            except RuntimeError as exc:
                # Bid became suppressed (hard long limit) partway through
                # this step's arrivals; stop applying further buy fills.
                logger.debug("Buy fill suppressed at t=%.4f: %s", current_time, exc)
                break

        sell_fill_count = 0
        for _ in range(n_sell):
            try:
                agent.on_sell_fill(mid_price=new_mid, current_time=current_time)
                sell_fill_count += 1
            except RuntimeError as exc:
                logger.debug("Sell fill suppressed at t=%.4f: %s", current_time, exc)
                break

        buy_filled = buy_fill_count > 0
        sell_filled = sell_fill_count > 0

        # ----------------------------------------------------------------
        # Stage 8 — compute P&L  and risk exposure
        # ----------------------------------------------------------------
        pnl = agent.pnl(new_mid)
        risk = agent.inventory.compute_risk(
            mid_price=new_mid,
            current_time=current_time
        )

        # Intensities (for diagnostics)
        lam_b = self._intensity(quote.bid_distance) if not math.isnan(quote.bid) else 0.0
        lam_a = self._intensity(quote.ask_distance) if not math.isnan(quote.ask) else 0.0

        record = StepRecord(
            step=step_idx,
            time=current_time,
            mid_price=new_mid,
            reservation=quote.reservation,
            spread=quote.spread if not math.isnan(quote.bid) else float("nan"),
            bid=quote.bid,
            ask=quote.ask,
            buy_filled=buy_filled,
            sell_filled=sell_filled,
            cash=agent.inventory.cash,
            inventory=agent.inventory.quantity,
            pnl=pnl,
            risk_exposure=risk,
            lambda_bid=lam_b,
            lambda_ask=lam_a,
            buy_fill_count=buy_fill_count,
            sell_fill_count=sell_fill_count,
        )
        return record, new_mid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        strategy: str = "inventory",
        fixed_half_spread: Optional[float] = None,
    ) -> SimulationResult:
        """
        Run a complete simulation from t=0 to t=T.

        Parameters
        ----------
        strategy : str
            'inventory'  — optimal Avellaneda-Stoikov strategy (default).
            'symmetric'  — benchmark: same spread, centred on mid-price.
        fixed_half_spread : float, optional
            Half-spread to use for the symmetric strategy.  If None and
            strategy='symmetric', the engine first runs an inventory pass
            to determine the time-averaged spread, then uses that value.

        Returns
        -------
        SimulationResult
        """
        if strategy not in {"inventory", "symmetric"}:
            raise ValueError("strategy must be 'inventory' or 'symmetric'.")

        # For symmetric strategy: determine the spread from an inventory run
        # if not supplied explicitly (mirrors section 3.3 of the paper).
        if strategy == "symmetric" and fixed_half_spread is None:
            logger.info(
                "Symmetric strategy: running inventory pass to determine "
                "average spread."
            )
            inv_result = self.run(strategy="inventory")
            valid_spreads = [
                r.spread for r in inv_result.path
                if not math.isnan(r.spread)
            ]
            avg_spread = float(np.mean(valid_spreads)) if valid_spreads else 0.0
            fixed_half_spread = avg_spread / 2.0
            logger.info("Average inventory spread=%.4f; half=%.4f",
                        avg_spread, fixed_half_spread)

        # ---- reset agent ----------------------------------------------------
        self.agent.reset(
            initial_quantity=self.initial_quantity,
            initial_cash=self.initial_cash,
        )

        # ---- main simulation loop -------------------------------------------
        mid_price = self.s0
        path: List[StepRecord] = []
        times = [i * self.dt for i in range(self.n_steps)]

        logger.info(
            "Starting %s simulation: s0=%.2f T=%.2f dt=%.4f n_steps=%d "
            "sigma=%.3f gamma=%.3f k=%.3f A=%.1f",
            strategy, self.s0, self.T, self.dt, self.n_steps,
            self.sigma, self.gamma, self.k, self.A,
        )

        for i, t in enumerate(times):
            record, mid_price = self._step(
                step_idx=i,
                current_time=t,
                mid_price=mid_price,
                strategy=strategy,
                fixed_half_spread=fixed_half_spread,
            )
            path.append(record)

            if (i + 1) % max(1, self.n_steps // 10) == 0:
                logger.info(
                    "  step %d/%d  t=%.3f  S=%.3f  q=%d  PnL=%.3f",
                    i + 1, self.n_steps, t, mid_price,
                    self.agent.inventory.quantity, record.pnl,
                )

        # ---- terminal P&L  (liquidate residual inventory at final mid) ------
        final_pnl = self.agent.pnl(mid_price)
        final_q = self.agent.inventory.quantity

        # Average spread (ignoring NaN steps)
        valid_spreads = [r.spread for r in path if not math.isnan(r.spread)]
        avg_spread = float(np.mean(valid_spreads)) if valid_spreads else 0.0

        # Sum actual fill COUNTS (not just steps-with->=1-fill) so that
        # multiple arrivals within a single dt are all counted.
        buy_fills  = sum(r.buy_fill_count for r in path)
        sell_fills = sum(r.sell_fill_count for r in path)

        logger.info(
            "Simulation complete: final_pnl=%.4f  final_q=%d  "
            "buy_fills=%d  sell_fills=%d  avg_spread=%.4f",
            final_pnl, final_q, buy_fills, sell_fills, avg_spread,
        )

        return SimulationResult(
            path=path,
            final_pnl=final_pnl,
            final_inventory=final_q,
            total_buy_fills=buy_fills,
            total_sell_fills=sell_fills,
            average_spread=avg_spread,
            strategy=strategy,
            params=dict(self._params),
        )

    def run_both(self) -> Tuple[SimulationResult, SimulationResult]:
        """
        Run the inventory strategy first, then the symmetric benchmark
        using the same averaged spread, matching the paper's Table 1–3
        comparison methodology.

        Both runs use the same RNG state (re-seeded before each run) so
        the mid-price paths are identical and the comparison is fair.

        Returns
        -------
        (inventory_result, symmetric_result)
        """
        # Run inventory strategy
        self._rng = np.random.default_rng(self.seed)
        inv_result = self.run(strategy="inventory")

        # Derive symmetric half-spread from the inventory run
        valid_spreads = [
            r.spread for r in inv_result.path if not math.isnan(r.spread)
        ]
        avg_spread = float(np.mean(valid_spreads)) if valid_spreads else 0.0
        half = avg_spread / 2.0

        # Re-seed so mid-price path is identical
        self._rng = np.random.default_rng(self.seed)
        sym_result = self.run(strategy="symmetric", fixed_half_spread=half)

        return inv_result, sym_result

    def run_monte_carlo(
        self,
        n_runs: int = 1000,
        strategy: str = "inventory",
        fixed_half_spread: Optional[float] = None,
    ) -> List[SimulationResult]:
        """
        Run n_runs independent simulations and return all results.

        Reproduces the Monte Carlo experiment from section 3.3 of the paper
        (Tables 1, 2, 3).  Each run uses a different RNG state drawn from
        the master seed sequence.

        Parameters
        ----------
        n_runs : int
            Number of Monte Carlo paths (paper uses 1000).
        strategy : str
            'inventory' or 'symmetric'.
        fixed_half_spread : float, optional
            Required when strategy='symmetric'.  Use the average spread
            from a corresponding inventory run.

        Returns
        -------
        list[SimulationResult]
            Length n_runs.
        """
        if strategy == "symmetric" and fixed_half_spread is None:
            raise ValueError(
                "fixed_half_spread is required for Monte Carlo with "
                "strategy='symmetric'.  Run inventory MC first and compute "
                "the mean spread."
            )

        results: List[SimulationResult] = []
        seed_seq = np.random.SeedSequence(self.seed)
        child_seeds = seed_seq.spawn(n_runs)

        logger.info(
            "Monte Carlo: n_runs=%d  strategy=%s  gamma=%.3f",
            n_runs, strategy, self.gamma,
        )

        for i, child_seed in enumerate(child_seeds):
            self._rng = np.random.default_rng(child_seed)
            result = self.run(
                strategy=strategy,
                fixed_half_spread=fixed_half_spread,
            )
            results.append(result)

            if (i + 1) % max(1, n_runs // 10) == 0:
                pnls = [r.final_pnl for r in results]
                logger.info(
                    "  MC run %d/%d  mean_pnl=%.3f  std_pnl=%.3f",
                    i + 1, n_runs, float(np.mean(pnls)), float(np.std(pnls)),
                )

        return results

    # ------------------------------------------------------------------
    # Convenience statistics (mirrors Tables 1-3 in the paper)
    # ------------------------------------------------------------------

    @staticmethod
    def summarise(results: List[SimulationResult]) -> dict:
        """
        Compute summary statistics over a list of Monte Carlo results.

        Returns a dict with keys matching the paper's table columns:
            average_spread, mean_profit, std_profit,
            mean_final_q, std_final_q, n_runs.

        Additionally (additive extension, does not remove/rename any
        existing key -- added to support the finite-vs-infinite horizon
        comparison in main.py without duplicating aggregation logic):
            mean_abs_inventory, max_abs_inventory : computed across every
                step of every run's path (not just the final step), giving
                a fuller picture of inventory risk carried over time.
            mean_buy_fills, mean_sell_fills, mean_total_fills : mean
                number of fills per run.

        Parameters
        ----------
        results : list[SimulationResult]

        Returns
        -------
        dict
        """
        if not results:
            raise ValueError("results list is empty.")

        pnls  = np.array([r.final_pnl for r in results])
        inv   = np.array([r.final_inventory for r in results])
        sprd  = np.array([r.average_spread for r in results])
        buys  = np.array([r.total_buy_fills for r in results], dtype=float)
        sells = np.array([r.total_sell_fills for r in results], dtype=float)

        # Path-level |inventory| statistics across every step of every run.
        abs_q_all = np.concatenate([
            np.abs([step.inventory for step in r.path]) for r in results
        ]) if any(r.path for r in results) else np.array([0.0])

        return {
            "n_runs":         len(results),
            "strategy":       results[0].strategy,
            "average_spread": float(np.mean(sprd)),
            "mean_profit":    float(np.mean(pnls)),
            "std_profit":     float(np.std(pnls)),
            "mean_final_q":   float(np.mean(inv)),
            "std_final_q":    float(np.std(inv)),
            "mean_abs_inventory": float(np.mean(abs_q_all)),
            "max_abs_inventory":  float(np.max(abs_q_all)),
            "mean_buy_fills":   float(np.mean(buys)),
            "mean_sell_fills":  float(np.mean(sells)),
            "mean_total_fills": float(np.mean(buys + sells)),
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SimulationEngine("
            f"s0={self.s0}, T={self.T}, dt={self.dt}, "
            f"sigma={self.sigma}, gamma={self.gamma}, "
            f"k={self.k}, A={self.A}, n_steps={self.n_steps})"
        )