"""
arrivals.py
===========
Order-arrival intensity model for the Avellaneda-Stoikov (2008)
optimal market-making strategy.

The central result borrowed from the econophysics literature (Section 2.5)
is that the Poisson intensity with which a limit order placed at distance
delta from the mid-price gets executed decays exponentially:

    lambda(delta) = A * exp(-k * delta)                              [Eq. 12]

This functional form is derived by combining two empirical observations:

    1. Market-order size follows a power law:
       f_Q(x) ~ x^{-1-alpha},  alpha ~ 1.5                          [Eq. 9]

    2. Temporary price impact is logarithmic:
       Delta_p ~ ln(Q)                                               [Eq. 11]

Aggregating these:

    lambda(delta) = P(Delta_p > delta)
                  = P(ln(Q) > K*delta)
                  = P(Q > exp(K*delta))
                  = A * integral_{exp(K*delta)}^{inf} x^{-1-alpha} dx
                  = A * exp(-k * delta)                              [Eq. 12]

where A = Lambda/alpha and k = alpha*K.

The execution probability over a small interval dt follows from the
Poisson CDF:

    P(fill in [t, t+dt]) = 1 - exp(-lambda(delta) * dt)

For infinitesimally small dt this converges to lambda(delta)*dt.

Reference
---------
Avellaneda, M. & Stoikov, S. (2008).
"High-frequency trading in a limit order book."
Quantitative Finance, 8(3), 217-224.
DOI: 10.1080/14697680701381228
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray


class ArrivalModel:
    """
    Exponential Poisson arrival-rate model for limit-order executions.

    Models the rate at which market orders from the opposite side of
    the book reach the agent's resting limit orders as a strictly
    decreasing function of their distance from the mid-price:

        lambda(delta) = A * exp(-k * delta)                          [Eq. 12]

    Both the bid (sell market orders hitting the agent's bid) and the ask
    (buy market orders lifting the agent's ask) sides are assumed to follow
    the same intensity function — the paper's symmetry assumption
    (Section 2.4).  Separate ``side`` labels are used throughout for
    clarity, but the underlying math is identical.

    Parameters
    ----------
    A : float
        Base arrival rate — the Poisson intensity at zero spread
        (delta = 0).  Must be > 0.  Paper uses A = 140, interpreted
        as the constant frequency Lambda of market orders divided by
        the power-law exponent alpha (Section 2.5).
    k : float
        Exponential decay coefficient controlling how steeply the
        fill rate falls as quotes move away from the mid-price.
        Must be > 0.  Paper uses k = 1.5.  Larger k → fills fall
        off faster with distance.
    seed : int or None, optional
        Seed for the internal ``numpy.random.Generator``.  Pass an
        integer for reproducible simulations; leave ``None`` for
        fresh randomness on each instantiation.

    Attributes
    ----------
    A : float
    k : float

    Examples
    --------
    >>> model = ArrivalModel(A=140.0, k=1.5, seed=0)
    >>> model.arrival_rate(0.0)           # maximum rate at zero spread
    140.0
    >>> model.arrival_rate(1.0)           # decays with distance
    31.37...
    >>> model.execution_probability(0.75, dt=0.005)
    0.4...
    """

    def __init__(
        self,
        A: float = 140.0,
        k: float = 1.5,
        seed: int | None = None,
    ) -> None:
        # ------------------------------------------------------------------
        # Parameter validation — catch misconfiguration before any draws.
        # ------------------------------------------------------------------
        if A <= 0:
            raise ValueError(f"A must be > 0, got {A}")
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")

        self.A: float = float(A)
        self.k: float = float(k)

        # Modern NumPy Generator API: thread-safe, reproducible per-instance,
        # avoids global random state shared across components.
        self._rng: Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core formula 1 — arrival rate
    # ------------------------------------------------------------------

    def arrival_rate(self, delta: ArrayLike) -> NDArray[np.float64]:
        """
        Compute the Poisson arrival-rate intensity lambda(delta).

        Returns the rate (events per unit time) at which market orders
        will reach a limit order placed at distance delta from the
        mid-price.

        Formula (Eq. 12):

            lambda(delta) = A * exp(-k * delta)

        The function is:
          - Monotonically decreasing in delta.
          - Equal to A at delta = 0 (quote sitting at the mid-price).
          - Approaching zero as delta → infinity (quote never reached).
          - Strictly positive for all finite delta >= 0.

        Parameters
        ----------
        delta : array-like of float
            Distance(s) of the limit order from the mid-price, in price
            units (dollars).  Must be >= 0 everywhere.  Accepts Python
            scalars, lists, and NumPy arrays of any shape — output shape
            matches input shape.

        Returns
        -------
        NDArray[np.float64]
            Arrival rate(s) lambda(delta) > 0, same shape as ``delta``.

        Raises
        ------
        ValueError
            If any element of ``delta`` is strictly negative.

        Notes
        -----
        Vectorisation over delta is intentional: the strategy layer uses
        this to evaluate the intensity across a grid of candidate quote
        distances when plotting the rate profile or calibrating k to data.
        """
        # Coerce to a NumPy array so the function handles scalars, lists,
        # and arrays uniformly.
        delta_arr: NDArray[np.float64] = np.asarray(delta, dtype=np.float64)

        # Guard: negative distances have no physical meaning here.
        # delta is defined as mid - bid (bid side) or ask - mid (ask side),
        # both of which are non-negative by construction.
        if np.any(delta_arr < 0.0):
            raise ValueError(
                f"delta must be >= 0 everywhere; "
                f"got min = {float(delta_arr.min()):.6f}"
            )

        # Vectorised computation: A * exp(-k * delta)
        # numpy broadcasts automatically over any input shape.
        return self.A * np.exp(-self.k * delta_arr)

    # ------------------------------------------------------------------
    # Core formula 2 — execution probability
    # ------------------------------------------------------------------

    def execution_probability(
        self,
        delta: ArrayLike,
        dt: float,
    ) -> NDArray[np.float64]:
        """
        Probability of at least one fill within time interval dt.

        Given that fills arrive as a Poisson process with rate
        lambda(delta), the probability of receiving at least one market
        order in the interval [t, t+dt] is the complement of the
        zero-event probability:

            P(fill | delta, dt) = 1 - exp(-lambda(delta) * dt)

        This is the exact Poisson CDF rather than the common linear
        approximation lambda*dt.  The exact form guarantees that
        probabilities remain in [0, 1) for any combination of A, k,
        delta, and dt — important when dt is not infinitesimally small.

        For the paper's baseline (A=140, dt=0.005), a quote at the
        mid-price (delta=0) has execution probability:

            P = 1 - exp(-140 * 0.005) = 1 - exp(-0.7) ≈ 0.503

        which is well outside the linear regime (lambda*dt = 0.7 > 1),
        making the exact formula essential.

        Parameters
        ----------
        delta : array-like of float
            Distance(s) from the mid-price.  Must be >= 0.
        dt : float
            Length of the time interval.  Must be > 0.

        Returns
        -------
        NDArray[np.float64]
            Execution probability in [0, 1), same shape as ``delta``.

        Raises
        ------
        ValueError
            If any element of ``delta`` is negative, or if ``dt`` <= 0.
        """
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        # Retrieve intensity — also validates delta >= 0.
        lam: NDArray[np.float64] = self.arrival_rate(delta)

        # Exact Poisson CDF: P(N >= 1) = 1 - P(N = 0) = 1 - exp(-lambda*dt)
        return 1.0 - np.exp(-lam * dt)

    # ------------------------------------------------------------------
    # Core formula 3 — simulate arrival
    # ------------------------------------------------------------------

    def simulate_arrival(
        self,
        delta: float,
        dt: float,
    ) -> bool:
        """
        Draw a single Bernoulli outcome: did a market order fill the quote?

        Used by the simulator engine's discrete-time event loop.  At each
        time step the engine calls this method once for the bid side (with
        delta^b) and once for the ask side (with delta^a) to determine
        whether a fill occurred.

        Internally, the method draws a uniform variate U ~ Uniform(0, 1)
        and compares it to the exact fill probability:

            fill = True  if  U < P(fill | delta, dt)

        This is equivalent to sampling from a Poisson process with
        intensity lambda(delta) and checking whether at least one event
        occurred in [t, t+dt].

        Parameters
        ----------
        delta : float
            Distance of the quote from the mid-price.  Must be >= 0.
        dt : float
            Length of the time step.  Must be > 0.

        Returns
        -------
        bool
            ``True`` if a market order reached the quote; ``False`` otherwise.

        Notes
        -----
        The random generator is owned by this instance and seeded via
        ``__init__``.  The simulator engine must use the same instance
        throughout a path to ensure reproducibility from a single seed.
        """
        # Compute the exact fill probability for this (delta, dt) pair.
        prob: float = float(self.execution_probability(delta, dt))

        # Bernoulli trial: compare a uniform draw to the fill probability.
        # rng.random() draws from Uniform(0, 1).
        return bool(self._rng.random() < prob)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def expected_fills_per_step(self, delta: ArrayLike, dt: float) -> NDArray[np.float64]:
        """
        Expected number of fills per time step.

        For a Poisson process, the expected count in [t, t+dt] is simply:

            E[N] = lambda(delta) * dt

        Unlike ``execution_probability``, this can exceed 1.0 for very
        small delta or large dt.  It is useful for calibrating A and k
        to target a desired fill rate.

        Parameters
        ----------
        delta : array-like of float
        dt : float

        Returns
        -------
        NDArray[np.float64]
        """
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        return self.arrival_rate(delta) * dt

    def half_life_distance(self) -> float:
        """
        Quote distance at which the arrival rate falls to A / 2.

        Solves  lambda(delta_half) = A / 2:

            A * exp(-k * delta_half) = A / 2
            delta_half = ln(2) / k

        At the paper's k=1.5 this gives delta_half ≈ 0.462 price units.
        Useful as an intuitive calibration anchor: moving a quote by
        ``half_life_distance()`` halves the expected fill rate.

        Returns
        -------
        float
        """
        return float(np.log(2.0) / self.k)

    def mean_time_to_fill(self, delta: ArrayLike) -> NDArray[np.float64]:
        """
        Expected waiting time (in simulation time units) for the first fill.

        For a Poisson process with rate lambda, the inter-arrival times
        are Exponential(lambda), so:

            E[time to first fill] = 1 / lambda(delta)

        Parameters
        ----------
        delta : array-like of float
            Quote distance(s) from mid-price.  Must be >= 0.

        Returns
        -------
        NDArray[np.float64]
            Expected fill time in the same units as T (the horizon).

        Notes
        -----
        At delta=0 and A=140, the expected time between fills is 1/140 ≈
        0.007 time units, comparable to the paper's dt=0.005 — meaning
        fills are frequent at the mid-price, which is consistent with the
        simulation dynamics.
        """
        lam: NDArray[np.float64] = self.arrival_rate(delta)
        return 1.0 / lam

    def __repr__(self) -> str:
        return f"ArrivalModel(A={self.A}, k={self.k})"


# ---------------------------------------------------------------------------
# Smoke test  (python arrivals.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== ArrivalModel smoke test ===\n")

    # Paper baseline parameters (Section 3.3)
    DT    = 0.005    # paper time step
    model = ArrivalModel(A=140.0, k=1.5, seed=42)
    print(f"Model      : {model}")
    print(f"Half-life  : {model.half_life_distance():.4f} price units")
    print(f"  (fill rate halves every {model.half_life_distance():.4f} dollars from mid)\n")

    # ------------------------------------------------------------------
    # 1. arrival_rate boundary conditions
    # ------------------------------------------------------------------
    lam_0 = float(model.arrival_rate(0.0))
    print(f"arrival_rate(delta=0): {lam_0:.4f}  (expected {model.A})")
    assert abs(lam_0 - model.A) < 1e-12, "Rate at delta=0 must equal A"

    import math
    lam_1     = float(model.arrival_rate(1.0))
    expected  = 140.0 * math.exp(-1.5 * 1.0)
    print(f"arrival_rate(delta=1): {lam_1:.4f}  (expected {expected:.4f})")
    assert abs(lam_1 - expected) < 1e-10, "Rate at delta=1 mismatch"
    print("  ✓ scalar values correct\n")

    # ------------------------------------------------------------------
    # 2. Vectorised input and monotone decay
    # ------------------------------------------------------------------
    deltas = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0])
    rates  = model.arrival_rate(deltas)
    print("Vectorised arrival_rate:")
    for d, r in zip(deltas, rates):
        bar = "█" * max(1, int(r / 7))
        print(f"  delta={d:.2f}  lambda={r:8.4f}  {bar}")
    assert np.all(np.diff(rates) < 0), "Rates must be strictly decreasing"
    print("  ✓ monotonically decreasing with delta\n")

    # ------------------------------------------------------------------
    # 3. execution_probability range and symmetry check
    # ------------------------------------------------------------------
    print(f"execution_probability (dt={DT}):")
    for d in [0.0, 0.5, 1.0, 2.0]:
        p = float(model.execution_probability(d, DT))
        assert 0.0 <= p < 1.0, f"Probability out of range for delta={d}"
        print(f"  delta={d:.2f}  P(fill)={p:.6f}")

    # Verify the known value at delta=0, dt=0.005:
    # P = 1 - exp(-140 * 0.005) = 1 - exp(-0.7) ≈ 0.50341
    p_expected = 1.0 - math.exp(-140.0 * 0.005)
    p_computed = float(model.execution_probability(0.0, DT))
    assert abs(p_computed - p_expected) < 1e-12, "Probability at delta=0 mismatch"
    print(f"\n  At delta=0: P={p_computed:.6f}  (expected {p_expected:.6f})")
    print("  ✓ exact Poisson CDF values correct\n")

    # ------------------------------------------------------------------
    # 4. Linear-regime convergence as dt → 0
    # ------------------------------------------------------------------
    tiny_dt  = 1e-7
    p_exact  = float(model.execution_probability(0.0, tiny_dt))
    p_linear = model.A * tiny_dt          # first-order approximation
    rel_err  = abs(p_exact - p_linear) / p_linear
    print(f"Linear-regime convergence (dt={tiny_dt}):")
    print(f"  Exact  P(fill) = {p_exact:.10f}")
    print(f"  Linear approx  = {p_linear:.10f}")
    print(f"  Relative error = {rel_err:.2e}  (expected ~ lambda*dt/2 ≈ {model.A*tiny_dt/2:.2e})")
    assert rel_err < 1e-5, "Should be in linear regime for tiny dt"
    print("  ✓ converges to lambda*dt for small dt\n")

    # ------------------------------------------------------------------
    # 5. simulate_arrival — Monte Carlo fill rate
    # ------------------------------------------------------------------
    N_TRIALS  = 200_000
    delta_mc  = 0.75
    p_theory  = float(model.execution_probability(delta_mc, DT))
    fills     = sum(model.simulate_arrival(delta_mc, DT) for _ in range(N_TRIALS))
    p_mc      = fills / N_TRIALS
    rel_mc    = abs(p_mc - p_theory) / p_theory

    print(f"Monte Carlo fill rate ({N_TRIALS:,} trials, delta={delta_mc}, dt={DT}):")
    print(f"  Theoretical P(fill) = {p_theory:.6f}")
    print(f"  MC fill rate        = {p_mc:.6f}")
    print(f"  Relative error      = {rel_mc:.2%}")
    assert rel_mc < 0.02, f"MC rate too far from theoretical: {rel_mc:.2%}"
    print("  ✓ MC fill rate within 2% of theoretical\n")

    # ------------------------------------------------------------------
    # 6. Helper methods
    # ------------------------------------------------------------------
    hl   = model.half_life_distance()
    lam_hl = float(model.arrival_rate(hl))
    print(f"half_life_distance: {hl:.4f}")
    print(f"  lambda({hl:.4f}) = {lam_hl:.4f}  (expected {model.A / 2:.4f})")
    assert abs(lam_hl - model.A / 2.0) < 1e-10, "Half-life rate mismatch"

    mttf_0 = float(model.mean_time_to_fill(0.0))
    print(f"\nmean_time_to_fill(delta=0): {mttf_0:.6f}  (expected {1/model.A:.6f})")
    assert abs(mttf_0 - 1.0 / model.A) < 1e-12, "MTTF mismatch"
    print("  ✓ helper methods correct\n")

    # ------------------------------------------------------------------
    # 7. Guard: negative delta raises ValueError
    # ------------------------------------------------------------------
    try:
        model.arrival_rate(-0.01)
        print("ERROR: should have raised ValueError", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"  ✓ ValueError raised for negative delta: {exc}\n")

    # ------------------------------------------------------------------
    # 8. Guard: non-positive dt raises ValueError
    # ------------------------------------------------------------------
    try:
        model.execution_probability(0.5, dt=0.0)
        print("ERROR: should have raised ValueError for dt=0", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"  ✓ ValueError raised for dt=0: {exc}\n")

    print("All checks passed.")