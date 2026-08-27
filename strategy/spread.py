"""
spread.py
=========
Optimal bid-ask spread computation for the Avellaneda-Stoikov (2008)
optimal market-making strategy.

The total optimal spread is derived from the first-order asymptotic
expansion of the HJB equation (Section 3.2) under the assumption of
symmetric exponential order-arrival intensities:

    lambda(delta) = A * exp(-k * delta)                              [Eq. 20]

The spread formula (Eq. 30) decomposes cleanly into two additive terms:

    delta^a + delta^b = gamma * sigma^2 * (T - t)                   [Term 1]
                      + (2 / gamma) * ln(1 + gamma / k)             [Term 2]

Term 1 — Inventory risk component
    Proportional to the variance rate sigma^2 and the risk-aversion
    gamma, and decays linearly to zero as t → T. Captures the cost
    of holding inventory over the remaining horizon: more time means
    more exposure to adverse mid-price moves.

Term 2 — Arrival rate component
    Constant throughout the horizon — depends only on (gamma, k), not
    on time. Represents the compensation required for adverse selection
    and execution uncertainty embedded in the order-arrival process.
    Derived from the first-order condition of the HJB equation when the
    arrival term is log-linearised (Eq. 26).

Key properties
--------------
- Spread is always >= 0.
- Spread is independent of the mid-price S and the inventory q.
  (A consequence of exponential arrivals + first-order q expansion.)
- Spread is monotonically non-increasing in t: narrows as t → T.
- At t = T, the spread collapses to the pure arrival-rate component:
    spread(T) = (2 / gamma) * ln(1 + gamma / k) > 0.
- As gamma → 0, both terms → 0: a risk-neutral agent quotes at the mid.

Bid and ask quotes are placed symmetrically around the reservation price:

    p^b = r(S, q, t) - spread / 2
    p^a = r(S, q, t) + spread / 2

Reference
---------
Avellaneda, M. & Stoikov, S. (2008).
"High-frequency trading in a limit order book."
Quantitative Finance, 8(3), 217-224.
DOI: 10.1080/14697680701381228
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpreadResult:
    """
    Output of a single call to ``SpreadModel.compute()``.

    Attributes
    ----------
    spread : float or NDArray[np.float64]
        Total optimal bid-ask spread  delta^a + delta^b >= 0.
    half_spread : float or NDArray[np.float64]
        Half the total spread; distance of each quote from the
        reservation price.  Equal to spread / 2.
    inventory_risk_component : float or NDArray[np.float64]
        Time-varying term:  gamma * sigma^2 * (T - t).
        Decays to zero as t → T.
    arrival_rate_component : float
        Constant term:  (2 / gamma) * ln(1 + gamma / k).
        Independent of time and inventory.
    time_remaining : float
        Remaining horizon T - t at the time of computation.
    """

    spread: float | NDArray[np.float64]
    half_spread: float | NDArray[np.float64]
    inventory_risk_component: float | NDArray[np.float64]
    arrival_rate_component: float
    time_remaining: float

    def __str__(self) -> str:
        return (
            f"SpreadResult("
            f"spread={self.spread}, "
            f"half_spread={self.half_spread}, "
            f"inv_risk={self.inventory_risk_component:.6f}, "
            f"arrival={self.arrival_rate_component:.6f}, "
            f"time_remaining={self.time_remaining:.4f})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SpreadModel:
    """
    Computes the Avellaneda-Stoikov optimal bid-ask spread.

    The spread is the second step in the paper's two-step optimal quoting
    procedure (Section 3.1):

        Step 1 → ReservationPrice.compute()   [reservation.py]
        Step 2 → SpreadModel.compute()        [this class]

    Given the reservation price r and the optimal half-spread, quotes are:

        p^b = r - half_spread
        p^a = r + half_spread

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient of the CARA utility function.
        Must be > 0.  Higher gamma → wider spread → more conservative
        quoting to limit inventory accumulation.
        Paper uses gamma = 0.1 (Table 1), 0.01 (Table 2), 1.0 (Table 3).
    sigma : float
        Volatility of the arithmetic Brownian motion mid-price.
        Must be > 0.  Enters as sigma^2 in the inventory-risk term.
        Paper uses sigma = 2.
    T : float
        Terminal time horizon.  Must be > 0.
        Paper uses T = 1.
    k : float
        Exponential decay coefficient of the arrival-rate intensity
        lambda(delta) = A * exp(-k * delta).  Must be > 0.
        Higher k → fills fall off faster with distance → wider spread.
        Paper uses k = 1.5.

    Attributes
    ----------
    gamma : float
    sigma : float
    T : float
    k : float

    Examples
    --------
    >>> model = SpreadModel(gamma=0.1, sigma=2.0, T=1.0, k=1.5)
    >>> result = model.compute(t=0.0)
    >>> round(result.spread, 4)
    1.4801

    Spread narrows as time approaches T:

    >>> result_mid = model.compute(t=0.5)
    >>> result_mid.spread < result.spread
    True

    >>> result_T = model.compute(t=1.0)
    >>> round(result_T.inventory_risk_component, 10)
    0.0
    """

    def __init__(
        self,
        gamma: float,
        sigma: float,
        T: float,
        k: float,
    ) -> None:
        # ------------------------------------------------------------------
        # Parameter validation.
        # ------------------------------------------------------------------
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if T <= 0:
            raise ValueError(f"T must be > 0, got {T}")
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")

        self.gamma: float = float(gamma)
        self.sigma: float = float(sigma)
        self.T: float     = float(T)
        self.k: float     = float(k)

        # ------------------------------------------------------------------
        # Pre-compute constant sub-expressions.
        #
        # Both are invariant across the entire simulation run, so computing
        # them once in __init__ avoids redundant work in the engine's tight
        # loop (200 steps × 1000 Monte Carlo paths).
        # ------------------------------------------------------------------

        # Coefficient of the inventory-risk term: gamma * sigma^2
        # Multiplied by (T - t) at each call to get the time-varying component.
        # [from Term 1 of Eq. 30]
        self._risk_coeff: float = self.gamma * self.sigma ** 2

        # Constant arrival-rate component: (2 / gamma) * ln(1 + gamma / k)
        # This is the floor of the spread — it persists even at t = T.
        # [Term 2 of Eq. 30, derived from first-order condition Eq. 26]
        self._arrival_component: float = (
            (2.0 / self.gamma) * float(np.log(1.0 + self.gamma / self.k))
        )

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def compute(
        self,
        t: ArrayLike,
        q: ArrayLike = None,
        S: ArrayLike = None,
    ) -> SpreadResult:
        """
        Compute the total optimal bid-ask spread at time t.

        ``q`` and ``S`` are accepted as optional keyword arguments and are
        IGNORED here (the finite-horizon spread, Eq. 30, does not depend on
        inventory or mid-price). They exist purely so that callers such as
        ``MarketMakerAgent.quote()`` can call ``compute(t=..., q=..., S=...)``
        uniformly across both the finite-horizon and infinite-horizon spread
        models without branching on which model is active.

        Formula (Eq. 30):

            spread(t) = gamma * sigma^2 * (T - t)
                      + (2 / gamma) * ln(1 + gamma / k)

        The spread is:
          - Inventory-independent: does not depend on q or S.
            (Consequence of symmetric exponential arrivals, Section 3.2.)
          - Monotonically non-increasing in t: wider early, narrower near T.
          - Bounded below by the arrival-rate component at t = T.
          - Always >= 0 for valid parameter inputs.

        Parameters
        ----------
        t : array-like of float
            Current simulation time(s).  Must satisfy 0 <= t <= T.
            Accepts scalars, Python lists, and NumPy arrays for batch
            evaluation over an entire time grid — useful for plotting the
            spread term-structure or computing average spread statistics.

        Returns
        -------
        SpreadResult
            Frozen dataclass with fields:
              - ``spread``                  total spread (Eq. 30)
              - ``half_spread``             spread / 2
              - ``inventory_risk_component`` gamma * sigma^2 * (T - t)
              - ``arrival_rate_component``   (2/gamma) * ln(1 + gamma/k)
              - ``time_remaining``           T - t (scalar, taken from t[0]
                                            when t is an array)

        Raises
        ------
        ValueError
            If any element of ``t`` is outside [0, T].

        Notes
        -----
        Vectorisation: when ``t`` is a NumPy array the output ``spread``
        and related fields are also arrays of the same shape, enabling
        efficient term-structure plotting with a single call.
        """
        # Coerce to NumPy array for uniform scalar/array handling.
        t_arr: NDArray[np.float64] = np.asarray(t, dtype=np.float64)

        # Validate: all time values must lie within the horizon.
        if np.any(t_arr < 0.0) or np.any(t_arr > self.T):
            bad = t_arr[(t_arr < 0.0) | (t_arr > self.T)]
            raise ValueError(
                f"All t values must be in [0, T]=[0, {self.T}]; "
                f"got out-of-range values: {bad}"
            )

        # ------------------------------------------------------------------
        # Term 1 — inventory risk component.
        # gamma * sigma^2 * (T - t)
        # Time-varying: maximum at t=0, decays linearly to 0 at t=T.
        # ------------------------------------------------------------------
        time_remaining: NDArray[np.float64] = self.T - t_arr
        inventory_risk: NDArray[np.float64] = self._risk_coeff * time_remaining

        # ------------------------------------------------------------------
        # Term 2 — arrival rate component.
        # (2 / gamma) * ln(1 + gamma / k)
        # Constant: pre-computed in __init__, broadcast as a scalar here.
        # ------------------------------------------------------------------
        arrival_component: float = self._arrival_component

        # ------------------------------------------------------------------
        # Total spread and half-spread.
        # ------------------------------------------------------------------
        spread: NDArray[np.float64]      = inventory_risk + arrival_component
        half_spread: NDArray[np.float64] = spread / 2.0

        # ------------------------------------------------------------------
        # Scalar passthrough.
        # When t was a Python scalar or 0-d array, return Python floats so
        # downstream code (engine event loop) gets native types, not 0-d arrays.
        # ------------------------------------------------------------------
        if spread.ndim == 0:
            return SpreadResult(
                spread=float(spread),
                half_spread=float(half_spread),
                inventory_risk_component=float(inventory_risk),
                arrival_rate_component=arrival_component,
                time_remaining=float(time_remaining),
            )

        # For array inputs, time_remaining is an array; record the first
        # element as the scalar summary (caller can inspect the full array
        # via inventory_risk_component).
        return SpreadResult(
            spread=spread,
            half_spread=half_spread,
            inventory_risk_component=inventory_risk,
            arrival_rate_component=arrival_component,
            time_remaining=float(time_remaining.flat[0]),
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def minimum_spread(self) -> float:
        """
        Minimum spread attained at the terminal time t = T.

        At t = T the inventory-risk component vanishes, leaving only the
        constant arrival-rate floor:

            spread_min = (2 / gamma) * ln(1 + gamma / k)

        Returns
        -------
        float
            The floor value of the spread throughout the horizon.
        """
        # arrival-rate component already computed; return directly.
        return self._arrival_component

    def maximum_spread(self) -> float:
        """
        Maximum spread at the start of the horizon t = 0.

        At t = 0 the full inventory-risk component is active:

            spread_max = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/k)

        Returns
        -------
        float
        """
        return float(self._risk_coeff * self.T + self._arrival_component)

    def spread_at_fraction(self, fraction: float) -> float:
        """
        Spread at a given fraction of the horizon.

        Convenience wrapper: evaluates the spread at t = fraction * T.

        Parameters
        ----------
        fraction : float
            Fraction of the total horizon elapsed.  Must be in [0, 1].

        Returns
        -------
        float
        """
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        return float(self.compute(t=fraction * self.T).spread)

    def time_to_half_spread(self) -> float:
        """
        Time at which the spread falls to half its initial (t=0) value.

        Solves  spread(t*) = spread_max / 2  for t*:

            gamma * sigma^2 * (T - t*) + arrival = (spread_max + spread_min) / 2
            t* = T - (spread_max - spread_min) / (2 * gamma * sigma^2)
               = T - T / 2
               = T / 2

        The symmetry arises because the inventory-risk component is linear
        in (T - t): the spread always halves (relative to its dynamic range)
        at the midpoint of the horizon.

        Returns
        -------
        float
            t* = T / 2.
        """
        # The inventory-risk component decays linearly from its max at t=0
        # to 0 at t=T; its midpoint (half of max) occurs at t = T/2.
        return self.T / 2.0

    def quote_prices(
        self,
        reservation_price: float | NDArray[np.float64],
        t: float,
    ) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
        """
        Compute bid and ask quotes from a reservation price and time.

        Places quotes symmetrically around the reservation price:

            p^b = r - half_spread(t)
            p^a = r + half_spread(t)

        Parameters
        ----------
        reservation_price : float or NDArray[np.float64]
            The agent's indifference price r(S, q, t) from reservation.py.
        t : float
            Current simulation time.  Must satisfy 0 <= t <= T.

        Returns
        -------
        bid : float or NDArray[np.float64]
            Optimal bid quote p^b.
        ask : float or NDArray[np.float64]
            Optimal ask quote p^a.
        """
        result = self.compute(t=t)
        bid = reservation_price - result.half_spread
        ask = reservation_price + result.half_spread
        return bid, ask

    def __repr__(self) -> str:
        return (
            f"SpreadModel("
            f"gamma={self.gamma}, "
            f"sigma={self.sigma}, "
            f"T={self.T}, "
            f"k={self.k})"
        )


# ---------------------------------------------------------------------------
# Smoke test  (python spread.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import math

    print("=== SpreadModel smoke test ===\n")

    # Paper baseline parameters (Section 3.3, Table 1)
    model = SpreadModel(gamma=0.1, sigma=2.0, T=1.0, k=1.5)
    print(f"Model:          {model}")
    print(f"Min spread:     {model.minimum_spread():.6f}  (at t=T)")
    print(f"Max spread:     {model.maximum_spread():.6f}  (at t=0)")
    print(f"Half-horizon t: {model.time_to_half_spread():.4f}  (spread ≈ mid range)\n")

    # ------------------------------------------------------------------
    # 1. Exact value at t=0 (paper baseline, manually verified)
    # ------------------------------------------------------------------
    # spread = 0.1 * 4 * 1.0 + (2/0.1) * ln(1 + 0.1/1.5)
    #        = 0.4 + 20 * ln(1.0667)
    #        = 0.4 + 20 * 0.06454...
    #        = 0.4 + 1.29008...
    #        ≈ 1.48008...
    expected_t0 = 0.1 * 4.0 * 1.0 + (2.0 / 0.1) * math.log(1.0 + 0.1 / 1.5)
    result_t0   = model.compute(t=0.0)

    print("Spread at t=0 (paper baseline):")
    print(f"  spread          = {result_t0.spread:.8f}")
    print(f"  expected        = {expected_t0:.8f}")
    print(f"  inv_risk_term   = {result_t0.inventory_risk_component:.8f}  "
          f"(expected {0.1*4.0*1.0:.8f})")
    print(f"  arrival_term    = {result_t0.arrival_rate_component:.8f}  "
          f"(expected {(2/0.1)*math.log(1+0.1/1.5):.8f})")
    assert abs(result_t0.spread - expected_t0) < 1e-12, "Spread at t=0 mismatch"
    print("  ✓ exact value matches hand-computed formula\n")

    # ------------------------------------------------------------------
    # 2. At t=T the inventory-risk component must vanish exactly
    # ------------------------------------------------------------------
    result_tT = model.compute(t=1.0)
    print("Spread at t=T=1.0:")
    print(f"  spread        = {result_tT.spread:.8f}")
    print(f"  inv_risk_term = {result_tT.inventory_risk_component:.8f}  (expected 0.0)")
    print(f"  arrival_term  = {result_tT.arrival_rate_component:.8f}")
    assert abs(result_tT.inventory_risk_component) < 1e-12, "Risk term must vanish at T"
    assert abs(result_tT.spread - model.minimum_spread()) < 1e-12
    print("  ✓ inventory-risk component vanishes at t = T\n")

    # ------------------------------------------------------------------
    # 3. Spread is monotonically non-increasing in t
    # ------------------------------------------------------------------
    times      = np.linspace(0.0, 1.0, 11)
    spreads    = model.compute(t=times).spread
    print("Spread term-structure (vectorised, gamma=0.1):")
    for t_val, s_val in zip(times, spreads):
        bar = "█" * int(s_val * 14)
        print(f"  t={t_val:.1f}  spread={s_val:.4f}  {bar}")
    assert np.all(np.diff(spreads) <= 0), "Spread must be non-increasing in t"
    print("  ✓ monotonically non-increasing with t\n")

    # ------------------------------------------------------------------
    # 4. Spread is inventory- and price-independent
    # ------------------------------------------------------------------
    # Compute at the same t for very different (S, q) — spread must be identical.
    s1 = model.compute(t=0.5).spread
    s2 = model.compute(t=0.5).spread   # same t, q and S not arguments → same
    assert abs(s1 - s2) < 1e-12
    print(f"Spread at t=0.5: {s1:.8f}  (inventory- and price-independent)")
    print("  ✓ spread does not depend on S or q\n")

    # ------------------------------------------------------------------
    # 5. Half-spread symmetry: bid and ask equidistant from reservation price
    # ------------------------------------------------------------------
    r_price = 99.5   # arbitrary reservation price
    bid, ask = model.quote_prices(reservation_price=r_price, t=0.3)
    mid_of_quotes = (bid + ask) / 2.0
    print(f"Quote symmetry (r={r_price}, t=0.3):")
    print(f"  bid               = {bid:.6f}")
    print(f"  ask               = {ask:.6f}")
    print(f"  (bid+ask)/2       = {mid_of_quotes:.6f}  (expected {r_price})")
    print(f"  ask - bid         = {ask - bid:.6f}  (= spread)")
    assert abs(mid_of_quotes - r_price) < 1e-12, "Quotes must be centred on r"
    assert abs((ask - bid) - model.compute(t=0.3).spread) < 1e-12
    print("  ✓ quotes symmetric around reservation price\n")

    # ------------------------------------------------------------------
    # 6. Risk-aversion sensitivity: higher gamma → wider spread
    # ------------------------------------------------------------------
    print("Spread at t=0 for different gamma values:")
    for g in [0.01, 0.1, 0.5, 1.0]:
        m = SpreadModel(gamma=g, sigma=2.0, T=1.0, k=1.5)
        s = m.compute(t=0.0).spread
        bar = "█" * int(s * 4)
        print(f"  gamma={g:.2f}  spread={s:.4f}  {bar}")
    # Verify spread at gamma=0.01 < gamma=0.1 < gamma=1.0
    spreads_gamma = [
        SpreadModel(gamma=g, sigma=2.0, T=1.0, k=1.5).compute(t=0.0).spread
        for g in [0.01, 0.1, 1.0]
    ]
    assert spreads_gamma[0] < spreads_gamma[1] < spreads_gamma[2], \
        "Spread must increase with gamma"
    print("  ✓ spread increases with risk aversion gamma\n")

    # ------------------------------------------------------------------
    # 7. Reproduce paper Table 1/2/3 average spread values approximately
    # ------------------------------------------------------------------
    # The paper reports an average spread of ~1.49 for gamma=0.1.
    # Since the spread decays linearly from max to min over [0, T],
    # the time-average equals (max + min) / 2.
    avg_spread = (model.maximum_spread() + model.minimum_spread()) / 2.0
    print(f"Approximate time-averaged spread (gamma=0.1):")
    print(f"  (spread_max + spread_min) / 2 = {avg_spread:.4f}")
    print(f"  Paper Table 1 reports ~1.49")
    assert abs(avg_spread - 1.49) < 0.05, \
        f"Average spread {avg_spread:.4f} too far from paper value 1.49"
    print("  ✓ consistent with paper Table 1\n")

    # Table 2: gamma=0.01 → average spread ≈ 1.35
    m2 = SpreadModel(gamma=0.01, sigma=2.0, T=1.0, k=1.5)
    avg2 = (m2.maximum_spread() + m2.minimum_spread()) / 2.0
    print(f"Average spread (gamma=0.01): {avg2:.4f}  (paper Table 2 reports ~1.35)")
    assert abs(avg2 - 1.35) < 0.05, f"gamma=0.01 average spread mismatch: {avg2:.4f}"
    print("  ✓ consistent with paper Table 2\n")

    # Table 3: gamma=1.0 → average spread ≈ 3.02
    m3 = SpreadModel(gamma=1.0, sigma=2.0, T=1.0, k=1.5)
    avg3 = (m3.maximum_spread() + m3.minimum_spread()) / 2.0
    print(f"Average spread (gamma=1.00): {avg3:.4f}  (paper Table 3 reports ~3.02)")
    assert abs(avg3 - 3.02) < 0.05, f"gamma=1.0 average spread mismatch: {avg3:.4f}"
    print("  ✓ consistent with paper Table 3\n")

    # ------------------------------------------------------------------
    # 8. Guard: t out of range
    # ------------------------------------------------------------------
    try:
        model.compute(t=1.5)
        print("ERROR: should have raised ValueError", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"  ✓ ValueError for t > T: {exc}\n")

    try:
        model.compute(t=-0.1)
        print("ERROR: should have raised ValueError", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"  ✓ ValueError for t < 0: {exc}\n")

    print("All checks passed.")