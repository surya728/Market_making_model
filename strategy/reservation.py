"""
reservation.py
==============
Reservation (indifference) price computation for the Avellaneda-Stoikov
(2008) optimal market-making strategy.

The reservation price is the agent's subjective fair value of the asset
given their current inventory position. It adjusts the mid-price to reflect
the inventory risk the agent is carrying:

    r(S, q, t) = S - q * gamma * sigma^2 * (T - t)                  [Eq. 8/29]

Derivation
----------
The reservation price arises from the agent's CARA exponential utility
maximisation problem. Given value function:

    v(x, s, q, t) = -exp(-gamma*x) * exp(-gamma*q*s)
                    * exp(gamma^2 * q^2 * sigma^2 * (T-t) / 2)      [Eq. 3]

The reservation bid price r^b is the price at which the agent is
indifferent between their current portfolio and current portfolio plus
one unit of stock (Definition 1):

    v(x - r^b, s, q+1, t) = v(x, s, q, t)                          [Eq. 4]

Solving symmetrically for the bid and ask reservation prices (Eq. 6-7)
and taking their average gives the reservation price (Eq. 8):

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

Intuition
---------
- At q = 0  (flat):       r = S          (no inventory adjustment)
- At q > 0  (long):       r < S          (agent wants to sell → quotes lower)
- At q < 0  (short):      r > S          (agent wants to buy  → quotes higher)
- As t → T  (near term):  r → S          (inventory risk vanishes at horizon)
- As gamma→0 (risk neutral): r → S       (agent ignores inventory)

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
class ReservationPriceResult:
    """
    Output of a single call to ``ReservationPrice.compute()``.

    Attributes
    ----------
    price : float or NDArray[np.float64]
        Reservation price r(S, q, t).  Scalar when scalar inputs are
        provided; array when array inputs are provided.
    inventory_adjustment : float or NDArray[np.float64]
        The inventory-risk correction term  -q * gamma * sigma^2 * (T-t).
        Negative when long (q > 0), positive when short (q < 0).
    time_remaining : float
        Remaining horizon T - t at the time of computation.
    mid_price : float or NDArray[np.float64]
        The raw mid-price S passed into the computation.
    """

    price: float | NDArray[np.float64]
    inventory_adjustment: float | NDArray[np.float64]
    time_remaining: float
    mid_price: float | NDArray[np.float64]

    def __str__(self) -> str:
        return (
            f"ReservationPriceResult("
            f"price={self.price}, "
            f"adjustment={self.inventory_adjustment:.6f}, "
            f"time_remaining={self.time_remaining:.4f})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ReservationPrice:
    """
    Computes the Avellaneda-Stoikov reservation (indifference) price.

    The reservation price is the agent's personal valuation of the
    asset adjusted for their current inventory and remaining time horizon.
    It is the first step in the paper's two-step optimal quoting procedure:

        Step 1 → ReservationPrice.compute()   [this class]
        Step 2 → Spread.compute()             [spread.py]

    The bid and ask quotes are then placed symmetrically around this
    reservation price at the optimal half-spread distance.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient of the CARA utility function.
        Must be > 0.  Higher gamma → stronger inventory adjustment →
        wider divergence of quotes from mid-price when holding inventory.
        Paper uses gamma = 0.1 (Table 1), 0.01 (Table 2), 1.0 (Table 3).
    sigma : float
        Volatility of the arithmetic Brownian motion mid-price process.
        Must be > 0.  Enters as sigma^2 (price variance per unit time).
        Paper uses sigma = 2.
    T : float
        Terminal time horizon.  Must be > 0.
        Paper uses T = 1.

    Attributes
    ----------
    gamma : float
    sigma : float
    T : float

    Examples
    --------
    >>> rp = ReservationPrice(gamma=0.1, sigma=2.0, T=1.0)

    Flat inventory — reservation price equals mid-price:

    >>> result = rp.compute(S=100.0, q=0, t=0.0)
    >>> result.price
    100.0

    Long inventory — reservation price below mid (agent wants to sell):

    >>> result = rp.compute(S=100.0, q=5, t=0.0)
    >>> result.price
    98.0

    Short inventory — reservation price above mid (agent wants to buy):

    >>> result = rp.compute(S=100.0, q=-5, t=0.0)
    >>> result.price
    102.0
    """

    def __init__(
        self,
        gamma: float,
        sigma: float,
        T: float,
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

        self.gamma: float = float(gamma)
        self.sigma: float = float(sigma)
        self.T: float     = float(T)

        # Pre-compute gamma * sigma^2 — the risk-adjustment coefficient.
        # This product appears in every call to compute() and is constant
        # for a given (gamma, sigma) pair, so caching it saves two
        # multiplications per tick across 200 steps × 1000 MC paths.
        self._risk_coeff: float = self.gamma * self.sigma ** 2

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def compute(
        self,
        S: ArrayLike,
        q: ArrayLike,
        t: float,
    ) -> ReservationPriceResult:
        """
        Compute the reservation (indifference) price.

        Formula (Eq. 8 / 29 of the paper):

            r(S, q, t) = S - q * gamma * sigma^2 * (T - t)

        The formula decomposes into:

            r = S  +  adjustment
            adjustment = -q * gamma * sigma^2 * (T - t)

        where the adjustment is:
          - Zero      when q = 0  (flat inventory)
          - Negative  when q > 0  (long: push quotes down to attract sells)
          - Positive  when q < 0  (short: push quotes up to attract buys)
          - Decays to zero as t → T  (inventory risk vanishes at horizon)

        Parameters
        ----------
        S : array-like of float
            Current market mid-price S_t.  Must be > 0.  Accepts
            scalars and NumPy arrays for batch evaluation over multiple
            scenarios (e.g., sensitivity analysis across price levels).
        q : array-like of float
            Current inventory in shares.  Positive = long position,
            negative = short position, zero = flat.  Must be broadcastable
            with S.
        t : float
            Current simulation time.  Must satisfy 0 <= t <= T.

        Returns
        -------
        ReservationPriceResult
            Frozen dataclass with fields: ``price``, ``inventory_adjustment``,
            ``time_remaining``, ``mid_price``.

        Raises
        ------
        ValueError
            If ``t`` is outside [0, T], or if any element of ``S`` is <= 0.

        Notes
        -----
        Vectorisation: ``S`` and ``q`` are passed through ``np.asarray``,
        so the method handles scalars, Python lists, and NumPy arrays of
        any shape uniformly.  Output shape follows NumPy broadcasting rules.
        """
        # ------------------------------------------------------------------
        # Input coercion and validation.
        # ------------------------------------------------------------------

        # Coerce to NumPy arrays for uniform handling of scalars and arrays.
        S_arr: NDArray[np.float64] = np.asarray(S, dtype=np.float64)
        q_arr: NDArray[np.float64] = np.asarray(q, dtype=np.float64)

        # Validate mid-price: negative or zero prices are nonsensical.
        if np.any(S_arr <= 0):
            raise ValueError(
                f"S must be > 0 everywhere; got min = {float(S_arr.min()):.6f}"
            )

        # Validate time: must be within the investment horizon.
        if t < 0.0 or t > self.T:
            raise ValueError(
                f"t = {t} is outside valid range [0, T] = [0, {self.T}]"
            )

        # ------------------------------------------------------------------
        # Core computation.
        # ------------------------------------------------------------------

        # Remaining time to terminal horizon T.
        # As (T - t) → 0 the inventory adjustment → 0, meaning the agent
        # values the stock at the mid-price regardless of position size
        # (Section 2.2, paragraph below Eq. 8).
        time_remaining: float = self.T - t

        # Inventory risk adjustment:  -q * gamma * sigma^2 * (T - t)
        # The sign ensures:
        #   long  (q > 0) → negative adjustment → quotes move below mid
        #   short (q < 0) → positive adjustment → quotes move above mid
        inventory_adjustment: NDArray[np.float64] = (
            -q_arr * self._risk_coeff * time_remaining
        )

        # Reservation price: mid-price corrected for inventory risk.
        reservation_price: NDArray[np.float64] = S_arr + inventory_adjustment

        # ------------------------------------------------------------------
        # Return scalars when scalar inputs were given.
        # Avoids surprising 0-d array outputs for the common single-step case.
        # ------------------------------------------------------------------
        if reservation_price.ndim == 0:
            return ReservationPriceResult(
                price=float(reservation_price),
                inventory_adjustment=float(inventory_adjustment),
                time_remaining=time_remaining,
                mid_price=float(S_arr),
            )

        return ReservationPriceResult(
            price=reservation_price,
            inventory_adjustment=inventory_adjustment,
            time_remaining=time_remaining,
            mid_price=S_arr,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def inventory_sensitivity(self, t: float) -> float:
        """
        Rate of change of reservation price with respect to inventory.

        The partial derivative of r with respect to q:

            dr/dq = -gamma * sigma^2 * (T - t)

        This is always <= 0: increasing inventory (going longer) always
        lowers the reservation price.  Its magnitude decreases toward
        zero as t → T (inventory becomes less risky near the horizon).

        Parameters
        ----------
        t : float
            Current simulation time.  Must satisfy 0 <= t <= T.

        Returns
        -------
        float
            dr/dq at time t.  Always <= 0.
        """
        if t < 0.0 or t > self.T:
            raise ValueError(
                f"t = {t} is outside valid range [0, {self.T}]"
            )
        return -self._risk_coeff * (self.T - t)

    def breakeven_inventory(self, S: float, r_target: float, t: float) -> float:
        """
        Inventory level q at which the reservation price equals r_target.

        Solves  r_target = S - q * gamma * sigma^2 * (T - t)  for q:

            q* = (S - r_target) / (gamma * sigma^2 * (T - t))

        Useful for computing how much inventory would push the reservation
        price to a specific level — e.g., to the current best bid or ask.

        Parameters
        ----------
        S : float
            Current mid-price.  Must be > 0.
        r_target : float
            Target reservation price.
        t : float
            Current simulation time.  Must satisfy 0 <= t < T.

        Returns
        -------
        float
            The inventory q* that achieves r = r_target.

        Raises
        ------
        ValueError
            If t == T (division by zero: adjustment vanishes at horizon).
        """
        if t >= self.T:
            raise ValueError(
                "breakeven_inventory is undefined at t = T: "
                "the inventory adjustment vanishes and q* diverges."
            )
        if S <= 0:
            raise ValueError(f"S must be > 0, got {S}")

        time_remaining: float = self.T - t
        return (S - r_target) / (self._risk_coeff * time_remaining)

    def __repr__(self) -> str:
        return (
            f"ReservationPrice("
            f"gamma={self.gamma}, "
            f"sigma={self.sigma}, "
            f"T={self.T})"
        )


# ---------------------------------------------------------------------------
# Smoke test  (python reservation.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== ReservationPrice smoke test ===\n")

    # Paper baseline parameters (Section 3.3)
    rp = ReservationPrice(gamma=0.1, sigma=2.0, T=1.0)
    print(f"Model: {rp}\n")

    # ------------------------------------------------------------------
    # 1. Flat inventory — r must equal S exactly
    # ------------------------------------------------------------------
    result_flat = rp.compute(S=100.0, q=0, t=0.0)
    print("Flat inventory (q=0, t=0):")
    print(f"  r = {result_flat.price:.6f}  (expected 100.000000)")
    print(f"  adjustment = {result_flat.inventory_adjustment:.6f}  (expected 0.0)")
    assert abs(result_flat.price - 100.0) < 1e-12, "Flat inventory must give r=S"
    assert abs(result_flat.inventory_adjustment) < 1e-12
    print("  ✓ r = S when q = 0\n")

    # ------------------------------------------------------------------
    # 2. Long inventory — r < S
    # ------------------------------------------------------------------
    result_long = rp.compute(S=100.0, q=5, t=0.0)
    # Manual:  r = 100 - 5 * 0.1 * 4 * 1.0 = 100 - 2.0 = 98.0
    print("Long inventory (q=+5, t=0):")
    print(f"  r = {result_long.price:.6f}  (expected 98.000000)")
    print(f"  adjustment = {result_long.inventory_adjustment:.6f}  (expected -2.0)")
    assert abs(result_long.price - 98.0) < 1e-12, "Long inventory must lower r"
    assert result_long.price < 100.0
    print("  ✓ r < S when q > 0\n")

    # ------------------------------------------------------------------
    # 3. Short inventory — r > S
    # ------------------------------------------------------------------
    result_short = rp.compute(S=100.0, q=-5, t=0.0)
    print("Short inventory (q=-5, t=0):")
    print(f"  r = {result_short.price:.6f}  (expected 102.000000)")
    print(f"  adjustment = {result_short.inventory_adjustment:.6f}  (expected +2.0)")
    assert abs(result_short.price - 102.0) < 1e-12, "Short inventory must raise r"
    assert result_short.price > 100.0
    print("  ✓ r > S when q < 0\n")

    # ------------------------------------------------------------------
    # 4. Adjustment vanishes at terminal time t = T
    # ------------------------------------------------------------------
    result_terminal = rp.compute(S=100.0, q=10, t=1.0)
    print("Terminal time (q=10, t=T=1.0):")
    print(f"  r = {result_terminal.price:.6f}  (expected 100.000000)")
    print(f"  adjustment = {result_terminal.inventory_adjustment:.6f}  (expected 0.0)")
    assert abs(result_terminal.price - 100.0) < 1e-12, "At t=T, r must equal S"
    print("  ✓ inventory adjustment vanishes at t = T\n")

    # ------------------------------------------------------------------
    # 5. Monotone decay of adjustment over time
    # ------------------------------------------------------------------
    times = np.linspace(0.0, 1.0, 6)
    print("Reservation price decaying toward S over time (q=5):")
    adjustments = []
    for t_val in times:
        res = rp.compute(S=100.0, q=5, t=float(t_val))
        adjustments.append(res.inventory_adjustment)
        print(f"  t={t_val:.2f}  r={res.price:.4f}  adj={res.inventory_adjustment:.4f}")
    # Adjustment must be non-decreasing (moving toward 0) as t → T
    assert all(adjustments[i] <= adjustments[i+1] for i in range(len(adjustments)-1)), \
        "Adjustment must move monotonically toward 0"
    print("  ✓ adjustment decays monotonically toward 0\n")

    # ------------------------------------------------------------------
    # 6. Vectorised inputs (batch evaluation)
    # ------------------------------------------------------------------
    q_grid = np.arange(-5, 6)      # inventory from -5 to +5
    result_batch = rp.compute(S=100.0, q=q_grid, t=0.0)
    print("Vectorised over inventory grid (S=100, t=0):")
    for qi, ri in zip(q_grid, result_batch.price):
        bar = "█" * int(abs(ri - 100.0) * 5)
        side = "↓" if ri < 100 else ("↑" if ri > 100 else "=")
        print(f"  q={qi:+d}  r={ri:.4f}  {side} {bar}")
    assert result_batch.price.shape == q_grid.shape, "Output shape must match input"
    print("  ✓ vectorised output shape correct\n")

    # ------------------------------------------------------------------
    # 7. Symmetry: long q and short -q produce equal and opposite adjustments
    # ------------------------------------------------------------------
    r_plus  = rp.compute(S=100.0, q=+3, t=0.5).price
    r_minus = rp.compute(S=100.0, q=-3, t=0.5).price
    assert abs((r_plus - 100.0) + (r_minus - 100.0)) < 1e-12, \
        "Adjustments for +q and -q must be equal and opposite"
    print(f"Symmetry check (q=±3, t=0.5):")
    print(f"  r(q=+3) = {r_plus:.6f}")
    print(f"  r(q=-3) = {r_minus:.6f}")
    print(f"  sum of adjustments = {(r_plus-100)+(r_minus-100):.2e}  (expected 0)")
    print("  ✓ adjustments are equal and opposite for ±q\n")

    # ------------------------------------------------------------------
    # 8. inventory_sensitivity helper
    # ------------------------------------------------------------------
    sens_t0 = rp.inventory_sensitivity(t=0.0)
    sens_tT = rp.inventory_sensitivity(t=1.0)
    print(f"inventory_sensitivity:")
    print(f"  at t=0.0: dr/dq = {sens_t0:.4f}  (expected {-rp.gamma * rp.sigma**2:.4f})")
    print(f"  at t=T=1: dr/dq = {sens_tT:.4f}  (expected 0.0)")
    assert abs(sens_t0 - (-rp.gamma * rp.sigma**2)) < 1e-12
    assert abs(sens_tT) < 1e-12
    print("  ✓ sensitivity correct at both endpoints\n")

    # ------------------------------------------------------------------
    # 9. breakeven_inventory helper
    # ------------------------------------------------------------------
    q_be = rp.breakeven_inventory(S=100.0, r_target=98.0, t=0.0)
    print(f"breakeven_inventory (S=100, r_target=98, t=0):")
    print(f"  q* = {q_be:.4f}  (expected 5.0)")
    assert abs(q_be - 5.0) < 1e-12
    # Verify: plugging q* back gives r = 98
    r_verify = rp.compute(S=100.0, q=q_be, t=0.0).price
    assert abs(r_verify - 98.0) < 1e-12
    print("  ✓ breakeven inventory is consistent with compute()\n")

    # ------------------------------------------------------------------
    # 10. Guard: t out of range
    # ------------------------------------------------------------------
    try:
        rp.compute(S=100.0, q=0, t=1.5)
        print("ERROR: should have raised ValueError", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"  ✓ ValueError for t > T: {exc}\n")

    # ------------------------------------------------------------------
    # 11. Guard: non-positive mid-price
    # ------------------------------------------------------------------
    try:
        rp.compute(S=0.0, q=0, t=0.0)
        print("ERROR: should have raised ValueError", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"  ✓ ValueError for S <= 0: {exc}\n")

    print("All checks passed.")