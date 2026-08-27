"""
strategy/spread_infinite.py
============================
Stationary, infinite-horizon bid-ask spread for the Avellaneda-Stoikov
(2008) market-making framework.

The paper's Section 2.3 gives closed-form stationary reservation prices
r_a(s,q) and r_b(s,q) but does **not** re-derive an infinite-horizon
analogue of the finite-horizon spread formula (Eq. 30) -- that derivation
is specific to the finite-horizon HJB asymptotic expansion in Section 3.2
and is never redone for the omega-discounted objective of Section 2.3.

Per the design decision recorded in the progress document (Section B.4,
"Option 1 / preferred"), this module takes the paper-faithful choice that
requires no invented formula: quote directly at the paper's own stationary
ask/bid prices, i.e.

    spread(s, q) = r_a(s, q) - r_b(s, q)
    half_spread  = spread / 2

    p^b = r_b(s, q)   =   r(s, q) - half_spread
    p^a = r_a(s, q)   =   r(s, q) + half_spread

where r_a, r_b, and r = (r_a + r_b) / 2 come from
``strategy.reservation_infinite.InfiniteHorizonReservationPrice``.

This spread is, in general, inventory-dependent (unlike the finite-horizon
spread of Eq. 30, which is inventory-independent) -- the mid-price terms
cancel in r_a - r_b, but the q-dependence inside the logarithms does not.
It does not depend on (T - t) anywhere, consistent with the stationary
nature of the infinite-horizon objective.

Reference
---------
Avellaneda, M. & Stoikov, S. (2008).
"High-frequency trading in a limit order book."
Quantitative Finance, 8(3), 217-224. Section 2.3.
DOI: 10.1080/14697680701381228
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from strategy.reservation_infinite import InfiniteHorizonReservationPrice


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InfiniteHorizonSpreadResult:
    """
    Output of a single call to ``InfiniteHorizonSpreadModel.compute()``.

    Field names mirror ``SpreadResult`` (spread.py) where a natural
    equivalent exists, so ``MarketMakerAgent.quote()`` can read
    ``.spread`` / ``.half_spread`` without caring which model produced
    the result.

    Attributes
    ----------
    spread : float or NDArray[np.float64]
        Total stationary bid-ask spread r_a(s,q) - r_b(s,q) >= 0.
    half_spread : float or NDArray[np.float64]
        spread / 2.
    r_a : float or NDArray[np.float64]
        Stationary reservation ASK price (also the optimal ask quote).
    r_b : float or NDArray[np.float64]
        Stationary reservation BID price (also the optimal bid quote).
    reservation_price : float or NDArray[np.float64]
        (r_a + r_b) / 2, for reference/diagnostics.
    omega : float
        The discount-rate parameter used for this computation.
    """

    spread: float | NDArray[np.float64]
    half_spread: float | NDArray[np.float64]
    r_a: float | NDArray[np.float64]
    r_b: float | NDArray[np.float64]
    reservation_price: float | NDArray[np.float64]
    omega: float

    def __str__(self) -> str:
        return (
            f"InfiniteHorizonSpreadResult("
            f"spread={self.spread}, half_spread={self.half_spread}, "
            f"omega={self.omega:.6f})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class InfiniteHorizonSpreadModel:
    """
    Computes the stationary Avellaneda-Stoikov bid-ask spread as
    r_a(s,q) - r_b(s,q) (see module docstring for the rationale).

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient. Must be > 0.
    sigma : float
        Mid-price volatility. Must be > 0.
    omega : float
        Exponential discount rate of the infinite-horizon objective.
        Must be > 0.

    Notes
    -----
    Unlike ``SpreadModel`` (finite-horizon), this class has no ``T`` or
    ``k`` parameter: it does not use the arrival-rate/order-book
    calibration term at all, since it quotes directly at the paper's own
    stationary reservation ask/bid prices rather than re-deriving a
    spread from the HJB first-order condition (which the paper never does
    for the infinite-horizon case -- see module docstring).
    """

    def __init__(self, gamma: float, sigma: float, omega: float) -> None:
        # InfiniteHorizonReservationPrice performs the gamma/sigma/omega
        # validation; reuse it rather than duplicating the checks.
        self._reservation = InfiniteHorizonReservationPrice(
            gamma=gamma, sigma=sigma, omega=omega
        )
        self.gamma: float = self._reservation.gamma
        self.sigma: float = self._reservation.sigma
        self.omega: float = self._reservation.omega

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def compute(
        self,
        t: Optional[float] = None,
        q: ArrayLike = 0.0,
        S: ArrayLike = 1.0,
    ) -> InfiniteHorizonSpreadResult:
        """
        Compute the total stationary bid-ask spread.

        Parameters
        ----------
        t : float, optional
            Accepted and **ignored**. Present only so this class can be
            used interchangeably with ``SpreadModel.compute(t)`` inside
            ``MarketMakerAgent``. The stationary spread does not depend
            on time.
        q : array-like of float, default 0.0
            Current inventory. Required for a meaningful result (the
            infinite-horizon spread is inventory-dependent) -- the
            default of 0.0 is provided only so the method has a sensible
            standalone default, not because q is optional in practice.
            Must lie in the valid domain for this omega (see
            ``InfiniteHorizonReservationPrice.validate_inventory``).
        S : array-like of float, default 1.0
            Current mid-price. Mathematically irrelevant to the spread
            magnitude (it cancels exactly in r_a - r_b) but required by
            ``InfiniteHorizonReservationPrice.compute`` to be > 0; accept
            it here purely for interface parity with
            ``MarketMakerAgent.quote()``, which always passes the real
            mid-price.

        Returns
        -------
        InfiniteHorizonSpreadResult

        Raises
        ------
        ValueError
            If ``q`` is outside the valid domain for this omega, or if
            ``S <= 0``.
        """
        res = self._reservation.compute(S=S, q=q, t=t)

        r_a = res.r_a
        r_b = res.r_b
        spread = r_a - r_b
        half_spread = spread / 2.0

        return InfiniteHorizonSpreadResult(
            spread=spread,
            half_spread=half_spread,
            r_a=r_a,
            r_b=r_b,
            reservation_price=res.price,
            omega=self.omega,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def quote_prices(
        self,
        reservation_price: float | NDArray[np.float64],
        t: Optional[float] = None,
        q: ArrayLike = 0.0,
        S: ArrayLike = 1.0,
    ) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
        """
        Compute bid and ask quotes.

        Provided for interface parity with ``SpreadModel.quote_prices``.
        Note that, unlike the finite-horizon model, the bid/ask returned
        here are simply r_b(s,q) and r_a(s,q) directly -- the
        ``reservation_price`` argument is accepted for signature
        compatibility but is not used to re-derive the quotes (the
        half-spread is already computed from, and centred on, the
        paper's own r_a/r_b).

        Parameters
        ----------
        reservation_price : float or NDArray[np.float64]
            Unused; accepted for interface parity with
            ``SpreadModel.quote_prices``.
        t, q, S : see ``compute()``.

        Returns
        -------
        bid, ask : float or NDArray[np.float64]
        """
        result = self.compute(t=t, q=q, S=S)
        return result.r_b, result.r_a

    def __repr__(self) -> str:
        return (
            f"InfiniteHorizonSpreadModel("
            f"gamma={self.gamma}, sigma={self.sigma}, omega={self.omega})"
        )
