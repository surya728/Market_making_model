"""
strategy/reservation_infinite.py
=================================
Stationary, infinite-horizon reservation (indifference) price for the
Avellaneda-Stoikov (2008) market-making framework.

This is the counterpart of ``strategy/reservation.py`` (the finite-horizon
``ReservationPrice`` class) for the infinite-horizon objective described in
Section 2.3 of the paper:

    v(x, s, q) = E[ int_0^inf -exp(-omega*t) * exp(-gamma*(x + q*S_t)) dt ]

Equations implemented (verified against the paper PDF, Section 2.3 --
notation matches exactly, no discrepancy found):

    r_a(s, q) = s + (1/gamma) * ln(
                    1 + (1 - 2q) * gamma^2 * sigma^2
                        / (2*omega - gamma^2 * q^2 * sigma^2)
                )

    r_b(s, q) = s + (1/gamma) * ln(
                    1 + (-1 - 2q) * gamma^2 * sigma^2
                        / (2*omega - gamma^2 * q^2 * sigma^2)
                )

    valid for omega > (1/2) * gamma^2 * sigma^2 * q^2

    r(s, q) = (r_a(s, q) + r_b(s, q)) / 2

Crucially -- and unlike the finite-horizon reservation price -- these
equations do **not** depend on (T - t) anywhere. The stationarity comes
from the exponential discount rate ``omega`` instead of a terminal time.
``compute()`` accepts a ``t`` keyword purely so that this class is a
drop-in, duck-typed replacement for ``ReservationPrice`` inside
``MarketMakerAgent`` (see simulator/agent.py); the value of ``t`` is
never read.

The paper suggests calibrating ``omega`` from a desired maximum inventory
``q_max`` via

    omega = (1/2) * gamma^2 * sigma^2 * (q_max + 1)^2

which is exposed here as ``InfiniteHorizonReservationPrice.omega_for_q_max``.

Reference
---------
Avellaneda, M. & Stoikov, S. (2008).
"High-frequency trading in a limit order book."
Quantitative Finance, 8(3), 217-224. Section 2.3.
DOI: 10.1080/14697680701381228
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Small positive tolerance used when checking that a denominator / log
# argument is strictly on the valid side of zero. Kept as a module-level
# constant so tests and callers can reference the exact threshold used.
_DOMAIN_EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InfiniteHorizonReservationPriceResult:
    """
    Output of a single call to ``InfiniteHorizonReservationPrice.compute()``.

    Mirrors the field names of ``ReservationPriceResult`` (reservation.py)
    where they have a natural equivalent, so downstream code that only
    reads ``.price`` works unchanged regardless of which model is active.

    Attributes
    ----------
    price : float or NDArray[np.float64]
        Stationary reservation price r(s, q) = (r_a + r_b) / 2.
    r_a : float or NDArray[np.float64]
        Stationary reservation ASK price.
    r_b : float or NDArray[np.float64]
        Stationary reservation BID price.
    inventory_adjustment : float or NDArray[np.float64]
        price - mid_price (kept for parity with the finite-horizon result).
    mid_price : float or NDArray[np.float64]
        The raw mid-price S passed into the computation.
    omega : float
        The discount-rate parameter used for this computation.
    time_remaining : None
        Always ``None``. Present only so callers that inspect this field
        (as they may for the finite-horizon result) fail loudly/obviously
        rather than silently reading a meaningless number -- the
        infinite-horizon price has no notion of "time remaining".
    """

    price: float | NDArray[np.float64]
    r_a: float | NDArray[np.float64]
    r_b: float | NDArray[np.float64]
    inventory_adjustment: float | NDArray[np.float64]
    mid_price: float | NDArray[np.float64]
    omega: float
    time_remaining: None = None

    def __str__(self) -> str:
        return (
            f"InfiniteHorizonReservationPriceResult("
            f"price={self.price}, r_a={self.r_a}, r_b={self.r_b}, "
            f"omega={self.omega:.6f})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class InfiniteHorizonReservationPrice:
    """
    Computes the Avellaneda-Stoikov *stationary* (infinite-horizon)
    reservation prices r_a, r_b and their average r.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient of the CARA utility function. Must be > 0.
    sigma : float
        Volatility of the arithmetic Brownian mid-price process. Must be > 0.
    omega : float
        Exponential discount rate of the infinite-horizon objective. Must
        be > 0. Interpretable as an upper bound on tolerable inventory: the
        paper's suggested calibration is
        ``omega = 0.5 * gamma^2 * sigma^2 * (q_max + 1)^2``.

    Notes
    -----
    This class intentionally has **no** ``T`` parameter: the whole point of
    the infinite-horizon formulation is that the reservation price does not
    depend on a terminal time or on how much time remains.
    """

    def __init__(self, gamma: float, sigma: float, omega: float) -> None:
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if omega <= 0:
            raise ValueError(f"omega must be > 0, got {omega}")

        self.gamma: float = float(gamma)
        self.sigma: float = float(sigma)
        self.omega: float = float(omega)

        # gamma^2 * sigma^2 -- appears throughout the formulas.
        self._X: float = self.gamma ** 2 * self.sigma ** 2

    # ------------------------------------------------------------------
    # Domain validation
    # ------------------------------------------------------------------

    def domain_gap(self, q: ArrayLike) -> NDArray[np.float64]:
        """
        Compute ``2*omega - gamma^2 * q^2 * sigma^2`` for the given q.

        This is exactly the denominator the paper requires to be positive
        (``omega > 0.5*gamma^2*sigma^2*q^2``). Positive values of this
        quantity are necessary (but, as ``validate_inventory`` checks, not
        always sufficient on their own) for r_a/r_b to be well defined.
        """
        q_arr = np.asarray(q, dtype=np.float64)
        return 2.0 * self.omega - self._X * q_arr ** 2

    def validate_inventory(self, q: ArrayLike) -> NDArray[np.float64]:
        """
        Validate that ``q`` lies in the domain where r_a and r_b are
        mathematically defined for this (gamma, sigma, omega), and return
        the domain gap ``2*omega - gamma^2*sigma^2*q^2``.

        Two conditions are checked, both required for the formulas to
        produce a finite, real result:

        1. The shared denominator must be strictly positive
           (the condition stated explicitly in the paper):

               2*omega - gamma^2*sigma^2*q^2 > 0

        2. The argument of each logarithm must be strictly positive
           (otherwise ``log`` is undefined). This is a direct consequence
           of condition 1 combined with the numerator terms and is checked
           explicitly rather than assumed, since condition 1 alone does not
           guarantee it near the edge of the admissible inventory range.

        Parameters
        ----------
        q : array-like of float
            Inventory value(s) to validate.

        Returns
        -------
        NDArray[np.float64]
            The domain gap ``2*omega - gamma^2*sigma^2*q^2`` (same shape as
            the broadcast of ``q``).

        Raises
        ------
        ValueError
            If any element of ``q`` violates condition 1 or condition 2.
            No clipping or silent correction is performed.
        """
        q_arr = np.asarray(q, dtype=np.float64)
        D = self.domain_gap(q_arr)

        if np.any(D <= _DOMAIN_EPS):
            bad_q = q_arr[D <= _DOMAIN_EPS]
            raise ValueError(
                "Invalid inventory for infinite-horizon model: "
                f"2*omega - gamma^2*sigma^2*q^2 must be > 0, but is <= 0 "
                f"for q in {np.unique(bad_q).tolist()} "
                f"(gamma={self.gamma}, sigma={self.sigma}, omega={self.omega}). "
                "Reduce |q| (tighten inventory limits) or increase omega."
            )

        # Log-argument positivity (see docstring condition 2). Computed
        # against the *validated* D, so no division-by-zero risk here.
        arg_a = 1.0 + (1.0 - 2.0 * q_arr) * self._X / D
        arg_b = 1.0 + (-1.0 - 2.0 * q_arr) * self._X / D
        if np.any(arg_a <= _DOMAIN_EPS) or np.any(arg_b <= _DOMAIN_EPS):
            bad_q = q_arr[(arg_a <= _DOMAIN_EPS) | (arg_b <= _DOMAIN_EPS)]
            raise ValueError(
                "Invalid inventory for infinite-horizon model: the "
                "logarithm argument in r_a or r_b is <= 0 for q in "
                f"{np.unique(bad_q).tolist()} "
                f"(gamma={self.gamma}, sigma={self.sigma}, omega={self.omega}). "
                "Reduce |q| (tighten inventory limits) or increase omega."
            )

        return D

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def compute(
        self,
        S: ArrayLike,
        q: ArrayLike,
        t: Optional[float] = None,
    ) -> InfiniteHorizonReservationPriceResult:
        """
        Compute the stationary reservation prices r_a, r_b and their
        average r(s, q).

        Parameters
        ----------
        S : array-like of float
            Current market mid-price. Must be > 0.
        q : array-like of float
            Current inventory in shares. Must satisfy
            ``omega > 0.5*gamma^2*sigma^2*q^2`` (checked via
            ``validate_inventory``); otherwise a ``ValueError`` is raised.
        t : float, optional
            Accepted and **ignored**. Present only so this class can be
            used interchangeably with ``ReservationPrice.compute(S, q, t)``
            inside ``MarketMakerAgent``. The infinite-horizon reservation
            price does not depend on time.

        Returns
        -------
        InfiniteHorizonReservationPriceResult

        Raises
        ------
        ValueError
            If ``S <= 0`` anywhere, or if ``q`` is outside the valid domain
            for this ``omega`` (see ``validate_inventory``).
        """
        S_arr = np.asarray(S, dtype=np.float64)
        q_arr = np.asarray(q, dtype=np.float64)

        if np.any(S_arr <= 0):
            raise ValueError(
                f"S must be > 0 everywhere; got min = {float(S_arr.min()):.6f}"
            )

        D = self.validate_inventory(q_arr)

        arg_a = 1.0 + (1.0 - 2.0 * q_arr) * self._X / D
        arg_b = 1.0 + (-1.0 - 2.0 * q_arr) * self._X / D

        r_a = S_arr + (1.0 / self.gamma) * np.log(arg_a)
        r_b = S_arr + (1.0 / self.gamma) * np.log(arg_b)
        price = (r_a + r_b) / 2.0
        adjustment = price - S_arr

        if price.ndim == 0:
            return InfiniteHorizonReservationPriceResult(
                price=float(price),
                r_a=float(r_a),
                r_b=float(r_b),
                inventory_adjustment=float(adjustment),
                mid_price=float(S_arr),
                omega=self.omega,
            )

        return InfiniteHorizonReservationPriceResult(
            price=price,
            r_a=r_a,
            r_b=r_b,
            inventory_adjustment=adjustment,
            mid_price=S_arr,
            omega=self.omega,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def max_valid_inventory(self) -> int:
        """
        Advisory (not strictly enforced) upper bound on |q| implied by the
        paper's domain condition ``omega > 0.5*gamma^2*sigma^2*q^2`` alone:

            q_bound = floor( sqrt(2*omega) / (gamma*sigma) )

        This is a *necessary* bound (condition 1 in ``validate_inventory``)
        but, as noted there, not always *sufficient* -- the log-argument
        check can bind slightly tighter near the edge. Use
        ``validate_inventory`` for authoritative validation; use this only
        as a quick, cheap estimate (e.g. for choosing default limits).

        Returns
        -------
        int
        """
        raw = math.sqrt(2.0 * self.omega) / (self.gamma * self.sigma)
        # Subtract a small epsilon before flooring to stay strictly inside
        # the open domain rather than landing exactly on the boundary.
        return int(math.floor(raw - 1e-9))

    @staticmethod
    def omega_for_q_max(gamma: float, sigma: float, q_max: int) -> float:
        """
        Paper's suggested omega calibration for a desired maximum inventory:

            omega = (1/2) * gamma^2 * sigma^2 * (q_max + 1)^2

        Parameters
        ----------
        gamma : float
        sigma : float
        q_max : int
            Desired (positive) maximum absolute inventory.

        Returns
        -------
        float
            The corresponding omega.
        """
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if q_max <= 0:
            raise ValueError(f"q_max must be > 0, got {q_max}")
        return 0.5 * (gamma ** 2) * (sigma ** 2) * (q_max + 1) ** 2

    def __repr__(self) -> str:
        return (
            f"InfiniteHorizonReservationPrice("
            f"gamma={self.gamma}, sigma={self.sigma}, omega={self.omega})"
        )
