"""
strategy/inventory.py

InventoryManager for the Avellaneda-Stoikov (2008) market-making framework.

Tracks the agent's share inventory, enforces position limits, and computes
inventory-related risk metrics as described in the paper:
  - reservation price adjustment:  r(s,q,t) = s - q * gamma * sigma^2 * (T - t)
  - inventory risk exposure:       0.5 * gamma * q^2 * sigma^2 * (T - t)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class InventorySnapshot:
    """Immutable record of inventory state at a single point in time."""
    time: float
    quantity: int
    cash: float
    mid_price: float
    risk_exposure: float
    reservation_price: float
    limit_breached: bool


@dataclass
class InventoryLimits:
    """Hard and soft inventory limits."""
    max_long: int = 10          # hard upper limit  (+q_max)
    max_short: int = -10        # hard lower limit  (-q_max)
    soft_long: Optional[int] = None   # warn threshold (long)
    soft_short: Optional[int] = None  # warn threshold (short)

    def __post_init__(self) -> None:
        if self.max_long <= 0:
            raise ValueError("max_long must be positive.")
        if self.max_short >= 0:
            raise ValueError("max_short must be negative.")
        # Default soft limits to 80 % of hard limits
        if self.soft_long is None:
            self.soft_long = max(1, int(self.max_long * 0.8))
        if self.soft_short is None:
            self.soft_short = min(-1, int(self.max_short * 0.8))


# ---------------------------------------------------------------------------
# InventoryManager
# ---------------------------------------------------------------------------

class InventoryManager:
    """
    Manages the market-maker's share inventory throughout a simulation.

    Parameters
    ----------
    gamma : float
        Absolute risk-aversion coefficient (γ in the paper).
    sigma : float
        Volatility of the mid-price process (σ).
    T : float
        Terminal horizon of the trading session.
    limits : InventoryLimits, optional
        Position limits.  Defaults to InventoryLimits() with q_max = 10.
    initial_quantity : int, optional
        Starting inventory (default 0).
    initial_cash : float, optional
        Starting cash position in dollars (default 0.0).

    Attributes
    ----------
    quantity : int
        Current share inventory q_t.
    cash : float
        Current cash position X_t.
    history : list[InventorySnapshot]
        Time-stamped record of every inventory update.
    """

    def __init__(
        self,
        gamma: float,
        sigma: float,
        T: float,
        limits: Optional[InventoryLimits] = None,
        initial_quantity: int = 0,
        initial_cash: float = 0.0,
    ) -> None:
        if gamma <= 0:
            raise ValueError("gamma (risk-aversion) must be positive.")
        if sigma <= 0:
            raise ValueError("sigma (volatility) must be positive.")
        if T <= 0:
            raise ValueError("T (horizon) must be positive.")

        self.gamma = gamma
        self.sigma = sigma
        self.T = T
        self.limits: InventoryLimits = limits if limits is not None else InventoryLimits()

        self.quantity: int = initial_quantity
        self.cash: float = initial_cash

        self.history: list[InventorySnapshot] = []
        self._limit_breach_count: int = 0

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def update(
        self,
        delta_q: int,
        delta_cash: float,
        mid_price: float,
        current_time: float,
    ) -> InventorySnapshot:
        """
        Apply an inventory and cash change resulting from an order fill.

        The sign convention follows the paper:
          - A *buy* fill  (agent buys):   delta_q = +1, delta_cash = -(p^b)
          - A *sell* fill (agent sells):  delta_q = -1, delta_cash = +(p^a)

        Parameters
        ----------
        delta_q : int
            Change in share inventory (+1 for buy fill, -1 for sell fill).
        delta_cash : float
            Corresponding cash change in dollars.
        mid_price : float
            Current mid-price s_t (used for risk calculations).
        current_time : float
            Current simulation time t ∈ [0, T].

        Returns
        -------
        InventorySnapshot
            State after the update.

        Raises
        ------
        ValueError
            If current_time is outside [0, T].
        """
        if not (0.0 <= current_time <= self.T):
            raise ValueError(
                f"current_time {current_time} outside [0, {self.T}]."
            )

        # Apply changes
        self.quantity += delta_q
        self.cash += delta_cash

        # Evaluate limits and compute risk
        breach = self.check_limits(warn=True)
        risk = self.compute_risk(mid_price, current_time)
        r = self.reservation_price(mid_price, current_time)

        snap = InventorySnapshot(
            time=current_time,
            quantity=self.quantity,
            cash=self.cash,
            mid_price=mid_price,
            risk_exposure=risk,
            reservation_price=r,
            limit_breached=breach,
        )
        self.history.append(snap)
        return snap

    def check_limits(self, warn: bool = False) -> bool:
        """
        Check whether the current inventory violates position limits.

        Parameters
        ----------
        warn : bool
            If True, emit logger warnings for soft and hard breaches.

        Returns
        -------
        bool
            True if a *hard* limit is breached, False otherwise.
        """
        q = self.quantity
        lim = self.limits
        hard_breach = False

        if q > lim.max_long:
            hard_breach = True
            self._limit_breach_count += 1
            if warn:
                logger.warning(
                    "HARD limit breached: q=%d > max_long=%d", q, lim.max_long
                )
        elif q < lim.max_short:
            hard_breach = True
            self._limit_breach_count += 1
            if warn:
                logger.warning(
                    "HARD limit breached: q=%d < max_short=%d", q, lim.max_short
                )
        elif warn:
            if q >= lim.soft_long:
                logger.warning(
                    "Soft long limit approached: q=%d >= soft_long=%d",
                    q, lim.soft_long,
                )
            elif q <= lim.soft_short:
                logger.warning(
                    "Soft short limit approached: q=%d <= soft_short=%d",
                    q, lim.soft_short,
                )

        return hard_breach

    def compute_risk(self, mid_price: float, current_time: float) -> float:
        """
        Inventory risk exposure as used in the Avellaneda-Stoikov model.

        Derived from the value function (equation 3 in the paper):

            risk(q, t) = 0.5 * gamma * q^2 * sigma^2 * (T - t)

        This is the penalty the agent incurs for holding inventory q over
        the remaining time horizon (T - t).  It equals zero at terminal
        time T (inventory can be liquidated at mid-price) and grows as
        the horizon lengthens or variance / risk-aversion increase.

        Parameters
        ----------
        mid_price : float
            Current mid-price (not used in the arithmetic model; included
            for the geometric / mean-variance extension).
        current_time : float
            Current simulation time t.

        Returns
        -------
        float
            Non-negative risk exposure value.
        """
        time_remaining = max(0.0, self.T - current_time)
        return 0.5 * self.gamma * (self.quantity ** 2) * (self.sigma ** 2) * time_remaining

    def reservation_price(self, mid_price: float, current_time: float) -> float:
        """
        Compute the agent's reservation (indifference) price.

        From equation (8) in the paper:

            r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

        A long position (q > 0) pushes the reservation price *below* the
        mid-price, reflecting a desire to sell.  A short position (q < 0)
        raises it above, reflecting a desire to buy.

        Parameters
        ----------
        mid_price : float
            Current mid-price s_t.
        current_time : float
            Current simulation time t.

        Returns
        -------
        float
            Reservation price r(s, q, t).
        """
        time_remaining = max(0.0, self.T - current_time)
        return mid_price - self.quantity * self.gamma * (self.sigma ** 2) * time_remaining

    # ------------------------------------------------------------------
    # Convenience / diagnostic helpers
    # ------------------------------------------------------------------

    def is_flat(self) -> bool:
        """Return True if the agent holds no inventory (q == 0)."""
        return self.quantity == 0

    def skew_direction(self) -> str:
        """
        Return a string indicating whether quotes should be skewed.

        Returns 'BUY'  when agent is short and wants to accumulate,
                'SELL' when agent is long and wants to liquidate,
                'FLAT' when inventory is zero.
        """
        if self.quantity > 0:
            return "SELL"
        if self.quantity < 0:
            return "BUY"
        return "FLAT"

    def mark_to_market(self, mid_price: float) -> float:
        """
        Total mark-to-market value of the agent's portfolio.

            MtM = cash + q * mid_price

        Parameters
        ----------
        mid_price : float
            Current mid-price for valuation.

        Returns
        -------
        float
            Portfolio value in dollars.
        """
        return self.cash + self.quantity * mid_price

    def reset(self, initial_quantity: int = 0, initial_cash: float = 0.0) -> None:
        """Reset inventory to starting state (useful between simulation runs)."""
        self.quantity = initial_quantity
        self.cash = initial_cash
        self.history.clear()
        self._limit_breach_count = 0

    @property
    def breach_count(self) -> int:
        """Total number of hard limit breaches observed during the session."""
        return self._limit_breach_count

    def summary(self, mid_price: float, current_time: float) -> dict:
        """
        Return a dictionary of key inventory metrics at the current instant.

        Parameters
        ----------
        mid_price : float
        current_time : float

        Returns
        -------
        dict with keys:
            quantity, cash, mark_to_market, reservation_price,
            risk_exposure, skew_direction, limit_breached, breach_count
        """
        return {
            "quantity": self.quantity,
            "cash": round(self.cash, 6),
            "mark_to_market": round(self.mark_to_market(mid_price), 6),
            "reservation_price": round(self.reservation_price(mid_price, current_time), 6),
            "risk_exposure": round(self.compute_risk(mid_price, current_time), 6),
            "skew_direction": self.skew_direction(),
            "limit_breached": self.check_limits(warn=False),
            "breach_count": self._limit_breach_count,
        }

    def __repr__(self) -> str:
        return (
            f"InventoryManager("
            f"q={self.quantity}, cash={self.cash:.4f}, "
            f"gamma={self.gamma}, sigma={self.sigma}, T={self.T})"
        )