"""
tests/test_strategy_infinite.py

Unit tests for the infinite-horizon (stationary) Avellaneda-Stoikov
extension: strategy/reservation_infinite.py and strategy/spread_infinite.py.

These are plain pytest-style ``test_*`` functions using bare ``assert``
(no pytest-only fixtures/parametrize used), so they can be collected by
pytest OR executed directly via ``python tests/test_strategy_infinite.py``
in environments where pytest is not installed (this sandbox has no network
access and pytest could not be installed here; the guard at the bottom of
this file provides a small manual runner for that situation).

Per the testing checklist in the progress document, this file covers, at
minimum:
  - q = 0
  - positive inventory
  - negative inventory
  - inventory close to the allowed maximum
  - invalid omega / inventory combinations
plus a few extra checks (r_a > r_b, t-independence, spread positivity,
consistency between InfiniteHorizonReservationPrice and
InfiniteHorizonSpreadModel).
"""

from __future__ import annotations

import math

import numpy as np

from strategy.reservation_infinite import InfiniteHorizonReservationPrice
from strategy.spread_infinite import InfiniteHorizonSpreadModel


GAMMA = 0.1
SIGMA = 2.0
Q_MAX = 10
# Same calibration the paper suggests, and the same one MarketMakerAgent
# uses by default when constructed with model="infinite_horizon", q_max=Q_MAX.
OMEGA = InfiniteHorizonReservationPrice.omega_for_q_max(GAMMA, SIGMA, Q_MAX)


# ---------------------------------------------------------------------------
# InfiniteHorizonReservationPrice
# ---------------------------------------------------------------------------

def test_q_zero_is_close_to_mid_price():
    """At q=0 the stationary price is close to (but not required to be
    exactly equal to) the mid-price -- unlike the finite-horizon formula's
    exact symmetry, the exact infinite-horizon r_a/r_b are not perfectly
    symmetric around s even at q=0 (ln(1+x)+ln(1-x) = ln(1-x^2) != 0)."""
    rp = InfiniteHorizonReservationPrice(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    result = rp.compute(S=100.0, q=0, t=0.0)
    assert abs(result.price - 100.0) < 0.01
    assert result.r_a > result.r_b


def test_positive_inventory_lowers_price():
    rp = InfiniteHorizonReservationPrice(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    r_long = rp.compute(S=100.0, q=5, t=0.0)
    r_flat = rp.compute(S=100.0, q=0, t=0.0)
    assert r_long.price < r_flat.price
    assert r_long.r_a > r_long.r_b  # ask still above bid


def test_negative_inventory_raises_price():
    rp = InfiniteHorizonReservationPrice(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    r_short = rp.compute(S=100.0, q=-5, t=0.0)
    r_flat = rp.compute(S=100.0, q=0, t=0.0)
    assert r_short.price > r_flat.price
    assert r_short.r_a > r_short.r_b


def test_inventory_close_to_max_is_valid_but_wide():
    """q = Q_MAX - 1 must remain valid (it's the default hard limit the
    agent derives for this omega), and the reservation spread should widen
    noticeably as q approaches the boundary."""
    rp = InfiniteHorizonReservationPrice(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    near_max = Q_MAX - 1
    result = rp.compute(S=100.0, q=near_max, t=0.0)
    assert math.isfinite(result.price)
    assert result.r_a > result.r_b

    small_q_spread = (
        rp.compute(S=100.0, q=1, t=0.0).r_a
        - rp.compute(S=100.0, q=1, t=0.0).r_b
    )
    near_max_spread = result.r_a - result.r_b
    assert near_max_spread > small_q_spread


def test_inventory_exactly_at_paper_q_max_is_rejected():
    """Numerically verified boundary case: at q == q_max under the paper's
    own suggested omega calibration, the reservation-bid log argument is
    <= 0. This must raise, not silently clip or emit NaN/Inf."""
    rp = InfiniteHorizonReservationPrice(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    try:
        rp.compute(S=100.0, q=Q_MAX, t=0.0)
        assert False, "expected ValueError at q == q_max"
    except ValueError:
        pass


def test_invalid_inventory_far_beyond_domain_is_rejected():
    rp = InfiniteHorizonReservationPrice(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    try:
        rp.compute(S=100.0, q=Q_MAX + 50, t=0.0)
        assert False, "expected ValueError for a wildly out-of-domain q"
    except ValueError:
        pass
    try:
        rp.compute(S=100.0, q=-(Q_MAX + 50), t=0.0)
        assert False, "expected ValueError for a wildly out-of-domain negative q"
    except ValueError:
        pass


def test_invalid_omega_construction_rejected():
    for bad_omega in (0.0, -1.0):
        try:
            InfiniteHorizonReservationPrice(gamma=GAMMA, sigma=SIGMA, omega=bad_omega)
            assert False, f"expected ValueError for omega={bad_omega}"
        except ValueError:
            pass


def test_invalid_gamma_sigma_construction_rejected():
    for bad in (0.0, -1.0):
        try:
            InfiniteHorizonReservationPrice(gamma=bad, sigma=SIGMA, omega=OMEGA)
            assert False
        except ValueError:
            pass
        try:
            InfiniteHorizonReservationPrice(gamma=GAMMA, sigma=bad, omega=OMEGA)
            assert False
        except ValueError:
            pass


def test_does_not_depend_on_t():
    """Core structural requirement: the infinite-horizon price must not
    depend on (T - t) at all -- passing wildly different t values (or None)
    must not change the result."""
    rp = InfiniteHorizonReservationPrice(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    p1 = rp.compute(S=100.0, q=3, t=0.0).price
    p2 = rp.compute(S=100.0, q=3, t=123456.789).price
    p3 = rp.compute(S=100.0, q=3, t=None).price
    assert p1 == p2 == p3


def test_nonpositive_mid_price_rejected():
    rp = InfiniteHorizonReservationPrice(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    try:
        rp.compute(S=0.0, q=0, t=0.0)
        assert False
    except ValueError:
        pass
    try:
        rp.compute(S=-5.0, q=0, t=0.0)
        assert False
    except ValueError:
        pass


def test_omega_for_q_max_matches_paper_formula():
    g, s, qm = 0.1, 2.0, 10
    expected = 0.5 * g ** 2 * s ** 2 * (qm + 1) ** 2
    actual = InfiniteHorizonReservationPrice.omega_for_q_max(g, s, qm)
    assert abs(actual - expected) < 1e-12


def test_vectorised_q_matches_scalar_calls():
    rp = InfiniteHorizonReservationPrice(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    q_grid = np.array([-5, -2, 0, 2, 5])
    batch = rp.compute(S=100.0, q=q_grid, t=0.0)
    for qi, expected_price in zip(q_grid, batch.price):
        scalar_price = rp.compute(S=100.0, q=int(qi), t=0.0).price
        assert abs(scalar_price - expected_price) < 1e-9


# ---------------------------------------------------------------------------
# InfiniteHorizonSpreadModel
# ---------------------------------------------------------------------------

def test_spread_is_positive_at_q_zero():
    sm = InfiniteHorizonSpreadModel(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    result = sm.compute(t=0.0, q=0, S=100.0)
    assert result.spread > 0
    assert abs(result.half_spread - result.spread / 2.0) < 1e-12


def test_spread_widens_with_inventory():
    sm = InfiniteHorizonSpreadModel(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    s_flat = sm.compute(t=0.0, q=0, S=100.0).spread
    s_5 = sm.compute(t=0.0, q=5, S=100.0).spread
    s_neg5 = sm.compute(t=0.0, q=-5, S=100.0).spread
    assert s_5 > s_flat
    assert s_neg5 > s_flat


def test_spread_does_not_depend_on_mid_price():
    sm = InfiniteHorizonSpreadModel(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    s1 = sm.compute(t=0.0, q=3, S=50.0).spread
    s2 = sm.compute(t=0.0, q=3, S=500.0).spread
    assert abs(s1 - s2) < 1e-9


def test_spread_does_not_depend_on_t():
    sm = InfiniteHorizonSpreadModel(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    s1 = sm.compute(t=0.0, q=3, S=100.0).spread
    s2 = sm.compute(t=999.0, q=3, S=100.0).spread
    assert s1 == s2


def test_quote_prices_equal_r_b_r_a():
    sm = InfiniteHorizonSpreadModel(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    result = sm.compute(t=0.0, q=2, S=100.0)
    bid, ask = sm.quote_prices(reservation_price=result.reservation_price, t=0.0, q=2, S=100.0)
    assert bid == result.r_b
    assert ask == result.r_a
    assert ask > bid


def test_spread_propagates_invalid_inventory():
    sm = InfiniteHorizonSpreadModel(gamma=GAMMA, sigma=SIGMA, omega=OMEGA)
    try:
        sm.compute(t=0.0, q=Q_MAX, S=100.0)
        assert False, "expected ValueError propagated from reservation model"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Manual runner (used only when pytest is unavailable, e.g. offline sandbox)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import traceback

    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    sys.exit(1 if failures else 0)
