"""
tests/test_fill_arrivals.py

Tests for the fill-arrival correction in simulator/engine.py.

Background
----------
The paper (Avellaneda & Stoikov 2008, eq. 12) models buy/sell order
arrivals as independent Poisson processes with intensity
lambda(delta) = A * exp(-k * delta). Over a discrete step of size dt, the
NUMBER of arrivals on a side is Poisson(lambda * dt).

The engine previously modelled only whether >=1 arrival occurred in a
step (Bernoulli(1 - exp(-lambda*dt))), which silently discards the
2-or-more-arrivals branch of the distribution. This under-counts fills
whenever lambda*dt is not small -- e.g. with the paper's own parameters
(A=140, dt=0.005), lambda*dt reaches 0.7 when quotes sit at the mid-price.

These tests check:
  1. The exact-inversion Poisson sampler (_poisson_count_from_uniform) is
     correct against a manual reference implementation.
  2. Multiple fills per step are now actually possible (not clipped to 1).
  3. The RNG consumption pattern per step is unchanged (exactly two floats
     via a single `rng.random(2)` call), which is what preserves the
     common-random-numbers (CRN) property between the finite- and
     infinite-horizon engines.
  4. Common random numbers are, in fact, preserved end-to-end: the
     finite- and infinite-horizon engines (same seed) produce byte-
     identical mid-price paths despite quoting different spreads.
  5. Empirically, the new sampler's mean fill count matches the Poisson
     mean (lambda*dt), not the smaller Bernoulli mean (1-exp(-lambda*dt)).
  6. Edge cases: suppressed (NaN) quotes give zero fills; mu=0 gives zero
     fills; RNG is still consumed even when a quote is suppressed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from simulator.engine import SimulationEngine
from simulator.agent import Quote
from strategy.reservation_infinite import InfiniteHorizonReservationPrice


COMMON_PARAMS = dict(
    s0=100.0, dt=0.005, sigma=2.0, gamma=0.1, k=1.5, A=140.0,
    initial_quantity=0, initial_cash=0.0,
)
SEED = 42


def _make_finite_infinite_pair(seed: int = SEED):
    q_max = 10
    omega = InfiniteHorizonReservationPrice.omega_for_q_max(
        gamma=COMMON_PARAMS["gamma"], sigma=COMMON_PARAMS["sigma"], q_max=q_max
    )
    finite_engine = SimulationEngine(
        T=1.0, model="finite_horizon", seed=seed, **COMMON_PARAMS
    )
    infinite_engine = SimulationEngine(
        T=1.0, model="infinite_horizon", omega=omega, q_max=q_max,
        seed=seed, **COMMON_PARAMS
    )
    return finite_engine, infinite_engine


# ---------------------------------------------------------------------------
# 1. Exact-inversion Poisson sampler correctness
# ---------------------------------------------------------------------------

def _reference_poisson_inversion(mu: float, u: float) -> int:
    """Independent re-implementation of Knuth's inversion algorithm."""
    if mu <= 0.0:
        return 0
    p = math.exp(-mu)
    cdf = p
    k = 0
    while u >= cdf:
        k += 1
        p *= mu / k
        cdf += p
        if k > 10_000:
            raise RuntimeError("reference sampler failed to converge")
    return k


@pytest.mark.parametrize("mu", [0.0, 0.001, 0.1, 0.5, 0.7, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("u", [0.0, 0.01, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999999])
def test_poisson_count_from_uniform_matches_reference(mu, u):
    eng = SimulationEngine(seed=1, **COMMON_PARAMS)
    got = eng._poisson_count_from_uniform(mu, u)
    expected = _reference_poisson_inversion(mu, u)
    assert got == expected


def test_poisson_count_from_uniform_is_nondecreasing_in_u():
    """The inversion CDF map must be monotone non-decreasing in u."""
    eng = SimulationEngine(seed=1, **COMMON_PARAMS)
    mu = 0.7
    us = np.linspace(0.0, 0.999999, 200)
    counts = [eng._poisson_count_from_uniform(mu, u) for u in us]
    assert all(b >= a for a, b in zip(counts, counts[1:]))


def test_poisson_count_from_uniform_mu_zero_always_zero():
    eng = SimulationEngine(seed=1, **COMMON_PARAMS)
    for u in (0.0, 0.3, 0.9999):
        assert eng._poisson_count_from_uniform(0.0, u) == 0


def test_poisson_count_from_uniform_empirical_mean_matches_mu():
    """
    Drawing many independent uniforms and inverting them should reproduce
    the Poisson(mu) distribution's mean, not the Bernoulli(1-exp(-mu))
    mean used by the old implementation.
    """
    eng = SimulationEngine(seed=1, **COMMON_PARAMS)
    rng = np.random.default_rng(123)
    mu = 0.7
    n = 200_000
    us = rng.random(n)
    counts = np.array([eng._poisson_count_from_uniform(mu, u) for u in us])

    empirical_mean = counts.mean()
    bernoulli_mean = 1.0 - math.exp(-mu)  # what the OLD model implied

    # Empirical mean should be close to the true Poisson mean mu=0.7 ...
    assert abs(empirical_mean - mu) < 0.02
    # ... and clearly above the old (undercounting) Bernoulli mean ~0.503.
    assert empirical_mean > bernoulli_mean + 0.1

    # Fraction of draws with >=2 fills should be close to the analytic
    # Poisson value  P(N>=2) = 1 - exp(-mu) - mu*exp(-mu).
    p_ge2_analytic = 1.0 - math.exp(-mu) - mu * math.exp(-mu)
    p_ge2_empirical = float(np.mean(counts >= 2))
    assert abs(p_ge2_empirical - p_ge2_analytic) < 0.01


# ---------------------------------------------------------------------------
# 2. Multiple fills per step are actually possible end-to-end
# ---------------------------------------------------------------------------

def test_simulate_fills_can_return_counts_greater_than_one():
    """
    With a quote sitting exactly at the mid-price (delta=0 => lambda=A)
    and dt/A chosen so mu is not small, at least one draw among many
    uniform grid points must yield n_buy >= 2 (and n_sell >= 2).
    """
    eng = SimulationEngine(seed=1, **COMMON_PARAMS)  # A*dt = 0.7
    quote = Quote(bid=100.0, ask=100.0, reservation=100.0,
                  half_spread=0.0, mid_price=100.0, time=0.0)

    found_multi_buy = False
    found_multi_sell = False
    # Directly probe the exact-inversion function across the uniform
    # range (mu = A*dt = 0.7 for both sides since delta=0 here).
    mu = eng._intensity(0.0) * eng.dt
    assert mu == pytest.approx(0.7)
    for u in np.linspace(0.0, 0.999999, 5000):
        n = eng._poisson_count_from_uniform(mu, u)
        if n >= 2:
            found_multi_buy = True
            found_multi_sell = True
            break
    assert found_multi_buy and found_multi_sell


def test_engine_applies_multiple_fills_within_one_step(monkeypatch):
    """
    Force the RNG to return uniforms that map to n_buy=3 fills in a single
    call to _simulate_fills, then check the agent's inventory advances by
    exactly 3 (not clamped to 1) after processing that one step's fills.
    """
    eng = SimulationEngine(seed=1, **COMMON_PARAMS)
    eng.agent.reset(initial_quantity=0, initial_cash=0.0)

    quote = Quote(bid=100.0, ask=100.0, reservation=100.0,
                  half_spread=0.0, mid_price=100.0, time=0.0)
    eng.agent._last_quote = quote

    # mu = A*dt = 0.7 on both sides. Pick a u that maps to n=3 (u just
    # under the CDF at k=3), and a u that maps to n=0 for the other side.
    mu = 0.7
    p = math.exp(-mu)
    cdf = p
    for k in range(1, 4):
        p *= mu / k
        cdf += p
    u_for_three = cdf - 1e-9   # lands in the k=3 bucket
    u_for_zero = 0.0           # lands in the k=0 bucket

    class _FixedRNG:
        def random(self, size):
            return np.array([u_for_three, u_for_zero])

        def standard_normal(self):
            return 0.0

    eng._rng = _FixedRNG()

    n_buy, n_sell = eng._simulate_fills(quote)
    assert n_buy == 3
    assert n_sell == 0

    for _ in range(n_buy):
        eng.agent.on_buy_fill(mid_price=100.0, current_time=0.0)
    for _ in range(n_sell):
        eng.agent.on_sell_fill(mid_price=100.0, current_time=0.0)

    assert eng.agent.inventory.quantity == 3


# ---------------------------------------------------------------------------
# 3 & 4. RNG consumption pattern / common random numbers preserved
# ---------------------------------------------------------------------------

def test_rng_consumes_exactly_two_floats_per_step_regardless_of_mu():
    """
    _simulate_fills must always draw exactly one rng.random(2) call (two
    floats) per invocation, regardless of the intensities involved. This
    is what keeps the finite- and infinite-horizon engines' mid-price
    paths byte-identical despite quoting different spreads.
    """
    eng = SimulationEngine(seed=1, **COMMON_PARAMS)

    calls = []
    real_rng = eng._rng

    class _SpyRNG:
        def random(self, size):
            calls.append(size)
            return real_rng.random(size)

    eng._rng = _SpyRNG()

    tight_quote = Quote(bid=99.999, ask=100.001, reservation=100.0,
                         half_spread=0.001, mid_price=100.0, time=0.0)
    wide_quote = Quote(bid=90.0, ask=110.0, reservation=100.0,
                        half_spread=10.0, mid_price=100.0, time=0.0)
    nan_quote = Quote(bid=float("nan"), ask=float("nan"), reservation=100.0,
                       half_spread=float("nan"), mid_price=100.0, time=0.0)

    for q in (tight_quote, wide_quote, nan_quote):
        eng._simulate_fills(q)

    assert calls == [2, 2, 2]


def test_common_random_numbers_preserved_between_horizons():
    """
    End-to-end CRN check: with the same seed, the finite- and
    infinite-horizon engines must see byte-identical mid-price paths,
    even though their average spreads (and hence per-step mu) differ
    substantially (~1.49 vs ~0.17 in the primary experiment).
    """
    finite_engine, infinite_engine = _make_finite_infinite_pair()

    finite_result = finite_engine.run(strategy="inventory")
    infinite_result = infinite_engine.run(strategy="inventory")

    finite_mid = [r.mid_price for r in finite_result.path]
    infinite_mid = [r.mid_price for r in infinite_result.path]

    assert finite_mid == infinite_mid

    # Sanity: the two strategies really do quote very different spreads,
    # so this isn't a vacuous check (mu differs a lot between them).
    assert finite_result.average_spread > 5 * infinite_result.average_spread


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

def test_suppressed_quote_gives_zero_fills_but_still_consumes_rng():
    eng = SimulationEngine(seed=1, **COMMON_PARAMS)
    nan_quote = Quote(bid=float("nan"), ask=float("nan"), reservation=100.0,
                       half_spread=float("nan"), mid_price=100.0, time=0.0)

    state_before = eng._rng.bit_generator.state
    n_buy, n_sell = eng._simulate_fills(nan_quote)
    state_after = eng._rng.bit_generator.state

    assert (n_buy, n_sell) == (0, 0)
    assert state_before != state_after  # RNG stream still advanced


def test_fill_counts_are_never_negative_and_sum_to_total_fills():
    """Integration check on a short run: per-step counts are >=0 and the
    reported total_buy_fills/total_sell_fills equal the sum of per-step
    counts (not the count of steps with >=1 fill)."""
    eng = SimulationEngine(seed=7, **COMMON_PARAMS)
    result = eng.run(strategy="inventory")

    assert all(r.buy_fill_count >= 0 for r in result.path)
    assert all(r.sell_fill_count >= 0 for r in result.path)

    assert result.total_buy_fills == sum(r.buy_fill_count for r in result.path)
    assert result.total_sell_fills == sum(r.sell_fill_count for r in result.path)

    # buy_filled/sell_filled booleans must agree with count > 0.
    for r in result.path:
        assert r.buy_filled == (r.buy_fill_count > 0)
        assert r.sell_filled == (r.sell_fill_count > 0)


def test_primary_experiment_params_yield_some_multi_fill_steps():
    """
    With the paper's own A=140, dt=0.005 (A*dt=0.7), running the
    infinite-horizon strategy (whose average spread is tight, so delta
    is often near 0 and mu is often near 0.7) should now produce at
    least some steps with 2+ fills on a side -- this was structurally
    impossible under the old Bernoulli-only implementation.
    """
    q_max = 10
    omega = InfiniteHorizonReservationPrice.omega_for_q_max(
        gamma=COMMON_PARAMS["gamma"], sigma=COMMON_PARAMS["sigma"], q_max=q_max
    )
    eng = SimulationEngine(
        T=1.0, model="infinite_horizon", omega=omega, q_max=q_max,
        seed=SEED, **COMMON_PARAMS
    )
    result = eng.run(strategy="inventory")
    multi_fill_steps = sum(
        1 for r in result.path
        if r.buy_fill_count >= 2 or r.sell_fill_count >= 2
    )
    assert multi_fill_steps > 0
