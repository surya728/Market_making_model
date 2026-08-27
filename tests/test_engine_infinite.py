"""
tests/test_engine_infinite.py

Integration tests for wiring the infinite-horizon strategy into the ACTUAL
simulation pipeline (SimulationEngine -> MarketMakerAgent, and the
SimulationEngine.summarise() extension used by main.py's finite-vs-infinite
experiment section).

These are pytest-collectable ``test_*`` functions using plain ``assert``;
run via ``pytest tests/test_engine_infinite.py -v`` or directly with
``python tests/test_engine_infinite.py`` (a manual runner is included at
the bottom, matching tests/test_strategy_infinite.py's convention, for
environments without pytest/network access).

Scope: this file is about the *pipeline integration* added on top of the
Phase C strategy-level work (already covered by
tests/test_strategy_infinite.py). It does not re-test the reservation/
spread math itself.
"""

from __future__ import annotations

import math

import numpy as np

from simulator.engine import SimulationEngine
from strategy.reservation_infinite import InfiniteHorizonReservationPrice


COMMON_PARAMS = dict(
    s0=100.0, dt=0.005, sigma=2.0, gamma=0.1, k=1.5, A=140.0,
    initial_quantity=0, initial_cash=0.0,
)
SEED = 42
N_RUNS = 5  # kept small/fast for unit-test purposes


def _build_engines():
    q_max = 10
    omega = InfiniteHorizonReservationPrice.omega_for_q_max(
        gamma=COMMON_PARAMS["gamma"], sigma=COMMON_PARAMS["sigma"], q_max=q_max
    )
    finite_engine = SimulationEngine(
        T=1.0, model="finite_horizon", seed=SEED, **COMMON_PARAMS
    )
    infinite_engine = SimulationEngine(
        T=1.0, model="infinite_horizon", omega=omega, q_max=q_max,
        seed=SEED, **COMMON_PARAMS
    )
    return finite_engine, infinite_engine


# ---------------------------------------------------------------------------
# Model selection through the actual pipeline
# ---------------------------------------------------------------------------

def test_engine_defaults_to_finite_horizon():
    eng = SimulationEngine(seed=1)
    assert eng.model == "finite_horizon"
    assert eng.agent.model == "finite_horizon"
    assert eng.agent.omega is None


def test_engine_selects_infinite_horizon():
    finite_engine, infinite_engine = _build_engines()
    assert finite_engine.agent.model == "finite_horizon"
    assert infinite_engine.agent.model == "infinite_horizon"
    assert infinite_engine.agent.omega is not None
    assert infinite_engine.agent.omega > 0


def test_both_models_run_through_the_same_engine_step_logic():
    """_step()/run() themselves are model-agnostic -- both should complete
    a full run without special-casing anywhere in engine.py."""
    finite_engine, infinite_engine = _build_engines()
    finite_result = finite_engine.run(strategy="inventory")
    infinite_result = infinite_engine.run(strategy="inventory")
    assert len(finite_result.path) == finite_engine.n_steps
    assert len(infinite_result.path) == infinite_engine.n_steps
    assert finite_result.strategy == "inventory"
    assert infinite_result.strategy == "inventory"


def test_symmetric_benchmark_still_works_for_both_models():
    """The 'inventory'/'symmetric' seam is orthogonal to horizon model --
    both must still work under model='infinite_horizon'."""
    finite_engine, infinite_engine = _build_engines()
    finite_inv, finite_sym = finite_engine.run_both()
    infinite_inv, infinite_sym = infinite_engine.run_both()
    for r in (finite_inv, finite_sym, infinite_inv, infinite_sym):
        assert math.isfinite(r.final_pnl)


# ---------------------------------------------------------------------------
# Numerical validity (NaN/Inf, bid<ask, inventory bounds, P&L presence)
# ---------------------------------------------------------------------------

def test_no_nan_or_inf_either_model():
    finite_engine, infinite_engine = _build_engines()
    for engine in (finite_engine, infinite_engine):
        result = engine.run(strategy="inventory")
        pnl_vals = [r.pnl for r in result.path]
        bid_vals = [r.bid for r in result.path if not math.isnan(r.bid)]
        ask_vals = [r.ask for r in result.path if not math.isnan(r.ask)]
        assert all(math.isfinite(v) for v in pnl_vals)
        assert all(math.isfinite(v) for v in bid_vals)
        assert all(math.isfinite(v) for v in ask_vals)


def test_bid_less_than_ask_either_model():
    finite_engine, infinite_engine = _build_engines()
    for engine in (finite_engine, infinite_engine):
        result = engine.run(strategy="inventory")
        for step in result.path:
            if not math.isnan(step.bid) and not math.isnan(step.ask):
                assert step.ask > step.bid


def test_inventory_updates_and_stays_within_limits():
    finite_engine, infinite_engine = _build_engines()
    for engine in (finite_engine, infinite_engine):
        result = engine.run(strategy="inventory")
        limits = engine.agent.inventory.limits
        assert any(step.inventory != 0 for step in result.path), (
            "inventory never moved off zero -- fills are not being applied"
        )
        assert all(
            limits.max_short <= step.inventory <= limits.max_long
            for step in result.path
        )


def test_pnl_produced_for_monte_carlo_pool_both_models():
    finite_engine, infinite_engine = _build_engines()
    for engine in (finite_engine, infinite_engine):
        results = engine.run_monte_carlo(n_runs=N_RUNS, strategy="inventory")
        assert len(results) == N_RUNS
        assert all(math.isfinite(r.final_pnl) for r in results)


# ---------------------------------------------------------------------------
# Fair-comparison / common random numbers
# ---------------------------------------------------------------------------

def test_same_seed_gives_identical_mid_price_paths():
    """Isolating the effect of the horizon formulation requires both
    engines to see the SAME mid-price path (and the same raw fill-decision
    draws) when constructed with the same seed -- only the quoting model
    should differ."""
    finite_engine, infinite_engine = _build_engines()
    finite_result = finite_engine.run(strategy="inventory")
    infinite_result = infinite_engine.run(strategy="inventory")
    finite_mid = [s.mid_price for s in finite_result.path]
    infinite_mid = [s.mid_price for s in infinite_result.path]
    assert finite_mid == infinite_mid


def test_different_seeds_give_different_mid_price_paths():
    """Sanity check on the above: this isn't trivially true for any input,
    i.e. seed actually controls the path."""
    engine_a = SimulationEngine(T=1.0, model="finite_horizon", seed=1, **COMMON_PARAMS)
    engine_b = SimulationEngine(T=1.0, model="finite_horizon", seed=2, **COMMON_PARAMS)
    result_a = engine_a.run(strategy="inventory")
    result_b = engine_b.run(strategy="inventory")
    mid_a = [s.mid_price for s in result_a.path]
    mid_b = [s.mid_price for s in result_b.path]
    assert mid_a != mid_b


def test_same_seed_gives_identical_monte_carlo_child_seeds():
    """run_monte_carlo derives per-run child seeds purely from self.seed,
    so two engines built with the same seed must produce pools whose
    mid-price paths line up run-by-run."""
    finite_engine, infinite_engine = _build_engines()
    finite_mc = finite_engine.run_monte_carlo(n_runs=N_RUNS, strategy="inventory")
    infinite_mc = infinite_engine.run_monte_carlo(n_runs=N_RUNS, strategy="inventory")
    for f_res, i_res in zip(finite_mc, infinite_mc):
        f_mid = [s.mid_price for s in f_res.path]
        i_mid = [s.mid_price for s in i_res.path]
        assert f_mid == i_mid


# ---------------------------------------------------------------------------
# SimulationEngine.summarise() extension (used by main.py's comparison table)
# ---------------------------------------------------------------------------

def test_summarise_backward_compatible_keys_present():
    finite_engine, _ = _build_engines()
    results = finite_engine.run_monte_carlo(n_runs=N_RUNS, strategy="inventory")
    stats = SimulationEngine.summarise(results)
    for key in ("n_runs", "strategy", "average_spread", "mean_profit",
                "std_profit", "mean_final_q", "std_final_q"):
        assert key in stats


def test_summarise_new_keys_present_and_sane():
    finite_engine, _ = _build_engines()
    results = finite_engine.run_monte_carlo(n_runs=N_RUNS, strategy="inventory")
    stats = SimulationEngine.summarise(results)
    for key in ("mean_abs_inventory", "max_abs_inventory",
                "mean_buy_fills", "mean_sell_fills", "mean_total_fills"):
        assert key in stats
        assert math.isfinite(stats[key])
    assert stats["mean_abs_inventory"] >= 0.0
    assert stats["max_abs_inventory"] >= stats["mean_abs_inventory"]
    assert stats["mean_total_fills"] == (
        stats["mean_buy_fills"] + stats["mean_sell_fills"]
    )


def test_summarise_works_for_infinite_horizon_pool_too():
    _, infinite_engine = _build_engines()
    results = infinite_engine.run_monte_carlo(n_runs=N_RUNS, strategy="inventory")
    stats = SimulationEngine.summarise(results)
    assert stats["n_runs"] == N_RUNS
    assert math.isfinite(stats["mean_profit"])
    assert math.isfinite(stats["max_abs_inventory"])


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
