"""
tests/test_load_data.py
========================

Unit tests for calibration.load_data.LOBSTERLoader.

Framework note: these tests use Python's standard-library `unittest`
rather than pytest. `unittest` ships with every CPython install, so these
tests are runnable anywhere without an extra dependency; if the project
later standardises on pytest, this file's test methods are trivially
pytest-discoverable/runnable as-is (pytest can run unittest.TestCase
classes natively), so nothing here needs to be thrown away.

Design decision: we do NOT depend on the real LOBSTER sample download
(https://lobsterdata.com/info/DataSamples.php) for these tests. Task 2's
own checklist (Phase 1A) treats "download and inspect real data" as a
separate, manual task (Task 1); this test module's job is to verify
LOBSTERLoader's parsing/validation/cleaning *logic* is correct, which we
can and should do deterministically and offline with small synthetic
files that exactly follow the documented LOBSTER format. This also means
these tests run in CI / any environment without network access, and their
failures point unambiguously at our code rather than at network flakiness
or upstream sample-data changes.

Synthetic file construction mirrors CALIBRATION_PHASE1.md section 2
exactly:
  - message file: 6 columns, no header, [time, event_type, order_id,
    size, price, direction], price in integer ticks of 1/10000 dollar.
  - orderbook file: 4*n_levels columns, no header,
    [ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...].
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.load_data import (
    LOBSTERData,
    LOBSTERFileNotFoundError,
    LOBSTERFormatError,
    LOBSTERLoader,
    _orderbook_columns,
)

logging.basicConfig(level=logging.WARNING)

N_LEVELS = 3
TRADING_START = 34200.0  # 09:30:00
TRADING_END = 57600.0  # 16:00:00


# ---------------------------------------------------------------------------
# Synthetic data builder (shared helper, not tied to any one TestCase)
# ---------------------------------------------------------------------------
def make_synthetic_files(
    tmp_path: Path,
    n_rows: int = 200,
    n_levels: int = N_LEVELS,
    ticker: str = "AAPL",
    date: str = "2012-06-21",
    include_out_of_hours: bool = True,
    include_halts_and_crosses: bool = True,
    seed: int = 42,
) -> tuple[Path, Path]:
    """
    Build a small, internally-consistent pair of synthetic LOBSTER files
    (message + orderbook) with matching row counts, and return their paths.

    The generated book is a simple random walk around $100 with a 1-cent
    spread, in LOBSTER's integer-tick units (i.e. $100.00 -> 1_000_000).
    """
    rng = np.random.default_rng(seed)

    base_price_ticks = 1_000_000  # $100.00 in 1/10000-dollar ticks
    mid_walk = base_price_ticks + np.cumsum(rng.integers(-5, 6, size=n_rows)) * 100

    times = np.linspace(TRADING_START, TRADING_END, n_rows)
    if include_out_of_hours:
        times[0] = TRADING_START - 100.0
        times[1] = TRADING_START - 1.0
        times[-1] = TRADING_END + 50.0

    event_types = rng.choice([1, 2, 3, 4, 5], size=n_rows)
    if include_halts_and_crosses:
        event_types[5] = 6
        event_types[6] = 7

    order_ids = np.arange(1, n_rows + 1)
    sizes = rng.integers(1, 500, size=n_rows)
    directions = rng.choice([-1, 1], size=n_rows)
    exec_price_ticks = mid_walk + directions * rng.integers(0, 50, size=n_rows)

    message_df = pd.DataFrame(
        {
            0: times,
            1: event_types,
            2: order_ids,
            3: sizes,
            4: exec_price_ticks,
            5: directions,
        }
    )

    ob_data = {}
    spread_ticks = 100  # 1 cent spread at level 1
    for level in range(1, n_levels + 1):
        level_offset = (level - 1) * 200
        ask_price = mid_walk + spread_ticks // 2 + level_offset
        bid_price = mid_walk - spread_ticks // 2 - level_offset
        ask_size = rng.integers(100, 1000, size=n_rows)
        bid_size = rng.integers(100, 1000, size=n_rows)
        base_col = 4 * (level - 1)
        ob_data[base_col + 0] = ask_price
        ob_data[base_col + 1] = ask_size
        ob_data[base_col + 2] = bid_price
        ob_data[base_col + 3] = bid_size

    orderbook_df = pd.DataFrame(ob_data)[sorted(ob_data.keys())]

    message_path = tmp_path / f"{ticker}_{date}_34200000_57600000_message_{n_levels}.csv"
    orderbook_path = tmp_path / f"{ticker}_{date}_34200000_57600000_orderbook_{n_levels}.csv"

    message_df.to_csv(message_path, header=False, index=False)
    orderbook_df.to_csv(orderbook_path, header=False, index=False)

    return message_path, orderbook_path


class TempDirTestCase(unittest.TestCase):
    """Base class providing a fresh tmp_path per test, unittest-style."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="lobster_test_")
        self.tmp_path = Path(self._tmpdir)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# _orderbook_columns
# ---------------------------------------------------------------------------
class TestOrderbookColumns(unittest.TestCase):
    def test_naming_and_order(self):
        cols = _orderbook_columns(2)
        self.assertEqual(
            cols,
            [
                "ask_price_1",
                "ask_size_1",
                "bid_price_1",
                "bid_size_1",
                "ask_price_2",
                "ask_size_2",
                "bid_price_2",
                "bid_size_2",
            ],
        )

    def test_length_matches_4n(self):
        for n in [1, 3, 10]:
            self.assertEqual(len(_orderbook_columns(n)), 4 * n)


# ---------------------------------------------------------------------------
# Ticker / date inference
# ---------------------------------------------------------------------------
class TestTickerDateInference(TempDirTestCase):
    def test_inferred_from_filename(self):
        message_path, orderbook_path = make_synthetic_files(self.tmp_path)
        loader_ = LOBSTERLoader(message_path, orderbook_path, n_levels=N_LEVELS)
        self.assertEqual(loader_.ticker, "AAPL")
        self.assertEqual(loader_.date, "2012-06-21")

    def test_explicit_override(self):
        message_path, orderbook_path = make_synthetic_files(self.tmp_path)
        loader_ = LOBSTERLoader(
            message_path, orderbook_path, n_levels=N_LEVELS, ticker="MSFT", date="2020-01-01"
        )
        self.assertEqual(loader_.ticker, "MSFT")
        self.assertEqual(loader_.date, "2020-01-01")

    def test_fallback_for_unparseable_filename(self):
        msg = self.tmp_path / "weird.csv"
        ob = self.tmp_path / "weirdob.csv"
        pd.DataFrame([[1, 2, 3]]).to_csv(msg, header=False, index=False)
        pd.DataFrame([[1, 2, 3]]).to_csv(ob, header=False, index=False)
        loader_ = LOBSTERLoader(msg, ob, n_levels=1)
        self.assertEqual(loader_.ticker, "UNKNOWN")
        self.assertEqual(loader_.date, "UNKNOWN")


# ---------------------------------------------------------------------------
# load_messages / load_orderbook / load_merged (happy path)
# ---------------------------------------------------------------------------
class TestLoaderHappyPath(TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.message_path, self.orderbook_path = make_synthetic_files(self.tmp_path)
        self.loader = LOBSTERLoader(self.message_path, self.orderbook_path, n_levels=N_LEVELS)

    # -- load_messages --
    def test_load_messages_columns(self):
        df = self.loader.load_messages()
        self.assertEqual(
            list(df.columns), ["time", "event_type", "order_id", "size", "price", "direction"]
        )

    def test_load_messages_filters_trading_hours(self):
        df = self.loader.load_messages()
        self.assertTrue((df["time"] >= TRADING_START).all())
        self.assertTrue((df["time"] <= TRADING_END).all())
        # 3 out-of-hours rows were deliberately injected by the fixture.
        self.assertEqual(len(df), 200 - 3)

    def test_load_messages_converts_price_to_dollars(self):
        df = self.loader.load_messages()
        frac_near_100 = df["price"].between(90, 110).mean()
        self.assertGreater(frac_near_100, 0.9)

    def test_load_messages_preserves_event_type_6_and_7(self):
        df = self.loader.load_messages()
        self.assertTrue(set(df["event_type"].unique()) & {6, 7})

    def test_load_messages_index_is_contiguous(self):
        df = self.loader.load_messages()
        self.assertEqual(list(df.index), list(range(len(df))))

    def test_load_messages_caches_raw_read(self):
        # Load once to populate the cache, then patch read_csv and load
        # again -- if caching works, the patched (failing) read_csv must
        # never be invoked on the second call.
        self.loader.load_messages()

        original_read_csv = pd.read_csv

        def failing_read_csv(*args, **kwargs):
            raise AssertionError("read_csv should not be called again (cache miss)")

        pd.read_csv = failing_read_csv
        try:
            self.loader.load_messages()  # should hit the cache, not raise
        finally:
            pd.read_csv = original_read_csv

    # -- load_orderbook --
    def test_load_orderbook_columns(self):
        df = self.loader.load_orderbook()
        self.assertEqual(list(df.columns), _orderbook_columns(N_LEVELS))

    def test_load_orderbook_row_count_matches_messages(self):
        messages = self.loader.load_messages()
        orderbook = self.loader.load_orderbook()
        self.assertEqual(len(messages), len(orderbook))

    def test_load_orderbook_ask_gte_bid_at_level_1(self):
        df = self.loader.load_orderbook()
        self.assertTrue((df["ask_price_1"] >= df["bid_price_1"]).all())

    def test_load_orderbook_prices_in_dollars(self):
        df = self.loader.load_orderbook()
        frac_near_100 = df["ask_price_1"].between(90, 110).mean()
        self.assertGreater(frac_near_100, 0.9)

    # -- load_merged --
    def test_load_merged_has_mid_price_and_spread(self):
        df = self.loader.load_merged()
        self.assertIn("mid_price", df.columns)
        self.assertIn("spread", df.columns)
        expected_mid = (df["ask_price_1"] + df["bid_price_1"]) / 2.0
        expected_spread = df["ask_price_1"] - df["bid_price_1"]
        pd.testing.assert_series_equal(df["mid_price"], expected_mid, check_names=False)
        pd.testing.assert_series_equal(df["spread"], expected_spread, check_names=False)

    def test_load_merged_removes_event_types_6_and_7(self):
        df = self.loader.load_merged()
        self.assertFalse(df["event_type"].isin([6, 7]).any())

    def test_load_merged_spread_non_negative(self):
        df = self.loader.load_merged()
        self.assertTrue((df["spread"] >= 0).all())

    def test_load_merged_row_count_consistent_with_filters(self):
        messages = self.loader.load_messages()
        merged = self.loader.load_merged()
        n_halts_crosses = messages["event_type"].isin([6, 7]).sum()
        self.assertEqual(len(merged), len(messages) - n_halts_crosses)

    # -- load() / LOBSTERData --
    def test_load_returns_lobster_data(self):
        data = self.loader.load()
        self.assertIsInstance(data, LOBSTERData)
        self.assertEqual(data.ticker, "AAPL")
        self.assertEqual(data.date, "2012-06-21")
        self.assertEqual(data.n_levels, N_LEVELS)
        self.assertEqual(data.n_events, len(data.merged))
        self.assertEqual(data.n_fills, int(data.merged["event_type"].isin([4, 5]).sum()))
        self.assertAlmostEqual(data.mean_mid_price, data.merged["mid_price"].mean())
        self.assertAlmostEqual(data.mean_spread, data.merged["spread"].mean())

    def test_load_source_files_recorded(self):
        data = self.loader.load()
        self.assertEqual(data.source_files["message_path"], str(self.message_path))
        self.assertEqual(data.source_files["orderbook_path"], str(self.orderbook_path))


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling(TempDirTestCase):
    def test_missing_message_file_raises(self):
        missing = self.tmp_path / "does_not_exist_message_1.csv"
        _, orderbook_path = make_synthetic_files(self.tmp_path, n_levels=1)
        loader_ = LOBSTERLoader(missing, orderbook_path, n_levels=1)
        with self.assertRaises(LOBSTERFileNotFoundError):
            loader_.load_messages()

    def test_missing_orderbook_file_raises(self):
        message_path, _ = make_synthetic_files(self.tmp_path, n_levels=1)
        missing = self.tmp_path / "does_not_exist_orderbook_1.csv"
        loader_ = LOBSTERLoader(message_path, missing, n_levels=1)
        with self.assertRaises(LOBSTERFileNotFoundError):
            loader_.load_orderbook()

    def test_wrong_message_column_count_raises(self):
        bad_msg = self.tmp_path / "BAD_2020-01-01_0_1_message_1.csv"
        pd.DataFrame(np.zeros((10, 5))).to_csv(bad_msg, header=False, index=False)
        _, orderbook_path = make_synthetic_files(self.tmp_path, n_levels=1, n_rows=10)
        loader_ = LOBSTERLoader(bad_msg, orderbook_path, n_levels=1)
        with self.assertRaisesRegex(LOBSTERFormatError, "columns"):
            loader_.load_messages()

    def test_wrong_orderbook_column_count_raises(self):
        message_path, _ = make_synthetic_files(self.tmp_path, n_levels=1, n_rows=10)
        bad_ob = self.tmp_path / "BAD_2020-01-01_0_1_orderbook_1.csv"
        pd.DataFrame(np.full((10, 3), 1_000_000)).to_csv(bad_ob, header=False, index=False)
        loader_ = LOBSTERLoader(message_path, bad_ob, n_levels=1)
        with self.assertRaisesRegex(LOBSTERFormatError, "columns"):
            loader_.load_orderbook()

    def test_row_count_mismatch_raises(self):
        message_path, _ = make_synthetic_files(self.tmp_path, n_levels=1, n_rows=50)
        ob = pd.DataFrame(np.tile([1_000_100, 100, 999_900, 100], (30, 1)))
        bad_ob_path = self.tmp_path / "MISMATCH_2020-01-01_0_1_orderbook_1.csv"
        ob.to_csv(bad_ob_path, header=False, index=False)
        loader_ = LOBSTERLoader(message_path, bad_ob_path, n_levels=1)
        with self.assertRaisesRegex(LOBSTERFormatError, "Row count mismatch"):
            loader_.load_orderbook()

    def test_invalid_direction_value_raises(self):
        n_rows = 10
        df = pd.DataFrame(
            {
                0: np.linspace(TRADING_START, TRADING_END, n_rows),
                1: np.full(n_rows, 1),
                2: np.arange(n_rows),
                3: np.full(n_rows, 100),
                4: np.full(n_rows, 1_000_000),
                5: np.full(n_rows, 0),  # invalid: must be -1 or +1
            }
        )
        msg_path = self.tmp_path / "BAD_2020-01-01_0_1_message_1.csv"
        df.to_csv(msg_path, header=False, index=False)
        ob_df = pd.DataFrame(np.tile([1_000_100, 100, 999_900, 100], (n_rows, 1)))
        ob_path = self.tmp_path / "BAD_2020-01-01_0_1_orderbook_1.csv"
        ob_df.to_csv(ob_path, header=False, index=False)

        loader_ = LOBSTERLoader(msg_path, ob_path, n_levels=1)
        with self.assertRaisesRegex(LOBSTERFormatError, "direction"):
            loader_.load_messages()

    def test_empty_message_file_raises(self):
        msg_path = self.tmp_path / "EMPTY_2020-01-01_0_1_message_1.csv"
        msg_path.write_text("")
        ob_path = self.tmp_path / "EMPTY_2020-01-01_0_1_orderbook_1.csv"
        ob_path.write_text("")
        loader_ = LOBSTERLoader(msg_path, ob_path, n_levels=1)
        with self.assertRaises(LOBSTERFormatError):
            loader_.load_messages()

    def test_non_numeric_message_values_raise(self):
        n_rows = 5
        lines = [f"{34300 + i},1,{i},100,1000000,NOT_A_NUMBER" for i in range(n_rows)]
        msg_path = self.tmp_path / "BAD_2020-01-01_0_1_message_1.csv"
        msg_path.write_text("\n".join(lines))
        ob_df = pd.DataFrame(np.tile([1_000_100, 100, 999_900, 100], (n_rows, 1)))
        ob_path = self.tmp_path / "BAD_2020-01-01_0_1_orderbook_1.csv"
        ob_df.to_csv(ob_path, header=False, index=False)
        loader_ = LOBSTERLoader(msg_path, ob_path, n_levels=1)
        with self.assertRaises(LOBSTERFormatError):
            loader_.load_messages()

    def test_constructor_rejects_invalid_n_levels(self):
        message_path, orderbook_path = make_synthetic_files(self.tmp_path)
        with self.assertRaises(ValueError):
            LOBSTERLoader(message_path, orderbook_path, n_levels=0)

    def test_constructor_rejects_invalid_tick_size(self):
        message_path, orderbook_path = make_synthetic_files(self.tmp_path)
        with self.assertRaises(ValueError):
            LOBSTERLoader(message_path, orderbook_path, tick_size=0.0)

    def test_constructor_rejects_bad_trading_window(self):
        message_path, orderbook_path = make_synthetic_files(self.tmp_path)
        with self.assertRaises(ValueError):
            LOBSTERLoader(message_path, orderbook_path, trading_start=100.0, trading_end=50.0)

    def test_all_rows_outside_trading_hours_raises_on_merge(self):
        n_rows = 5
        df = pd.DataFrame(
            {
                0: np.full(n_rows, TRADING_START - 10.0),
                1: np.full(n_rows, 1),
                2: np.arange(n_rows),
                3: np.full(n_rows, 100),
                4: np.full(n_rows, 1_000_000),
                5: np.full(n_rows, 1),
            }
        )
        msg_path = self.tmp_path / "OOH_2020-01-01_0_1_message_1.csv"
        df.to_csv(msg_path, header=False, index=False)
        ob_df = pd.DataFrame(np.tile([1_000_100, 100, 999_900, 100], (n_rows, 1)))
        ob_path = self.tmp_path / "OOH_2020-01-01_0_1_orderbook_1.csv"
        ob_df.to_csv(ob_path, header=False, index=False)

        loader_ = LOBSTERLoader(msg_path, ob_path, n_levels=1)
        with self.assertRaisesRegex(LOBSTERFormatError, "empty"):
            loader_.load_merged()


# ---------------------------------------------------------------------------
# save_parquet
# ---------------------------------------------------------------------------
class TestSaveParquet(TempDirTestCase):
    def test_rejects_empty_dataframe(self):
        with self.assertRaises(ValueError):
            LOBSTERLoader.save_parquet(pd.DataFrame(), self.tmp_path / "out.parquet")

    def test_writes_file_or_raises_importerror(self):
        message_path, orderbook_path = make_synthetic_files(self.tmp_path)
        loader_ = LOBSTERLoader(message_path, orderbook_path, n_levels=N_LEVELS)
        df = loader_.load_merged()
        out_path = self.tmp_path / "out" / "merged.parquet"
        try:
            LOBSTERLoader.save_parquet(df, out_path)
        except ImportError:
            # Sandboxed test environment may lack pyarrow/fastparquet with
            # no network access to install them -- an ImportError here is
            # an acceptable, clearly-signalled outcome; we just confirm
            # it's a proper ImportError and not some other failure mode.
            self.skipTest("pyarrow/fastparquet not installed in this environment")
        else:
            self.assertTrue(out_path.exists())


# ---------------------------------------------------------------------------
# Deterministic, hand-computed example
# ---------------------------------------------------------------------------
class TestHandComputedExample(TempDirTestCase):
    def test_two_row_example(self):
        """
        A fully deterministic, hand-checkable example: 2 rows, 1 level.

        Row 0: time=34300, event_type=1 (submission, not a fill),
               price=1234560 ticks = $123.456, direction=+1.
               Orderbook: ask=1234600 ($123.46), bid=1234500 ($123.45).
               -> mid = (123.46+123.45)/2 = 123.455, spread = 0.01

        Row 1: time=34301, event_type=4 (visible execution -> a fill),
               price=1234550 ticks = $123.455, direction=-1.
               Orderbook: ask=1234610 ($123.461), bid=1234510 ($123.451).
               -> mid = 123.456
        """
        msg_df = pd.DataFrame(
            {
                0: [34300.0, 34301.0],
                1: [1, 4],
                2: [1, 2],
                3: [10, 20],
                4: [1234560, 1234550],
                5: [1, -1],
            }
        )
        ob_df = pd.DataFrame(
            {
                0: [1234600, 1234610],
                1: [500, 600],
                2: [1234500, 1234510],
                3: [700, 800],
            }
        )
        msg_path = self.tmp_path / "TEST_2020-01-01_0_1_message_1.csv"
        ob_path = self.tmp_path / "TEST_2020-01-01_0_1_orderbook_1.csv"
        msg_df.to_csv(msg_path, header=False, index=False)
        ob_df.to_csv(ob_path, header=False, index=False)

        loader_ = LOBSTERLoader(msg_path, ob_path, n_levels=1)
        merged = loader_.load_merged()

        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged.loc[0, "price"], 123.456)
        self.assertAlmostEqual(merged.loc[0, "mid_price"], 123.455)
        self.assertAlmostEqual(merged.loc[0, "spread"], 0.01)
        self.assertAlmostEqual(merged.loc[1, "mid_price"], 123.456)

        data = loader_.load()
        self.assertEqual(data.n_events, 2)
        self.assertEqual(data.n_fills, 1)  # only row 1 has event_type in {4, 5}


if __name__ == "__main__":
    unittest.main(verbosity=2)
