"""
load_data.py — LOBSTERLoader
=============================

Parses and cleans raw LOBSTER (Limit Order Book System — The Efficient
Reconstructor) CSV file pairs into standardised, analysis-ready pandas
DataFrames.

Design intent (per CALIBRATION_PHASE1.md, section 5.1):
    This is the ONLY module in the calibration package that knows about the
    LOBSTER file format (column ordering, tick-to-dollar conversion, the
    lack of header rows, the event-type coding, etc). Every downstream
    module (mid_price_series.py, estimate_sigma.py, estimate_lambda.py, ...)
    consumes the clean DataFrames / LOBSTERData produced here and never
    touches a raw LOBSTER CSV directly. Concentrating format knowledge in
    one place means that if LOBSTER ever changes its export format, or we
    add a new venue with a different format, only this file needs to change.

Scope note:
    This file implements Task 2 only: data loading, validation, cleaning
    and the mid_price/spread derived columns. Sigma and lambda estimation
    are explicitly OUT of scope here and live in later modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger.
#
# Design decision: we configure a logger named after this module rather than
# calling logging.basicConfig() here. Library code should never configure
# the root logging setup (handlers, format, level) — that is the
# application's / notebook's responsibility. We just emit records; the
# caller decides where they go. A NullHandler is attached so that, in the
# absence of any caller configuration, we don't get "No handlers could be
# found" warnings, while still allowing `logging.basicConfig()` in a
# notebook or script to pick everything up.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class LOBSTERFormatError(ValueError):
    """
    Raised when a LOBSTER file does not match the expected structure
    (wrong column count, unreadable, mismatched row counts between the
    message and orderbook files, etc).

    Design decision: we define a dedicated exception type rather than
    raising bare ValueError/RuntimeError everywhere. This lets calling
    code (e.g. a calibration pipeline or a CLI) catch
    "this data was malformed" specifically, and distinguish it from other
    failure modes (missing file, permissions, out-of-memory), without
    resorting to string-matching on error messages.
    """


class LOBSTERFileNotFoundError(FileNotFoundError):
    """
    Raised when a message or orderbook file path does not exist.

    Design decision: subclassing the built-in FileNotFoundError (rather than
    inventing an unrelated exception) means existing `except FileNotFoundError`
    handlers in caller code keep working, while callers who want
    LOBSTER-specific handling can catch this narrower type.
    """


# ---------------------------------------------------------------------------
# Column name templates
# ---------------------------------------------------------------------------
# Design decision: message-file column names are fixed by the LOBSTER spec
# (section 2.1 of the roadmap) and never vary with n_levels, so they are a
# simple module-level constant rather than something computed per instance.
MESSAGE_COLUMNS: list[str] = [
    "time",
    "event_type",
    "order_id",
    "size",
    "price",
    "direction",
]

# Event type codes that represent an actual execution (fill) against a
# resting limit order, as opposed to submissions/cancellations/halts/
# auctions. Defined once here so every module that needs "is this row a
# fill?" logic (this module's load_merged, and later mid_price_series.py)
# uses the same authoritative list instead of re-typing magic numbers.
FILL_EVENT_TYPES: frozenset[int] = frozenset({4, 5})

# Event types to be dropped unconditionally when building the merged/clean
# view: 6 = auction trade (opening/closing cross), 7 = trading halt
# indicator. These do not represent continuous double-auction trading and
# would contaminate both the sigma and lambda estimators if left in.
EXCLUDED_EVENT_TYPES: frozenset[int] = frozenset({6, 7})


def _orderbook_columns(n_levels: int) -> list[str]:
    """
    Build the 4*n_levels orderbook column names in LOBSTER's fixed order.

    Design decision: this is a free function, not a method, because it has
    no dependency on instance state — it is a pure mapping from n_levels to
    a list of names, useful both inside the class and inside unit tests
    that want to construct synthetic LOBSTER files without instantiating a
    loader.

    LOBSTER orders columns as, per level i (1-indexed, 1 = top of book):
        ask_price_i, ask_size_i, bid_price_i, bid_size_i
    i.e. level 1 occupies columns [0, 1, 2, 3], level 2 occupies
    [4, 5, 6, 7], and so on.
    """
    columns: list[str] = []
    for level in range(1, n_levels + 1):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )
    return columns


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class LOBSTERData:
    """
    Container for the fully-loaded, cleaned output of LOBSTERLoader.

    Design decision: we use a plain (mutable-by-default) dataclass rather
    than a NamedTuple or a raw dict for two reasons:
      1. DataFrame fields don't have a meaningful notion of equality/hash
         that would benefit from NamedTuple's tuple semantics, so nothing
         is lost by not using it.
      2. A dataclass gives us named, typed, IDE-discoverable attributes
         (data.messages, data.n_fills, ...) which is exactly the ergonomic
         "structured result object" the roadmap specifies in section 5.1,
         and downstream modules (MidPriceBuilder etc.) are documented to
         accept a LOBSTERData instance as their input.

    We do not freeze this dataclass (unlike SimConfig, which the roadmap
    explicitly marks frozen=True) because LOBSTERData is a data-loading
    result, not a configuration object passed around and relied upon to be
    immutable; callers may reasonably want to attach extra columns to
    `merged` during exploratory analysis without fighting the dataclass.
    """

    messages: pd.DataFrame
    orderbook: pd.DataFrame
    merged: pd.DataFrame
    ticker: str
    date: str
    n_levels: int
    n_events: int
    n_fills: int
    mean_mid_price: float
    mean_spread: float

    # Free-form metadata slot (e.g. source file paths). Not in the roadmap's
    # spec table, but harmless to include as an optional convenience for
    # debugging / provenance without disturbing the documented fields.
    source_files: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LOBSTERLoader
# ---------------------------------------------------------------------------
class LOBSTERLoader:
    """
    Parses and cleans a raw LOBSTER message/orderbook CSV file pair.

    Parameters
    ----------
    message_path : str or pathlib.Path
        Path to the `*_message_*.csv` file.
    orderbook_path : str or pathlib.Path
        Path to the `*_orderbook_*.csv` file.
    n_levels : int, default 10
        Number of price levels present in the orderbook file. Must match
        the file's actual level count (i.e. orderbook file must have
        exactly 4 * n_levels columns), or load_orderbook() raises
        LOBSTERFormatError.
    tick_size : float, default 1e-4
        Conversion factor from LOBSTER integer price ticks to dollars.
        LOBSTER encodes prices in units of 1/10000 of a dollar, so the
        default of 1e-4 is correct for the standard LOBSTER export and is
        exposed as a parameter only in case a differently-scaled or
        differently-denominated export is ever encountered.
    trading_start : float, default 34200.0
        Start of the trading-hours filter window, in seconds since
        midnight (34200.0 = 9:30:00.000 exactly, the NASDAQ regular open).
    trading_end : float, default 57600.0
        End of the trading-hours filter window, in seconds since midnight
        (57600.0 = 16:00:00.000, the NASDAQ regular close).
    ticker : str, optional
        Ticker symbol for labelling output. If not supplied, we attempt to
        infer it from the message file name (LOBSTER's own naming
        convention places the ticker first, e.g.
        "AAPL_2012-06-21_34200000_57600000_message_10.csv"). Falls back to
        "UNKNOWN" if it cannot be inferred and none is supplied.
    date : str, optional
        Trading date for labelling output. Same inference/fallback
        behaviour as `ticker`.

    Design decision — why a class instead of free functions:
        The roadmap's architecture table names an explicit `LOBSTERLoader`
        class with a fixed constructor signature and three loading methods
        that are meant to be called independently or in combination
        (`load_messages()`, `load_orderbook()`, `load_merged()`). A class
        lets us:
          (a) parse each raw file at most once and cache the raw
              DataFrame, so calling load_messages() and then load_merged()
              doesn't re-read the message CSV from disk twice, and
          (b) validate cross-file invariants (matching row counts) once,
              at first-load time, rather than duplicating that check in
              three separate free functions.

    Design decision — why raw parsing is separated from public methods:
        Internally we keep a "raw" (unfiltered, but already priced-in-
        dollars and column-named) copy of each file, and only apply the
        trading-hours filter, halt/cross removal, etc. in the public
        methods. This matters for correctness, not just style: the
        roadmap is explicit that "Row i in the message file and row i in
        the orderbook file describe the same event" and that this
        alignment "must be preserved during loading". If load_messages()
        and load_orderbook() each computed their own independent trading-
        hours mask, floating point time comparisons or future edits could
        cause the two masks to diverge and silently break row alignment.
        Instead, the trading-hours mask is computed exactly once from the
        message file's `time` column, cached, and reused by both
        load_messages() (implicitly, since it operates on that same
        column) and load_orderbook() (explicitly passed in), guaranteeing
        by construction that both filtered DataFrames keep matching
        integer positions.
    """

    def __init__(
        self,
        message_path: Union[str, Path],
        orderbook_path: Union[str, Path],
        n_levels: int = 10,
        tick_size: float = 1e-4,
        trading_start: float = 34200.0,
        trading_end: float = 57600.0,
        ticker: Optional[str] = None,
        date: Optional[str] = None,
    ) -> None:
        self.message_path = Path(message_path)
        self.orderbook_path = Path(orderbook_path)

        if n_levels < 1:
            raise ValueError(f"n_levels must be >= 1, got {n_levels}")
        self.n_levels = n_levels

        if tick_size <= 0:
            raise ValueError(f"tick_size must be > 0, got {tick_size}")
        self.tick_size = tick_size

        if trading_end <= trading_start:
            raise ValueError(
                "trading_end must be strictly after trading_start "
                f"(got start={trading_start}, end={trading_end})"
            )
        self.trading_start = trading_start
        self.trading_end = trading_end

        inferred_ticker, inferred_date = self._infer_ticker_and_date(self.message_path)
        self.ticker = ticker or inferred_ticker
        self.date = date or inferred_date

        # Caches populated lazily by _load_raw_*(). Design decision: lazy
        # loading means constructing a LOBSTERLoader never touches disk —
        # only calling one of the public methods does. This is convenient
        # for tests/tooling that want to construct a loader (e.g. to
        # inspect .ticker / .date) without requiring the files to exist
        # yet, and avoids paying I/O cost for callers who only need one of
        # messages/orderbook/merged.
        self._raw_messages: Optional[pd.DataFrame] = None
        self._raw_orderbook: Optional[pd.DataFrame] = None

        logger.debug(
            "Initialised LOBSTERLoader(ticker=%s, date=%s, n_levels=%d, "
            "message_path=%s, orderbook_path=%s)",
            self.ticker,
            self.date,
            self.n_levels,
            self.message_path,
            self.orderbook_path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_ticker_and_date(message_path: Path) -> tuple[str, str]:
        """
        Best-effort extraction of ticker and date from a LOBSTER file name.

        LOBSTER's naming convention (roadmap section 2, "Naming
        convention") is:
            TICKER_DATE_STARTTIME_ENDTIME_message_NLEVELS.csv

        Design decision: this is deliberately "best effort" — filename
        parsing is inherently fragile (users rename files, re-export with
        different tools, etc), so on any failure to match the expected
        pattern we fall back to "UNKNOWN" / "UNKNOWN" rather than raising.
        Ticker/date are used only for labelling (LOBSTERData.ticker/date,
        log messages, output file names in later tasks) — never for any
        numerical computation — so silently degrading here is safe and
        preferable to forcing every caller to pass ticker= and date=
        explicitly for what is, in the common case, a solved problem.
        """
        stem = message_path.stem  # filename without extension
        parts = stem.split("_")
        if len(parts) >= 2:
            ticker, date = parts[0], parts[1]
            return ticker, date
        return "UNKNOWN", "UNKNOWN"

    def _require_file(self, path: Path, label: str) -> None:
        """Raise a clear, labelled error if a required input file is missing."""
        if not path.exists():
            raise LOBSTERFileNotFoundError(
                f"LOBSTER {label} file not found: {path}. "
                "See CALIBRATION_PHASE1.md section 1 for download "
                "instructions (https://lobsterdata.com/info/DataSamples.php)."
            )
        if not path.is_file():
            raise LOBSTERFileNotFoundError(f"LOBSTER {label} path is not a file: {path}")

    def _load_raw_messages(self) -> pd.DataFrame:
        """
        Read and column-name the message CSV, with prices converted to
        dollars, but WITHOUT any trading-hours or event-type filtering.

        Cached on `self._raw_messages` after first call.
        """
        if self._raw_messages is not None:
            return self._raw_messages

        self._require_file(self.message_path, "message")
        logger.info("Reading LOBSTER message file: %s", self.message_path)

        try:
            df = pd.read_csv(self.message_path, header=None)
        except pd.errors.EmptyDataError as exc:
            raise LOBSTERFormatError(
                f"Message file is empty: {self.message_path}"
            ) from exc
        except pd.errors.ParserError as exc:
            raise LOBSTERFormatError(
                f"Message file could not be parsed as CSV: {self.message_path} ({exc})"
            ) from exc

        if df.shape[1] != len(MESSAGE_COLUMNS):
            raise LOBSTERFormatError(
                f"Message file {self.message_path} has {df.shape[1]} columns; "
                f"expected exactly {len(MESSAGE_COLUMNS)} "
                f"({MESSAGE_COLUMNS}). The file may not be a standard "
                "LOBSTER message export, or may already have a header row."
            )
        if df.empty:
            raise LOBSTERFormatError(f"Message file has no rows: {self.message_path}")

        df.columns = MESSAGE_COLUMNS

        # Design decision: enforce numeric dtypes explicitly rather than
        # trusting pandas' type inference. A LOBSTER file with a stray
        # header row, a truncated download, or a corrupted row can produce
        # object-dtype columns that silently break every downstream
        # arithmetic operation (e.g. price / 10000 raising or, worse,
        # doing string repetition). Coercing here, with errors="raise",
        # converts that silent corruption into an immediate, informative
        # failure at load time.
        numeric_columns = ["time", "event_type", "order_id", "size", "price", "direction"]
        for col in numeric_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError) as exc:
                raise LOBSTERFormatError(
                    f"Column '{col}' in {self.message_path} contains "
                    f"non-numeric values and could not be parsed: {exc}"
                ) from exc

        # Price conversion: LOBSTER encodes prices in integer ticks of
        # 1/10000 dollar. Converting here, once, at the raw-loading stage
        # means every downstream consumer (this module's own load_merged,
        # and later mid_price_series.py / estimate_*.py) works in dollars
        # and never has to remember the tick_size convention again.
        df["price"] = df["price"] * self.tick_size

        if (df["price"] <= 0).any():
            n_bad = int((df["price"] <= 0).sum())
            logger.warning(
                "%d rows in %s have non-positive price after tick conversion; "
                "these likely correspond to non-price events and will not be "
                "excluded automatically here (only in load_merged's event-type "
                "filtering), but are flagged for visibility.",
                n_bad,
                self.message_path,
            )

        if not df["direction"].isin([-1, 1]).all():
            bad_values = sorted(df.loc[~df["direction"].isin([-1, 1]), "direction"].unique())
            raise LOBSTERFormatError(
                f"Message file {self.message_path} has 'direction' values "
                f"outside {{-1, 1}}: {bad_values}. This indicates the file is "
                "not a standard LOBSTER export or was corrupted."
            )

        self._raw_messages = df
        logger.debug("Loaded %d raw message rows from %s", len(df), self.message_path)
        return df

    def _load_raw_orderbook(self) -> pd.DataFrame:
        """
        Read and column-name the orderbook CSV, with prices converted to
        dollars, but WITHOUT any trading-hours filtering.

        Cached on `self._raw_orderbook` after first call. Also validates,
        on first call, that the row count matches the message file's row
        count — this is the "must be preserved during loading" alignment
        invariant called out in the roadmap.
        """
        if self._raw_orderbook is not None:
            return self._raw_orderbook

        self._require_file(self.orderbook_path, "orderbook")
        logger.info("Reading LOBSTER orderbook file: %s", self.orderbook_path)

        try:
            df = pd.read_csv(self.orderbook_path, header=None)
        except pd.errors.EmptyDataError as exc:
            raise LOBSTERFormatError(
                f"Orderbook file is empty: {self.orderbook_path}"
            ) from exc
        except pd.errors.ParserError as exc:
            raise LOBSTERFormatError(
                f"Orderbook file could not be parsed as CSV: {self.orderbook_path} ({exc})"
            ) from exc

        expected_ncols = 4 * self.n_levels
        if df.shape[1] != expected_ncols:
            raise LOBSTERFormatError(
                f"Orderbook file {self.orderbook_path} has {df.shape[1]} "
                f"columns; expected 4 * n_levels = {expected_ncols} for "
                f"n_levels={self.n_levels}. Either n_levels was passed "
                "incorrectly, or this is not a standard LOBSTER orderbook "
                "export."
            )
        if df.empty:
            raise LOBSTERFormatError(f"Orderbook file has no rows: {self.orderbook_path}")

        df.columns = _orderbook_columns(self.n_levels)

        try:
            df = df.apply(pd.to_numeric, errors="raise")
        except (ValueError, TypeError) as exc:
            raise LOBSTERFormatError(
                f"Orderbook file {self.orderbook_path} contains non-numeric "
                f"values and could not be parsed: {exc}"
            ) from exc

        # Row-count / alignment check. This is deliberately a hard error
        # (not a warning) because every downstream computation in this
        # package assumes row i of messages and row i of orderbook refer
        # to the same event; silently truncating or padding would produce
        # plausible-looking but WRONG mid-prices and fill deltas.
        n_messages = len(self._load_raw_messages())
        if len(df) != n_messages:
            raise LOBSTERFormatError(
                f"Row count mismatch between message file "
                f"({n_messages} rows, {self.message_path}) and orderbook "
                f"file ({len(df)} rows, {self.orderbook_path}). LOBSTER "
                "guarantees these files have identical row counts; a "
                "mismatch means at least one file is truncated, corrupted, "
                "or the two files do not actually correspond to the same "
                "session."
            )

        # Price conversion for all price columns (ask_price_i, bid_price_i).
        # Size columns (ask_size_i, bid_size_i) are share counts and are
        # left untouched.
        price_cols = [c for c in df.columns if c.startswith("ask_price_") or c.startswith("bid_price_")]
        df[price_cols] = df[price_cols] * self.tick_size

        # Sanity check: for every level, ask should be >= bid (a crossed
        # top-of-book is a known, if rare, data-quality issue rather than
        # a hard error — LOBSTER reconstruction can momentarily show a
        # crossed book around certain event sequences). We log rather than
        # raise, matching the "flag for visibility, decide filtering later"
        # approach also used for non-positive prices above; load_merged()
        # is where callers can choose to filter these out (e.g. via its
        # own spread-based cleaning) rather than losing rows silently here.
        crossed = df["ask_price_1"] < df["bid_price_1"]
        if crossed.any():
            logger.warning(
                "%d rows in %s have a crossed top-of-book (ask_price_1 < "
                "bid_price_1). These are left in place by load_orderbook(); "
                "downstream cleaning (e.g. spread-based filtering in "
                "mid_price_series.py) should handle them.",
                int(crossed.sum()),
                self.orderbook_path,
            )

        self._raw_orderbook = df
        logger.debug("Loaded %d raw orderbook rows from %s", len(df), self.orderbook_path)
        return df

    def _trading_hours_mask(self) -> pd.Series:
        """
        Boolean mask, indexed like the raw message DataFrame, that is True
        for rows within [trading_start, trading_end].

        Design decision: this mask is computed exactly once, from the
        message file's `time` column only, and is the single source of
        truth used by both load_messages() and load_orderbook() to select
        rows. Because the orderbook file has no time column of its own,
        deriving its trading-hours filter from the message file's mask (by
        integer position) is the only way to filter it at all — and doing
        so guarantees the two filtered frames stay row-aligned.
        """
        messages = self._load_raw_messages()
        mask = (messages["time"] >= self.trading_start) & (messages["time"] <= self.trading_end)
        return mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_messages(self) -> pd.DataFrame:
        """
        Load the message file, filtered to trading hours.

        Returns
        -------
        pandas.DataFrame
            Columns: time, event_type, order_id, size, price (dollars),
            direction. Index is reset to a contiguous RangeIndex starting
            at 0 after filtering (see Design decision below).
            event_type values are preserved as-is (i.e. types 6 and 7 are
            NOT removed here — that filtering happens in load_merged();
            see the roadmap's section 5.1 spec, which explicitly documents
            "EventType preserved as-is" for this method).

        Raises
        ------
        LOBSTERFileNotFoundError
            If the message file does not exist.
        LOBSTERFormatError
            If the file cannot be parsed, has the wrong column count, is
            empty, or contains non-numeric / out-of-range values.

        Design decision — index reset:
            We call `.reset_index(drop=True)` after filtering rather than
            leaving the original (pre-filter) integer index in place. This
            is intentional: load_orderbook() performs the identical
            filter-then-reset sequence, so the two returned DataFrames
            remain aligned by position (row i <-> row i) even though rows
            were dropped from the middle of the original file — matching
            the "row i in messages and row i in orderbook describe the
            same event" invariant from the roadmap, now expressed in
            0..N-1 terms rather than the original file's row numbers.
        """
        mask = self._trading_hours_mask()
        messages = self._load_raw_messages()
        filtered = messages.loc[mask].reset_index(drop=True)
        logger.info(
            "load_messages: %d rows after trading-hours filter "
            "[%.1f, %.1f] (from %d raw rows)",
            len(filtered),
            self.trading_start,
            self.trading_end,
            len(messages),
        )
        return filtered

    def load_orderbook(self) -> pd.DataFrame:
        """
        Load the orderbook file, filtered to the same trading-hours window
        (and therefore the same rows) as load_messages().

        Returns
        -------
        pandas.DataFrame
            Columns: ask_price_1, ask_size_1, bid_price_1, bid_size_1,
            ask_price_2, ... up to level n_levels. All *_price_* columns
            are in dollars; all *_size_* columns are share counts.
            Guaranteed to have exactly the same number of rows, in the
            same order, as load_messages().

        Raises
        ------
        LOBSTERFileNotFoundError
            If the orderbook file does not exist.
        LOBSTERFormatError
            If the file cannot be parsed, has the wrong column count
            (i.e. != 4 * n_levels), is empty, contains non-numeric values,
            or its row count does not match the message file's row count.
        """
        mask = self._trading_hours_mask()
        orderbook = self._load_raw_orderbook()
        filtered = orderbook.loc[mask].reset_index(drop=True)
        logger.info("load_orderbook: %d rows after trading-hours filter", len(filtered))
        return filtered

    def load_merged(self) -> pd.DataFrame:
        """
        Load and join the message and orderbook files into a single
        DataFrame, with derived mid_price and spread columns, and with
        event types 6 (auction trade) and 7 (trading halt indicator)
        removed.

        Returns
        -------
        pandas.DataFrame
            All message columns + all orderbook columns, joined on row
            position, plus:
              - mid_price : (ask_price_1 + bid_price_1) / 2, in dollars
              - spread    : ask_price_1 - bid_price_1, in dollars
            Filtered to trading hours AND with event_type in {6, 7}
            removed. Index reset to a contiguous RangeIndex.

        Raises
        ------
        LOBSTERFileNotFoundError, LOBSTERFormatError
            Same conditions as load_messages() / load_orderbook().

        Design decision — join strategy:
            We join purely by integer position (via `pd.concat(...,
            axis=1)` after both frames have been independently filtered
            to trading hours and index-reset identically), not by any
            business key. LOBSTER does not provide one — row-position
            correspondence IS the join key, by construction of the file
            format. Using `pd.merge` here would be both unnecessary and
            actively risky, since it invites an accidental key-based join
            that could silently reorder or duplicate rows if any join
            column were less unique than expected.

        Design decision — why remove event types 6/7 only in load_merged():
            load_messages() and load_orderbook() are documented (and
            tested) to preserve full trading-hours data, including
            crosses and halts, for callers who specifically want to see
            them (e.g. to plot where the opening/closing auction happened
            in an exploratory notebook). load_merged() is the "ready for
            estimation" view described in the roadmap's data-flow diagram
            (section 4.2) and its column-mapping tables (section 3),
            which explicitly call for excluding types 6 and 7 before
            anything downstream (sigma/lambda estimation) touches the
            data. Keeping the two behaviours in different methods avoids
            forcing every caller of load_messages()/load_orderbook() to
            re-derive the halt/cross filter themselves.
        """
        messages = self.load_messages()
        orderbook = self.load_orderbook()

        if len(messages) != len(orderbook):
            # Defensive check: should be unreachable given both methods
            # filter with the identical mask derived in
            # _trading_hours_mask(), but we check explicitly rather than
            # silently trusting that invariant forever, since a future
            # edit to either method could break it.
            raise LOBSTERFormatError(
                "Internal alignment error: filtered messages "
                f"({len(messages)} rows) and filtered orderbook "
                f"({len(orderbook)} rows) have different lengths. This "
                "should not happen; please report this as a bug."
            )

        merged = pd.concat([messages, orderbook], axis=1)

        merged["mid_price"] = (merged["ask_price_1"] + merged["bid_price_1"]) / 2.0
        merged["spread"] = merged["ask_price_1"] - merged["bid_price_1"]

        before = len(merged)
        merged = merged.loc[~merged["event_type"].isin(EXCLUDED_EVENT_TYPES)].reset_index(drop=True)
        removed = before - len(merged)
        logger.info(
            "load_merged: removed %d rows with event_type in %s "
            "(auction trade / trading halt); %d rows remain",
            removed,
            sorted(EXCLUDED_EVENT_TYPES),
            len(merged),
        )

        if merged.empty:
            raise LOBSTERFormatError(
                "load_merged produced an empty DataFrame after filtering "
                "trading hours and removing halt/auction events. Check "
                "trading_start/trading_end against the data's actual "
                "timestamps."
            )

        return merged

    def load(self) -> LOBSTERData:
        """
        Convenience method: run load_messages(), load_orderbook(), and
        load_merged(), and package the results into a LOBSTERData instance
        along with summary statistics.

        Design decision: this is not explicitly named in the roadmap's
        method list for LOBSTERLoader (which lists load_messages,
        load_orderbook, load_merged, save_parquet), but the roadmap DOES
        specify a `LOBSTERData` output dataclass with fields like
        n_events, n_fills, mean_mid_price, mean_spread that are naturally
        computed from the combination of all three loads. Rather than
        force every caller to manually call all three methods and compute
        those summary statistics themselves, we provide this one
        convenience method. It is purely additive — load_messages(),
        load_orderbook(), and load_merged() remain independently callable
        exactly as specified.

        Returns
        -------
        LOBSTERData
        """
        messages = self.load_messages()
        orderbook = self.load_orderbook()
        merged = self.load_merged()

        n_fills = int(merged["event_type"].isin(FILL_EVENT_TYPES).sum())

        return LOBSTERData(
            messages=messages,
            orderbook=orderbook,
            merged=merged,
            ticker=self.ticker,
            date=self.date,
            n_levels=self.n_levels,
            n_events=len(merged),
            n_fills=n_fills,
            mean_mid_price=float(merged["mid_price"].mean()),
            mean_spread=float(merged["spread"].mean()),
            source_files={
                "message_path": str(self.message_path),
                "orderbook_path": str(self.orderbook_path),
            },
        )

    @staticmethod
    def save_parquet(df: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """
        Save a cleaned DataFrame to parquet for fast reloading.

        Parameters
        ----------
        df : pandas.DataFrame
            Typically the result of load_merged(), but any DataFrame
            produced by this loader may be saved.
        output_path : str or pathlib.Path
            Destination path. Parent directories are created if they do
            not already exist.

        Design decision — why a @staticmethod:
            Saving doesn't depend on any instance state (message_path,
            n_levels, etc) — it operates purely on the DataFrame the
            caller passes in, exactly as specified in the roadmap
            ("save_parquet(df, output_path) -> None"). Making it a
            staticmethod signals that clearly and lets it be called
            without a loader instance at all, e.g.
            `LOBSTERLoader.save_parquet(some_df, "out.parquet")`.

        Raises
        ------
        ValueError
            If df is empty.
        OSError
            If the file cannot be written (propagated from pandas/pyarrow).
        """
        if df.empty:
            raise ValueError("Refusing to save an empty DataFrame to parquet.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df.to_parquet(output_path)
        except ImportError as exc:
            raise ImportError(
                "Saving to parquet requires 'pyarrow' or 'fastparquet' to "
                "be installed (e.g. `pip install pyarrow`)."
            ) from exc

        logger.info("Saved %d rows to %s", len(df), output_path)
