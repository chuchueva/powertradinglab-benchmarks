"""
Tests for the v1 data interface.

Synthetic tables only — the point is the SHAPE of what a reader gets: one row
per delivery day, collapsed to the newest pass, in bytes that do not move when
the data does not.

Run:
    uv run pytest tests/test_summary.py -q
    uv run python tests/test_summary.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.summary as S  # noqa: E402


class _Log:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


def _day(day: str, knownby: str, seed: int = 0, area: str = "NL",
         n: int = 96, acted: bool = True) -> pd.DataFrame:
    """One graded delivery day, with the oracles built the way the method says."""
    from src.period import market_day_bounds

    rng = np.random.default_rng(seed)
    # The MARKET day, not the UTC one: NL's 05 August starts at 22:00 UTC on the
    # 4th, and a grid that ignores that spills into the next delivery day.
    start = pd.Timestamp(market_day_bounds(pd.Timestamp(day).date(), area)[0])
    ts = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    buy = rng.normal(0, 10, n)
    sell = rng.normal(0, 10, n)
    action = pd.array([1] * n if acted else [None] * n, dtype="Int8")
    return pd.DataFrame({
        "delivery_ts": ts, "knownby_ts": pd.Timestamp(knownby, tz="UTC"),
        "area": area, "action": action,
        "action_pnl": buy if acted else np.nan,
        "always_buy": buy, "always_sell": sell,
        "ideal": np.maximum(buy, sell), "devil": np.minimum(buy, sell),
        "schema_version": 6, "methodology_version": 3})


def _collect(frame, floor=None):
    """The public floor is patched too: these tests are built on August days,
    which the real settings now keep out of v1."""
    S_read, S_floor = S.read_results, S.cfg.public_from
    S.read_results = lambda kind: frame
    S.cfg.public_from = lambda: floor
    try:
        return S.collect(_Log())
    finally:
        S.read_results, S.cfg.public_from = S_read, S_floor


def test_nothing_before_the_public_floor_reaches_a_reader():
    """Days before publication.public_from are drafts. They are dropped whole,
    and the day that survives keeps the pass ordinal it had over the full
    table — the cut must not renumber anything."""
    from datetime import date
    frame = pd.concat([
        _day("2026-09-16", "2026-09-17T09:40:00", seed=1),
        _day("2026-09-17", "2026-09-18T09:40:00", seed=2),
        _day("2026-09-17", "2026-10-20T09:40:00", seed=3),
    ], ignore_index=True)
    rows = _collect(frame, floor=date(2026, 9, 17))
    assert list(pd.to_datetime(rows["delivery_day"]).dt.strftime("%Y-%m-%d")) == ["2026-09-17"]
    assert int(rows.iloc[0]["calc_iteration"]) == 2


def test_the_oracles_and_the_passives_are_the_same_two_numbers():
    """
    Per PTU, ideal is the max of the two passive positions and adverse the min,
    so over any day `extractable + adverse == passive_buy + passive_sell`
    exactly. It is the cheapest possible check that the public names were not
    crossed on the way out: swap two of them and this breaks at once.
    """
    rows = _collect(_day("2026-08-05", "2026-08-06T09:40:00"))
    assert len(rows) == 1
    r = rows.iloc[0]
    assert abs((r["extractable_value"] + r["adverse_value"])
               - (r["passive_da_buy_value"] + r["passive_da_sell_value"])) < 1e-9


def test_a_day_collapses_to_its_newest_pass():
    """The store keeps every version; the shop window shows one. Which one it is
    has to be stated, not guessed, so `pass` and `calc_iteration` are columns."""
    first = _day("2026-08-05", "2026-08-06T09:40:00", seed=1)
    final = _day("2026-08-05", "2026-09-20T08:00:00", seed=2)
    rows = _collect(pd.concat([first, final], ignore_index=True))

    assert len(rows) == 1, "one delivery day is one row, whatever its history"
    r = rows.iloc[0]
    assert r["pass"] == "as-final" and r["calc_iteration"] == 2
    assert pd.Timestamp(r["knownby_ts"]) == pd.Timestamp("2026-09-20T08:00:00", tz="UTC")

    only_first = _collect(first).iloc[0]
    assert only_first["pass"] == "as-known" and only_first["calc_iteration"] == 1


def test_a_day_with_no_decision_keeps_its_market_values():
    """RO's first eleven days: graded on the market, empty where we decided
    nothing. An empty cell is the honest rendering of that, not a zero."""
    rows = _collect(_day("2026-08-05", "2026-08-06T09:40:00", acted=False))
    r = rows.iloc[0]
    assert pd.isna(r["trailing_bias_value"]) and r["ptu_with_action"] == 0
    assert not pd.isna(r["extractable_value"])
    assert ",,96,0,as-known," in S._csv(rows), "an empty cell, not a zero"


def test_a_short_day_says_so_in_a_column():
    """A day the source left short sums to less for a reason that has nothing to
    do with the market, so the denominator travels with the number."""
    rows = _collect(_day("2026-08-05", "2026-08-06T09:40:00", n=95))
    assert rows.iloc[0]["ptu"] == 95


def test_the_bytes_do_not_move_when_the_data_does_not():
    """No wall clock anywhere: a run that changes nothing writes nothing, so the
    daily backup commit does not record that time passed."""
    rows = _collect(_day("2026-08-05", "2026-08-06T09:40:00"))
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "NL_scores.csv"
        assert S._put(p, S._csv(rows)) is True
        assert S._put(p, S._csv(rows)) is False, "identical content, second write"


def test_the_flat_names_keep_the_zone_separable():
    """Zone codes carry hyphens of their own (DE-LU, IT-North), so the boundary
    is an underscore and it appears exactly once."""
    for zone in ("NL", "DE-LU", "IT-North"):
        for name in (f"{zone}_scores.csv", f"{zone}_scores-90d.csv",
                     f"{zone}_scores-latest.csv"):
            assert name.count("_") == 1
            assert name.split("_", 1)[0] == zone
    assert "_" not in "scores-latest.csv", "no prefix means every zone"


def _fc(day: str, knownby: str, actions, area: str = "NL",
        reason=None) -> pd.DataFrame:
    """One forecast for a delivery day: one action per PTU (1, -1 or None)."""
    from src.period import market_day_bounds
    start = pd.Timestamp(market_day_bounds(pd.Timestamp(day).date(), area)[0])
    n = len(actions)
    return pd.DataFrame({
        "delivery_ts": pd.date_range(start, periods=n, freq="15min", tz="UTC"),
        "knownby_ts": pd.Timestamp(knownby, tz="UTC"), "area": area,
        "action": pd.array(actions, dtype="Int8"),
        "data_through": pd.Timestamp(day, tz="UTC") - pd.Timedelta(days=2),
        "skip_reason": pd.array([reason] * n, dtype="str"),
        "schema_version": 6, "methodology_version": 3})


def _collect_fc(frame, floor=None):
    S_read, S_floor = S.read_results, S.cfg.public_from
    S.read_results = lambda kind: frame
    S.cfg.public_from = lambda: floor
    try:
        return S.collect_forecasts(_Log())
    finally:
        S.read_results, S.cfg.public_from = S_read, S_floor


def test_a_forecast_counts_its_decision_in_ptu_and_the_counts_add_up():
    """decided, partly decided, declined — the three states the site must
    state, read from three counters that always sum to ptu."""
    frame = pd.concat([
        _fc("2026-09-20", "2026-09-19T09:40:00", [1] * 48 + [-1] * 48),
        _fc("2026-09-21", "2026-09-20T09:40:00", [1] * 40 + [None] * 56),
        _fc("2026-09-22", "2026-09-21T09:40:00", [None] * 96,
            reason="input_incomplete:2026-09-20:DA"),
    ], ignore_index=True)
    rows = _collect_fc(frame).set_index(
        pd.to_datetime(_collect_fc(frame)["delivery_day"]).dt.strftime("%m-%d"))
    for _, r in rows.iterrows():
        assert r["ptu_buy"] + r["ptu_sell"] + r["ptu_no_action"] == r["ptu"]
    assert (rows.loc["09-20", "ptu_buy"], rows.loc["09-20", "ptu_sell"]) == (48, 48)
    assert rows.loc["09-21", "ptu_no_action"] == 56
    assert rows.loc["09-22", "ptu_no_action"] == 96
    assert rows.loc["09-22", "skip_reason"] == "input_incomplete:2026-09-20:DA"
    assert rows.loc["09-20", "skip_reason"] == "", "unformed is not refused"


def test_a_forecast_with_nothing_to_cite_has_no_row():
    """No signal on any PTU and nothing refused: publish writes no file, so v1
    has no row — a row could not name the document behind it."""
    rows = _collect_fc(_fc("2026-09-20", "2026-09-19T09:40:00", [None] * 96))
    assert rows.empty


def test_forecasts_obey_the_public_floor_and_the_newest_wins():
    frame = pd.concat([
        _fc("2026-09-16", "2026-09-15T09:40:00", [1] * 96),
        _fc("2026-09-17", "2026-09-16T09:40:00", [1] * 96),
        _fc("2026-09-17", "2026-09-16T11:00:00", [-1] * 96),
    ], ignore_index=True)
    from datetime import date
    rows = _collect_fc(frame, floor=date(2026, 9, 17))
    assert len(rows) == 1
    r = rows.iloc[0]
    assert r["ptu_sell"] == 96, "the newest forecast of the day wins"
    csv = S._csv_forecasts(rows)
    assert "NL,2026-09-17,2026-09-16T11:00:00Z,96,0,96,0,2026-09-15T00:00:00Z,," in csv


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
