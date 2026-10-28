"""
NL daily workflow, compute step: forecast(current_ts) / score(current_ts).

Thin orchestration only — read the store, call benchmarks_core, append a table.
No maths lives here; the maths is the SAME code the report calls, so backtest
and live differ in exactly one value: current_ts.

    forecast(current_ts)  what to do on the DA auction closing today, for the
                          delivery day that auction serves (tomorrow, local)
    score(current_ts)     what an already-issued action actually earned, once
                          the delivery day has settled

Time entities — there are only TWO in this project, and these tables reuse them:
    delivery_ts   the 15-min PTU being traded / settled
    knownby_ts    when the row became known
For a price row knownby_ts is when ENTSO-E published it. For a row WE compute,
knownby_ts IS current_ts: the moment we knew it. So our own results are the same
shape as our inputs, and as_of() reads them the same way — which is how score()
guarantees it only ever grades an action that had already been issued.

Output (append-only, parquet, key = (area, knownby_ts, delivery_ts)):
    data/processed/results/forecasts.parquet
        delivery_ts, knownby_ts, area, action, sig_buy, sig_sell,
        n_obs, data_through, n_days, skip_reason,
        schema_version, methodology_version
    data/processed/results/scores.parquet
        delivery_ts, knownby_ts, area, action, action_pnl,
        always_buy, always_sell, ideal, devil, n_days,
        schema_version, methodology_version

The two version numbers are stamped into the ROW, at the moment of computation,
and the publication layer renders them from there. They used to be read from
settings.json at RENDER time — which meant that raising a version by hand
relabelled the whole published history with it. See `_version_cols` below.

`action` is numeric by design (+1 buy, -1 sell, later 0 idle, and a continuous
score in [-1, +1] once the harder maths arrives) — never a word, so nothing
downstream branches on strings. <NA> means no action was issued, which is
different from a decision to sit out. `skip_reason` says WHICH of the two kinds
of <NA> a row is: empty means the signal simply did not form (a short trailing
window), set means the forecast was REFUSED because its input day was not whole.
Both are honest absences; only the second one is our own doing.

Re-running the same current_ts overwrites its own rows rather than adding new
ones: the computation is deterministic, so a second run must reproduce the first.
A difference is a bug, not a new version.

Run (from repo root):
    uv run python -m src.workflow_compute forecast [2026-08-04T09:30]
    uv run python -m src.workflow_compute score    [2026-08-04T09:30]
"""

import os
import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

import src.benchmarks_core as C
import src.config as cfg
from src.parse_fx import fx_as_of, rate_due
from src.period import auction_gate, market_day_bounds, market_tz
from src.workflow import (PROCESSED, ROOT, for_each_area, load_store,
                          setup_logging, store_dir)

# The RESULTS are one table per kind, across all zones — not one per zone like
# the store. The split is by what the thing is. The store is machinery: 235k
# rows per zone per product, in a file parquet rewrites whole on every append,
# so twenty-five zones in one file would be six million rows rewritten daily for
# the sake of ninety-six new ones. The results are the PRODUCT: small, already
# keyed by area, and the entire point of the project is comparing zones — which
# a reader should not have to do by opening twenty-five files.
RESULTS_DIR = os.path.join(ROOT, cfg.path("results"))
FORECASTS = os.path.join(RESULTS_DIR, "forecasts.parquet")
SCORES = os.path.join(RESULTS_DIR, "scores.parquet")
# `area` belongs in the key even while there is only one zone. Without it, an FR
# row and an NL row for the same PTU computed at the same instant are the same
# key, and _append's drop_duplicates would discard one WITHOUT a word — a loss
# that surfaces months later, if at all. Today it changes nothing; that is the
# point of fixing it today.
KEY = ["area", "knownby_ts", "delivery_ts"]

VERSION_COLS = ["schema_version", "methodology_version"]
# The settings that reach the published <parameters> header. They are stamped
# into the row for the same reason the version numbers are: `publish` re-renders
# every historical group on every run, so a value read from settings at RENDER
# time silently rewrites what past files claim the moment the setting changes.
# This was documented on 2026-09-01 as the version stamp's twin and left open;
# moving run_time_utc from 09:30 to 09:40 is what finally came for it.
# `causality` is not here: it is a statement about the method, not a tunable.
PARAM_COLS = ["n_days_trailing_bias_default", "final_day_of_month", "run_time_utc"]
# The publisher is the third of the same kind, and the last one still read at
# render time. Identity was kept OUT of the structure on purpose — a name and a
# domain get rebranded and move, while element names are frozen the moment
# anything is published — but "can change freely" was only ever true of the
# FORMAT. Read from settings while rendering, a rename would rewrite what every
# past file claims, and the frozen-file check would refuse the lot. So the
# publisher a file was published BY belongs to the file, like its version and
# its parameters.
PUBLISHER_COLS = ["publisher_name", "publisher_url"]
FORECAST_COLS = ["delivery_ts", "knownby_ts", "area", "action", "sig_buy",
                 "sig_sell", "n_obs", "data_through", "n_days",
                 "skip_reason"] + VERSION_COLS + PARAM_COLS + PUBLISHER_COLS
SCORE_COLS = ["delivery_ts", "knownby_ts", "area", "action", "action_pnl",
              "always_buy", "always_sell", "ideal", "devil",
              "n_days", "skip_reason"] + VERSION_COLS + PARAM_COLS + PUBLISHER_COLS


def _version_cols() -> dict:
    """
    The two published version numbers, frozen into the row as it is computed.

    They belong to the RESULT, not to the current configuration. Reading them
    from settings at render time looked equivalent and was not: `publish`
    re-renders every historical group on every run, so on 2026-09-01, the first
    morning after the numbers were raised by hand, all sixty already-published
    files differed from their own re-render and the frozen-file check refused
    every one of them. Worse than the noise is what a `--force` would have done
    there — relabelled July's numbers with September's methodology, which is
    precisely the claim these two numbers exist to make impossible.

    A row carries the version it was computed under, forever. An old file then
    re-renders byte-identical, and a bump touches nothing but what comes after.
    """
    v = cfg.versions()
    return {"schema_version": v["schema"], "methodology_version": v["methodology"]}


def _param_cols() -> dict:
    """The tunables in force at computation time, frozen into the row."""
    p = cfg.publication_parameters()
    return {c: p[c] for c in PARAM_COLS}


def _publisher_cols() -> dict:
    """Who published this row, frozen at the moment it was computed."""
    pub = cfg.publisher()
    return {"publisher_name": pub["name"], "publisher_url": pub.get("url", "")}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_fx(current_ts: pd.Timestamp) -> pd.DataFrame:
    """Daily FX table as it stood at current_ts (parse_fx.fx_as_of): rates
    known before it, the last known rate carried onto today's not-yet-due
    dates. EUR is a unit and never looked up (build_matrix)."""
    fx = pd.read_feather(os.path.join(PROCESSED, "fx_daily.feather"))
    return fx_as_of(fx, current_ts)


def _as_of_legs(current_ts: pd.Timestamp, area: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two price legs as they stood at current_ts, still separate.

    Kept apart so a refusal can say WHICH leg was missing. The check itself is
    on the joined matrix and has to be — a PTU is only usable when both sides of
    it exist — but the joined matrix cannot tell a reader whether the hole was
    in day-ahead or in imbalance, and that is the first thing anyone asks.
    """
    da, im = load_store(area)
    return C.as_of(da, current_ts), C.as_of(im, current_ts)


def _matrix_from_legs(da: pd.DataFrame, im: pd.DataFrame,
                      current_ts: pd.Timestamp, area: str | None = None,
                      log=None) -> pd.DataFrame:
    if da.empty or im.empty:
        raise ValueError(f"Store is empty as of {current_ts} "
                         f"(DA rows={len(da)}, IM rows={len(im)})")
    # The live path DROPS what it cannot value instead of raising: a missing
    # rate for one day must not take the zone down, it must make that day
    # incomplete — and the gates below already know what incomplete means.
    m = C.build_matrix(da, im, _load_fx(current_ts), on=["delivery_ts"], on_missing_fx="drop")
    missing = m.attrs.get("fx_missing", [])
    if missing and log is not None:
        for d in sorted({t[2] for t in missing}):
            # "Not yet" is not "missing". The rate for a date appears on the
            # afternoon of that date, so today's PTU — which a zone publishing
            # imbalance intraday puts in the store all morning — have no rate
            # yet and are not supposed to. Those PTU belong to a day that is not
            # settled and cannot be scored anyway; the scorer already says so.
            if not rate_due(date.fromisoformat(d), current_ts.to_pydatetime()):
                continue
            which = ", ".join(f"{leg} in {cur}" for leg, cur, dd in missing if dd == d)
            log.warning("%s: no FX rate for %s (%s), and the ECB published for "
                        "that date hours ago — those PTU cannot be valued; "
                        "nothing is filled in. Run: uv run python -m src.fx",
                        area or "?", d, which)
    return C.add_pnl(m)


def _matrix_as_of(current_ts: pd.Timestamp, area: str, log=None) -> pd.DataFrame:
    """The settled DA/IM matrix as it stood at current_ts, with pnl attached."""
    return _matrix_from_legs(*_as_of_legs(current_ts, area), current_ts, area, log)


def _target_grid(current_ts: pd.Timestamp, area: str) -> pd.DatetimeIndex:
    """
    The delivery PTUs of the day the auction closing TODAY serves.

    Anchored to the market's local calendar, not to UTC dates: a market day runs
    local midnight to local midnight, so DST days are 92 or 100 PTUs. Never 96
    by assumption.
    """
    local_day = current_ts.astimezone(market_tz(area)).date()
    start, end = market_day_bounds(local_day + timedelta(days=1), area)
    return pd.date_range(start, end, freq="15min", inclusive="left")


def _append(path: str, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Append rows, de-duped on (knownby_ts, delivery_ts), newest write wins."""
    # reindex, not df[cols]: a frame written before a column existed is missing
    # it, not broken, and the column list grows over time (skip_reason,
    # publisher_name). Selecting raised a KeyError on exactly the old rows a
    # migration is there to bring forward.
    df = df.reindex(columns=cols)
    if os.path.exists(path):
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df = (df.drop_duplicates(KEY, keep="last")
            .sort_values(KEY)
            .reindex(columns=cols)      # a file written before a column existed
            .reset_index(drop=True))    # must not decide the column order
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def market_day(delivery_ts: pd.Series, area: str) -> pd.Series:
    """The market day a PTU belongs to (local midnight to local midnight)."""
    tz = market_tz(area)
    return delivery_ts.dt.tz_convert(tz).dt.normalize().dt.tz_localize(None)


def final_ts(day: pd.Timestamp, final_day: int) -> pd.Timestamp:
    """
    The instant after which a delivery day's imbalance is taken as settled for
    good: `final_day` of the FOLLOWING month, at UTC midnight.
    """
    nxt = pd.Timestamp(day).replace(day=1) + pd.DateOffset(months=1)
    return pd.Timestamp(nxt.replace(day=final_day), tz="UTC")


def score_reason(day: pd.Timestamp, scorable: bool, first_done: bool,
                 final_done: bool, current_ts: pd.Timestamp,
                 final_day: int) -> str | None:
    """
    Should this delivery day be scored right now, and on what grounds?

    Exactly TWO passes per day, and no more:
      as-known  — the first time the day can be scored at all;
      as-final  — once, after its revision window closes.

    Anything in between would be re-grading a number that has not moved. The
    naive "score everything settled, every day" grows quadratically: 96 rows on
    day one, 192 on day two, and about six million after a year, all but a
    handful of them identical to the row before.

    `scorable` means the market day has ENDED and left something behind. It used
    to mean COMPLETE — every PTU present — and that was right while the only
    reason for a short day was that imbalance was still arriving. RO broke the
    assumption: it publishes intraday and on 2026-09-25 simply skipped the
    quarter-hour at 10:00 UTC, which never came in any later poll. A day that is
    permanently 95/96 would never be scored under the old rule, and would say
    "not settled yet" in the log for ever.

    So a short day is scored on the PTU it HAS, and the published file carries
    the count (`ptu="95"`), which is the honest statement: this is the sum over
    the quarter-hours that exist. The owner's call, 2026-09-26. The accepted
    cost is that an as-known pass taken while a day was still filling can differ
    from its as-final pass — and that difference is information, recorded in
    two files, not a defect.
    """
    if not scorable:
        return None
    if not first_done:
        return "as-known"
    if not final_done and current_ts >= final_ts(day, final_day):
        return "as-final"
    return None


def _parse_ts(text: str | None) -> pd.Timestamp:
    """CLI timestamp -> tz-aware UTC. No argument means now."""
    if text is None:
        return pd.Timestamp(datetime.now(timezone.utc))
    ts = pd.Timestamp(text)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


# --------------------------------------------------------------------------- #
# The input window
# --------------------------------------------------------------------------- #
# A forecast for tomorrow stands on a window of n_days observations per UTC
# hour, built by models/features.py and tied to TIMESTAMPS: d1 is today's hour
# if it is whole and known (zones that publish imbalance intraday), otherwise
# yesterday's; d2, d3 are exactly one and two days before d1. A hole anywhere
# is an <NA> in a named cell, and the hour gets no action.
#
# What this replaced, and why (kept because the failures are the reasons):
# signal_for took the last n_days observations from the END of whatever was in
# the store. A missing day did not shorten the window, it SHIFTED it, with
# n_obs still reading n_days — 2026-08-30 (08-29 absent) and 2026-09-15 (09-13
# absent in the middle). The per-hour gate added on 2026-09-26 caught holes,
# but not surplus: a partly published hour of TODAY entered the window as d1
# (found 2026-10-27, DE_LU / SE1 / SI 09:00 UTC daily).
#
# One missing day still refuses n_days forecasts in a row, because it sits in
# that many consecutive windows. That is the accepted price: no interpolation.
#
# A refused day is refused for good. _target_grid always aims at "tomorrow
# relative to the run", so no later run can target it again. The one way back
# is a manual backfill and a re-run on the SAME day, before the gate — which
# is what the refusal writes into the log, as a command to run.
def input_day_for(current_ts: pd.Timestamp, area: str) -> date:
    """The market day a forecast stands on: yesterday, market-local."""
    return current_ts.astimezone(market_tz(area)).date() - timedelta(days=1)


def day_completeness(m: pd.DataFrame, day: date, area: str) -> tuple[int, int]:
    """
    (PTU of `day` present in the matrix, PTU it should have).

    The matrix is the INNER join of DA and IM, so a PTU counted here is present
    on BOTH legs and one count answers both questions. The expected number comes
    from the market day's own bounds, never from a constant: a DST day is 92 or
    100 PTU, and a hard-coded 96 would call it incomplete twice a year.
    """
    start, end = market_day_bounds(day, area)
    expect = int((pd.Timestamp(end) - pd.Timestamp(start)) / pd.Timedelta("15min"))
    have = int(((m["delivery_ts"] >= start) & (m["delivery_ts"] < end)).sum())
    return have, expect


def signal_window(need: date, n_days: int) -> list[date]:
    """The days the signal is supposed to stand on: `need` and the n_days-1
    before it, newest first. Calendar days, contiguous, no gaps allowed."""
    return [need - timedelta(days=i) for i in range(n_days)]


def hour_grid(day: date, area: str) -> pd.Series:
    """How many PTU each UTC hour of one market day is supposed to hold.

    Counted from the market day's own grid, never assumed to be four: on a
    daylight-saving day one UTC hour-of-day occurs twice or not at all, and a
    hard-coded 4 would refuse a good hour twice a year.
    """
    start, end = market_day_bounds(day, area)
    grid = pd.date_range(start, end, freq="15min", inclusive="left")
    return pd.Series(1, index=grid).groupby(grid.hour).size()


def leg_completeness(leg: pd.DataFrame, day: date, area: str) -> int:
    """How many PTU of `day` one leg holds on its own."""
    start, end = market_day_bounds(day, area)
    return int(((leg["delivery_ts"] >= start) & (leg["delivery_ts"] < end)).sum())


def _no_forecast(current_ts: pd.Timestamp, area: str, ptus: pd.DatetimeIndex,
                 m: pd.DataFrame, reason: str, detail: str,
                 n_days: int, log, persist: bool) -> pd.DataFrame:
    """
    Record that no action was issued for the target day, and why.

    One row per PTU with action <NA>, in forecasts.parquet rather than in a
    table of its own: a delivery day should have ONE history in ONE place. If a
    manual backfill lets a real forecast follow later the same day, the two
    passes then number together — calc_iteration 1 refused, 2 issued — instead
    of being split across two files a reader has to join by hand. score() drops
    <NA> actions, so nothing downstream can mistake this for a decision.

    `reason` is the published skip_reason; `detail` is the sentence for the log.
    Two callers, two refusals: the evidence is not all there, or the auction has
    already closed. Both say the same thing in the end — we do not publish a
    decision we could not have made.
    """
    n = len(ptus)
    out = (pd.DataFrame({"delivery_ts": ptus})
             .assign(knownby_ts=current_ts, area=area,
                     action=pd.array([pd.NA] * n, dtype="Int8"),
                     sig_buy=np.nan, sig_sell=np.nan,
                     n_obs=pd.array([pd.NA] * n, dtype="Int64"),
                     data_through=m["delivery_ts"].max().floor("h"),
                     n_days=n_days, skip_reason=reason,
                     **_version_cols(), **_param_cols(), **_publisher_cols()))
    if persist:
        _append(FORECASTS, out, FORECAST_COLS)
    log.warning("forecast %s @ %s: NO FORECAST — %s", area, current_ts, detail)
    return out[FORECAST_COLS]


def _recovery(short: list[str], days: list[date], area: str) -> str:
    """The command that would fill the hole — which depends on which leg it is.

    Named zone, not all of them: with more than one zone a bare `update` would
    re-request every zone to fix one, and at twenty-five that is not a fix, it
    is a morning.
    """
    steps = []
    if "FX" in short:
        steps.append("uv run python -m src.fx")
    if set(short) - {"FX"} or not short:
        steps.append(f"uv run python -m src.update --from {min(days)} "
                     f"--to {max(days)} {area}")
    return ("refill and re-run: " + " && ".join(steps)
            + " && uv run python -m src.workflow_compute forecast")


def _refuse_incomplete(need: date, missing: list[tuple[date, int, int]], area: str,
                       n_days: int, da, im) -> tuple[str, str]:
    """skip_reason and log sentence for a window that is not whole."""
    target_day = need + timedelta(days=2)
    days = [d for d, _, _ in missing]

    # Which leg is short, per missing day. Legs matter: the 09-13 gap was in
    # DAY-AHEAD while imbalance was complete, and "0 of 96" alone sent the owner
    # digging through the previous day's log to find that out.
    short, per_day = [], []
    if da is not None and im is not None:
        for d, have, expect in missing:
            n_da = leg_completeness(da, d, area)
            n_im = leg_completeness(im, d, area)
            legs = [g for g, v in (("DA", n_da), ("IM", n_im)) if v < expect]
            # Both price legs whole and the day still short: the prices are
            # there and could not be VALUED. That is the exchange rate — the
            # third leg, which only a non-euro zone has.
            if not legs and have < expect:
                legs = ["FX"]
            per_day.append(f"{d:%m-%d}: DA {n_da}/{expect}, IM {n_im}/{expect}"
                           + (", FX missing" if legs == ["FX"] else ""))
            short += [g for g in legs if g not in short]

    reason = ("input_incomplete:" + ",".join(f"{d:%Y-%m-%d}" for d in days)
              + (f":{','.join(short)}" if short else ""))
    window = signal_window(need, n_days)
    detail = (
        f"delivery day {target_day} — the signal needs {window[-1]}..{window[0]} "
        f"whole ({n_days} days) and {', '.join(f'{d:%Y-%m-%d}' for d in sorted(days))} "
        f"{'is' if len(days) == 1 else 'are'} not. {' | '.join(per_day)}. "
        f"This delivery day cannot be recovered by a later run. To recover it, "
        f"{_recovery(short, days, area)} BEFORE the auction gate")
    return reason, detail


def _short_legs(bad: dict, area: str, da, im) -> list[str]:
    """Which leg was short across the refused hours — DA, IM or FX."""
    legs: list[str] = []
    if da is None or im is None:
        return legs
    for hod, shorts in bad.items():
        for d, _, _ in shorts:
            start, end = market_day_bounds(d, area)
            in_hour = lambda leg: int((  # noqa: E731
                (leg["delivery_ts"] >= start) & (leg["delivery_ts"] < end)
                & (leg["delivery_ts"].dt.hour == hod)).sum())
            expect = int(hour_grid(d, area)[hod])
            n_da, n_im = in_hour(da), in_hour(im)
            found = [g for g, v in (("DA", n_da), ("IM", n_im)) if v < expect]
            # Both price legs whole and the hour still short: the prices exist
            # and could not be VALUED — the exchange rate, the third leg.
            legs += [g for g in (found or ["FX"]) if g not in legs]
    return legs


def _hour_reason(bad: dict, legs: list[str]) -> str:
    """The per-row skip_reason for an hour the window could not support.

    Same three-part shape the day-level refusal uses, so one parser reads both:
    kind, the days that were short, the legs. WHICH hour it is needs no saying
    — the row is that hour.
    """
    days = sorted({d for shorts in bad.values() for d, _, _ in shorts})
    return ("hour_incomplete:" + ",".join(f"{d:%Y-%m-%d}" for d in days)
            + (f":{','.join(legs)}" if legs else ""))


def _hour_detail(bad: dict) -> str:
    return " | ".join(
        f"{hod:02d}:00 UTC short in " + ", ".join(
            f"{d:%m-%d} ({n}/{e})" for d, n, e in shorts)
        for hod, shorts in sorted(bad.items()))


def _bad_cells(x: pd.DataFrame, area: str,
               n_days: int) -> dict[int, list[tuple[date, int, int]]]:
    """UTC hour-of-day -> (market day, PTU present, 4) for every SHORT window
    cell (fewer than 4 quarter-hours) of an unformed matrix row. A whole cell
    in an unformed row is a consequence, not a cause, and is not listed."""
    bad: dict[int, list[tuple[date, int, int]]] = {}
    for _, r in x[x["action"].isna()].iterrows():
        hod = int(r["delivery_ts"].hour)
        for k in range(1, n_days + 1):
            if int(r[f"ptu_d{k}"]) < 4:
                ts = r[f"delivery_ts_d{k}"]
                day = ts.tz_convert(market_tz(area)).date()
                cell = (day, int(r[f"ptu_d{k}"]), 4)
                if cell not in bad.setdefault(hod, []):
                    bad[hod].append(cell)
    return bad


def _refuse_after_gate(target_day: date, gate: pd.Timestamp,
                       current_ts: pd.Timestamp) -> tuple[str, str]:
    """skip_reason and log sentence for a decision that is simply too late."""
    late = (current_ts - gate).total_seconds() / 60
    detail = (
        f"delivery day {target_day} — the day-ahead auction for it closed at "
        f"{gate:%Y-%m-%dT%H:%M:%SZ}, {late:.0f} min ago. The numbers would have "
        f"been causal, but a bid nobody could place earns nothing, so no "
        f"decision is issued. Nothing to recover: this window is gone.")
    return "after_gate", detail


# --------------------------------------------------------------------------- #
# forecast
# --------------------------------------------------------------------------- #
def forecast(current_ts: pd.Timestamp, area: str,
             log=None, persist: bool = True) -> pd.DataFrame:
    """Decide the DA action for every PTU of the next delivery day."""
    log = log or setup_logging()
    n_days = cfg.n_days(area)

    da, im = _as_of_legs(current_ts, area)
    m = _matrix_from_legs(da, im, current_ts, area, log)
    ptus = _target_grid(current_ts, area)

    need = input_day_for(current_ts, area)
    target_day = need + timedelta(days=2)

    # Could this decision still be placed? Checked FIRST, because no amount of
    # data rescues a closed auction.
    gate = pd.Timestamp(auction_gate(target_day))
    if current_ts >= gate:
        reason, detail = _refuse_after_gate(target_day, gate, current_ts)
        return _no_forecast(current_ts, area, ptus, m, reason, detail,
                            n_days, log, persist)

    # The signal comes from the MATRIX (models/features.py): one row per
    # delivery hour, the window tied to timestamps — d1 is today's hour if it
    # is whole, else yesterday's; d2, d3 exactly one and two days before d1.
    # A hole anywhere in the window is an <NA> in a named cell, and the rule
    # (models/trailing_bias.py) then issues no action for that hour. The same
    # two functions build the training matrix, so live and backtest share them.
    # (Imported here: models.features imports this module.)
    from models.features import matrix_from
    from models.trailing_bias import decide
    x = decide(matrix_from(m, current_ts, area, fresh=True), n_days)

    # Hours the window could not support, as {hod: [(market day, ptu, 4)]}
    # from the cells that are <NA> — the shape the reason writers read.
    bad = _bad_cells(x, area, n_days)
    if bad and x["action"].isna().all():
        days = sorted({d for shorts in bad.values() for d, _, _ in shorts})
        per_day = [(d, *day_completeness(m, d, area)) for d in days]
        reason, detail = _refuse_incomplete(need, per_day, area, n_days, da, im)
        return _no_forecast(current_ts, area, ptus, m, reason, detail,
                            n_days, log, persist)

    sig = x[["delivery_ts", "action", "sig_buy", "sig_sell", "n_obs",
             "data_through"]].rename(columns={"delivery_ts": "hour_ts"})
    out = (pd.DataFrame({"delivery_ts": ptus})
             .assign(hour_ts=lambda d: d["delivery_ts"].dt.floor("h"))
             .merge(sig, how="left", on="hour_ts")
             .assign(knownby_ts=current_ts, area=area, n_days=n_days,
                     skip_reason=pd.NA, **_version_cols(), **_param_cols(),
                     **_publisher_cols()))

    # The PARTIAL case — some hours decided, some not.
    if bad:
        mask = out["action"].isna()
        legs = _short_legs(bad, area, da, im)
        out.loc[mask, "skip_reason"] = _hour_reason(bad, legs)
        log.warning("forecast %s @ %s: %d of %d hours refused — %s. The rest of "
                    "the day is decided; an hour needs its four quarters in "
                    "every day of its window, and these did not have them.",
                    area, current_ts, len(bad), len(x), _hour_detail(bad))

    if persist:
        _append(FORECASTS, out, FORECAST_COLS)
    formed = int(out["action"].notna().sum())
    log.info("forecast %s @ %s: %d PTU (%d with a signal), target %s..%s, "
             "data through %s, n_days=%d", area, current_ts, len(out), formed,
             ptus.min(), ptus.max(), out["data_through"].max(), n_days)
    # Only the genuinely UNFORMED ones belong in this warning. An hour withdrawn
    # by the gate above already has its own line and its own reason; counting it
    # here again would blame a short trailing window for a hole in the source.
    unformed = int((out["action"].isna() & out["skip_reason"].isna()).sum())
    if unformed:
        log.warning("forecast %s @ %s: %d PTU without a formed signal "
                    "(trailing window shorter than %d observations)",
                    area, current_ts, unformed, n_days)
    return out[FORECAST_COLS]


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def _forecasts_as_of(current_ts: pd.Timestamp, area: str) -> pd.DataFrame:
    """Our own forecast table as it stood at current_ts — one row per PTU.

    The same as_of we apply to prices, applied to our own output: only actions
    issued before now, and per delivery_ts the latest pass. That is what makes
    score() incapable of grading a decision it has not yet made.
    """
    if not os.path.exists(FORECASTS):
        return pd.DataFrame(columns=FORECAST_COLS + ["day"])
    fc = pd.read_parquet(FORECASTS)
    fc = C.as_of(fc.loc[fc["area"] == area], current_ts)
    if fc.empty:
        return fc.reindex(columns=list(fc.columns) + ["day"])
    return fc.assign(day=market_day(fc["delivery_ts"], area))


# --------------------------------------------------------------------------- #
# Scoring is driven by the STORE, not by the forecast table
# --------------------------------------------------------------------------- #
# It used to be driven by the forecasts: the first thing score() did was drop
# every row whose action was <NA>, so a refused day vanished before anything was
# computed. That threw away four benchmarks to protect one. always_buy,
# always_sell, ideal and devil are functions of the day's DA and IM prices and
# of nothing else — they say what the market did, not what we decided — and they
# are computable for every settled day, including the days our own input gate
# refused to forecast. Only action_pnl needs an action, because only it asks
# what OUR rule earned.
#
# Found on 2026-09-18: delivery days 09-15 and 09-17 were absent from the
# published series altogether, each because a hole in day-ahead on 09-13 sat in
# its signal window. The prices for both days were in the store the whole time.
#
# So the candidate days are: every day the store holds, from `scoring.score_from`
# onwards, PLUS every day an action was issued for. The second half is not
# decoration — it is what keeps the days that predate the floor coming back for
# their as-final pass.
def score(current_ts: pd.Timestamp, area: str,
          log=None, persist: bool = True) -> pd.DataFrame:
    """Grade every settled delivery day: the four market benchmarks always, the
    trailing-bias one only where an action was actually issued."""
    log = log or setup_logging()

    m = _matrix_as_of(current_ts, area, log)
    m = m.assign(day=market_day(m["delivery_ts"], area))
    fc = _forecasts_as_of(current_ts, area)

    acted = set(fc.loc[fc["action"].notna(), "day"]) if len(fc) else set()
    days = pd.DatetimeIndex(pd.unique(m["day"]))
    floor = cfg.score_from()
    if floor is not None:
        days = days[days >= pd.Timestamp(floor)]
    days = sorted(set(days) | acted)
    if not days:
        log.info("score %s @ %s: nothing to score", area, current_ts)
        return pd.DataFrame(columns=SCORE_COLS)

    # (present, expected) per candidate day. Expected comes from the market
    # day's own bounds, so a DST day is 92 or 100 PTU and never a constant 96.
    settled = {d: day_completeness(m, d.date(), area) for d in days}
    # And whether the day is OVER. That is what makes it gradable now: a day
    # still being delivered is not short, it is unfinished.
    ended = {d: pd.Timestamp(market_day_bounds(d.date(), area)[1]) <= current_ts
             for d in days}
    final_day = cfg.final_day_of_month()

    # What has already been recorded, per delivery day.
    prior = pd.read_parquet(SCORES) if os.path.exists(SCORES) else None
    if prior is not None and len(prior):
        prior = prior.loc[prior["area"] == area].assign(
            day=lambda d: market_day(d["delivery_ts"], area))
        prior = prior.assign(
            after_final=[k >= final_ts(d, final_day)
                         for k, d in zip(prior["knownby_ts"], prior["day"])])
        first_done = set(prior["day"])
        final_done = set(prior.loc[prior["after_final"], "day"])
    else:
        first_done, final_done = set(), set()

    reasons = {}
    for day in days:
        have, _ = settled[day]
        r = score_reason(day, ended[day] and have > 0, day in first_done,
                         day in final_done, current_ts, final_day)
        if r:
            reasons[day] = r

    if not reasons:
        _log_waiting(log, area, current_ts, days, settled, first_done,
                     final_done, final_day, set())
        log.info("score %s @ %s: nothing to score", area, current_ts)
        return pd.DataFrame(columns=SCORE_COLS)

    todo = m.loc[m["day"].isin(reasons)]
    side = ["delivery_ts", "action", "n_days", "skip_reason"]
    if len(fc):
        j = todo.merge(fc.reindex(columns=side), how="left", on="delivery_ts",
                       validate="one_to_one")
    else:
        j = todo.assign(action=pd.NA, n_days=pd.NA, skip_reason=pd.NA)

    # benchmarks_core.action_pnl: +1 buy, -1 sell, 0 stay out (0.0), <NA> NaN.
    out = j.assign(action_pnl=C.action_pnl(j["action"], j["always_buy"], j["always_sell"]),
                   knownby_ts=current_ts, area=area,
                   **_version_cols(), **_param_cols(),
                     **_publisher_cols())

    if persist:
        _append(SCORES, out, SCORE_COLS)
    _log_scored(log, area, current_ts, out, reasons, settled)
    _log_waiting(log, area, current_ts, days, settled, first_done, final_done,
                 final_day, set(reasons))
    return out[SCORE_COLS]


def _days_list(days: list, cap: int = 5) -> str:
    """Dates on one line: all of them while they fit, then a count."""
    names = [str(d.date()) for d in days]
    if len(names) <= cap:
        return ", ".join(names)
    return ", ".join(names[:cap]) + f", and {len(names) - cap} more"


def _log_day(log, area: str, current_ts: pd.Timestamp, out: pd.DataFrame,
             day: pd.Timestamp, reason: str, settled: dict) -> None:
    """One scored delivery day, in full."""
    d = out.loc[out["day"] == day]
    have, expect = settled[day]
    short = "" if have == expect else f" of {expect} — the source skipped {expect - have}"
    if d["action"].notna().any():
        log.info("score %s @ %s: delivery day %s (%s), %d PTU%s, "
                 "action_pnl=%.2f EUR/MW, ideal=%.2f, devil=%.2f",
                 area, current_ts, day.date(), reason, len(d), short,
                 d["action_pnl"].sum(), d["ideal"].sum(), d["devil"].sum())
    else:
        why = d["skip_reason"].dropna().unique()
        log.info("score %s @ %s: delivery day %s (%s), %d PTU%s, NO ACTION was "
                 "issued (%s) — market values only, ideal=%.2f, devil=%.2f",
                 area, current_ts, day.date(), reason, len(d), short,
                 why[0] if len(why) else "no forecast row",
                 d["ideal"].sum(), d["devil"].sum())


def _log_scored(log, area: str, current_ts: pd.Timestamp, out: pd.DataFrame,
                reasons: dict, settled: dict) -> None:
    """
    Report the pass at CONSTANT size: one full day per reason, the rest counted.

    A run normally scores one day, and then a line each was exactly right. Two
    situations score many at once. A zone in SHADOW stores nothing, so every day
    since the floor comes back every morning — eighteen lines on 2026-09-25, ten
    on 09-26, and one more every day for as long as the zone stays in the dark.
    The as-final pass on the 20th takes a whole month in one go, which is the
    same shape and will arrive in every zone that goes live.

    Neither was telling a reader anything to act on, because the numbers are in
    the results table, which is where numbers belong. So within each reason the
    NEWEST day is reported in full and the others are summarised in one line:
    how many, over what span, how many carried an action, and how many were
    short of a full day. The last of those is the only per-day fact worth
    keeping — a source that skips PTUs is news — and a count preserves it.
    """
    by_reason: dict[str, list] = {}
    for day, reason in sorted(reasons.items()):
        by_reason.setdefault(reason, []).append(day)

    for reason, group in by_reason.items():
        _log_day(log, area, current_ts, out, group[-1], reason, settled)
        rest = group[:-1]
        if not rest:
            continue
        acted = sum(1 for d in rest
                    if out.loc[out["day"] == d, "action"].notna().any())
        n_short = sum(1 for d in rest if settled[d][0] != settled[d][1])
        tail = f", {n_short} short of a full day" if n_short else ""
        log.info("score %s @ %s: %d more delivery day(s) in the same %s pass, "
                 "%s..%s (%d with an action, %d market values only%s)",
                 area, current_ts, len(rest), reason,
                 rest[0].date(), rest[-1].date(), acted, len(rest) - acted, tail)


def _log_waiting(log, area: str, current_ts: pd.Timestamp, days: list,
                 settled: dict, first_done: set, final_done: set,
                 final_day: int, just_scored: set) -> None:
    """
    Say what is waiting and why — but at CONSTANT size.

    Days already scored and queued for their final pass accumulate forever, so a
    line each grew the log by one line per day per day. Those are summarised into
    one line naming only the earliest due date; the detail is derivable and was
    never read. A day being delivered RIGHT NOW stays itemised: there is one per
    zone at most, and it is a real question ("did the source deliver?"). A day
    that has not begun, and a day that ended empty, are counted instead — the
    first because tomorrow's day-ahead prices make tomorrow a candidate before
    it exists, the second because it waits for ever and saying so every morning
    for ever adds nothing.
    """
    queued, empty, unstarted = [], [], []
    for day in days:
        if day in just_scored:
            continue
        have, expect = settled[day]
        start, end = market_day_bounds(day.date(), area)
        if pd.Timestamp(start) > current_ts:
            unstarted.append(day)
        elif pd.Timestamp(end) > current_ts:
            log.info("score %s @ %s: delivery day %s still being delivered, "
                     "%d of %d PTU published", area, current_ts, day.date(),
                     have, expect)
        elif have == 0:
            # Ended and left nothing. A different thing from a short day, and
            # the only case that now waits for ever — correctly, since there is
            # nothing to grade.
            empty.append(day)
        elif day in first_done and day not in final_done:
            # A day that has had BOTH passes is finished, not waiting. Counting
            # it made the summary claim a due date that had already been served.
            queued.append(day)
    if unstarted:
        log.info("score %s @ %s: %d delivery day(s) not yet started (%s)",
                 area, current_ts, len(unstarted), _days_list(unstarted))
    if empty:
        log.info("score %s @ %s: %d delivery day(s) ended with nothing published "
                 "and cannot be graded: %s", area, current_ts, len(empty),
                 _days_list(empty))
    if queued:
        log.info("score %s @ %s: %d day(s) scored and waiting for the final pass,"
                 " earliest due %s", area, current_ts, len(queued),
                 min(final_ts(d, final_day) for d in queued).date())


def writes_results(area: str, persist: bool = True) -> bool:
    """Whether this zone's results are stored and published on this run.

    Two independent reasons not to write, and they multiply: the RUN may be a
    simulation (a hand-supplied current_ts), and the ZONE may be in shadow.
    """
    return persist and cfg.zone_mode(area) == "live"


def migrate_results(log=None) -> None:
    """
    Fold any per-zone results tables into the shared ones. Idempotent.

    They lived beside the store while there was one zone. Rows carry `area` and
    `_append` de-dupes on (area, knownby_ts, delivery_ts), so folding them in is
    a concatenation and cannot merge two zones' rows into one. The per-zone
    files are left on disk, like the pre-versioning feathers before them: a
    migration that deletes its own input cannot be checked afterwards.
    """
    log = log or setup_logging()
    for kind, target, cols in (("forecasts", FORECASTS, FORECAST_COLS),
                               ("scores", SCORES, SCORE_COLS)):
        moved = 0
        for area in cfg.AREAS:
            old = os.path.join(store_dir(area), f"{kind}.parquet")
            if not os.path.exists(old):
                continue
            df = pd.read_parquet(old)
            before = len(pd.read_parquet(target)) if os.path.exists(target) else 0
            after = len(_append(target, df, cols))
            moved += len(df)
            log.info("migrate results: %s %s: %d rows, table %d -> %d",
                     area, kind, len(df), before, after)
        if not moved:
            log.info("migrate results: %s already shared, nothing to move", kind)

    # The publisher into the row, for rows written before it lived there. Run
    # this BEFORE renaming the publisher in settings: it stamps what settings
    # say NOW, which is what those files were actually published by. Afterwards
    # a rename touches nothing but what comes after it — the same shape as the
    # version stamp of 2026-09-01 and the parameters of 2026-09-17.
    for kind, target, cols in (("forecasts", FORECASTS, FORECAST_COLS),
                               ("scores", SCORES, SCORE_COLS)):
        if not os.path.exists(target):
            continue
        df = pd.read_parquet(target)
        missing = [c for c in PUBLISHER_COLS if c not in df.columns
                   or df[c].isna().any()]
        if not missing:
            log.info("migrate publisher: %s already stamped", kind)
            continue
        pub = _publisher_cols()
        for c in PUBLISHER_COLS:
            df[c] = df[c].fillna(pub[c]) if c in df.columns else pub[c]
        _append(target, df, cols)
        log.info("migrate publisher: %s, %d rows stamped %s / %r",
                 kind, len(df), pub["publisher_name"], pub["publisher_url"])


def main() -> None:
    log = setup_logging()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "forecast"
    explicit = len(sys.argv) > 2
    current_ts = _parse_ts(sys.argv[2] if explicit else None)

    # A run with a hand-supplied current_ts is a SIMULATION: it asks what would
    # have been decided at some past instant. It computes and reports, but it
    # must not write, because `knownby_ts` in the tables is `current_ts` — and a
    # stored row would then claim we knew something at a moment when this code
    # did not yet exist. That claim eventually gets published. (It did: a score
    # file dated 2026-07-25, from a command run on 5 August.) Looking at the
    # past without writing is what modelling/validate_as_of is for.
    persist = not explicit
    if explicit:
        log.info("explicit current_ts %s — simulation: computed and logged, "
                 "NOT stored", current_ts)

    if cmd == "migrate":
        migrate_results(log)
        return
    def run_one(fn, area: str):
        write = writes_results(area, persist)
        if not write and persist:
            log.info("%s is in SHADOW mode: computed and logged, nothing stored "
                     "and nothing published", area)
        return fn(current_ts, area, log=log, persist=write)

    if cmd == "forecast":
        step = lambda a: run_one(forecast, a)   # noqa: E731
    elif cmd == "score":
        step = lambda a: run_one(score, a)      # noqa: E731
    else:
        log.error("unknown command %r (expected: forecast | score | migrate)", cmd)
        raise SystemExit(1)
    raise SystemExit(1 if for_each_area(step, log, cmd) else 0)


if __name__ == "__main__":
    main()
