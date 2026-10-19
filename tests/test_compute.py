"""
Tests for the as-of compute path. Synthetic data only — no feather, no network.

The one that matters is test_signal_for_matches_shift: it proves the live path
(as_of filter, no shift) and the backtest path (shift(2), no filter) produce the
SAME action when publication is regular. That is the "backtest == live" claim,
checked rather than asserted in a docstring.

Run:
    uv run pytest tests/test_compute.py -q
    uv run python tests/test_compute.py        # same checks, no pytest
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.benchmarks_core as C          # noqa: E402
from src.workflow_compute import (_no_forecast, _refuse_after_gate,  # noqa: E402
                                  _refuse_incomplete, _target_grid,
                                  day_completeness, input_day_for,
                                  leg_completeness, signal_window,
                                  window_completeness)

N_DAYS = 3


def _synthetic_matrix(days: int = 20, seed: int = 0) -> pd.DataFrame:
    """Hourly-varying pnl over `days` full UTC days, one row per 15-min PTU."""
    rng = np.random.default_rng(seed)
    ptus = pd.date_range("2026-01-01", periods=days * 96, freq="15min", tz="UTC")
    # A per-hour-of-day bias plus noise, so the signal has something to find.
    bias = np.cos(ptus.hour / 24 * 2 * np.pi)
    always_buy = bias + rng.normal(0, 0.3, len(ptus))
    return pd.DataFrame({"delivery_ts": ptus,
                         "always_buy": always_buy,
                         "always_sell": -always_buy + rng.normal(0, 0.1, len(ptus))})


def test_as_of_filters_and_keeps_latest_version():
    """Step 1 drops the unpublished; step 2 keeps the newest version, not the first."""
    ts = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    df = pd.DataFrame({
        "delivery_ts": [ts, ts, ts + timedelta(minutes=15)],
        "knownby_ts": [ts + timedelta(hours=1),    # as-known
                       ts + timedelta(hours=5),    # as-final, same PTU
                       ts + timedelta(hours=9)],   # not published yet at cut
        "price_da": [10.0, 11.0, 12.0],
    })
    cut = ts + timedelta(hours=6)

    out = C.as_of(df, cut)
    assert len(out) == 1, "the row published after the cut must be gone"
    assert out.loc[0, "price_da"] == 11.0, "must keep the LATEST version, not the first"

    # Strictly before: a row published exactly at current_ts is not yet knowable.
    assert C.as_of(df, ts + timedelta(hours=1)).empty


def test_as_of_is_empty_before_anything_is_published():
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    df = pd.DataFrame({"delivery_ts": [ts], "knownby_ts": [ts], "x": [1.0]})
    assert C.as_of(df, ts - timedelta(days=1)).empty


def test_signal_for_matches_shift():
    """
    Live (as_of + lag_days=0) == backtest (shift(2)) under regular publication.

    Backtest: forecast_actions labels hour X of day D using a window ending at
    D-2. Live: standing at D-1 with everything through D-2 settled, signal_for
    aims at day D. Same window, so the actions must be identical.
    """
    m = _synthetic_matrix()
    target_day = pd.Timestamp("2026-01-15", tz="UTC")

    backtest = C.forecast_actions(m, N_DAYS, lag_days=2)
    backtest = backtest[(backtest["hour_ts"] >= target_day)
                        & (backtest["hour_ts"] < target_day + timedelta(days=1))]

    # Everything settled through D-2, i.e. nothing from D-1 onwards.
    settled = m[m["delivery_ts"] < target_day - timedelta(days=1)]
    target_hours = pd.date_range(target_day, periods=24, freq="h", tz="UTC")
    live = C.signal_for(settled, N_DAYS, target_hours)

    merged = backtest.merge(live, on="hour_ts", validate="one_to_one")
    assert len(merged) == 24, "every target hour must be labelled by both paths"
    assert (merged["stat_action_x"] == merged["stat_action_y"]).all(), \
        "as_of path and shift(2) path disagree"
    assert (merged["n_obs"] == N_DAYS).all()
    # data_through is per hour-of-day: the freshest observation feeding THAT
    # hour's window sits exactly 48 h before it — the lag, measured not assumed.
    assert (merged["data_through"] == merged["hour_ts"] - timedelta(days=2)).all()


def test_signal_for_leaves_short_window_unformed():
    """A window shorter than n_days yields <NA>, never a default -1."""
    m = _synthetic_matrix(days=2)          # only 2 days, n_days=3
    target_hours = pd.date_range("2026-01-03", periods=24, freq="h", tz="UTC")
    live = C.signal_for(m, N_DAYS, target_hours)
    assert live["stat_action"].isna().all()
    assert (live["n_obs"] == 2).all()


def test_double_lag_is_visible():
    """Guard on the trap: filtering AND shifting moves the window twice."""
    m = _synthetic_matrix()
    a = C.forecast_actions(m, N_DAYS, lag_days=0)
    b = C.forecast_actions(m, N_DAYS, lag_days=2)
    merged = a.merge(b, on="hour_ts", validate="one_to_one")
    assert not (merged["stat_action_x"] == merged["stat_action_y"]).all(), \
        "lag_days must actually move the window"


def test_zone_registry_is_the_only_place_a_zone_is_defined():
    """
    Area code, EIC and timezone travel together, so they live together.

    They used to be in three places: the EIC as a constant in two modules, the
    timezone in a third keyed BY the EIC. Adding a zone meant remembering all
    three, and the EIC leaked into logs and filenames where nobody can read it.
    """
    import src.config as cfg
    from src.period import market_tz

    assert cfg.eic("NL") == "10YNL----------L"
    assert str(market_tz("RO")) == "Europe/Bucharest"

    # a zone that is not registered fails by NAME, listing what is known
    try:
        cfg.eic("XX")
    except KeyError as e:
        assert "settings.json" in str(e) and "NL" in str(e)
    else:
        raise AssertionError("an unknown zone must be refused")


def test_market_day_differs_by_zone():
    """The whole point of the registry: RO is UTC+3 in summer, NL UTC+2, so the
    same delivery date is a different UTC window."""
    from src.period import market_day_bounds
    from datetime import date

    nl = market_day_bounds(date(2026, 7, 25), "NL")
    ro = market_day_bounds(date(2026, 7, 25), "RO")
    assert (nl[0] - ro[0]).total_seconds() == 3600, (nl, ro)


def test_target_grid_is_the_market_day_not_the_utc_day():
    """23/25-hour DST days give 92/100 PTU. 96 is never assumed."""
    # Deciding on D-1 for the spring-forward day 2026-03-29 (23 h).
    spring = _target_grid(pd.Timestamp("2026-03-28 09:30", tz="UTC"), "NL")
    assert len(spring) == 92, len(spring)
    assert str(spring[0]) == "2026-03-28 23:00:00+00:00"

    # ... and for the fall-back day 2026-10-25 (25 h).
    autumn = _target_grid(pd.Timestamp("2026-10-24 09:30", tz="UTC"), "NL")
    assert len(autumn) == 100, len(autumn)

    # An ordinary summer day: 96 PTU starting at local midnight = 22:00Z.
    summer = _target_grid(pd.Timestamp("2026-07-25 09:30", tz="UTC"), "NL")
    assert len(summer) == 96
    assert str(summer[0]) == "2026-07-25 22:00:00+00:00"


def _eur_legs(day: str = "2026-08-05"):
    """Four PTU of a euro zone: DA 100, IM 110/90."""
    ptus = pd.date_range(day, periods=4, freq="15min", tz="UTC")
    da = pd.DataFrame({"delivery_ts": ptus, "currency": "EUR", "price_da": 100.0})
    im = pd.DataFrame({"delivery_ts": ptus, "currency": "EUR",
                       "price_im_pos": 110.0, "price_im_neg": 90.0})
    return da, im


def test_eur_does_not_depend_on_the_fx_table():
    """
    A euro zone must not be blocked by a stale ECB download.

    EUR is the unit, not a rate — one euro is worth one euro on every date. The
    FX table used to supply EUR=1.0 rows, but only over the calendar of the last
    downloaded file, so NL stopped computing the day the prices ran past it.
    Here the FX table knows nothing but USD in 2020, and it still works.
    """
    da, im = _eur_legs()
    fx = pd.DataFrame({"rate_date": pd.to_datetime(["2020-01-01"], utc=True),
                       "currency": ["USD"], "rate": [1.1]})
    m = C.add_pnl(C.build_matrix(da, im, fx, on=["delivery_ts"]))
    assert (m["price_da"] == 100.0).all()
    assert (m["always_buy"] == (90.0 - 100.0) * 0.25).all()
    assert (m["ideal"] == -2.5).all()


def test_missing_real_rate_says_which_dates():
    """A genuinely missing rate must name the uncovered dates, not just the code."""
    da, im = _eur_legs()
    da = da.assign(currency="PLN")                  # DA now needs a real rate
    fx = pd.DataFrame({"rate_date": pd.to_datetime(["2020-01-01"], utc=True),
                       "currency": ["PLN"], "rate": [4.3]})
    try:
        C.build_matrix(da, im, fx, on=["delivery_ts"])
    except ValueError as e:
        msg = str(e)
        assert "2026-08-05" in msg, msg      # what was needed
        assert "2020-01-01" in msg, msg      # what the table actually has
        assert "PLN" in msg, msg
    else:
        raise AssertionError("a missing non-EUR rate must still be refused")


def test_a_day_is_scored_exactly_twice():
    """
    as-known when it first settles, as-final once the revision window shuts,
    and never in between — otherwise the table grows quadratically with days.
    """
    from src.workflow_compute import score_reason

    day = pd.Timestamp("2026-08-06")
    just_settled = pd.Timestamp("2026-08-07 09:30", tz="UTC")
    before_final = pd.Timestamp("2026-09-19 09:30", tz="UTC")
    after_final = pd.Timestamp("2026-09-20 09:30", tz="UTC")

    # first pass
    assert score_reason(day, True, False, False, just_settled, 20) == "as-known"
    # every day in between: nothing to add, the value has not moved
    assert score_reason(day, True, True, False, just_settled, 20) is None
    assert score_reason(day, True, True, False, before_final, 20) is None
    # once the window shuts, exactly one more pass
    assert score_reason(day, True, True, False, after_final, 20) == "as-final"
    assert score_reason(day, True, True, True, after_final, 20) is None


def test_a_day_still_being_delivered_is_not_scored_but_a_finished_short_one_is():
    """
    The rule changed on 2026-09-26, and the reason was RO.

    `scorable` used to mean COMPLETE, which was right while the only reason for
    a short day was that imbalance had not all arrived yet. RO publishes
    intraday and on 09-25 skipped the quarter-hour at 10:00 UTC for good: under
    the old rule that day would never be scored and would print "not settled
    yet" for ever. So the test is now whether the day is OVER — a day still
    being delivered is not short, it is unfinished — and a finished day is
    scored on the PTU it has, with the count in the file.
    """
    from src.workflow_compute import score_reason

    day = pd.Timestamp("2026-08-06")
    now = pd.Timestamp("2026-08-07 09:30", tz="UTC")

    assert score_reason(day, False, False, False, now, 20) is None   # still running
    assert score_reason(day, True, False, False, now, 20) == "as-known"  # finished, even if short


# --------------------------------------------------------------------------- #
# The input gate — see the block comment above it in workflow_compute.py
# --------------------------------------------------------------------------- #
def _nl_matrix(days, area: str = "NL") -> pd.DataFrame:
    """A DA/IM matrix covering exactly the market days listed, that zone's."""
    from src.period import market_day_bounds

    frames = []
    for d in days:
        start, end = market_day_bounds(d, area)
        ptus = pd.date_range(start, end, freq="15min", inclusive="left")
        bias = np.cos(ptus.hour / 24 * 2 * np.pi)
        frames.append(pd.DataFrame({"delivery_ts": ptus,
                                    "always_buy": bias, "always_sell": -bias}))
    return pd.concat(frames, ignore_index=True)


def test_a_missing_input_day_still_forms_a_confident_signal():
    """
    WHY the gate is on freshness and not on n_obs.

    signal_for takes the last n_days observations per hour-of-day from the END
    of what is available, so a hole does not shorten the window — it shifts it.
    n_obs stays at n_days, the short-window guard cannot fire, and the forecast
    comes out fully formed on evidence a day older than it looks. This is the
    2026-08-30 failure, reproduced: 08-29 absent, the signal formed anyway.
    """
    from datetime import date

    m = _nl_matrix([date(2026, 8, 26), date(2026, 8, 27), date(2026, 8, 28)])
    current = pd.Timestamp("2026-08-30 09:30", tz="UTC")     # 08-29 is missing

    sig = C.signal_for(m, N_DAYS, _target_grid(current, "NL").floor("h").unique())
    assert sig["stat_action"].notna().all(), "the old guard sees nothing wrong"
    assert (sig["n_obs"] == N_DAYS).all(), "and n_obs cannot betray the hole"

    # The gate looks at a different thing entirely, and does see it.
    need = input_day_for(current, "NL")
    assert need == date(2026, 8, 29)
    assert day_completeness(m, need, "NL") == (0, 96)


def test_day_completeness_never_assumes_96():
    """A DST day is 92 or 100 PTU. A hard-coded 96 would refuse to forecast
    twice a year, and would accept a short day as whole the other time."""
    from datetime import date

    ordinary = _nl_matrix([date(2026, 8, 28)])
    assert day_completeness(ordinary, date(2026, 8, 28), "NL") == (96, 96)

    spring = _nl_matrix([date(2026, 3, 29)])          # 23 h
    assert day_completeness(spring, date(2026, 3, 29), "NL") == (92, 92)

    autumn = _nl_matrix([date(2026, 10, 25)])         # 25 h
    assert day_completeness(autumn, date(2026, 10, 25), "NL") == (100, 100)


def test_half_a_day_is_not_a_day():
    """The forecast twin of test_a_half_published_day_is_not_scored: imbalance
    arrives in pieces, and half the evidence is not the published method."""
    from datetime import date

    m = _nl_matrix([date(2026, 8, 28)])
    half = m[m["delivery_ts"] < pd.Timestamp("2026-08-28 10:00", tz="UTC")]
    have, expect = day_completeness(half, date(2026, 8, 28), "NL")
    assert have < expect and have > 0, (have, expect)


def test_completeness_needs_both_legs():
    """One count answers two questions only because the matrix is an INNER join.
    If that ever became a left join, a day present in DA and absent in IM would
    read as whole — so the join type is pinned here, not just documented."""
    from datetime import date

    ptus = pd.date_range("2026-08-27 22:00", periods=96, freq="15min", tz="UTC")
    fx = pd.DataFrame({"rate_date": pd.to_datetime(["2020-01-01"], utc=True),
                       "currency": ["USD"], "rate": [1.1]})
    da = pd.DataFrame({"delivery_ts": ptus, "currency": "EUR", "price_da": 100.0})
    im = pd.DataFrame({"delivery_ts": ptus[:40], "currency": "EUR",
                       "price_im_pos": 110.0, "price_im_neg": 90.0})

    m = C.build_matrix(da, im, fx, on=["delivery_ts"])
    assert day_completeness(m, date(2026, 8, 28), "NL") == (40, 96)


def test_a_refused_forecast_is_recorded_not_silent():
    """
    A refusal is a fact about the delivery day and has to survive as one.

    Rows go into the forecasts table with action <NA> and a skip_reason naming
    the day that was missing, so the refusal is frozen next to the days that did
    produce an action instead of leaving a hole a reader has to explain.
    """
    import logging
    from datetime import date

    from src.workflow_compute import FORECAST_COLS

    m = _nl_matrix([date(2026, 8, 26), date(2026, 8, 27), date(2026, 8, 28)])
    current = pd.Timestamp("2026-08-30 09:30", tz="UTC")
    ptus = _target_grid(current, "NL")
    need = input_day_for(current, "NL")

    reason, detail = _refuse_incomplete(need, [(need, 0, 96)], "NL", N_DAYS,
                                        None, None)
    out = _no_forecast(current, "NL", ptus, m, reason, detail, N_DAYS,
                       logging.getLogger("test"), persist=False)

    assert list(out.columns) == FORECAST_COLS
    assert len(out) == len(ptus) == 96
    assert out["action"].isna().all(), "a refusal must not look like a decision"
    assert (out["skip_reason"] == "input_incomplete:2026-08-29").all()
    # data_through records how stale we actually were — the evidence, not a guess
    assert out["data_through"].max() == pd.Timestamp("2026-08-28 21:00", tz="UTC")
    # a refusal is a published result too, so it carries its version like any other
    import src.config as cfg
    v = cfg.versions()
    assert (out["schema_version"] == v["schema"]).all()
    assert (out["methodology_version"] == v["methodology"]).all()


def test_a_refusal_names_which_leg_was_missing():
    """
    "0 of 96 on both legs" is true and useless.

    On 2026-09-14 the gap was in DAY-AHEAD — Sunday 09-13 was simply not
    published — while imbalance was complete. Every earlier gap had been in
    imbalance, so the owner read the line, assumed imbalance again, and went
    digging through the previous day's log to find out otherwise. The check
    stays on the joined matrix, because a PTU is usable only when both sides
    exist; only the DIAGNOSIS gains the two counts.
    """
    import logging
    from datetime import date

    days = [date(2026, 9, 10), date(2026, 9, 11), date(2026, 9, 12)]
    im = _nl_matrix([*days, date(2026, 9, 13)])      # imbalance has the day
    da = _nl_matrix(days)                            # day-ahead does not
    m = _nl_matrix(days)                             # so the join has nothing

    current = pd.Timestamp("2026-09-14 09:30", tz="UTC")
    need = input_day_for(current, "NL")
    assert need == date(2026, 9, 13)
    assert leg_completeness(da, need, "NL") == 0
    assert leg_completeness(im, need, "NL") == 96

    records = []
    log = logging.getLogger("test_legs")
    log.warning = lambda msg, *a: records.append(msg % a)

    reason, detail = _refuse_incomplete(need, [(need, 0, 96)], "NL", N_DAYS,
                                        da, im)
    assert "09-13: DA 0/96, IM 96/96" in detail, detail
    assert reason == "input_incomplete:2026-09-13:DA", reason


def test_the_whole_signal_window_must_be_whole_not_just_its_newest_day():
    """
    The 2026-09-15 lesson: the gate guarded one day, the signal uses n_days.

    Day-ahead for Sunday 09-13 was never published. The newest input day (09-14)
    was complete, so the gate opened — but signal_for takes the last n_days
    observations per hour-of-day from the END of what exists, so the window
    silently skipped the hole and reached back to 09-11. Three calendar days of
    evidence spanning four, and n_obs still reading 3. Exactly the failure the
    gate was built to stop, one level up: it is not enough for the freshest day
    to be there, the whole window has to be.
    """
    from datetime import date

    # 09-13 missing from the middle, everything else present
    m = _nl_matrix([date(2026, 9, 11), date(2026, 9, 12), date(2026, 9, 14)])
    current = pd.Timestamp("2026-09-15 09:30", tz="UTC")
    need = input_day_for(current, "NL")
    assert need == date(2026, 9, 14)

    # the old, one-day check is happy
    assert day_completeness(m, need, "NL") == (96, 96)

    # and the signal really does reach past the hole
    sig = C.signal_for(m, N_DAYS, _target_grid(current, "NL").floor("h").unique())
    assert (sig["n_obs"] == N_DAYS).all(), "n_obs cannot see the hole"

    # the window check does see it
    window = window_completeness(m, need, "NL", N_DAYS)
    assert [d for d, _, _ in window] == [date(2026, 9, 14), date(2026, 9, 13),
                                         date(2026, 9, 12)]
    missing = [(d, h, e) for d, h, e in window if h < e]
    assert missing == [(date(2026, 9, 13), 0, 96)], missing


def test_one_hole_refuses_three_forecasts_and_that_is_the_price():
    """A day missing sits in n_days consecutive windows, so it costs n_days
    forecasts. Accepted deliberately: the alternative is averaging over the gap,
    which at this volatility is fitting, not measuring."""
    from datetime import date

    hole = date(2026, 9, 13)
    refused = [t for t in (date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17))
               if hole in signal_window(t - timedelta(days=2), N_DAYS)]
    assert refused == [date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17)]
    # ... and the fourth day is clear again
    assert hole not in signal_window(date(2026, 9, 18) - timedelta(days=2), N_DAYS)


def test_a_refusal_lists_every_missing_day_of_the_window():
    """More than one hole is one refusal naming both, not two refusals."""
    import logging
    from datetime import date

    m = _nl_matrix([date(2026, 9, 12), date(2026, 9, 14)])   # 09-13 AND 09-11 gone
    current = pd.Timestamp("2026-09-15 09:30", tz="UTC")
    need = input_day_for(current, "NL")
    missing = [(d, h, e) for d, h, e in window_completeness(m, need, "NL", N_DAYS)
               if h < e]
    assert [d for d, _, _ in missing] == [date(2026, 9, 13)]

    reason, detail = _refuse_incomplete(need, missing, "NL", N_DAYS, None, None)
    out = _no_forecast(current, "NL", _target_grid(current, "NL"), m, reason,
                       detail, N_DAYS, logging.getLogger("test"), persist=False)
    assert (out["skip_reason"] == "input_incomplete:2026-09-13").all()


def test_a_decision_is_not_issued_after_the_auction_has_closed():
    """
    The numbers would still be causal; the bid would not exist.

    The day-ahead auction for delivery day D closes at 12:00 market-local on
    D-1 — 10:00 UTC in summer for NL. A forecast computed after that uses only
    data through D-2, so nothing about it is lookahead, but the benchmark claims
    to measure what a mechanical rule would have EARNED, and an order nobody
    could submit earns nothing. Checked BEFORE the data, because no amount of
    backfilling reopens an auction.
    """
    from datetime import date

    from src.period import auction_gate

    # summer: 12:00 Brussels = 10:00 UTC on the day before delivery
    gate = auction_gate(date(2026, 9, 18))
    assert str(gate) == "2026-09-17 10:00:00+00:00", gate
    # winter: the same rule, an hour later in UTC, computed not written down
    assert str(auction_gate(date(2026, 1, 18))) == "2026-01-17 11:00:00+00:00"

    reason, detail = _refuse_after_gate(date(2026, 9, 18), pd.Timestamp(gate),
                                        pd.Timestamp("2026-09-17 14:00", tz="UTC"))
    assert reason == "after_gate"
    assert "240 min ago" in detail, detail
    assert "Nothing to recover" in detail


def test_the_gate_leaves_the_retries_room_but_not_much():
    """The decision run plus its retries must finish before the gate. At 09:40
    with four attempts five minutes apart the last one starts at 09:55, and in
    summer the gate is 10:00 — which is exactly why the gate is CHECKED rather
    than assumed to be respected by arithmetic."""
    from datetime import date

    import src.config as cfg
    from src.period import auction_gate

    gate = auction_gate(date(2026, 9, 18))
    h, m = (int(x) for x in cfg.RUN_TIME_UTC.split(":"))
    last_attempt = h * 60 + m + (cfg.retry_attempts() - 1) * cfg.retry_delay_s() // 60
    assert last_attempt < gate.hour * 60 + gate.minute, (last_attempt, gate)


def test_a_refused_day_can_never_be_forecast_later():
    """
    The reason a refusal is permanent rather than deferred.

    _target_grid always aims at "tomorrow relative to the run", so no later run
    can target the refused day again: by the next run its hours are already
    delivering, and issuing an action for them is the non-causality as_of
    exists to prevent. Only a re-run on the SAME day can still reach it.
    """
    refused_day = pd.Timestamp("2026-08-31 22:00", tz="UTC")   # first PTU of 09-01

    same_day = _target_grid(pd.Timestamp("2026-08-31 14:00", tz="UTC"), "NL")
    assert same_day.min() == refused_day, "a same-day re-run can still recover it"

    next_day = _target_grid(pd.Timestamp("2026-09-01 09:30", tz="UTC"), "NL")
    assert next_day.min() > refused_day, "and no later run can ever reach it"


def test_final_date_is_the_20th_of_the_following_month():
    from src.workflow_compute import final_ts

    assert final_ts(pd.Timestamp("2026-08-06"), 20) == pd.Timestamp("2026-09-20", tz="UTC")
    assert final_ts(pd.Timestamp("2026-08-31"), 20) == pd.Timestamp("2026-09-20", tz="UTC")
    # across the year boundary
    assert final_ts(pd.Timestamp("2026-12-01"), 20) == pd.Timestamp("2027-01-20", tz="UTC")


def test_result_tables_are_keyed_by_zone():
    """
    `area` must stay in the key of forecasts / scores.

    Drop it and an FR row and an NL row for the same PTU, computed at the same
    instant, become the same key — `_append`'s drop_duplicates then discards one
    silently. No error, no warning, just one row fewer, noticed months later.
    """
    from src.workflow_compute import KEY

    assert "area" in KEY, KEY


def test_slot_shows_a_late_start():
    """A punctual run and a catch-up after wake must not look alike."""
    from src.workflow import scheduled_slot

    times = ["05:30", "07:30", "09:30", "11:30", "13:30"]
    on_time = datetime(2026, 8, 6, 5, 30, 11, tzinfo=timezone.utc)
    slot, late = scheduled_slot(on_time, times)
    assert slot == "05:30Z" and late < 1

    woke_up = datetime(2026, 8, 6, 8, 23, 18, tzinfo=timezone.utc)
    slot, late = scheduled_slot(woke_up, times)
    assert slot == "07:30Z" and 53 <= late <= 54

    # just after UTC midnight the run belongs to yesterday's last slot
    slot, late = scheduled_slot(datetime(2026, 8, 6, 0, 10, tzinfo=timezone.utc), times)
    assert slot == "13:30Z" and late > 600


def test_every_scheduled_time_counts_as_a_slot():
    """
    The slot list must cover the monthly re-fetch too, not just the polls.

    With one poll a day, an 08:00 re-fetch matched against a 09:30-only list
    falls back to YESTERDAY's slot and reports itself twenty-two hours late —
    a false "machine asleep" warning on every monthly run, which is exactly the
    kind of alarm that teaches people to ignore alarms.
    """
    import src.config as cfg
    from src.workflow import scheduled_slot

    times = cfg.scheduled_times_utc()
    assert cfg.refetch_time_utc() in times, times
    assert cfg.RUN_TIME_UTC in times, times

    h, m = (int(x) for x in cfg.refetch_time_utc().split(":"))
    slot, late = scheduled_slot(datetime(2026, 8, 20, h, m, 5, tzinfo=timezone.utc))
    assert late < 1, (slot, late)


def test_the_market_benchmarks_do_not_need_an_action():
    """
    Four of the five values are properties of the day's prices, so a settled day
    is gradable on them with no decision of ours anywhere in sight. Only
    action_pnl needs an action — and where there is none it is NaN, which is
    what keeps it out of the published file.
    """
    import numpy as np

    j = pd.DataFrame({"always_buy": [1.0, 2.0, 3.0],
                      "always_sell": [-1.0, -2.0, -3.0],
                      "action": pd.array([1, -1, None], dtype="Int8")})
    act = pd.array(j["action"], dtype="Float64").to_numpy(dtype="float64",
                                                          na_value=np.nan)
    pnl = np.where(act == 1, j["always_buy"],
                   np.where(act == -1, j["always_sell"], np.nan))

    assert pnl[0] == 1.0 and pnl[1] == -2.0
    assert np.isnan(pnl[2]), "no action is not the same as an action to sell"


def test_the_scoring_floor_keeps_six_years_of_seed_out_of_the_first_run():
    """
    The store is seeded back to 2020. Scoring every settled day it holds would
    write the whole history in one pass, stamped with today's knownby_ts. The
    floor is what bounds that — and days before it still come back if an action
    was issued for them, which is how the existing series keeps its as-final
    pass.
    """
    import src.config as cfg

    floor = cfg.score_from()
    assert floor is not None, "settings must name the day the live series began"

    store_days = pd.DatetimeIndex(["2020-01-01", "2026-09-16", "2026-09-17"])
    kept = store_days[store_days >= pd.Timestamp(floor)]
    assert pd.Timestamp("2020-01-01") not in kept
    assert pd.Timestamp("2026-09-17") in kept

    acted = {pd.Timestamp("2026-09-16")}          # scored before the floor
    assert pd.Timestamp("2026-09-16") in set(kept) | acted


def test_a_refused_day_is_still_scored_on_what_the_market_did():
    """
    2026-09-18, the defect in one line: delivery day 09-17 was refused a
    forecast because of a day-ahead hole on 09-13, and then vanished from the
    scores entirely. The prices were in the store the whole time.

    score() drops nothing now; it starts from the store and treats the action as
    an optional extra, so the day is graded on the four market benchmarks and
    carries the refusal's own words as its reason.
    """
    from src.workflow_compute import SCORE_COLS

    assert "skip_reason" in SCORE_COLS, (
        "a scored day with no action has to say why, in its own row")


def test_one_broken_zone_does_not_stop_the_others():
    """
    The contract that pays for having ONE process instead of a job per zone.

    A zone that raises is logged and counted; the zones behind it still run.
    The return value is a COUNT, not a flag, because at twenty-five zones "2 of
    25 failed" is the alarm and the log is the diagnosis.
    """
    import logging

    from src.workflow import for_each_area

    seen = []

    def step(area):
        seen.append(area)
        if area == "RO":
            raise RuntimeError("source refused")

    log = logging.getLogger("test_isolation")
    failed = for_each_area(step, log, "update", ["NL", "RO", "BE"])

    assert seen == ["NL", "RO", "BE"], "a failure must not abort the loop"
    assert failed == 1, failed


def test_the_zone_list_comes_from_settings():
    """No default zone anywhere: a caller that forgets to say which zone it
    means must fail, not quietly work on whichever one happens to be first."""
    import inspect

    import src.config as cfg
    from src.workflow import load_store, write_store
    from src.workflow_compute import forecast, score

    assert cfg.AREAS, "settings must name at least one area"
    for fn in (load_store, write_store, forecast, score):
        area = inspect.signature(fn).parameters["area"]
        assert area.default is inspect.Parameter.empty, f"{fn.__name__} has a default zone"


def test_the_gate_belongs_to_the_auction_not_to_the_zone():
    """
    The bug this replaced, pinned so it cannot come back.

    Day-ahead is ONE European auction, so every zone sits behind the SAME
    instant. It is the local reading that moves: 12:00 in Brussels is 13:00 in
    Bucharest, and OPCOM publishes the Romanian gate as exactly that. The old
    code took noon in the zone's own timezone — right for NL by coincidence,
    and an hour early for RO, which would have refused every RO forecast as
    "after_gate" at a 09:40 run with nothing whatever wrong.

    Note what the old test did: it asserted 09:00 UTC for RO. It pinned the
    behaviour instead of the rule, so it agreed with the defect.
    """
    from datetime import date
    from zoneinfo import ZoneInfo

    from src.period import auction_gate

    for day, utc, local_ro in ((date(2026, 9, 18), "10:00", 13),    # summer
                               (date(2026, 1, 18), "11:00", 13)):   # winter
        gate = auction_gate(day)
        assert gate.strftime("%H:%M") == utc, (day, gate)
        assert gate.astimezone(ZoneInfo("Europe/Bucharest")).hour == local_ro
        assert gate.astimezone(ZoneInfo("Europe/Amsterdam")).hour == 12


# --------------------------------------------------------------------------- #
# Exchange rates in the live path
# --------------------------------------------------------------------------- #
def test_the_ecb_calendar_is_computed_and_matches_the_published_one():
    """
    Closed = weekends + the six TARGET closing days, computed (four fixed dates,
    two from Easter). Checked against the ECB's own published list for
    2026-2028: the starred TARGET days are closed, the office holidays are not
    — rates ARE published on Ascension, Whit Monday, 24 and 31 December.
    (Against eurofxref-hist.xml the same function reproduces every publication
    day since 2002 without a single disagreement.)
    """
    from datetime import date

    from src.parse_fx import ecb_closed

    closed = ["2026-01-01", "2026-04-03", "2026-04-06", "2026-05-01", "2026-12-25",
              "2026-12-26", "2027-03-26", "2027-03-29", "2028-04-14", "2028-04-17"]
    open_ = ["2026-05-14", "2026-05-25", "2026-06-04", "2026-10-02", "2026-12-24",
             "2026-12-31", "2027-05-06", "2028-06-05"]
    for d in closed:
        assert ecb_closed(date.fromisoformat(d)), d
    for d in open_:
        assert not ecb_closed(date.fromisoformat(d)), d
    assert ecb_closed(date(2026, 9, 19)) and ecb_closed(date(2026, 9, 20))   # weekend


def test_a_rate_carries_over_closed_days_and_stops_at_the_first_open_one():
    """
    The owner's rule, as the table enforces it: a rate extends over days on
    which the ECB did not publish because it was CLOSED, and never across a day
    on which a publication was due. Friday covers the weekend, not Monday; 24
    December covers Christmas, St Stephen's and the Sunday after — and a
    currency that stopped being quoted does not come back to life.
    """
    from src.parse_fx import build_fx_daily

    def obs(last_day, retired=None):
        days = pd.date_range(end=last_day, periods=5, freq="B", tz="UTC")
        rows = [(d, "RON", 5.0) for d in days]
        if retired:
            rows += [(pd.Timestamp(retired, tz="UTC"), "HRK", 7.5)]
        return pd.DataFrame(rows, columns=["rate_date", "currency", "rate"])

    fx = build_fx_daily(obs("2026-09-18", retired="2026-09-15"))
    ron = fx.loc[fx["currency"] == "RON"]
    assert ron["rate_date"].max() == pd.Timestamp("2026-09-20", tz="UTC")
    assert set(ron.loc[ron["rate_date"] > pd.Timestamp("2026-09-18", tz="UTC"),
                       "source"]) == {"ffill"}
    assert fx.loc[fx["currency"] == "HRK", "rate_date"].max() == \
        pd.Timestamp("2026-09-15", tz="UTC"), "a retired currency must not extend"

    xmas = build_fx_daily(obs("2026-12-24"))
    assert xmas["rate_date"].max() == pd.Timestamp("2026-12-27", tz="UTC")


def _legs(days, im_currency):
    ptus = pd.DatetimeIndex(
        [t for d in days
         for t in pd.date_range(f"{d} 00:00", periods=96, freq="15min", tz="UTC")])
    da = pd.DataFrame({"delivery_ts": ptus, "currency": "EUR", "price_da": 100.0})
    im = pd.DataFrame({"delivery_ts": ptus, "currency": im_currency,
                       "price_im_pos": 550.0, "price_im_neg": 450.0})
    return da, im


def test_the_live_matrix_drops_what_it_cannot_value_and_the_report_still_raises():
    """
    Two callers, two meanings of a missing rate. The report must never build a
    frozen table on a partial valuation, so it raises. The live path leaves out
    the PTU it cannot value and says which dates — the completeness gates then
    refuse the forecast and hold the score. Nothing is filled in either way.
    """
    da, im = _legs(["2026-09-21", "2026-09-22"], "RON")
    fx = pd.DataFrame({"rate_date": pd.to_datetime(["2026-09-21"], utc=True),
                       "currency": ["RON"], "rate": [5.0]})

    m = C.build_matrix(da, im, fx, on=["delivery_ts"], on_missing_fx="drop")
    assert len(m) == 96, len(m)
    assert m.attrs["fx_missing"] == [("IM", "RON", "2026-09-22")]
    assert (m["price_im_pos_eur"] == 110.0).all()

    try:
        C.build_matrix(da, im, fx, on=["delivery_ts"])
    except ValueError as e:
        assert "IM in RON" in str(e), e
    else:
        raise AssertionError("the report path must still raise on a missing rate")

    # A euro zone needs no table at all: EUR is a unit, never looked up.
    da, im = _legs(["2026-09-22"], "EUR")
    empty = pd.DataFrame(columns=["rate_date", "currency", "rate"])
    empty["rate_date"] = pd.to_datetime(empty["rate_date"], utc=True)
    assert len(C.build_matrix(da, im, empty, on=["delivery_ts"])) == 96


def test_a_refusal_names_fx_when_both_price_legs_are_whole():
    """Prices present on both legs and the day still short: they could not be
    VALUED. The refusal says leg FX and prints the command that fixes it — the
    rate refresh, not an ENTSO-E backfill that would change nothing."""
    from datetime import date

    from src.workflow_compute import _refuse_incomplete

    d = date(2026, 9, 21)
    legs = _nl_matrix([d])
    reason, detail = _refuse_incomplete(d, [(d, 0, 96)], "NL", 1, legs, legs)
    assert reason == "input_incomplete:2026-09-21:FX", reason
    assert "FX missing" in detail and "src.fx" in detail, detail
    assert "src.update --from" not in detail, detail


def test_the_recovery_command_names_the_zone():
    """One zone is refused; re-requesting all twenty-five to fix it is not a
    fix, it is a morning."""
    from datetime import date

    from src.workflow_compute import _recovery

    cmd = _recovery(["DA"], [date(2026, 9, 20), date(2026, 9, 22)], "RO")
    assert cmd.endswith("--from 2026-09-20 --to 2026-09-22 RO && "
                        "uv run python -m src.workflow_compute forecast"), cmd


def test_a_zone_says_what_it_is_for_and_shadow_writes_nothing(monkeypatch):
    """
    A zone joins by being watched. In shadow it runs the whole chain and stores
    nothing, so several days of log answer what no reasoning can — when the TSO
    publishes, whether the gate opens, whether the rate is there — and any code
    still shaped like the first zone fails in the log rather than in a
    published file.

    No default mode, in either direction: "live" would publish a zone nobody
    has watched, "shadow" would quietly unpublish a live one if the field went
    missing. And a simulation never writes, whatever the zone is.
    """
    import src.config as cfg
    from src.workflow_compute import writes_results

    for area in cfg.AREAS:
        assert cfg.zone_mode(area) in cfg.ZONE_MODES, area

    # The RULE, not today's roster. This line used to read
    # `writes_results("RO") is False, "RO is being watched"`, and it failed on
    # 2026-09-28 the moment RO went live — a correct change breaking a test
    # means the test was pinning the configuration instead of the behaviour.
    for area in cfg.AREAS:
        assert writes_results(area) is (cfg.zone_mode(area) == "live"), area
        assert writes_results(area, persist=False) is False, area

    # A zone without a mode, made up here rather than taken from the registry:
    # this used to name BE, and failed on 2026-10-19 the day BE was given one —
    # the same mistake as the RO line above, pinning the roster, not the rule.
    monkeypatch.setattr(cfg, "zone", lambda area: {"eic": "X", "tz": "UTC"})
    try:
        cfg.zone_mode("XX")
    except KeyError as e:
        assert "mode" in str(e), e
    else:
        raise AssertionError("a zone without a mode must fail by name")


def test_a_rate_that_is_not_out_yet_is_not_a_missing_rate():
    """
    The difference between "not yet" and "missing", and only the second is
    worth a word. RO publishes imbalance intraday, so today's PTU sit in the
    store from the morning while the ECB dates today's rate in the afternoon —
    warning about that would print one line every single day, and a warning
    that appears daily is one nobody reads on the day it matters.
    """
    from datetime import date, datetime, timezone

    from src.parse_fx import rate_due

    morning = datetime(2026, 9, 24, 9, 40, tzinfo=timezone.utc)   # our run
    assert not rate_due(date(2026, 9, 24), morning), "today's rate is not out yet"
    assert rate_due(date(2026, 9, 23), morning), "yesterday's has been out since 16:00 CET"
    assert not rate_due(date(2026, 9, 20), morning), "a Sunday never gets one"

    evening = datetime(2026, 9, 24, 15, 30, tzinfo=timezone.utc)  # 17:30 CEST
    assert rate_due(date(2026, 9, 24), evening), "by the evening it is due"


# --------------------------------------------------------------------------- #
# The gate, at the granularity the signal actually works on
# --------------------------------------------------------------------------- #
def test_one_missing_quarter_hour_costs_one_hour_not_three_days():
    """
    RO, 2026-09-25: one quarter-hour absent out of 96, and under the whole-day
    rule that refused THREE forecasts, because the day sits in three
    consecutive windows. The hour is what the signal is computed on, so the hour
    is what a hole should cost.
    """
    from datetime import date

    from src.workflow_compute import incomplete_hours

    days = [date(2026, 9, 23), date(2026, 9, 24), date(2026, 9, 25)]
    m = _nl_matrix(days)
    hole = pd.Timestamp("2026-09-25 10:00", tz="UTC")
    m = m.loc[m["delivery_ts"] != hole]

    bad = incomplete_hours(m, date(2026, 9, 25), "NL", 3)
    assert list(bad) == [10], bad
    assert bad[10] == [(date(2026, 9, 25), 3, 4)], bad[10]


def test_an_absent_day_still_fails_every_hour():
    """The old behaviour is the special case, not a second rule. A missing DAY
    is the dangerous one — it SHIFTS the window instead of shortening it — and
    it fails all twenty-four hours, so nothing forms and the day is refused
    exactly as before."""
    from datetime import date

    from src.workflow_compute import incomplete_hours

    m = _nl_matrix([date(2026, 9, 23), date(2026, 9, 25)])     # 09-24 absent
    bad = incomplete_hours(m, date(2026, 9, 25), "NL", 3)
    assert len(bad) == 24, sorted(bad)
    assert all(shorts[0][1] == 0 for shorts in bad.values())


def test_the_expected_count_per_hour_comes_from_the_grid_not_from_four():
    """A daylight-saving day has one UTC hour-of-day occurring twice or not at
    all. Hard-coding four would refuse a perfectly good hour twice a year."""
    from datetime import date

    from src.workflow_compute import hour_grid

    ordinary = hour_grid(date(2026, 9, 25), "NL")
    assert ordinary.sum() == 96 and set(ordinary) == {4}

    autumn = hour_grid(date(2026, 10, 25), "NL")          # 25 h
    assert autumn.sum() == 100 and autumn[22] == 8

    spring = hour_grid(date(2026, 3, 29), "NL")           # 23 h
    assert spring.sum() == 92


class _Log:
    """A logger that keeps the rendered lines, so a test can count them."""

    def __init__(self):
        self.lines: list[str] = []

    def _add(self, fmt, *a):
        self.lines.append(fmt % a if a else fmt)

    info = warning = error = _add


def _scored_rows(days, ptu: int = 96):
    """One scored row per PTU per day, no action anywhere."""
    rows = []
    for d in days:
        rows.append(pd.DataFrame({"day": [d] * ptu,
                                  "action": pd.array([None] * ptu, dtype="Int8"),
                                  "action_pnl": [np.nan] * ptu,
                                  "ideal": [1.0] * ptu,
                                  "devil": [-1.0] * ptu,
                                  "skip_reason": [None] * ptu}))
    return pd.concat(rows, ignore_index=True)


def test_a_pass_over_many_days_costs_a_fixed_number_of_lines():
    """
    A zone in shadow stores nothing, so every day since the floor is scored
    again every morning: eighteen lines on 2026-09-25, ten on 09-26, one more
    every day after. The as-final pass on the 20th has the same shape in a live
    zone. One full day per reason plus one summary is the whole report; the
    numbers live in the results table.
    """
    from src.workflow_compute import _log_scored

    days = pd.DatetimeIndex([f"2026-09-{d:02d}" for d in range(17, 27)], tz="UTC")
    reasons = {d: ("as-known" if d == days[-1] else "as-final") for d in days}
    settled = {d: (96, 96) for d in days}
    settled[days[0]] = (95, 96)                 # the source skipped one PTU

    log = _Log()
    _log_scored(log, "RO", pd.Timestamp("2026-09-27 09:40", tz="UTC"),
                _scored_rows(days), reasons, settled)

    assert len(log.lines) == 3, log.lines          # as-final full + summary, as-known full
    assert "2026-09-25 (as-final)" in log.lines[0]
    assert "8 more delivery day(s) in the same as-final pass" in log.lines[1]
    assert "2026-09-17..2026-09-24" in log.lines[1]
    assert "1 short of a full day" in log.lines[1], "a source that skips PTUs is news"
    assert "2026-09-26 (as-known)" in log.lines[2]

    longer = dict(reasons)
    for d in pd.DatetimeIndex(["2026-09-15", "2026-09-16"], tz="UTC"):
        longer[d], settled[d] = "as-final", (96, 96)
    log2 = _Log()
    _log_scored(log2, "RO", pd.Timestamp("2026-09-27 09:40", tz="UTC"),
                _scored_rows(list(longer)), longer, settled)
    assert len(log2.lines) == len(log.lines), "two more days must not mean two more lines"


def test_a_day_that_has_not_begun_is_not_a_day_with_nothing_in_it():
    """
    Tomorrow's day-ahead prices put tomorrow in the store, so the day is a
    candidate before it exists, and "still being delivered, 0 of 96 PTU
    published" was true and useless. A day genuinely in flight keeps its line —
    that one is a question. A day that ended empty waits for ever and is
    counted, not repeated every morning for ever.
    """
    from src.workflow_compute import _log_waiting

    now = pd.Timestamp("2026-09-27 09:40", tz="UTC")
    days = pd.DatetimeIndex(["2026-09-13", "2026-09-27", "2026-09-28"], tz="UTC")
    settled = {days[0]: (0, 96), days[1]: (0, 96), days[2]: (0, 96)}

    log = _Log()
    _log_waiting(log, "NL", now, list(days), settled, set(), set(), 20, set())
    text = "\n".join(log.lines)

    assert len(log.lines) == 3, log.lines
    assert sum("still being delivered" in ln for ln in log.lines) == 1
    assert "delivery day 2026-09-27 still being delivered" in text
    assert "not yet started (2026-09-28)" in text
    assert "ended with nothing published and cannot be graded: 2026-09-13" in text

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
