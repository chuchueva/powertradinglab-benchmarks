"""
The matrix (models/features.py), the action (models/actions.py) and what an
action earned (benchmarks_core.action_pnl). Synthetic data only.
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.benchmarks_core as C                     # noqa: E402
from models.actions import to_action                # noqa: E402
from models.features import hourly, rows_for, window  # noqa: E402
from models.trailing_bias import decide             # noqa: E402
from src.period import market_day_bounds           # noqa: E402

N = 3


def _m(days, area="NL", drop=()):
    """Quarter-hour matrix covering the market days listed; `drop` = UTC PTU to leave out."""
    frames = []
    for d in days:
        start, end = market_day_bounds(d, area)
        ptus = pd.date_range(start, end, freq="15min", inclusive="left")
        bias = np.cos(ptus.hour / 24 * 2 * np.pi) + ptus.day / 100
        frames.append(pd.DataFrame({"delivery_ts": ptus, "always_buy": bias,
                                    "always_sell": -bias}))
    m = pd.concat(frames, ignore_index=True)
    return m[~m["delivery_ts"].isin(pd.DatetimeIndex(drop, tz="UTC"))]


def _x(m, knownby, area="NL", fresh=True):
    t = pd.Timestamp(knownby, tz="UTC")
    return decide(window(rows_for(t, area), hourly(m), N, fresh, area), N)


def _days(a, b):
    return [d.date() for d in pd.date_range(a, b)]


def test_matrix_backtest_equals_the_report_shift_path():
    """backtest == live: the report's forecast_actions(lag_days=2) and the
    matrix with d1 = yesterday give the same action, hour by hour."""
    m = _m(_days("2026-01-01", "2026-01-20"))
    rep = C.forecast_actions(m, N, lag_days=2)
    x = _x(m, "2026-01-14 09:40", fresh=False)          # decides 2026-01-15
    j = x.merge(rep, left_on="delivery_ts", right_on="hour_ts")
    assert len(j) == 24
    assert (j["action"] == j["stat_action"]).all()


def test_yesterday_missing_is_a_hole_not_a_shifted_window():
    """2026-08-30: 08-29 absent. The old tail() reached back to 08-26; the
    matrix asks for 08-29 by timestamp, finds nothing, issues no action."""
    m = _m([date(2026, 8, 26), date(2026, 8, 27), date(2026, 8, 28)])
    x = _x(m, "2026-08-30 09:40")
    assert x["action"].isna().all()
    assert (x["ptu_d1"] == 0).all()


def test_a_hole_in_the_middle_is_a_hole():
    """2026-09-15: 09-13 absent, 09-14 whole. d2 is 09-13 and empty."""
    m = _m([date(2026, 9, 11), date(2026, 9, 12), date(2026, 9, 14)])
    x = _x(m, "2026-09-15 09:40")
    assert x["action"].isna().all()
    assert x["always_buy_d1"].notna().all() and x["always_buy_d2"].isna().all()


def test_one_missing_quarter_hour_costs_one_hour():
    m = _m(_days("2026-09-22", "2026-09-25"), drop=["2026-09-25 10:00"])
    x = _x(m, "2026-09-26 09:40")
    assert x.loc[x["action"].isna(), "delivery_ts"].dt.hour.tolist() == [10]


def test_a_whole_hour_of_today_is_used_and_a_partial_one_is_not():
    """Live: today's 00..08 UTC are whole, 09 has one quarter of four."""
    m = _m(_days("2026-09-21", "2026-09-25"))
    today = pd.date_range("2026-09-25 22:00", "2026-09-26 09:00", freq="15min", tz="UTC")
    m = pd.concat([m, pd.DataFrame({"delivery_ts": today, "always_buy": 5.0,
                                    "always_sell": -5.0})], ignore_index=True)
    x = _x(m, "2026-09-26 09:40")                     # target 2026-09-27 (NL)
    by_hour = x.set_index(x["delivery_ts"].dt.hour)
    assert by_hour.loc[3, "d1_today"] == 1             # 2026-09-26 03:00 whole
    assert by_hour.loc[9, "d1_today"] == 0             # 09:00 partial -> yesterday
    assert by_hour.loc[9, "ptu_d1"] == 4


def test_the_backtest_never_takes_today():
    m = _m(_days("2026-09-21", "2026-09-26"))
    x = _x(m, "2026-09-26 09:40", fresh=False)
    assert (x["d1_today"] == 0).all()


def test_spring_23_hour_day_takes_the_day_before_yesterday_for_the_missing_hour():
    """NL 2026-03-29 has no 01:00 UTC... in local terms one UTC hour is absent
    from the 23-hour day. That hour, and only it, reaches one day further."""
    m = _m(_days("2026-03-24", "2026-03-30"))
    x = _x(m, "2026-03-30 09:40", fresh=False)        # need = 2026-03-29
    assert x["action"].notna().all(), x.loc[x["action"].isna(), "delivery_ts"]


def test_to_action_with_and_without_sensitivity():
    p = pd.Series([0.9, 0.52, 0.5, 0.3, None])
    assert to_action(p).tolist() == [1, 1, -1, -1, pd.NA]
    assert to_action(p, sensitivity=0.05).tolist() == [1, 0, 0, -1, pd.NA]


def test_action_pnl_is_one_function_for_every_action():
    a = pd.array([1, -1, 0, None], dtype="Int8")
    out = C.action_pnl(a, [10.0] * 4, [-3.0] * 4)
    assert out[:3].tolist() == [10.0, -3.0, 0.0] and np.isnan(out[3])
