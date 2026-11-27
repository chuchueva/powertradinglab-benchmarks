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
from models.features import calendar, hourly, rows_for, slots, window  # noqa: E402
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


# --- 15-minute rows (step="15min", 2026-11-19) --------------------------------

def _x15(m, knownby, area="NL", fresh=True):
    t = pd.Timestamp(knownby, tz="UTC")
    rows = rows_for(t, area, "15min")
    s15 = slots(m, "15min")
    return window(rows, s15, N, fresh, area, "15min")


def test_a_15min_day_has_one_row_per_quarter_hour():
    m = _m(_days("2026-09-22", "2026-09-25"))
    x = _x15(m, "2026-09-26 09:40")
    assert len(x) == 96
    assert x["delivery_ts"].is_unique


def test_the_autumn_daylight_saving_day_has_100_quarter_hours():
    """NL, delivery day 2026-10-25 (clocks go back): 100 quarter-hours."""
    m = _m(_days("2026-10-20", "2026-10-24"))
    x = _x15(m, "2026-10-24 09:40")
    assert len(x) == 100


def test_one_missing_quarter_hour_costs_only_that_quarter_at_15min():
    """Hourly, one missing quarter kills its hour; at 15 minutes only itself."""
    m = _m(_days("2026-09-22", "2026-09-25"), drop=["2026-09-25 10:00"])
    x = _x15(m, "2026-09-26 09:40")
    hole = x[x["always_buy_d1"].isna()]
    assert hole["delivery_ts"].dt.strftime("%H:%M").tolist() == ["10:00"]


def test_four_quarters_add_up_to_their_hour():
    """The 15-minute lag of an hour's four quarters sums to the hourly lag."""
    m = _m(_days("2026-09-22", "2026-09-25"))
    t = "2026-09-26 09:40"
    x15 = _x15(m, t)
    xh = window(rows_for(pd.Timestamp(t, tz="UTC"), "NL"), hourly(m), N, True, "NL")
    x15 = x15.assign(hour_ts=x15["delivery_ts"].dt.floor("h"))
    summed = x15.groupby("hour_ts")["always_buy_d1"].sum()
    hourly_d1 = xh.set_index("delivery_ts")["always_buy_d1"]
    assert np.allclose(summed.loc[hourly_d1.index].to_numpy(), hourly_d1.to_numpy())


def test_bare_prices_are_carried_and_buy_is_the_higher_imbalance_price():
    m = _m(_days("2026-09-22", "2026-09-25"))
    m = m.assign(price_da=50.0, price_im_pos_eur=70.0, price_im_neg_eur=40.0)
    x = _x15(m, "2026-09-26 09:40")
    assert (x["price_da_d1"] == 50.0).all()
    assert (x["price_im_buy_d1"] == 70.0).all()
    assert (x["price_im_sell_d1"] == 40.0).all()


def test_quarter_of_hour_is_a_calendar_column():
    m = _m(_days("2026-09-22", "2026-09-25"))
    x = calendar(_x15(m, "2026-09-26 09:40"), "NL")
    assert x["quarter_of_hour"].tolist()[:4] == [0, 1, 2, 3]


def test_action_zero_earns_zero():
    """A decision to stay out (0) earns 0.0; no decision (<NA>) earns nothing."""
    a = to_action(pd.Series([0.5, 0.9, 0.1, None]), sensitivity=0.1)
    earned = C.action_pnl(a, [10.0, 10.0, 10.0, 10.0], [-3.0, -3.0, -3.0, -3.0])
    assert earned[0] == 0.0 and earned[1] == 10.0 and earned[2] == -3.0
    assert np.isnan(earned[3])



# --- logistic regression from saved coefficients (2026-11-21) ---------------

def test_coefficients_give_exactly_what_scikit_learn_gives():
    """decide() computes p itself; it must equal predict_proba."""
    import pytest
    sklearn_lm = pytest.importorskip("sklearn.linear_model")
    from models.logreg import decide as lr_decide
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = (x["a"] - 0.5 * x["b"] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    clf = sklearn_lm.LogisticRegression().fit(x.to_numpy(), y.to_numpy())
    model = {"intercept": float(clf.intercept_[0]),
             "coefficients": dict(zip(["a", "b"], clf.coef_[0].tolist()))}
    out = lr_decide(x, model, ["a", "b"])
    expected = clf.predict_proba(x.to_numpy())[:, 1]
    assert np.allclose(out["p_buy"].to_numpy(dtype=float), expected, atol=1e-12)


def test_a_hole_in_an_input_gives_no_action():
    from models.logreg import decide as lr_decide
    x = pd.DataFrame({"a": [1.0, None], "b": [0.5, 0.5]})
    model = {"intercept": 0.0, "coefficients": {"a": 1.0, "b": 1.0}}
    out = lr_decide(x, model, ["a", "b"])
    assert out["action"].iloc[0] == 1 and pd.isna(out["action"].iloc[1])


def test_coefficients_round_trip_through_the_model_folder(tmp_path, monkeypatch):
    import models.model_utils as mu
    monkeypatch.setattr(mu, "STORAGE", str(tmp_path))
    coef = {"intercept": 0.2, "coefficients": {"a": 1.5, "b": -0.3}}
    mu.save_model("logreg_test", ["a", "b"], {"areas": ["EE"]}, coefficients=coef)
    meta, scaler, model = mu.load_model("logreg_test")
    assert model == coef and meta["features"] == ["a", "b"] and scaler is None


def test_coefficients_that_do_not_match_the_features_are_refused(tmp_path, monkeypatch):
    import pytest
    import models.model_utils as mu
    monkeypatch.setattr(mu, "STORAGE", str(tmp_path))
    coef = {"intercept": 0.0, "coefficients": {"b": 1.0, "a": 1.0}}
    with pytest.raises(ValueError):
        mu.save_model("logreg_bad", ["a", "b"], {}, coefficients=coef)


def test_trailing_columns_follow_the_rule_and_refuse_a_short_window():
    from models.features import trailing_columns
    x = pd.DataFrame({"always_buy_d1": [1.0, 1.0], "always_buy_d2": [2.0, None],
                      "always_buy_d3": [-1.0, 0.0],
                      "always_sell_d1": [-1.0, -1.0], "always_sell_d2": [-2.0, 0.0],
                      "always_sell_d3": [1.0, 0.0]})
    x = trailing_columns(x)
    assert x["trail_diff"].iloc[0] == 4.0 and x["trail_bias"].iloc[0] == 1
    assert pd.isna(x["trail_diff"].iloc[1]) and pd.isna(x["trail_bias"].iloc[1])


# --- a model in shadow (2026-11-22) -------------------------------------------

def _fake_model(name):
    meta = {"features": ["trail_bias"], "sensitivity": 0.05, "n_lags": 3, "step": "15min"}
    coefficients = {"intercept": 0.0, "coefficients": {"trail_bias": 1.0}}
    return meta, None, coefficients


def test_shadow_decision_carries_both_actions(monkeypatch):
    """p = sigmoid(trail_bias): 0.73 where the rule says buy, 0.5 where it says
    sell. Without sensitivity 0.5 is a sell; with 0.05 it is no position."""
    import models.model_utils as mu
    import src.workflow_compute as wc
    monkeypatch.setattr(mu, "load_model", _fake_model)
    m = _m(_days("2026-09-22", "2026-09-25"))
    t = pd.Timestamp("2026-09-26 09:40", tz="UTC")
    spec = {"mode": "shadow", "name": "fake"}
    x = wc.shadow_decision(m, t, "NL", "logistic_regression", spec)
    assert len(x) == 96
    buy = x["trail_bias"] == 1
    assert (x.loc[buy, "action"] == 1).all() and (x.loc[buy, "action_sens"] == 1).all()
    assert (x.loc[~buy, "action"] == -1).all() and (x.loc[~buy, "action_sens"] == 0).all()
    assert (x["model_name"] == "fake").all() and (x["sensitivity"] == 0.05).all()


def test_a_failing_shadow_model_does_not_stop_the_rule(monkeypatch, caplog):
    import logging
    import src.config as cfg
    import src.workflow_compute as wc

    def models_of(area):
        return {"logistic_regression": {"mode": "shadow", "name": "broken"}}

    def broken(*args, **kwargs):
        raise RuntimeError("model file missing")

    monkeypatch.setattr(cfg, "zone_models", models_of)
    monkeypatch.setattr(wc, "shadow_decision", broken)
    m = _m(_days("2026-09-22", "2026-09-25"))
    t = pd.Timestamp("2026-09-26 09:40", tz="UTC")
    with caplog.at_level(logging.ERROR):
        wc.shadow_models(m, t, "NL", logging.getLogger("test"), persist=False)
    assert "FAILED" in caplog.text


def test_a_model_with_an_unknown_mode_is_refused(monkeypatch):
    import pytest
    import src.config as cfg

    def zone_of(area):
        return {"models": {"logistic_regression": {"mode": "maybe", "name": "x"}}}

    monkeypatch.setattr(cfg, "zone", zone_of)
    with pytest.raises(ValueError):
        cfg.zone_models("EE")


def test_price_change_from_d3_to_d2():
    from models.features import derived_columns
    x = pd.DataFrame({"price_da_d2": [110.0, None], "price_da_d3": [100.0, 90.0]})
    x = derived_columns(x)
    assert x["price_da_d3_d2_diff"].iloc[0] == 10.0
    assert pd.isna(x["price_da_d3_d2_diff"].iloc[1])


def test_price_change_from_d2_to_d1():
    from models.features import derived_columns
    x = pd.DataFrame({"price_da_d1": [95.0, 80.0], "price_da_d2": [100.0, None]})
    x = derived_columns(x)
    assert x["price_da_d2_d1_diff"].iloc[0] == -5.0
    assert pd.isna(x["price_da_d2_d1_diff"].iloc[1])


def test_an_event_switches_at_its_instant_and_interacts_with_trail_bias(monkeypatch):
    import src.config as cfg
    from models.features import event_columns

    def events_of(area):
        return {"picasso": pd.Timestamp("2026-09-30 22:00", tz="UTC")} if area == "HU" else {}

    monkeypatch.setattr(cfg, "zone_events", events_of)
    ts = pd.to_datetime(["2026-09-30 21:45", "2026-09-30 22:00", "2026-10-01 10:00"], utc=True)
    x = pd.DataFrame({"delivery_ts": ts, "trail_bias": pd.array([1, 1, 0], dtype="Int64")})
    hu = event_columns(x, "HU")
    assert hu["event_picasso"].tolist() == [0, 1, 1]
    assert hu["event_picasso_x_trail_bias"].tolist() == [0, 1, 0]
    ee = event_columns(x, "EE")
    assert "event_picasso" not in ee.columns


def test_an_event_without_a_timezone_is_refused(monkeypatch):
    import pytest
    import src.config as cfg

    def zone_of(area):
        return {"events": {"picasso": "2026-10-01 00:00"}}

    monkeypatch.setattr(cfg, "zone", zone_of)
    with pytest.raises(ValueError):
        cfg.zone_events("HU")


def test_a_bid_outside_its_price_limit_is_not_filled():
    from models.actions import filled_action
    action = pd.array([1, 1, -1, -1, 0, None], dtype="Int8")
    price = [800.0, 700.0, -300.0, -100.0, 900.0, 900.0]
    out = filled_action(pd.Series(action), price, buy_limit=750, sell_limit=-250)
    assert out.tolist()[:5] == [0, 1, 0, -1, 0]
    assert pd.isna(out.iloc[5])


def test_no_limit_changes_nothing():
    from models.actions import filled_action
    action = pd.Series(pd.array([1, -1], dtype="Int8"))
    out = filled_action(action, [5000.0, -5000.0])
    assert out.tolist() == [1, -1]


def test_negative_price_indicator_and_its_slope():
    from models.features import derived_columns
    x = pd.DataFrame({"price_da_d1": [-30.0, 40.0, None]})
    x = derived_columns(x)
    assert x["price_da_d1_negative"].iloc[0] == 1 and x["price_da_d1_negative"].iloc[1] == 0
    assert x["price_da_d1_x_negative"].iloc[0] == -30.0 and x["price_da_d1_x_negative"].iloc[1] == 0.0
    assert pd.isna(x["price_da_d1_negative"].iloc[2]) and pd.isna(x["price_da_d1_x_negative"].iloc[2])

