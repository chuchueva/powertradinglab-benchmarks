"""FX known-by and the live carry of the last known rate (owner's rule (b), 2026-10-27)."""
import pandas as pd

from src.parse_fx import build_fx_daily, fx_as_of

UTC = "UTC"


def _obs(days, cur="RON", rate=5.0):
    return pd.DataFrame({"rate_date": pd.DatetimeIndex(days, tz=UTC),
                         "currency": cur, "rate": rate})


def test_knownby_is_the_fetch_in_the_raw_era_and_the_rule_before_it():
    obs = _obs(["2026-09-18", "2026-09-21", "2026-09-22"])
    seen = pd.DataFrame({"currency": ["RON", "RON"],
                         "rate_date": pd.DatetimeIndex(["2026-09-18", "2026-09-22"], tz=UTC),
                         "knownby_ts": pd.DatetimeIndex(["2026-09-22 10:16", "2026-09-22 14:00"], tz=UTC)})
    fx = build_fx_daily(obs, first_seen=seen).set_index(["currency", "rate_date"])["knownby_ts"]
    # before the first saved fetch: 17:00 Brussels (CEST) = 15:00 UTC
    assert fx[("RON", pd.Timestamp("2026-09-18", tz=UTC))] == pd.Timestamp("2026-09-18 15:00", tz=UTC)
    # the weekend carries Friday's knownby
    assert fx[("RON", pd.Timestamp("2026-09-20", tz=UTC))] == pd.Timestamp("2026-09-18 15:00", tz=UTC)
    # raw era: what we actually saw
    assert fx[("RON", pd.Timestamp("2026-09-22", tz=UTC))] == pd.Timestamp("2026-09-22 14:00", tz=UTC)


def test_todays_hours_get_the_last_known_rate_before_the_ecb_publishes():
    fx = build_fx_daily(_obs(["2026-10-19", "2026-10-20"]))     # Mon, Tue
    out = fx_as_of(fx, pd.Timestamp("2026-10-21 09:40", tz=UTC))  # Wed morning
    ron = out[out["currency"] == "RON"].set_index("rate_date")
    wed = ron.loc[pd.Timestamp("2026-10-21", tz=UTC)]
    assert wed["source"] == "carried" and wed["rate"] == 5.0
    assert wed["knownby_ts"] == ron.loc[pd.Timestamp("2026-10-20", tz=UTC), "knownby_ts"]
    assert ron.index.max() == pd.Timestamp("2026-10-21", tz=UTC), "never past today"


def test_a_rate_not_yet_known_is_not_used():
    fx = build_fx_daily(_obs(["2026-10-19", "2026-10-20"]))
    out = fx_as_of(fx, pd.Timestamp("2026-10-20 09:40", tz=UTC))  # Tue's rate comes at 15:00
    tue = out[(out["currency"] == "RON") & (out["rate_date"] == pd.Timestamp("2026-10-20", tz=UTC))]
    assert tue["source"].tolist() == ["carried"], "Monday's rate, carried — not Tuesday's"


def test_a_due_rate_that_is_missing_is_not_carried_over():
    fx = build_fx_daily(_obs(["2026-10-19"]))                   # Tue never arrived
    out = fx_as_of(fx, pd.Timestamp("2026-10-21 09:40", tz=UTC))
    assert out.loc[out["currency"] == "RON", "rate_date"].max() == pd.Timestamp("2026-10-19", tz=UTC)


def test_a_retired_currency_does_not_come_back():
    fx = build_fx_daily(pd.concat([_obs(["2026-10-20"]), _obs(["2022-12-30"], cur="HRK", rate=7.5)]))
    out = fx_as_of(fx, pd.Timestamp("2026-10-21 09:40", tz=UTC))
    assert out.loc[out["currency"] == "HRK", "rate_date"].max() < pd.Timestamp("2023-01-02", tz=UTC)
