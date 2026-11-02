"""
Benchmark CORE — pure functions, no I/O. Shared by the report
(report/benchmarks.py), the live workflow (score) and the model checks.

    da, im = as_of(da, current_ts), as_of(im, current_ts)   # live path only
    matrix = build_matrix(da, im, fx, on=...)      # DA + IM(EUR) per 15-min PTU
    matrix = add_pnl(matrix)                       # always_buy/sell, ideal, devil
    action_pnl(action, always_buy, always_sell)    # what an action earned

The trailing-bias DECISION is not here. It lives in models/trailing_bias.py
and runs on the matrix from models/features.py — the same two for the live
forecast, the backtest and training. One exception, kept for the report:

    actions = forecast_actions(matrix, n_days)     # report only, shift(lag_days)
    matrix = apply_actions(matrix, actions)        # stat_action_pnl per PTU

forecast_actions is the report's shortcut (d1 = the day before yesterday,
hard-coded as shift(2)); tests/test_models.py checks that it agrees with the
matrix backtest hour by hour. It goes when the report moves to the matrix.
Definitions: docs/METHOD.md.

The two ways to be causal — pick ONE, never both:
  - as_of(df, current_ts): keeps only what was published by current_ts, so the
    lag emerges from reality. The live workflow.
  - lag_days=2 in forecast_actions: the backtest shortcut that hardcodes the
    same 48 h. The report, which does not filter by knownby_ts.
Applying the filter AND the shift would lag the signal twice over.

Notes:
  - `0` in an IM price is a REAL 0 EUR/MWh (NL). min/max is label-agnostic.
  - build_matrix keeps BOTH knownby timestamps (knownby_ts_da / knownby_ts_im)
    when present; the report ignores them.
"""


import numpy as np
import pandas as pd

H = 0.25  # MWh per 15-min PTU at 1 MW

# Germany relabel (2023): same market, both codes changed.
IM_ALIAS = {"DE": "DE_LU"}
IM_ALIAS_EIC = {"10Y1001A1001A83F": "10Y1001A1001A82H"}

_EUR_COLS = ["price_im_pos_eur", "price_im_neg_eur"]
YEARLY_COLS = ["ideal", "devil", "always_sell", "always_buy", "stat_action_pnl"]


def as_of(df: pd.DataFrame, current_ts: pd.Timestamp,
          keys: list[str] | tuple[str, ...] = ("delivery_ts",)) -> pd.DataFrame:
    """
    Point-in-time view: what a caller standing at `current_ts` could have seen.

    TWO steps, and the second one is the one that is easy to forget:
      1. keep rows published strictly BEFORE current_ts (knownby_ts < current_ts);
      2. per `keys`, keep the row with the LARGEST knownby_ts — i.e. the newest
         version that existed by then, not the first one.

    Step 2 only bites once a series carries several versions per delivery_ts
    (as-known vs as-final, calc_iteration 0/1). With one version it is a no-op,
    which is exactly why omitting it stays invisible until it corrupts results.

    Works on any table with a knownby_ts, including our OWN outputs: the
    forecasts table is scored through the same filter, so score() can only ever
    use an action that had already been issued.
    """
    keys = list(keys)
    if df.empty:
        return df.copy()
    assert str(df["knownby_ts"].dt.tz) == "UTC", "knownby_ts is not UTC"
    assert getattr(current_ts, "tzinfo", None) is not None, "current_ts is not tz-aware"

    known = df.loc[df["knownby_ts"] < current_ts]
    return (known.sort_values("knownby_ts")
                 .drop_duplicates(subset=keys, keep="last")
                 .sort_values(keys)
                 .reset_index(drop=True))


def apply_im_alias(im: pd.DataFrame) -> pd.DataFrame:
    """Map Germany's IM codes to the DA bidding-zone codes (area + area_eic)."""
    im = im.copy()
    if "area" in im.columns:
        im["area"] = im["area"].replace(IM_ALIAS)
    if "area_eic" in im.columns:
        im["area_eic"] = im["area_eic"].replace(IM_ALIAS_EIC)
    return im


def _raise_missing_fx(m: pd.DataFrame, fx: pd.DataFrame) -> None:
    """
    Explain WHICH dates are uncovered, not just which currency is involved.

    The first version of this named the currency only ("Missing FX rate for
    ['EUR']"), which reads as a currency problem when it is almost always a
    coverage problem: the FX table simply ends before the price data does,
    because the ECB file is a manual download that goes stale. Say the spans.
    """
    parts = []
    for leg in ("da", "im"):
        gap = m.loc[m[f"rate_{leg}"].isna()]
        for cur, g in gap.groupby(f"currency_{leg}"):
            have = fx.loc[fx["currency"] == cur, "rate_date"]
            span = (f"{have.min().date()}..{have.max().date()}" if len(have)
                    else "no rows at all")
            parts.append(f"{leg.upper()} in {cur}: need "
                         f"{g['rate_date'].min().date()}..{g['rate_date'].max().date()}, "
                         f"FX table has {span}")
    raise ValueError("FX table does not cover the price data. "
                     + "; ".join(parts)
                     + ". Refresh data/fx/ from the ECB and re-run parse_fx.py.")


def missing_fx(m: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(leg, currency, UTC date) for every rate the matrix could not find."""
    out = set()
    for leg in ("da", "im"):
        gap = m.loc[m[f"rate_{leg}"].isna()]
        for (cur, d), _ in gap.groupby([f"currency_{leg}", "rate_date"]):
            out.add((leg.upper(), cur, f"{d:%Y-%m-%d}"))
    return sorted(out, key=lambda t: (t[2], t[0], t[1]))


def build_matrix(da: pd.DataFrame, im: pd.DataFrame, fx: pd.DataFrame,
                 on: list[str] | tuple[str, ...] = ("delivery_ts",),
                 on_missing_fx: str = "raise") -> pd.DataFrame:
    """
    Join DA + IM on `on`, convert BOTH legs to EUR via the daily FX table.

    Every price carries its own currency: most zones are EUR, but some publish
    in local currency (e.g. UA_IPS DA & IM in UAH). We convert each leg by ITS
    OWN currency (price_eur = price_local / rate). A leg without a `currency`
    column is assumed EUR (rate 1.0) — keeps the EUR-only NL store working.
    The returned matrix is fully EUR-denominated: `price_da` is overwritten with
    its EUR value (whole matrix is EUR per this module's contract).

    `on` is the merge key: the report uses ["delivery_ts","area","area_eic"];
    the single-zone workflow store uses ["delivery_ts"]. Both are 1:1 after the
    store/history de-dup. knownby_ts columns (if present) are kept, renamed
    knownby_ts_da / knownby_ts_im, and never enter the pnl maths.

    `on_missing_fx` decides what a missing rate means, and the two callers want
    different things. The REPORT wants "raise": a frozen annual table must never
    be built on a partial valuation, and the loud error naming both spans is
    what has already caught a stale FX file once. The LIVE path wants "drop":
    PTU that cannot be valued are left out, the dates are listed in
    `attrs["fx_missing"]`, and the completeness gates that already exist do the
    rest — the forecast refuses (leg FX), the score waits for the rate. Nothing
    is ever filled in; "drop" only means "not today".
    """
    on = list(on)
    da = da.rename(columns={"knownby_ts": "knownby_ts_da"}) if "knownby_ts" in da.columns else da.copy()
    im = im.rename(columns={"knownby_ts": "knownby_ts_im"}) if "knownby_ts" in im.columns else im.copy()

    # Give each leg its own currency column (default EUR), then drop the raw
    # `currency` so the DA/IM join does not collide on it.
    da["currency_da"] = da["currency"] if "currency" in da.columns else "EUR"
    im["currency_im"] = im["currency"] if "currency" in im.columns else "EUR"
    da = da.drop(columns=["currency"], errors="ignore")
    im = im.drop(columns=["currency"], errors="ignore")

    m = da.merge(im, how="inner", on=on, validate="one_to_one")
    m = m.assign(rate_date=m["delivery_ts"].dt.floor("D"))

    # One FX table, joined once per leg on (rate_date, that leg's currency).
    fx_da = fx.rename(columns={"currency": "currency_da", "rate": "rate_da"})[["rate_date", "currency_da", "rate_da"]]
    fx_im = fx.rename(columns={"currency": "currency_im", "rate": "rate_im"})[["rate_date", "currency_im", "rate_im"]]
    m = m.merge(fx_da, how="left", on=["rate_date", "currency_da"], validate="many_to_one")
    m = m.merge(fx_im, how="left", on=["rate_date", "currency_im"], validate="many_to_one")

    # EUR is the UNIT, not a rate: the euro is worth one euro on every date that
    # has ever existed or will. So it is set here rather than looked up. The FX
    # table does carry EUR=1.0 rows, but only across the calendar of whichever
    # ECB file was last downloaded — which made euro zones fail the moment the
    # data ran past that download, for a conversion that needs no source at all.
    for leg in ("da", "im"):
        m.loc[m[f"currency_{leg}"] == "EUR", f"rate_{leg}"] = 1.0

    gap = m["rate_da"].isna() | m["rate_im"].isna()
    missing = []
    if gap.any():
        if on_missing_fx == "raise":
            _raise_missing_fx(m, fx)
        if on_missing_fx != "drop":
            raise ValueError(f"on_missing_fx must be 'raise' or 'drop', not {on_missing_fx!r}")
        missing = missing_fx(m)
        m = m.loc[~gap]

    out = m.assign(price_da=m["price_da"] / m["rate_da"],             # DA -> EUR
                   price_im_pos_eur=m["price_im_pos"] / m["rate_im"],
                   price_im_neg_eur=m["price_im_neg"] / m["rate_im"])
    out.attrs["fx_missing"] = missing
    return out


def add_pnl(m: pd.DataFrame, h: float = H) -> pd.DataFrame:
    """Add the four per-PTU benchmarks (EUR per 1 MW)."""
    buy = m[_EUR_COLS].max(axis=1)    # price to buy in IM (worse side)
    sell = m[_EUR_COLS].min(axis=1)   # price to sell in IM (worse side)
    price_da = m["price_da"]
    pnls = pd.concat([sell - price_da, price_da - buy], axis=1)  # long, short
    return m.assign(
        always_buy=(sell - price_da) * h,    # decision BUY on DA; IM settles selling
        always_sell=(price_da - buy) * h,    # decision SELL on DA; IM settles buying
        ideal=pnls.max(axis=1) * h,          # best forced direction (no clip)
        devil=pnls.min(axis=1) * h,          # worst forced direction
    )


def _hourly(m: pd.DataFrame) -> pd.DataFrame:
    """PTU pnl -> hourly totals [hour_ts, always_buy, always_sell, hod]."""
    hourly = (m.assign(hour_ts=m["delivery_ts"].dt.floor("h"))
                .groupby("hour_ts", as_index=False)
                .agg(always_buy=("always_buy", "sum"),
                     always_sell=("always_sell", "sum")))
    return hourly.assign(hod=hourly["hour_ts"].dt.hour)


def forecast_actions(m: pd.DataFrame, n_days: int, lag_days: int = 2) -> pd.DataFrame:
    """
    Hourly trailing-bias signal -> DA action for hours PRESENT in `m` (backtest).

    Returns [hour_ts (UTC, = delivery_ts floored to hour), stat_action (+1/-1)].
    Causal via rolling(n_days) over the same hour-of-day, then shift(lag_days).

    lag_days=2 is the report's shortcut for the as_of rule: deciding for day D on
    D-1 morning, the freshest fully settled day is D-2. Pass lag_days=0 when the
    frame has already been filtered by as_of, or the signal is lagged twice.
    Requires add_pnl() first (needs always_buy / always_sell).
    """
    hourly = _hourly(m).sort_values(["hod", "hour_ts"]).reset_index(drop=True)

    grp = hourly.groupby("hod")
    hourly["sig_buy"] = grp["always_buy"].transform(
        lambda s: s.rolling(n_days, min_periods=n_days).mean().shift(lag_days))
    hourly["sig_sell"] = grp["always_sell"].transform(
        lambda s: s.rolling(n_days, min_periods=n_days).mean().shift(lag_days))

    hourly = hourly.dropna(subset=["sig_buy", "sig_sell"])
    action = np.where(hourly["sig_buy"] > hourly["sig_sell"], 1, -1)
    return pd.DataFrame({"hour_ts": hourly["hour_ts"].to_numpy(),
                         "stat_action": action})


def action_pnl(action, always_buy, always_sell) -> np.ndarray:
    """
    What an action earned, per 1 MW — the ONE place this is computed
    (live score, report, model checks).

        +1    -> always_buy   (bought on the day-ahead)
        -1    -> always_sell  (sold on the day-ahead)
         0    -> 0.0          (decided to stay out: no position, no pnl)
        <NA>  -> NaN          (no decision was possible)
    """
    act = pd.array(action, dtype="Float64").to_numpy(dtype="float64", na_value=np.nan)
    return np.where(act == 1, np.asarray(always_buy, dtype=float),
                    np.where(act == -1, np.asarray(always_sell, dtype=float),
                             np.where(act == 0, 0.0, np.nan)))


def apply_actions(m: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Attach the chosen action's realised pnl per PTU (stat_action_pnl).
    An hour without an action is NaN now, not a sell (2026-10-28)."""
    m = m.assign(hour_ts=m["delivery_ts"].dt.floor("h"))
    m = m.merge(actions, how="left", on="hour_ts", validate="many_to_one")
    m["stat_action_pnl"] = action_pnl(m["stat_action"], m["always_buy"], m["always_sell"])
    return m
