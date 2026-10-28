"""
The matrix — one row per (delivery_ts, knownby_ts), every input a column.

    delivery_ts   the delivery HOUR the row decides (UTC, start of the hour)
    knownby_ts    the instant of the decision; everything in the row was
                  known strictly before it (as_of). Same meaning as in the
                  forecasts table: no third kind of time.

Trailing-bias window, per row, tied to timestamps (not "the last N rows"):

    d1   the freshest COMPLETE observation of the same UTC hour, taken from
         delivery_ts - 1 .. - 4 days, and only from an allowed market day:
           fresh=True  (live):     today or yesterday (`need`)
           fresh=False (backtest): yesterday (`need`) only
         Usually that is -1 day (today) or -2 days (yesterday). The -3 days
         candidate exists for the daylight-saving week: on the 25-hour day
         the same UTC hour two days back can fall into TODAY's market day,
         which a zone without intraday publication does not have yet. On the
         23-hour day one UTC hour does not occur in yesterday's market day at
         all; for that hour only, the day before yesterday is allowed.
    dk   d1 - (k - 1) days, exactly; never "the next one available"

    always_buy_dk, always_sell_dk   hourly sum, EUR per 1 MW; <NA> unless the
                                    hour is whole (4 quarter-hours)
    delivery_ts_dk                  the hour the window asks for
    ptu_dk                          quarter-hours of it actually present

A hole anywhere in the window is an <NA> in a named cell, and the rule then
gives no action. A partly published hour never enters the window.

UTC hours are always 4 quarter-hours long, and one day back is 24 h back;
the market-day test above is the only place local time enters.
"""
import pandas as pd

import src.benchmarks_core as C
import src.config as cfg
import src.workflow_compute as wc
from src.period import market_tz
from src.workflow import load_store

PTU_PER_HOUR = 4
N_LAGS = 14   # the matrix carries d1..d14; each model takes what it needs
DAY = pd.Timedelta(days=1)


def hourly(m: pd.DataFrame) -> pd.DataFrame:
    """Quarter-hour matrix -> one row per delivery hour: sums and PTU count."""
    h = m.assign(hour_ts=m["delivery_ts"].dt.floor("h"))
    return h.groupby("hour_ts", as_index=False).agg(
        always_buy=("always_buy", "sum"),
        always_sell=("always_sell", "sum"),
        ptu=("delivery_ts", "size"),
    )


def rows_for(knownby_ts: pd.Timestamp, area: str) -> pd.DataFrame:
    """The empty matrix rows of one decision: the delivery hours of the day
    the auction closing on knownby_ts serves, plus `need` (yesterday's market
    day relative to the decision, naive date)."""
    hours = wc._target_grid(knownby_ts, area).floor("h").unique()
    rows = pd.DataFrame({"delivery_ts": hours})
    rows["knownby_ts"] = knownby_ts
    rows["area"] = area
    rows["need"] = pd.Timestamp(wc.input_day_for(knownby_ts, area))
    return rows


def window(rows: pd.DataFrame, h: pd.DataFrame, n_lags: int, fresh: bool,
           area: str) -> pd.DataFrame:
    """Add the d1..dN columns to `rows` (delivery_ts, need; any number of
    decisions at once)."""
    whole = h[h["ptu"] == PTU_PER_HOUR].set_index("hour_ts")
    ptu = h.set_index("hour_ts")["ptu"]
    need = rows["need"]
    today = need + DAY

    # Oldest candidate first, so the newest acceptable one wins. Where none
    # is acceptable, the window is still laid out from yesterday's slot
    # (delivery_ts - 2 days) so the row shows WHAT was missing; its values
    # are then all <NA>.
    d1 = rows["delivery_ts"] - 2 * DAY
    found = pd.Series(False, index=rows.index)
    backs = (4, 3, 2, 1)
    days = {b: wc.market_day(rows["delivery_ts"] - b * DAY, area) for b in backs}

    def is_allowed(day):
        return ((day == need) | (day == today)) if fresh else (day == need)

    # Spring daylight-saving day (23 h): one UTC hour does not occur in
    # yesterday's market day AT ALL, by the calendar. Only for such a row is
    # the day before yesterday allowed — a hole in the DATA never qualifies.
    calendar_gap = ~pd.concat([is_allowed(days[b]) for b in backs], axis=1).any(axis=1)

    for back in backs:
        ts = rows["delivery_ts"] - back * DAY
        allowed = is_allowed(days[back]) | (calendar_gap & (days[back] == need - DAY))
        ok = allowed & ts.isin(whole.index)
        d1 = d1.where(~ok, ts)
        found = found | ok

    rows = rows.copy()
    rows["d1_today"] = (found & (wc.market_day(d1, area) == today)).astype(int)
    for k in range(1, n_lags + 1):
        ts = d1 - (k - 1) * DAY
        rows[f"delivery_ts_d{k}"] = ts
        rows[f"ptu_d{k}"] = ts.map(ptu).fillna(0).astype(int)
        rows[f"always_buy_d{k}"] = ts.map(whole["always_buy"]).where(found)
        rows[f"always_sell_d{k}"] = ts.map(whole["always_sell"]).where(found)
    return rows.drop(columns=["need"])


def matrix_from(m: pd.DataFrame, knownby_ts: pd.Timestamp, area: str,
                n_lags: int = None, fresh: bool = True) -> pd.DataFrame:
    """
    Matrix rows for one decision, from the quarter-hour matrix `m` as it
    stood at knownby_ts (already as_of, valued, with add_pnl).

    This is the function the live forecast calls.
    """
    n_lags = n_lags or N_LAGS
    x = window(rows_for(knownby_ts, area), hourly(m), n_lags, fresh, area)
    return calendar(x, area)


def matrix_at(da: pd.DataFrame, im: pd.DataFrame, knownby_ts: pd.Timestamp,
              area: str, n_lags: int = None, fresh: bool = True) -> pd.DataFrame:
    """matrix_from, starting from the whole store (as_of applied here)."""
    m = wc._matrix_from_legs(C.as_of(da, knownby_ts), C.as_of(im, knownby_ts),
                             knownby_ts, area)
    return matrix_from(m, knownby_ts, area, n_lags, fresh)


def build_matrix(area: str, knownby_list, n_lags: int = None, fresh: bool = True) -> pd.DataFrame:
    """Matrix rows for many decisions. The store is read once."""
    n_lags = n_lags or N_LAGS
    da, im = load_store(area)
    parts = [matrix_at(da, im, pd.Timestamp(t), area, n_lags, fresh) for t in knownby_list]
    return pd.concat(parts, ignore_index=True)


def calendar(x: pd.DataFrame, area: str) -> pd.DataFrame:
    """
    Calendar columns of the delivery hour — known at any instant, so they are
    inputs. Local time is the market's own (the zone's timezone).

        hour_utc, hour_local, weekday_local (0 = Monday), month_local,
        is_weekend

    Public holidays are NOT here yet: they need a per-country calendar, and
    the `holidays` package is not a dependency. Placeholder: is_holiday = 0.
    """
    local = x["delivery_ts"].dt.tz_convert(market_tz(area))
    x = x.copy()
    x["hour_utc"] = x["delivery_ts"].dt.hour
    x["hour_local"] = local.dt.hour
    x["weekday_local"] = local.dt.weekday
    x["month_local"] = local.dt.month
    x["is_weekend"] = (x["weekday_local"] >= 5).astype(int)
    x["is_holiday"] = 0          # placeholder, see docstring
    return x


def decision_instants(start, end) -> list[pd.Timestamp]:
    """One decision per day in [start, end], at run_time_utc."""
    hh, mm = (int(v) for v in cfg.RUN_TIME_UTC.split(":"))
    days = pd.date_range(pd.Timestamp(start).date(), pd.Timestamp(end).date(), freq="D")
    return [pd.Timestamp(d.date(), tz="UTC") + pd.Timedelta(hours=hh, minutes=mm)
            for d in days]


def _latest_matrix(area: str) -> pd.DataFrame:
    """The store's latest versions, valued, with add_pnl."""
    da, im = load_store(area)
    now = pd.Timestamp.now(tz="UTC")
    return wc._matrix_from_legs(C.as_of(da, now), C.as_of(im, now), now, area)


def backtest_matrix(areas, start, end, n_lags: int = None) -> pd.DataFrame:
    """
    The training / backtest matrix — inputs only. One decision per day in
    [start, end], for one zone or a list of zones (stacked, `area` column).
    By the GENERAL rule: d1 is always yesterday (fresh=False), values are the
    store's latest versions.

    Why not as_of per decision: before the service ran, the store's
    knownby_ts comes from the bulk archive, whose stamps are revision times,
    not publication times. Live may differ (fresher data); that is the point
    of live.

    The outcome of each hour is NOT here: it is the benchmark itself, see
    outcome().
    """
    areas = [areas] if isinstance(areas, str) else list(areas)
    n_lags = n_lags or N_LAGS
    parts = []
    for area in areas:
        h = hourly(_latest_matrix(area))
        rows = pd.concat([rows_for(t, area) for t in decision_instants(start, end)],
                         ignore_index=True)
        parts.append(calendar(window(rows, h, n_lags, fresh=False, area=area), area))
    return pd.concat(parts, ignore_index=True)


def outcome(areas) -> pd.DataFrame:
    """
    What each delivery hour actually paid, per 1 MW, from the benchmark
    (benchmarks_core.add_pnl): always_buy, always_sell, ideal, devil, ptu.
    Store's latest versions. Join to the matrix on (area, delivery_ts).
    """
    areas = [areas] if isinstance(areas, str) else list(areas)
    parts = []
    for area in areas:
        m = _latest_matrix(area)
        h = (m.assign(delivery_ts=m["delivery_ts"].dt.floor("h"))
               .groupby("delivery_ts", as_index=False)
               .agg(always_buy=("always_buy", "sum"), always_sell=("always_sell", "sum"),
                    ideal=("ideal", "sum"), devil=("devil", "sum"),
                    ptu=("always_buy", "size")))
        h.insert(0, "area", area)
        parts.append(h)
    return pd.concat(parts, ignore_index=True)


def lag_columns(n: int) -> list[str]:
    """always_buy_d1..dn and always_sell_d1..dn — the lag inputs."""
    return ([f"always_buy_d{k}" for k in range(1, n + 1)]
            + [f"always_sell_d{k}" for k in range(1, n + 1)])


CALENDAR_COLUMNS = ["hour_local", "weekday_local", "month_local", "is_weekend", "is_holiday"]


def area_columns(x: pd.DataFrame, areas: list[str]) -> pd.DataFrame:
    """One-hot zone columns area_<ZONE> for a model trained on several zones.
    `areas` is fixed by the model (its meta.json), not by what x contains."""
    x = x.copy()
    for a in areas:
        x[f"area_{a}"] = (x["area"] == a).astype(int)
    return x


def fit_scaler(dd: pd.DataFrame, features: list[str]) -> dict:
    """Mean and standard deviation per feature, from the TRAINING rows only."""
    return {f: {"mean": float(dd[f].mean()), "std": float(dd[f].std())} for f in features}


def scale(x: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    """(value - mean) / std for every feature in the scaler; the same function
    for training and for the live run. A feature with std 0 is left at 0."""
    x = x.copy()
    for f, s in scaler.items():
        std = s["std"] if s["std"] else 1.0
        x[f] = (x[f] - s["mean"]) / std
    return x
