"""
FX parser + daily cache: ECB euro reference rates -> one long feather.

Source: ECB gesmes XML in data/fx/ (eurofxref-hist.xml = full history since 1999;
eurofxref-daily.xml = latest day). Both share the same nested-Cube structure:
    <Cube time="YYYY-MM-DD"><Cube currency="USD" rate="1.1367"/> ... </Cube>
`rate` = units of the currency per 1 EUR, so conversion to EUR is:
    price_eur = price_local / rate

Output: data/processed/fx_daily.feather, one row per (currency, calendar day):
    rate_date  (UTC, tz-aware, midnight of the day the rate applies to)
    currency   (ISO code)
    rate       (units of currency per 1 EUR)
    source     ('ecb'      = published that day,
                'ffill'    = carried over a weekend / TARGET holiday,
                'peg'      = EUR-pegged currency board rate (not in ECB),
                'euro_conv'= locked euro conversion rate for the tail day(s)
                             before a currency's euro adoption (e.g. HRK 2022-12-31),
                'eur'      = EUR itself, always 1.0)

Design / decisions:
  - UTC-only: rate_date is tz-aware UTC midnight. No `knownby_ts` here - ECB rates
    are a canonical daily reference, not snapshotted/revised like our price feeds,
    so the date IS the key. Provenance is the ECB file itself.
  - ECB publishes on business days only. We forward-fill each currency over the
    calendar (weekends/holidays), but ONLY within that currency's own active span
    (first..last observed date) - never before it appears nor after it stops
    (e.g. HRK ends when Croatia joined the euro). No fabricated rates.
  - Pegged currencies NOT in ECB: BAM and BGN are fixed to the euro by currency
    board (1 EUR = 1.95583). We supply these as constants for any day the ECB does
    not cover them.
  - EUR is included as rate 1.0 so a price already in EUR converts uniformly.

To join to prices (in the benchmark): map a price row's UTC delivery date
    rate_date = delivery_ts.dt.floor("D")
then merge on (currency, rate_date); price_eur = price_local / rate.

Run:
    uv run python src/parse_fx.py
"""

import glob
import os

from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dateutil.easter import easter
from lxml import etree

# Every path in this project is anchored to the REPOSITORY, never to the
# current directory. Relative paths do not fail when a script is run from
# elsewhere — they quietly build a parallel tree (a stray data/raw and logs/
# appeared under modelling/ exactly this way, and hid two fetched files long
# enough to send us looking for a network fault that was not there).
ROOT = Path(__file__).resolve().parents[1]

FX_DIR = ROOT / "data/fx"
OUT_DIR = ROOT / "data/processed"
OUT = os.path.join(OUT_DIR, "fx_daily.feather")

_NS = {"e": "http://www.ecb.int/vocabulary/2002-08-01/eurofxref"}
# EUR-pegged currencies absent (or only partly present) in ECB reference rates.
_PEGS = {"BAM": 1.95583, "BGN": 1.95583}
# Currencies retired at euro adoption. ECB stops quoting a day or two before the
# switch, but IM can trade the currency up to the last pre-adoption day. We fill
# that tail gap (last ECB quote .. day before adoption) with the LOCKED euro
# conversion rate. HRK: Croatia joined the euro 2023-01-01 at 1 EUR = 7.53450 HRK.
_EURO_LEGACY = {"HRK": {"conv_rate": 7.53450, "adopt_date": "2023-01-01"}}


# --------------------------------------------------------------------------- #
# When the ECB does NOT publish
# --------------------------------------------------------------------------- #
# Measured, not assumed (2026-09-19, eurofxref-hist.xml 1999-01-04..2026-09-18):
# the weekdays with no reference rate are exactly the TARGET closing days —
# 1 January, Good Friday, Easter Monday, 1 May, 25 and 26 December — and
# nothing else. The rates follow TARGET, not the ECB office calendar: Ascension,
# Whit Monday, Corpus Christi, Schuman day, 3 October, 1 November, 24 and 31
# December all carry rates although the office is shut.
#
# So the calendar is COMPUTED — four fixed dates and two from Easter — and there
# is no hand-kept list of holidays to go stale in some future year. If the ECB
# ever adds a closing day, the cost is one loud morning ("no rate on a working
# day"), dated and fixed in a line. The dangerous direction — silently accepting
# a stale download as a holiday — cannot happen.
def ecb_closed(d: date) -> bool:
    """True if the ECB publishes no reference rate on `d` because it is CLOSED."""
    if d.weekday() >= 5:
        return True
    e = easter(d.year)
    return (d.month, d.day) in {(1, 1), (5, 1), (12, 25), (12, 26)} or \
        d in {e - timedelta(days=2), e + timedelta(days=1)}


# The concertation is at 14:15 CET and the rates appear on the ECB site around
# 16:00 CET, on the day they are dated. So the rate FOR a date exists only from
# that afternoon — which is after both of our runs, and therefore perfectly
# normal in the morning. An hour of margin, because "about 16:00" is what the
# ECB says and not a promise.
ECB_PUBLISH_LOCAL = time(17, 0)
ECB_TZ = ZoneInfo("Europe/Brussels")


def rate_due(d: date, now: datetime) -> bool:
    """
    Should the rate for `d` already exist at `now`?

    This is the difference between "not yet" and "missing", and only the second
    one is worth a word in the log. A zone that publishes imbalance intraday —
    RO does — puts today's PTU in the store hours before the ECB dates today's
    rate, so complaining about it would produce one warning every single day.
    A warning that appears daily is one nobody reads on the day it matters.
    """
    if ecb_closed(d):
        return False                      # the ECB never publishes on that day
    return now >= datetime.combine(d, ECB_PUBLISH_LOCAL, tzinfo=ECB_TZ)


def _closed_run_after(last: date) -> list[date]:
    """The closed days immediately following `last`, up to the first open one.

    This is the owner's rule, stated once: a rate extends over days on which
    the ECB did not publish because it was CLOSED, and never across a day on
    which a publication was due. The first open day after the last publication
    is where the table stops — if its rate has not arrived, that is a stale
    download and an error downstream, not a gap to be filled.
    """
    out, d = [], last + timedelta(days=1)
    while ecb_closed(d):
        out.append(d)
        d += timedelta(days=1)
    return out


def parse_ecb_gesmes(path: str) -> pd.DataFrame:
    """Parse one ECB gesmes XML into long records: rate_date (UTC), currency, rate."""
    tree = etree.parse(path)
    recs: list[tuple] = []
    for day in tree.findall(".//e:Cube[@time]", _NS):
        d = day.get("time")
        for c in day.findall("e:Cube[@currency]", _NS):
            rate = c.get("rate")
            if rate in (None, "", "N/A"):
                continue
            recs.append((d, c.get("currency"), float(rate)))
    df = pd.DataFrame(recs, columns=["rate_date", "currency", "rate"])
    df["rate_date"] = pd.to_datetime(df["rate_date"], utc=True)
    return df


def _read_all_observations() -> pd.DataFrame:
    """Union every ECB XML in FX_DIR (hist + daily), de-duped per (currency, date)."""
    paths = sorted(glob.glob(os.path.join(FX_DIR, "*.xml")))
    if not paths:
        raise FileNotFoundError(f"No ECB XML under {FX_DIR}/*.xml")
    obs = pd.concat([parse_ecb_gesmes(p) for p in paths], ignore_index=True)
    # If hist and daily overlap, keep one row per (currency, date).
    obs = obs.drop_duplicates(["currency", "rate_date"], keep="last")
    return obs


def build_fx_daily(obs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the full daily FX table (ffilled ECB + EUR + pegs).

    `obs` defaults to every ECB file under data/fx; passing it is for tests.
    """
    obs = _read_all_observations() if obs is None else obs
    start, last_pub = obs["rate_date"].min(), obs["rate_date"].max()
    # The table runs past the last publication over the CLOSED days that follow
    # it, and no further: a Friday rate covers Saturday and Sunday, a rate of
    # 24 December covers 25, 26 and the weekend after. Monday is not covered
    # until Monday's rate exists.
    tail = _closed_run_after(last_pub.date())
    end = pd.Timestamp(tail[-1], tz="UTC") if tail else last_pub
    cal = pd.date_range(start, end, freq="D", tz="UTC")

    parts: list[pd.DataFrame] = []

    # ECB currencies: forward-fill within each currency's own active span.
    for cur, g in obs.groupby("currency"):
        s = g.set_index("rate_date")["rate"].sort_index()
        # A currency still quoted on the last publication carries over the
        # closed days after it; a retired one (HRK) stops at its own last quote.
        until = end if s.index.max() == last_pub else s.index.max()
        active = cal[(cal >= s.index.min()) & (cal <= until)]
        r = s.reindex(active).ffill()
        source = pd.Series("ffill", index=active)
        source.loc[s.index.intersection(active)] = "ecb"
        parts.append(pd.DataFrame({"rate_date": active, "currency": cur,
                                   "rate": r.to_numpy(), "source": source.to_numpy()}))

    # EUR itself, always 1.0 across the whole calendar.
    parts.append(pd.DataFrame({"rate_date": cal, "currency": "EUR",
                               "rate": 1.0, "source": "eur"}))

    fx = pd.concat(parts, ignore_index=True)

    # Pegged currencies: fill any day the ECB does not already cover.
    for cur, val in _PEGS.items():
        have = set(fx.loc[fx["currency"] == cur, "rate_date"])
        missing = cal[~cal.isin(have)]
        if len(missing):
            parts.append(pd.DataFrame({"rate_date": missing, "currency": cur,
                                       "rate": val, "source": "peg"}))

    # Euro-legacy tail: fill the gap between the last ECB quote and euro adoption
    # with the locked conversion rate (e.g. HRK's 2022-12-31).
    for cur, spec in _EURO_LEGACY.items():
        cur_obs = obs.loc[obs["currency"] == cur, "rate_date"]
        if cur_obs.empty:
            continue
        adopt = pd.Timestamp(spec["adopt_date"], tz="UTC")
        gap = pd.date_range(cur_obs.max() + pd.Timedelta(days=1),
                            adopt - pd.Timedelta(days=1), freq="D", tz="UTC")
        have = set(fx.loc[fx["currency"] == cur, "rate_date"])
        gap = [d for d in gap if d not in have]
        if gap:
            parts.append(pd.DataFrame({"rate_date": gap, "currency": cur,
                                       "rate": spec["conv_rate"], "source": "euro_conv"}))

    fx = pd.concat(parts, ignore_index=True)

    fx = fx.sort_values(["currency", "rate_date"]).reset_index(drop=True)
    return fx


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fx = build_fx_daily()

    # Enforce the UTC-only rule before writing.
    assert str(fx["rate_date"].dt.tz) == "UTC", "rate_date is not UTC"
    assert not fx.duplicated(["currency", "rate_date"]).any(), "duplicate (currency, date)"

    fx.to_feather(OUT)
    span = f"{fx['rate_date'].min().date()} .. {fx['rate_date'].max().date()}"
    print(f"Wrote {OUT}")
    print(f"  rows={len(fx):,}  currencies={fx['currency'].nunique()}  span={span}")
    print(f"  sources={fx['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
