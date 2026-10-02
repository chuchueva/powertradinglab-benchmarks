"""
NBU FX parser: Ukrainian hryvnia (UAH) daily EUR rate -> feather.

Ukraine (UA_IPS / UA_BEI) trades DA and publishes IM, but UAH is NOT in the ECB
reference rates, so we use the National Bank of Ukraine (NBU) official rate — the
canonical, free, public source (the ECB-equivalent for Ukraine). One-off: Ukraine
is NOT in daily monitoring; this is for a historical look only.

Source file (download once into data/fx/): nbu_fx_uah.xml, NBU "exchange" format:
    <exchange><currency>
        <exchangedate>DD.MM.YYYY</exchangedate>   <- date the rate applies
        <cc>EUR</cc>
        <rate_per_unit>26.422</rate_per_unit>     <- UAH per 1 EUR
        <units>1</units> ...
    </currency> ... </exchange>

Output: data/processed/fx_uah.feather, SAME schema as fx_daily.feather so it can
be concatenated onto the FX table for the UA run:
    rate_date (UTC, tz-aware midnight), currency ("UAH"),
    rate (UAH per 1 EUR -> price_eur = price_uah / rate), source ("nbu" | "ffill")

Caveats (for reading the numbers): the NBU rate is the OFFICIAL rate — pegged to
USD 2022..Oct-2023, managed float since. And UAH devalued hard, so EUR-converted
UA benchmarks mix market volatility with currency depreciation.

Run:
    uv run python src/parse_fx_nbu.py
"""

from __future__ import annotations

import os

from pathlib import Path
import pandas as pd
from lxml import etree

# Every path in this project is anchored to the REPOSITORY, never to the
# current directory. Relative paths do not fail when a script is run from
# elsewhere — they quietly build a parallel tree (a stray data/raw and logs/
# appeared under modelling/ exactly this way, and hid two fetched files long
# enough to send us looking for a network fault that was not there).
ROOT = Path(__file__).resolve().parents[1]

NBU_FILE = ROOT / "data/fx/nbu_fx_uah.xml"
OUT_DIR = ROOT / "data/processed"
OUT = os.path.join(OUT_DIR, "fx_uah.feather")


def parse_nbu(path: str = NBU_FILE) -> pd.DataFrame:
    """Parse the NBU exchange XML -> observed daily records (rate_date, currency, rate)."""
    tree = etree.parse(path)
    recs: list[tuple] = []
    for cur in tree.findall(".//currency"):
        if (cur.findtext("cc") or "").strip() != "EUR":
            continue
        date_txt = (cur.findtext("exchangedate") or "").strip()
        rpu = cur.findtext("rate_per_unit")
        if rpu in (None, ""):  # fall back to rate/units if needed
            rate = float(cur.findtext("rate")) / float(cur.findtext("units") or 1)
        else:
            rate = float(rpu)
        recs.append((date_txt, rate))

    df = pd.DataFrame(recs, columns=["rate_date", "rate"])
    # NBU exchangedate is DD.MM.YYYY, the date the rate applies to (local Kyiv day);
    # we key it as a UTC midnight date, consistent with the UTC-only rule.
    df["rate_date"] = pd.to_datetime(df["rate_date"], format="%d.%m.%Y", utc=True)
    df["currency"] = "UAH"
    return df.drop_duplicates("rate_date", keep="last").sort_values("rate_date")


def build_uah_daily(path: str = NBU_FILE) -> pd.DataFrame:
    """Full daily UAH table: NBU observations forward-filled over the calendar."""
    obs = parse_nbu(path)
    cal = pd.date_range(obs["rate_date"].min(), obs["rate_date"].max(),
                        freq="D", tz="UTC")
    s = obs.set_index("rate_date")["rate"].reindex(cal).ffill()
    source = pd.Series("ffill", index=cal)
    source.loc[obs["rate_date"]] = "nbu"
    return pd.DataFrame({"rate_date": cal, "currency": "UAH",
                         "rate": s.to_numpy(), "source": source.to_numpy()})


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fx = build_uah_daily()
    assert str(fx["rate_date"].dt.tz) == "UTC", "rate_date is not UTC"
    assert not fx.duplicated(["currency", "rate_date"]).any(), "duplicate (currency, date)"
    fx.to_feather(OUT)
    print(f"Wrote {OUT}")
    print(f"  rows={len(fx):,}  span={fx['rate_date'].min().date()} .. "
          f"{fx['rate_date'].max().date()}")
    print(f"  UAH/EUR range: {fx['rate'].min()} .. {fx['rate'].max()}  "
          f"sources={fx['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
