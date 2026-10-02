"""
Monthly renewable share per zone: VRE forecast over actual load.

Built for one argument in the report. `ideal` — the market's fat — has been
shrinking year after year in most zones, while the share of wind and solar has
been rising. The expectation, stated to the owner by practitioners, was the
opposite: more renewables, harder forecasting, more volatility, more fat. The
data says otherwise, and this table is the second axis of that picture.

    share = (Solar + Wind Onshore + Wind Offshore, day-ahead forecast) / actual load

Why monthly and not the 15-minute grid: the full VRE history is ~31 million rows
and about 700 MB, and nothing in this argument looks below a month. Each month is
therefore reduced as soon as it is parsed and only the aggregate is kept. The
bundle stays on disk, so a later need for the fine grid is a re-parse, not a loss.

THE TRAP THIS FILE IS BUILT AROUND — coverage. If a zone reported 60% of its MTUs
in 2020 and 95% in 2026, its share rises without a single new turbine, and the
report's main claim would rest on an artefact. So:

  - the share is computed only on MTUs where BOTH series exist (inner join), so
    numerator and denominator always describe the same instants;
  - `n_ptu_matched` and `coverage` travel with every row, so a rise in share can
    be told apart from a rise in reporting;
  - the three production types are kept as separate columns, so a series that
    disappears for a while shows up as its own column going to zero rather than
    as "less renewables".

Numerator is a FORECAST, denominator is an ACTUAL. That is deliberate, not a
compromise: the day-ahead auction prices what the market EXPECTED, so expected
renewable share is a driver of the price, while a realised figure is partly a
consequence of it. The report should still say which is which.

Levels of aggregation are matched exactly (area_eic AND area_type), never mixed:
in 2026-07, 58 of the 60 (eic, area_type) pairs in 14.1.D have a load twin, and
no pair disagrees about its level.

Output: data/processed/vre_share_monthly.feather, one row per
(month, area_eic, area, area_type).

Run (from repo root, AFTER build_history has refreshed load):
    uv run python src/vre_share.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from config import history_months
from parse_filelibrary import iter_zip_frames, parse_vre_frame

# Every path in this project is anchored to the REPOSITORY, never to the
# current directory. Relative paths do not fail when a script is run from
# elsewhere — they quietly build a parallel tree.
ROOT = Path(__file__).resolve().parents[1]

BUNDLE = ROOT / "data/archive/GenerationForecastsForWindAndSolar_14.1.D_r3.zip"
LOAD = ROOT / "data/processed/history_load.feather"
OUT = ROOT / "data/processed/vre_share_monthly.feather"

PTU_HOURS = 0.25          # every series is on the 15-min grid by now
TYPES = {"Solar": "solar_mwh",
         "Wind Onshore": "wind_onshore_mwh",
         "Wind Offshore": "wind_offshore_mwh"}


def _load_by_month() -> dict[pd.Period, pd.DataFrame]:
    """Actual load, split per calendar month, only the columns needed.

    Four columns instead of six: the load feather is 14 million rows, and `area`
    and `knownby_ts` come from the VRE side or are not needed at all.
    """
    ld = pd.read_feather(LOAD, columns=["delivery_ts", "area_eic",
                                        "area_type", "load_mw"])
    # tz dropped explicitly for the month key: everything here is UTC, and a
    # period has no timezone to carry. Doing it silently earns a warning that
    # would eventually be ignored along with the ones that matter.
    ld["month"] = ld["delivery_ts"].dt.tz_convert(None).dt.to_period("M")
    return dict(tuple(ld.groupby("month", observed=True)))


def _one_month(vre: pd.DataFrame, load: pd.DataFrame) -> pd.DataFrame:
    """Reduce one month to one row per (area_eic, area, area_type)."""
    keys = ["area_eic", "area", "area_type"]

    # Per-type MWh on the FULL vre grid is not what we want — it has to be the
    # matched grid, or the types would not add up to vre_mwh.
    total = (vre.groupby([*keys, "delivery_ts"], observed=True)["vre_mw"]
                .sum().reset_index())
    m = total.merge(load, how="inner",
                    on=["area_eic", "area_type", "delivery_ts"],
                    validate="one_to_one")

    agg = (m.groupby(keys, observed=True)
             .agg(n_ptu_matched=("delivery_ts", "size"),
                  vre_mwh=("vre_mw", lambda s: s.sum() * PTU_HOURS),
                  load_mwh=("load_mw", lambda s: s.sum() * PTU_HOURS))
             .reset_index())

    # Per type, over exactly the matched instants.
    matched = vre.merge(m[[*keys, "delivery_ts"]], how="inner",
                        on=[*keys, "delivery_ts"])
    per_type = (matched.groupby([*keys, "production_type"], observed=True)["vre_mw"]
                       .sum().mul(PTU_HOURS).unstack("production_type"))
    per_type = per_type.rename(columns=TYPES).reset_index()
    for col in TYPES.values():
        if col not in per_type.columns:
            per_type[col] = pd.NA

    out = agg.merge(per_type[[*keys, *TYPES.values()]], how="left", on=keys)
    # How many MTUs each side had on its own — the difference from
    # n_ptu_matched is what the inner join threw away, and why.
    out = out.merge(total.groupby(keys, observed=True).size()
                         .rename("n_ptu_vre").reset_index(), how="left", on=keys)
    out = out.merge(load.groupby(["area_eic", "area_type"], observed=True).size()
                        .rename("n_ptu_load").reset_index(),
                    how="left", on=["area_eic", "area_type"])
    return out


def build() -> pd.DataFrame:
    m_from, m_to = history_months()
    print(f"  bundle: {BUNDLE.name}")
    print(f"  window: {m_from} .. {m_to} (inclusive)")
    print(f"  load  : {LOAD.name}")

    by_month = _load_by_month()
    parts = []
    for member, raw in iter_zip_frames(str(BUNDLE), m_from, m_to):
        month = pd.Period(member[:7].replace("_", "-"), freq="M")
        res = parse_vre_frame(raw)
        load = by_month.get(month)
        if load is None or res.df.empty:
            print(f"  {member:<58} SKIPPED (no load for {month})"
                  if load is None else f"  {member:<58} SKIPPED (no VRE rows)")
            continue
        one = _one_month(res.df, load).assign(month=month.to_timestamp())
        parts.append(one)
        print(f"  {member:<58} zones={len(one):>3}  "
              f"share={one['vre_mwh'].sum() / one['load_mwh'].sum():.1%}")

    if not parts:
        raise FileNotFoundError(f"Nothing built between {m_from} and {m_to}")

    df = pd.concat(parts, ignore_index=True)
    df["share"] = df["vre_mwh"] / df["load_mwh"]
    # A UTC month has exactly days*96 quarter-hours: no DST to reason about,
    # which is one more reason everything in this project is UTC.
    n_expected = (df["month"].dt.days_in_month * 96).astype("int64")
    df["coverage"] = df["n_ptu_matched"] / n_expected

    cols = ["month", "area_eic", "area", "area_type", "share",
            "vre_mwh", "load_mwh", *TYPES.values(),
            "n_ptu_matched", "n_ptu_vre", "n_ptu_load", "coverage"]
    return df[cols].sort_values(["area", "area_type", "month"]).reset_index(drop=True)


def main() -> None:
    print("\n=== Building monthly VRE share ===")
    df = build()
    os.makedirs(OUT.parent, exist_ok=True)
    df.to_feather(OUT)
    print(f"\nWrote {OUT}")
    print(f"  rows={len(df):,}  zones={df['area_eic'].nunique()}  "
          f"months={df['month'].nunique()}  "
          f"span={df['month'].min():%Y-%m} .. {df['month'].max():%Y-%m}")
    print(f"  coverage: min={df['coverage'].min():.1%} "
          f"median={df['coverage'].median():.1%} max={df['coverage'].max():.1%}")
    low = (df["coverage"] < 0.9).sum()
    print(f"  rows below 90% coverage: {low:,} of {len(df):,} "
          f"— read those with care, the share may be an artefact")


if __name__ == "__main__":
    main()
