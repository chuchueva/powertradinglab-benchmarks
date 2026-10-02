"""
Every table of the report, from one script -> six csv files.

    table_by_countries.csv     Extractable Value by zone and year, both groups
    adverse_value.csv          Adverse Value, only the zones that were ever dual
    passive_da_s.csv           Passive DA Value, single-price zones, one row each
    passive_da_d.csv           Passive DA Values, dual-price zones, two rows each
    trailing_bias_values.csv   Trailing Bias Value, one row per zone, years across
    tb_efficiency.csv          share of Extractable Value picked up by Trailing Bias

Conventions, the same in all four and all deliberate:

* Every benchmark is the MEAN MONTH of the year, in EUR per 1 MW of trading
  quantity. Not a yearly sum: 2026 stops in August, and only a mean month puts a
  part-year in the same column as a whole one. The 2026 figure rests on 8 months.
* Rows are chart lines, not raw bidding zones: DK, NO and SE each enter once, as
  the mean of their zones, exactly as in the figures.
* `S/D` is read from the data, never from a list. The imbalance band recovered as
  -4*(ideal + devil)/N is zero in a single-price zone and positive in a dual one;
  a year counts as dual when more than half of its months carry a band above
  1 EUR/MWh, and a zone that switched inside the window gets an arrow and a year.
* `adverse_value.csv` holds only the zones that used dual pricing at some point.
  Everywhere else Adverse Value is Extractable Value with the opposite sign, so a
  row there would carry no information. The difference between the two, where it
  exists, IS the imbalance band: AV = -EV - 0.25 * band * N.

The Passive DA tables are split by pricing mechanism because the pair carries
different amounts of information in the two cases. In a single-price zone
always_sell = -always_buy exactly, so one row per zone says everything; in a
dual-price zone the two sides are independent and each gets its own row.

`tb_efficiency.csv` is the share stat_action_pnl / ideal, a fraction (0.10 = 10%).
Per year it is the mean-month Trailing Bias Value over the mean-month Extractable
Value of that year. Two summaries over EFF_FROM..2026:

    sums_{from}-2026    THE ONE THE REPORT QUOTES. The yearly Trailing Bias Values
                        of the span added up and divided by the yearly Extractable
                        Values of the same span added up -- that is, the printed
                        columns of trailing_bias_values.csv over the printed columns
                        of table_by_countries.csv. A reader recomputes it from the
                        two tables of the report with one division, which is the
                        whole point of choosing it.
    mean_{from}-2026    a control: the mean of the yearly shares. It weighs a thin
                        year as much as a fat one and cannot be recomputed from the
                        printed tables, so it stays out of the text. Where the two
                        columns disagree, one zone's single year is carrying the
                        summary: CZ 2023 is the example.

A year whose Extractable Value is zero or negative (RO 2020) gets no share: a ratio
of two negatives would read as a positive efficiency. Such a year is left out of
both summaries, numerator and denominator alike.

Run from the report folder:

    python tables.py
    python tables.py yearly_update.csv --outdir .
"""

import argparse
import calendar
from pathlib import Path

import pandas as pd

GROUPS = {
    1: ["BE", "CZ", "DE_LU", "DK", "EE", "FI", "FR", "HR", "HU", "LV", "PL", "RO", "RS", "SE"],
    2: ["AT", "NL", "NO"],
}
YEARS = list(range(2020, 2027))
BAND_DUAL = 1.0          # EUR/MWh: above this a month is dual, below it is single
EFF_FROM = 2022          # first year of the efficiency summaries


def load(path):
    y = pd.read_csv(path)
    y["N"] = [calendar.monthrange(int(a), int(m))[1] * 96 for a, m in zip(y.year, y.month)]
    y["band"] = -4 * (y.ideal + y.devil) / y.N
    y["dual"] = y.band > BAND_DUAL
    y["Group"] = y.area.map({a: g for g, lines in GROUPS.items() for a in lines})
    y = y[y.Group.notna()].copy()
    y["Group"] = y.Group.astype(int)
    return y


def regime_label(dual_by_year):
    """'S', 'D', or the switches with the year each one happened."""
    years = sorted(dual_by_year)
    marks = ["D" if dual_by_year[y] else "S" for y in years]
    if len(set(marks)) == 1:
        return marks[0]
    label = marks[0]
    for i in range(1, len(marks)):
        if marks[i] != marks[i - 1]:
            label += f"->{marks[i]} ({years[i]})"
    return label


def yearly(y, col, how="mean"):
    """Mean month of the year, one row per zone, years as columns."""
    t = y.groupby(["Group", "area", "year"])[col].agg(how).unstack("year")
    return t.reindex(columns=YEARS)


def order(t):
    """Group 1 first, then group 2, each block sorted by zone name."""
    return t.reset_index().sort_values(["Group", "area"]).rename(columns={"area": "Country"})


def build_countries(y, out):
    ev = yearly(y, "ideal")
    t = order(ev)
    t["2022/2026"] = (ev[2022] / ev[2026]).round(2).values
    t["S/D"] = [regime_label(y[y.area == a].groupby("year").dual.mean().gt(0.5).to_dict())
                for a in t.Country]
    t[YEARS] = t[YEARS].round(0)
    f = Path(out) / "table_by_countries.csv"
    t.to_csv(f, index=False)
    print(f"{f}  ({len(t)} rows)")
    return t


def build_adverse(y, out):
    ever_dual = [a for a, g in y.groupby("area")
                 if g.groupby("year").dual.mean().gt(0.5).any()]
    z = y[y.area.isin(ever_dual)]
    # no groups here: duality cuts across them, so the rows are simply alphabetical
    t = (yearly(z, "devil").reset_index().drop(columns="Group")
         .rename(columns={"area": "Country"}).sort_values("Country"))
    t["S/D"] = [regime_label(y[y.area == a].groupby("year").dual.mean().gt(0.5).to_dict())
                for a in t.Country]
    t[YEARS] = t[YEARS].round(0)
    f = Path(out) / "adverse_value.csv"
    t.to_csv(f, index=False)
    print(f"{f}  ({len(t)} rows: {', '.join(t.Country)})")
    return t


def build_passive(y, out, dual_zones):
    """Years across, one row per single-price zone, two per dual-price zone."""
    for tag, zones, sides in (("s", [a for a in y.area.unique() if a not in dual_zones],
                               ["always_sell"]),
                              ("d", dual_zones, ["always_buy", "always_sell"])):
        blocks = []
        for side in sides:
            b = order(yearly(y[y.area.isin(zones)], side))
            b.insert(2, "Side", side)
            blocks.append(b)
        t = (pd.concat(blocks)
               .sort_values(["Group", "Country", "Side"])
               .reset_index(drop=True))
        t[YEARS] = t[YEARS].round(0)
        f = Path(out) / f"passive_da_{tag}.csv"
        t.to_csv(f, index=False)
        print(f"{f}  ({len(t)} rows, {len(zones)} zones)")


def build_wide(y, out, col, name):
    """One row per zone, years across, group 1 first and each block by name."""
    t = order(yearly(y, col))
    t[YEARS] = t[YEARS].round(0)
    f = Path(out) / name
    t.to_csv(f, index=False)
    print(f"{f}  ({len(t)} rows)")
    return t


def build_efficiency(y, out, start=EFF_FROM):
    """Trailing Bias Value as a share of Extractable Value, by zone and year."""
    ev = yearly(y, "ideal")
    tb = yearly(y, "stat_action_pnl")
    share = (tb / ev).where(ev > 0)                     # no share for EV <= 0
    t = order(share)
    span = [c for c in YEARS if c >= start]
    keep = share[span].notna()                          # the years that carry a share
    t[f"sums_{start}-2026"] = (tb[span].where(keep).sum(axis=1)
                               / ev[span].where(keep).sum(axis=1)).values
    t[f"mean_{start}-2026"] = share[span].mean(axis=1).values
    num = YEARS + [f"sums_{start}-2026", f"mean_{start}-2026"]
    t[num] = t[num].round(3)
    f = Path(out) / "tb_efficiency.csv"
    t.to_csv(f, index=False)
    print(f"{f}  ({len(t)} rows)")
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="?", default="yearly_update.csv")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()

    y = load(a.data)
    out = Path(a.outdir)
    out.mkdir(exist_ok=True)

    countries = build_countries(y, out)
    dual = build_adverse(y, out)
    build_passive(y, out, list(dual.Country))
    build_wide(y, out, "stat_action_pnl", "trailing_bias_values.csv")
    build_efficiency(y, out)

    print()
    print(countries.to_string(index=False))


if __name__ == "__main__":
    main()
