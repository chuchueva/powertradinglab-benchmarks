"""
Per-zone eyeball charts: two stacked panels per bidding zone -> figures/<area>.png

Reads the monthly table written by benchmarks.py (yearly_update.csv) and draws,
for every area in it:

    top     ideal  +  price_da
    bottom  ideal  +  gap_vs_price_da

Nothing is aggregated, renamed or cleaned. Every series comes from the CSV as it
is; the single derived column is gap_vs_price_da = gap_mean / price_da,
the ratio built by hand in Excel. Months absent from the CSV (dropped by the 90%
coverage filter in benchmarks.py, or missing at the source) are left as breaks in
the line rather than joined across, so a coverage hole is visible as a hole.

Run from the report folder:

    python plot_02_zones.py                        # yearly_update.csv -> figures/
    python plot_02_zones.py path/to/table.csv -o figures --from 2020 --to 2025
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

IDEAL_COLOR = "#1F6F78"   # same teal as the report
OTHER_COLOR = "#D1671F"

REQUIRED = ["year", "month", "area", "ideal", "price_da", "gap_mean"]


def load(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{path}: columns missing: {', '.join(missing)}. "
            "Re-run benchmarks.py — gap_mean is written there."
        )
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    if "gap_vs_price_da" not in df.columns:
        df["gap_vs_price_da"] = df.gap_mean / df.price_da.where(df.price_da > 0)
    return df


def monthly_grid(g):
    """One row per calendar month between the zone's first and last month."""
    g = g.sort_values("date").set_index("date")
    full = pd.date_range(g.index.min(), g.index.max(), freq="MS")
    return g.reindex(full)


def panel(ax, x, left, right, left_label, right_label, right_pct=False):
    ax.plot(x, left, color=IDEAL_COLOR, lw=1.6, label=left_label)
    ax.set_ylabel(left_label, color=IDEAL_COLOR, fontsize=9)
    ax.tick_params(axis="y", labelcolor=IDEAL_COLOR, labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", color="#D8DEE0", lw=0.6)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.plot(x, right, color=OTHER_COLOR, lw=1.6, label=right_label)
    ax2.set_ylabel(right_label, color=OTHER_COLOR, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=OTHER_COLOR, labelsize=8)
    if right_pct:
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    for side in ("top",):
        ax.spines[side].set_visible(False)
        ax2.spines[side].set_visible(False)
    return ax2


def draw(area, g, outdir):
    x = g.index
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

    panel(ax_top, x, g.ideal, g.price_da,
          "ideal, EUR/MW per month", "price_da, EUR/MWh")
    panel(ax_bot, x, g.ideal, g.gap_vs_price_da,
          "ideal, EUR/MW per month", "gap_vs_price_da", right_pct=True)

    peak = g.ideal.idxmax()
    by_year = g.groupby(g.index.year).ideal.sum()
    fig.suptitle(
        f"{area}   |   peak month {peak:%Y-%m} = {g.ideal.max():,.0f}"
        f"   |   peak year {by_year.idxmax()} = {by_year.max():,.0f}"
        f"   |   {int(g.ideal.notna().sum())} months",
        fontsize=11, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(Path(outdir) / f"{area}.png", dpi=150)
    plt.close(fig)
    return peak, by_year.idxmax()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default="yearly_update.csv")
    p.add_argument("-o", "--out", default="figures")
    p.add_argument("--from", dest="y0", type=int, default=None)
    p.add_argument("--to", dest="y1", type=int, default=None)
    a = p.parse_args()

    df = load(a.csv)
    if a.y0:
        df = df[df.year >= a.y0]
    if a.y1:
        df = df[df.year <= a.y1]

    out = Path(a.out)
    out.mkdir(exist_ok=True)

    print(f"{'area':<14}{'months':>7}{'peak month':>12}{'peak year':>11}")
    for area, g in df.groupby("area"):
        g = monthly_grid(g)
        if g.ideal.notna().sum() < 6:
            print(f"{area:<14}{int(g.ideal.notna().sum()):>7}   skipped (<6 months)")
            continue
        peak, peak_year = draw(area, g, out)
        print(f"{area:<14}{int(g.ideal.notna().sum()):>7}"
              f"{peak.strftime('%Y-%m'):>12}{peak_year:>11}")
    print(f"\n-> {out.resolve()}")


if __name__ == "__main__":
    main()
