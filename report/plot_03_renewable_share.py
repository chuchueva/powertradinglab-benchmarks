"""
The second axis of the main claim: wind and solar as a share of load, per group.

    figures/group_1_vre.png
    figures/group_2_vre.png

Same shape as plot_01_ev_decay.py — grey lines are the zones, the heavy line is
the median across them — but the series is NOT normalised to its own maximum.
A share is already comparable between zones, and the level is the point: the
claim is that this line rises while the ceiling falls.

Two things this script does deliberately:

* the multi-zone families are aggregated by SUMMING energy and dividing once
  (DK = DK1+DK2, NO = NO1..NO5, SE = SE1..SE4). A mean of shares would let a
  small-load zone dominate — SE2 alone runs above 100% of its own load.
* the default smoothing is 12 months, not 3. Solar has a season, and a
  3-month window leaves a sawtooth that hides the trend the chart is about.

The series is wind and solar only — variable generation, not "renewables".
Norway is almost entirely renewable and sits at ~0 here, which is correct.

Run from the report folder:

    python plot_03_renewable_share.py
    python plot_03_renewable_share.py vre_share_monthly.csv --smooth 3
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

GROUPS = {
    "1": ["BE", "CZ", "DE_LU", "DK", "EE", "FI", "FR", "HR", "HU", "LV", "PL", "RO", "RS", "SE"],
    "2": ["AT", "NL", "NO"],
}
FAMILIES = {**{z: "DK" for z in ["DK1", "DK2"]},
            **{z: "NO" for z in ["NO1", "NO2", "NO3", "NO4", "NO5"]},
            **{z: "SE" for z in ["SE1", "SE2", "SE3", "SE4"]}}

MEDIAN_COLOR = "#1F6F78"
SPAGHETTI = "#C9D0D4"


def load(path):
    p = Path(path)
    if not p.exists():
        for alt in (Path("vre_share_monthly.csv"), Path("vre_share_monthly.feather"),
                    Path("data/processed/vre_share_monthly.feather")):
            if alt.exists():
                p = alt
                break
        else:
            raise SystemExit(f"{path} not found")
    v = pd.read_feather(p) if p.suffix == ".feather" else pd.read_csv(p)
    v = v[v.area_type.fillna("").str.contains("BZN")].copy()   # one aggregation level only
    v["month"] = pd.to_datetime(v.month)
    v["line"] = v.area.map(lambda a: FAMILIES.get(a, a))
    a = v.groupby(["line", "month"], as_index=False)[["vre_mwh", "load_mwh"]].sum()
    a["share"] = a.vre_mwh / a.load_mwh
    print(f"read {p}  lines={a.line.nunique()}  {a.month.min():%Y-%m}..{a.month.max():%Y-%m}")
    return a


def draw(a, name, lines, out, smooth):
    w = a[a.line.isin(lines)].pivot(index="month", columns="line", values="share")
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="MS"))
    missing = [l for l in lines if l not in w.columns]
    if missing:
        print(f"  not in the data: {', '.join(missing)}")
    if smooth > 1:
        w = w.rolling(smooth, center=True, min_periods=max(2, smooth // 2)).median()
    med = w.median(axis=1)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for c in w.columns:
        ax.plot(w.index, w[c], color=SPAGHETTI, lw=0.7, alpha=0.7, zorder=1)
    ax.plot(med.index, med, color=MEDIAN_COLOR, lw=2.4, zorder=3,
            label="median share of wind and solar")

    ax.set_ylim(0, max(1.0, float(w.max().max()) * 1.05))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("wind and solar, share of load", fontsize=10)
    ax.grid(axis="y", color="#D8DEE0", lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    smoothed = f", {smooth}-month rolling median" if smooth > 1 else ""
    ax.set_title(f"Group {name} — {len(lines)} lines: {', '.join(lines)}{smoothed}",
                 fontsize=10, loc="left", color="#5B6770")

    fig.tight_layout()
    f = Path(out) / f"group_{name}_vre.png"
    fig.savefig(f, dpi=150)
    plt.close(fig)

    n = w.notna().sum(axis=1)
    print(f"group {name}: lines behind the median, by year")
    print(n.groupby(n.index.year).agg(["min", "max"]).to_string())
    print(f"  -> {f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="?", default="vre_share_monthly.csv")
    p.add_argument("-o", "--out", default="figures")
    p.add_argument("--smooth", type=int, default=12,
                   help="rolling median over N months; 12 by default because solar "
                        "has a season, 1 turns it off")
    a = p.parse_args()
    v = load(a.data)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    for name, lines in GROUPS.items():
        draw(v, name, lines, out, a.smooth)


if __name__ == "__main__":
    main()
