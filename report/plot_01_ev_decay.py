"""
The main decay figure: one chart per report group -> figures/group_<name>_decay.png

Every zone's monthly series is divided by its OWN maximum, so zones of very
different size can share one axis and a peak reads as "share of this zone's best
month". Grey lines in the background are the zones themselves.

DK, NO and SE enter as ONE line each, the mean of their zones. Otherwise Sweden
would carry four votes and Hungary one, and the summary would drift towards
whichever country happens to be split into more bidding zones.

The ceiling is drawn twice, in one colour: the MEAN solid and the MEDIAN in long
dashes. They answer different questions -- what the average zone did, and what the
typical zone did -- and the distance between them is information: it opens where a
minority of zones runs hotter than the rest.

Three series are drawn on the same 0..1 scale, each normalised the same way:

    ideal             the ceiling
    price_da          the day-ahead price level
    gap_vs_price_da   the deviation between day-ahead and imbalance, relative to price

The last two are drawn as the median only: they are commentary on the ceiling,
not the subject of the chart.

Read together they say which driver carries which hump: in 2022 the ceiling
follows the price, later it follows the deviation while the price is low.

Every group is drawn the same way, whatever its size, so the two figures of the
report are read off one legend.

Run from the report folder (or give the path):

    python plot_01_ev_decay.py
    python plot_01_ev_decay.py yearly_update.csv --smooth 3 --csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

GROUPS = {
    "A": ["BE", "CZ", "DE_LU", "DK", "EE", "FI", "FR", "HR", "HU", "LV", "PL", "RO", "RS", "SE"],
    "B": ["AT", "NL", "NO"],
}

# column -> (label, colour, linestyle, linewidth)
# ideal is the subject of the chart and is drawn solid; the two explanatory
# series are dashed, so the eye reads them as commentary on the teal line.
SERIES = {
    "ideal":           ("Extractable Value",           "#157A7F", "-",            2.4),
    "price_da":        ("DA price",                    "#E39B2F", (0, (5, 2)),    1.8),
    "gap_vs_price_da": ("Volatility",                  "#0E2A33", (0, (1.5, 1.8)), 1.8),
}
SPAGHETTI = "#C9D0D4"


def load(path):
    p = Path(path)
    if not p.exists():
        for alt in (Path("yearly_update.csv"), Path("report/yearly_update.csv"),
                    Path("yearly_update.feather"), Path("report/yearly_update.feather")):
            if alt.exists():
                p = alt
                break
        else:
            raise SystemExit(f"{path} not found")
    y = pd.read_feather(p) if p.suffix == ".feather" else pd.read_csv(p)
    if "gap_vs_price_da" not in y.columns:
        y["gap_vs_price_da"] = y.gap_mean / y.price_da.where(y.price_da > 0)
    y["date"] = pd.to_datetime(dict(year=y.year, month=y.month, day=1))
    print(f"read {p}  rows={len(y)}  areas={y.area.nunique()}")
    return y


def shares(y, areas, col, smooth):
    """One column per zone, divided by that zone's own maximum."""
    w = y[y.area.isin(areas)].pivot(index="date", columns="area", values=col)
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="MS"))
    missing = [a for a in areas if a not in w.columns]
    if missing:
        print(f"  not in the data, skipped: {', '.join(missing)}")
    w = w / w.max()
    if smooth > 1:
        w = w.rolling(smooth, center=True, min_periods=1).median()
    return w


def draw(y, name, areas, out, smooth, write_csv):
    print(f"group {name}: {len(areas)} lines")
    fig, ax = plt.subplots(figsize=(13, 6.5))

    med = {}
    for col, (label, colour, dash, lw) in SERIES.items():
        w = shares(y, areas, col, smooth)
        med[label] = w.median(axis=1)
        if col == "ideal":                       # the spaghetti, ideal only
            for a in w.columns:
                ax.plot(w.index, w[a], color=SPAGHETTI, lw=0.6, alpha=0.55, zorder=1)
            n = w.notna().sum(axis=1)
            print("  zones behind the median, by year:")
            print(n.groupby(n.index.year).agg(["min", "max"]).to_string())
            # Two summaries of the SAME quantity, in one colour: the mean solid,
            # the median in long dashes. Where they separate, the average zone and
            # the typical zone disagree -- a minority is running hotter than the
            # rest, and the text names those zones.
            med["mean " + label] = w.mean(axis=1)
            ax.plot(w.index, med["mean " + label], color=colour, lw=2.0,
                    linestyle="-", label=f"mean {label}", zorder=3)
            # the dashes go ON TOP: where the two nearly touch, a wider solid line
            # would otherwise cover them and the chart would look like one series
            ax.plot(med[label].index, med[label], color=colour, lw=1.6,
                    linestyle=(0, (9, 3)), label=f"median {label}", zorder=5)
            continue
        ax.plot(med[label].index, med[label], color=colour, lw=lw, linestyle=dash,
                label=f"median {label}", zorder=3)

    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Share of the zone's own maximum", fontsize=15)
    ax.tick_params(axis="x", labelsize=15)
    ax.grid(axis="y", color="#D8DEE0", lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(bbox_to_anchor=(0.05, 0.95), fontsize=15, loc="upper left")
    # No title on the figure: the report supplies the caption.

    fig.tight_layout()
    f = Path(out) / f"group_{name}_decay.png"
    fig.savefig(f, dpi=150)
    plt.close(fig)
    print(f"  -> {f}")

    if write_csv:
        t = pd.DataFrame(med)
        t.to_csv(Path(out) / f"group_{name}_median.csv")
        print(f"  -> {Path(out) / f'group_{name}_median.csv'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="?", default="yearly_update.feather")
    p.add_argument("-o", "--out", default="figures")
    p.add_argument("--smooth", type=int, default=3,
                   help="rolling median over N months applied before taking the "
                        "cross-zone median; 3 by default, 1 turns it off")
    p.add_argument("--csv", action="store_true", help="also write the medians as csv")
    a = p.parse_args()

    y = load(a.data)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    for name, areas in GROUPS.items():
        draw(y, name, areas, out, a.smooth, a.csv)


if __name__ == "__main__":
    main()
