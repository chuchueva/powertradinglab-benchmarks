"""
Build the price history caches (feather) from the File Library archive.

Cache / publication layer - separate from the pure parsers. It walks the File
Library zips in data/archive/, runs the parsers over every month (all areas),
concatenates the whole history per product into one long DataFrame and writes it
to data/processed/.

SOURCE: every product now comes from the platform's FULL-HISTORY bundle - one
download each instead of one per month (`EnergyPrices_12.1.D_r3.1.zip`,
`ImbalancePrices_17.1.G_r3.1.zip`, `ActualTotalLoad_6.1.A_r3.zip`). The
per-month zips are deliberately NOT read: they
are the July snapshot, kept as evidence of what the archive held before this
refresh, and two live sources for one product is exactly the pair that drifts
apart. The month window comes from settings (`history`), so 2020 as the start
and "last closed month" as the end are stated once, not spelled into a glob.

The feathers are DISPOSABLE dev caches, rebuildable any time from the archive +
parsers (the real source of truth). Benchmarks read them instead of re-parsing.

Products / schemas (one row per 15-min MTU per area):
    DA   -> history_da.feather   : delivery_ts, knownby_ts, area, area_eic, price_da
    IM   -> history_im.feather   : delivery_ts, knownby_ts, area, area_eic, currency,
                                   price_im_pos, price_im_neg
    LOAD -> history_load.feather : delivery_ts, knownby_ts, area, area_eic, area_type,
                                   load_mw
`area` is the human-readable short code (NL, BE, DE_LU, SE3, IT-NORTH...);
`area_eic` is the canonical EIC (the de-dup / join key).
Both timestamps are UTC tz-aware (asserted before writing).

IM note: price is in the zone's NATIVE currency (`currency` column). EUR
conversion is a later derived step against the ECB FX table, never baked in here.
LOAD note: volume in MW; `area_type` (BZN/CTA/CTY) keeps aggregation levels apart.

Run:
    uv run python src/build_history.py            # all: DA, IM, LOAD
    uv run python src/build_history.py da          # DA only
    uv run python src/build_history.py im          # IM only
    uv run python src/build_history.py load        # LOAD only
"""

from __future__ import annotations

import glob
import os
import sys

from pathlib import Path
import pandas as pd

from config import history_months
from parse_filelibrary import (
    iter_zip_frames,
    parse_da_frame,
    parse_im_frame,
    parse_load_frame,
)

# Every path in this project is anchored to the REPOSITORY, never to the
# current directory. Relative paths do not fail when a script is run from
# elsewhere — they quietly build a parallel tree (a stray data/raw and logs/
# appeared under modelling/ exactly this way, and hid two fetched files long
# enough to send us looking for a network fault that was not there).
ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_DIR = ROOT / "data/archive"
OUT_DIR = ROOT / "data/processed"

# The DA/IM globs have NO leading "*_" on purpose: that is the whole difference
# between the bundle (`EnergyPrices_12.1.D_r3.1.zip`) and a monthly zip
# (`2020_01_EnergyPrices_12.1.D_r3.1.zip`), so the pattern picks the bundle and
# leaves the July per-month snapshot alone.
PRODUCTS = {
    "da": {
        "glob": "EnergyPrices_12.1.D_*.zip",
        "parse": parse_da_frame,
        "out": "history_da.feather",
    },
    "im": {
        "glob": "ImbalancePrices_17.1.G_*.zip",
        "parse": parse_im_frame,
        "out": "history_im.feather",
    },
    "load": {
        "glob": "ActualTotalLoad_6.1.A_*.zip",
        "parse": parse_load_frame,
        "out": "history_load.feather",
    },
}


def build_product(key: str, zones: list[str] | None = None) -> pd.DataFrame:
    """Parse every month of one product, in window, and return the history."""
    cfg = PRODUCTS[key]
    paths = sorted(glob.glob(os.path.join(ARCHIVE_DIR, cfg["glob"])))
    if not paths:
        raise FileNotFoundError(f"No {key.upper()} zips under {ARCHIVE_DIR}/{cfg['glob']}")

    m_from, m_to = history_months()
    print(f"  source: {', '.join(os.path.basename(p) for p in paths)}")
    print(f"  window: {m_from} .. {m_to} (inclusive)")

    parts: list[pd.DataFrame] = []
    for p in paths:
        for member, raw in iter_zip_frames(p, m_from, m_to):
            res = cfg["parse"](raw, zones=zones)
            parts.append(res.df)
            extra = ""
            if key == "im":
                extra = f"  cur={res.meta.get('currencies_seen')}"
            print(f"  {member:<48} rows={len(res.df):>8}  "
                  f"areas={res.meta.get('n_areas')}  "
                  f"res={res.meta.get('resolutions_seen')}{extra}")

    if not parts:
        raise FileNotFoundError(
            f"No {key.upper()} month between {m_from} and {m_to} inside "
            f"{[os.path.basename(p) for p in paths]}")

    n_months = len(parts)
    hist = pd.concat(parts, ignore_index=True)

    # Final safety de-dup across month boundaries: keep the latest UpdateTime per
    # (area_eic, delivery_ts). Normally a no-op, but guards month-edge overlaps
    # and any row republished in a later monthly extract. De-dup on the EIC, the
    # canonical identity, not the readable code.
    before = len(hist)
    hist = (hist.sort_values("knownby_ts")
                .drop_duplicates(["area_eic", "delivery_ts"], keep="last")
                .sort_values(["area_eic", "delivery_ts"])
                .reset_index(drop=True))
    print(f"\n{key.upper()}: concatenated {n_months} months -> {before} rows; "
          f"final de-dup dropped {before - len(hist)}; kept {len(hist)}.")
    return hist


def write_feather(key: str, hist: pd.DataFrame) -> None:
    # Enforce the UTC-only rule explicitly before writing.
    for col in ("delivery_ts", "knownby_ts"):
        assert str(hist[col].dt.tz) == "UTC", f"{col} is not UTC"

    out = os.path.join(OUT_DIR, PRODUCTS[key]["out"])
    hist.to_feather(out)
    span = f"{hist['delivery_ts'].min()} .. {hist['delivery_ts'].max()}"
    print(f"Wrote {out}")
    print(f"  rows={len(hist):,}  areas={hist['area_eic'].nunique()}  span={span}")
    if key == "im":
        print(f"  currencies={sorted(hist['currency'].dropna().unique())}")


def main() -> None:
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    keys = list(PRODUCTS) if which == "all" else [which]
    if any(k not in PRODUCTS for k in keys):
        print(f"Usage: build_history.py [da|im|all]  (got {which!r})")
        raise SystemExit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    for key in keys:
        print(f"\n=== Building {key.upper()} history ===")
        hist = build_product(key, zones=None)  # None = all areas
        write_feather(key, hist)


if __name__ == "__main__":
    main()
