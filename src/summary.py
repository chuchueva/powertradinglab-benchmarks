"""
The v1 data interface: the same numbers as the published files, in a shape a
program can read without a parser.

WHY THIS EXISTS. The frozen XML files are the RECORD — one file per zone, per
delivery day, per pass, never rewritten, carrying every PTU and the parameters
it was computed under. That is the right shape for an archive and the wrong
shape for a reader who wants "the last ninety days of RO". This layer is the
reader's shape: a handful of small CSVs plus two JSON files, rebuilt from the
results table after every publication, describing the same facts.

The record stays authoritative. Every row here names the frozen file it came
from, in `source_file`, so any number on the site or in anyone's notebook can be
walked back to the document that established it. The promise is kept
mechanically rather than by assertion.

WHAT IS AND IS NOT IN IT

- Derived values only, per delivery DAY. Never a raw day-ahead or imbalance
  price: those are ENTSO-E's to license, not ours to republish, and the rule
  holds for summary files exactly as for everything else. Per-PTU detail stays
  in the frozen XML, which is the archive's job.
- One row per zone per delivery day, collapsed to the NEWEST pass, with `pass`
  saying which one it is. That is the store's own rule — keep the versions,
  collapse at read time — applied to the shop window.
- Sums over the day's PTU. A day the source left short is therefore
  mechanically smaller than a whole one, which is why `ptu` and
  `ptu_with_action` are columns and not footnotes: 92 of 96 PTU carrying a
  decision makes `trailing_bias_value` smaller for a reason that has nothing to
  do with the market.
- Nothing before `publication.public_from`. Earlier days are working drafts:
  they stay in the frozen archive and the private backup, and a reader never
  meets them — here or on the site, which takes only the files v1 cites.

FLAT, NOT NESTED. `NL_scores-90d.csv`, not `NL/scores-90d.csv`. Twenty-five
zones give seventy-odd files in one directory, which is nothing for a machine
and what `index.json` is a map for. The underscore appears EXACTLY once and
separates the zone from the rest, because zone codes contain hyphens of their
own (`DE-LU`, `IT-North`) and a hyphen would leave the boundary ambiguous. A
name with no zone prefix — `scores-latest.csv` — means every zone.

NO WALL CLOCK ANYWHERE. `index.json` reports `generated_from`, the newest
`knownby_ts` in the data, not the time this ran. So the whole directory is a
pure function of its inputs: identical results produce identical bytes, a run
that changes nothing writes nothing, and the daily backup commit stays quiet
instead of recording that time passed.

Run (from repo root, after publish):
    uv run python -m src.summary
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

import src.config as cfg
from src.publish import (OUT_DIR, PUBLISHED_NAME, _path, _skip_reason,
                         read_results, rel)
from src.workflow import setup_logging
from src.workflow_compute import final_ts, market_day

V1 = OUT_DIR / "v1"
DECIMALS = 4
RECENT_DAYS = 90

# The public vocabulary, in the order a reader meets it: the two passive
# positions, the two oracles, then ours.
VALUE_COLS = ["passive_da_buy_value", "passive_da_sell_value",
              "extractable_value", "adverse_value", "trailing_bias_value"]
INTERNAL = {v: k for k, v in PUBLISHED_NAME.items()}
COLS = (["zone", "delivery_day"] + VALUE_COLS +
        ["ptu", "ptu_with_action", "pass", "calc_iteration", "knownby_ts",
         "schema_version", "methodology_version", "source_file"])

UNIT = "EUR per MW of trading capacity, summed over the delivery day"
DEFINITIONS = {
    "passive_da_buy_value":
        "Buy 1 MW on the day-ahead auction for every PTU of the day and settle "
        "the position in imbalance. No foresight: the same side every PTU.",
    "passive_da_sell_value":
        "Sell 1 MW on the day-ahead auction for every PTU of the day and settle "
        "the position in imbalance. No foresight: the same side every PTU.",
    "extractable_value":
        "The better of the two passive positions, chosen per PTU with perfect "
        "hindsight. It can be negative where dual imbalance pricing leaves the "
        "day-ahead price inside the penalty band, and that negative is real.",
    "adverse_value":
        "The worse of the two passive positions, chosen per PTU with perfect "
        "hindsight. Where a zone prices imbalance single-sided it is the exact "
        "mirror of extractable_value.",
    "trailing_bias_value":
        "The result of the hourly trailing-bias rule, which picks the day-ahead "
        "side from a trailing mean per hour of day and uses only what was known "
        "before the auction closed. Empty where no decision was issued.",
}
COLUMNS = {
    "zone": "Bidding zone code.",
    "delivery_day": "The delivery day in the zone's own market time, ISO date.",
    "ptu": "How many 15-minute periods the day was graded on. Less than a full "
           "day means the source published fewer.",
    "ptu_with_action": "How many of those carried a decision of ours. The "
                       "denominator for trailing_bias_value.",
    "pass": "as-known (graded the morning after delivery) or as-final (regraded "
            "after imbalance settled).",
    "calc_iteration": "Which pass this is for this delivery day, counting from 1.",
    "knownby_ts": "The instant the row was computed — everything in it was "
                  "knowable then.",
    "source_file": "The frozen published document this row was derived from.",
}
FORECAST_COLS = ["zone", "delivery_day", "knownby_ts", "ptu", "ptu_buy",
                 "ptu_sell", "ptu_no_action", "data_through", "skip_reason",
                 "schema_version", "methodology_version", "source_file"]
FORECAST_COLUMNS = {
    "zone": "Bidding zone code.",
    "delivery_day": "The day the decision is FOR, in the zone's own market time.",
    "knownby_ts": "When the decision was made. Always before the day-ahead gate "
                  "of the day before delivery_day.",
    "ptu": "15-minute periods in the delivery day (92, 96 or 100).",
    "ptu_buy": "Periods with action +1: buy on the day-ahead auction, settle in "
               "imbalance.",
    "ptu_sell": "Periods with action -1: sell on the day-ahead auction, settle in "
                "imbalance.",
    "ptu_no_action": "Periods with no action. ptu = ptu_buy + ptu_sell + "
                     "ptu_no_action, always.",
    "data_through": "The freshest fact the signal saw.",
    "skip_reason": "Empty: where there is no action, the signal did not form. "
                   "Set: the forecast was REFUSED, and the value says why "
                   "(kind:missing day[:leg]).",
    "source_file": "The frozen published forecast this row was derived from; "
                   "per-period actions are there.",
}

TERMS = ("Free to use with attribution. Derived values only: no day-ahead or "
         "imbalance prices are republished here or anywhere else in this "
         "dataset.")


def collect(log) -> pd.DataFrame:
    """One row per zone and delivery day, newest pass only."""
    df = read_results("score")
    if df.empty:
        return pd.DataFrame(columns=COLS)

    final_day = cfg.final_day_of_month()
    floor = _floor()
    rows = []
    for area, per_area in df.groupby("area"):
        per_area = per_area.assign(day=market_day(per_area["delivery_ts"], area))
        if floor is not None:
            # Whole days are dropped, so the per-day pass ordinal below is
            # unaffected by where the cut falls.
            per_area = per_area[per_area["day"] >= floor]
            if per_area.empty:
                continue
        # Same ordinal as the publisher computes, over the whole table, so the
        # two agree on which pass a file is.
        order = (per_area[["day", "knownby_ts"]].drop_duplicates()
                 .sort_values(["day", "knownby_ts"]))
        order["calc_iteration"] = order.groupby("day").cumcount() + 1
        order = order.set_index(["day", "knownby_ts"])["calc_iteration"]
        newest = per_area.groupby("day")["knownby_ts"].max()

        for (day, knownby), g in per_area.groupby(["day", "knownby_ts"]):
            if knownby != newest[day]:
                continue                      # collapse at read time
            path = _path("score", area, day, knownby)
            if not path.exists():
                log.warning("summary: %s %s has no published file yet; the row "
                            "is included with no source_file", area, day.date())
            values = {pub: g[col].sum(min_count=1)
                      for pub, col in INTERNAL.items()}
            rows.append({"zone": area, "delivery_day": day, **values,
                         "ptu": len(g),
                         "ptu_with_action": int(g["action"].notna().sum()),
                         "pass": ("as-final" if knownby >= final_ts(day, final_day)
                                  else "as-known"),
                         "calc_iteration": int(order.loc[(day, knownby)]),
                         "knownby_ts": knownby,
                         "schema_version": int(g["schema_version"].iloc[0]),
                         "methodology_version": int(g["methodology_version"].iloc[0]),
                         "source_file": rel(path) if path.exists() else ""})
    return pd.DataFrame(rows, columns=COLS).sort_values(["zone", "delivery_day"])


def _floor():
    f = cfg.public_from()
    return pd.Timestamp(f) if f is not None else None


def collect_forecasts(log) -> pd.DataFrame:
    """One row per zone and delivery day: the decision, counted in PTU.

    A forecast is issued once, before the gate; if a manual backfill ever adds
    a second one, the newest wins, as everywhere in v1. A day on which no
    signal formed on any PTU and nothing was refused has no published file —
    publish writes none — and so no row here: there is nothing to cite.
    """
    df = read_results("forecast")
    if df.empty:
        return pd.DataFrame(columns=FORECAST_COLS)

    floor = _floor()
    rows = []
    for area, per_area in df.groupby("area"):
        per_area = per_area.assign(day=market_day(per_area["delivery_ts"], area))
        if floor is not None:
            per_area = per_area[per_area["day"] >= floor]
        if per_area.empty:
            continue
        newest = per_area.groupby("day")["knownby_ts"].max()
        for (day, knownby), g in per_area.groupby(["day", "knownby_ts"]):
            if knownby != newest[day]:
                continue
            reason = _skip_reason(g)
            path = _path("forecast", area, day, knownby)
            if not path.exists():
                if not g["action"].notna().any() and reason is None:
                    continue                  # publish wrote no file, by rule
                log.warning("summary: forecast %s %s has no published file yet; "
                            "the row is included with no source_file",
                            area, day.date())
            act = g["action"]
            through = g["data_through"].max() if "data_through" in g else pd.NaT
            rows.append({"zone": area, "delivery_day": day,
                         "knownby_ts": knownby, "ptu": len(g),
                         "ptu_buy": int((act == 1).sum()),
                         "ptu_sell": int((act == -1).sum()),
                         "ptu_no_action": int(act.isna().sum()),
                         "data_through": through,
                         "skip_reason": reason or "",
                         "schema_version": int(g["schema_version"].iloc[0]),
                         "methodology_version": int(g["methodology_version"].iloc[0]),
                         "source_file": rel(path) if path.exists() else ""})
    if not rows:
        return pd.DataFrame(columns=FORECAST_COLS)
    return pd.DataFrame(rows, columns=FORECAST_COLS).sort_values(["zone", "delivery_day"])


def _ts(t) -> str:
    return "" if pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def _csv_forecasts(df: pd.DataFrame) -> str:
    out = df.copy()
    out["delivery_day"] = out["delivery_day"].map(lambda d: f"{pd.Timestamp(d):%Y-%m-%d}")
    out["knownby_ts"] = out["knownby_ts"].map(_ts)
    out["data_through"] = out["data_through"].map(_ts)
    return out[FORECAST_COLS].to_csv(index=False, lineterminator="\n")


def _csv(df: pd.DataFrame) -> str:
    """Formatted to fixed decimals, because identical numbers must produce
    identical bytes — the same reason the published files format their own."""
    out = df.copy()
    for c in VALUE_COLS:
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{float(v):.{DECIMALS}f}")
    out["delivery_day"] = out["delivery_day"].map(lambda d: f"{pd.Timestamp(d):%Y-%m-%d}")
    out["knownby_ts"] = out["knownby_ts"].map(
        lambda t: pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ"))
    return out[COLS].to_csv(index=False, lineterminator="\n")


def _put(path: Path, text: str) -> bool:
    """Write only if the bytes differ. Rule 6, applied to the shop window."""
    data = text.encode("utf-8")
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True


def _zones_json(rows: pd.DataFrame) -> str:
    out = {}
    for zone, g in rows.groupby("zone"):
        out[zone] = {
            "eic": cfg.eic(zone),
            "timezone": cfg.zone_tz_name(zone),
            "day_ahead_currency": cfg.expected_currency(zone, "DA"),
            "imbalance_currency": cfg.expected_currency(zone, "IM"),
            "first_delivery_day": f"{pd.Timestamp(g['delivery_day'].min()):%Y-%m-%d}",
            "last_delivery_day": f"{pd.Timestamp(g['delivery_day'].max()):%Y-%m-%d}",
            "delivery_days": int(len(g)),
        }
    return json.dumps(out, indent=2, sort_keys=True) + "\n"


def _index_json(rows: pd.DataFrame) -> str:
    files = []
    for f in sorted(V1.iterdir()):
        if f.name == "index.json" or not f.is_file():
            continue
        data = f.read_bytes()
        files.append({"name": f.name, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    pub = cfg.publisher()
    return json.dumps({
        "dataset": "PowerTradingLab benchmarks",
        "interface_version": 1,
        "publisher": {"name": pub.get("name"), "url": pub.get("url")},
        "terms": TERMS,
        "generated_from": pd.Timestamp(rows["knownby_ts"].max()).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "unit": UNIT,
        "granularity": "one row per zone per delivery day, newest pass only",
        "benchmarks": DEFINITIONS,
        "columns": COLUMNS,
        "forecast_columns": FORECAST_COLUMNS,
        "zones": sorted(rows["zone"].unique().tolist()),
        "files": files,
    }, indent=2, sort_keys=False) + "\n"


def build(log=None) -> int:
    """Rebuild outputs/v1. Returns how many files changed."""
    log = log or setup_logging()
    rows = collect(log)
    if rows.empty:
        log.info("summary: no scored day in any zone yet, nothing written")
        return 0

    V1.mkdir(parents=True, exist_ok=True)
    changed = []
    for zone, g in rows.groupby("zone"):
        g = g.sort_values("delivery_day")
        for name, part in ((f"{zone}_scores.csv", g),
                           (f"{zone}_scores-90d.csv", g.tail(RECENT_DAYS)),
                           (f"{zone}_scores-latest.csv", g.tail(1))):
            if _put(V1 / name, _csv(part)):
                changed.append(name)

    latest = rows.groupby("zone").tail(1).sort_values("zone")
    for name, text in (("scores-latest.csv", _csv(latest)),
                       ("zones.json", _zones_json(rows))):
        if _put(V1 / name, text):
            changed.append(name)

    # Forecasts, by analogy. The newest row is normally TOMORROW: the decision
    # made before the gate, published before the prices exist.
    fc = collect_forecasts(log)
    for zone, g in fc.groupby("zone"):
        g = g.sort_values("delivery_day")
        for name, part in ((f"{zone}_forecasts.csv", g),
                           (f"{zone}_forecasts-90d.csv", g.tail(RECENT_DAYS)),
                           (f"{zone}_forecasts-latest.csv", g.tail(1))):
            if _put(V1 / name, _csv_forecasts(part)):
                changed.append(name)
    if not fc.empty:
        fc_latest = fc.groupby("zone").tail(1).sort_values("zone")
        if _put(V1 / "forecasts-latest.csv", _csv_forecasts(fc_latest)):
            changed.append("forecasts-latest.csv")
    # Last, because it checksums the others.
    if _put(V1 / "index.json", _index_json(rows)):
        changed.append("index.json")

    log.info("summary: v1 rebuilt — %d zones, %d scored days, %d forecast days, "
             "%d file(s) changed", rows["zone"].nunique(), len(rows), len(fc),
             len(changed))
    return len(changed)


if __name__ == "__main__":
    build()
