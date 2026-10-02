"""
Publication step: the internal parquet tables -> the files we give to the world.

This layer COMPUTES NOTHING. It reads `forecasts.parquet` / `scores.parquet` and
renders them. If publication did any arithmetic of its own, numbers would be
born in two places and would eventually disagree. The useful consequence: when
the layout changes, every file in history can be regenerated from the tables
without recomputing a single value.

One file per (area, delivery day, knownby_ts) — the same triple that keys the
tables. `knownby_ts` is in the FILENAME rather than a pass number, because a
counter is state that has to be derived from whatever happens to be in the
directory: move a file to an archive, or write two at once, and two different
files can claim to be "calc_02". The timestamp is carried by the row itself, so
it cannot collide, and sorting by name gives chronology for free. The ordinal
still exists — inside the file, as `calc_iteration`, where it is a plain count
of passes and makes no claim about whether anything changed.

Filed by DELIVERY month, so both passes over one delivery day sit side by side.
(Note this is the opposite of `data/raw`, filed by FETCH month — that tree is
about requests, this one is about delivery days.)

Nothing computed means nothing published — with ONE deliberate exception. A
forecast whose signal simply did not form produces no file: an empty file looks
like a result when there is none. But a forecast REFUSED because its input day
was not whole does produce a file, carrying <no_forecast> and no points. The
difference is whose absence it is. An unformed signal is the method saying "not
enough history yet"; a refusal is us saying "we ran, the data was not there, and
this delivery day will never have an action". A reader who finds a hole in the
series cannot tell those apart, and the second one is the one they need.

Immutability is mechanical, not a promise: if the target exists and the content
is identical, the file is left alone; if the content differs, the run STOPS and
names the file. `--force` overwrites, which makes republishing a deliberate act
rather than a side effect of an ordinary run.

Run (from repo root):
    uv run python -m src.publish                          # everything missing
    uv run python -m src.publish --from 2026-08-06 --to 2026-08-06
    uv run python -m src.publish --kind score --force
"""

import argparse
import os
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

import src.config as cfg
from src.workflow_compute import FORECASTS, SCORES, market_day
from src.workflow import setup_logging

# Every path in this project is anchored to the REPOSITORY, never to the
# current directory. Relative paths do not fail when a script is run from
# elsewhere — they quietly build a parallel tree (a stray data/raw and logs/
# appeared under modelling/ exactly this way, and hid two fetched files long
# enough to send us looking for a network fault that was not there).
ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "outputs"

UNIT = "EUR per 1 MW of trading quantity"

ACTION_CODES = ("+1 = buy on the day-ahead auction and settle in imbalance; "
                "-1 = sell. The attribute is absent when no action was issued.")

ATTRIBUTION = (
    "Derived from data published on the ENTSO-E Transparency Platform "
    "(transparency.entsoe.eu). Imbalance prices (A85) are reused under CC-BY 4.0. "
    "Day-ahead prices (A44) belong to the respective power exchanges and are NOT "
    "republished here — this file contains only values derived from them."
)

NOTICE = (
    "This file reports a measurement of market conditions, not a trading "
    "recommendation and not investment advice."
)

NOTICE_MUTABLE = (
    "The only file here that changes. Everything inside a dated folder is "
    "frozen: written once and never rewritten."
)

# Order matters only for readability of the output.
SCORE_VALUES = ["action_pnl", "ideal", "devil", "always_buy", "always_sell"]
# The four that need no decision of ours: they are functions of the day's prices
# and are published for every settled day, including one we refused to forecast.
MARKET_VALUES = [c for c in SCORE_VALUES if c != "action_pnl"]

# The PUBLIC vocabulary, and the only place it exists. Internal column names
# stay the terse working ones; here they become the contract that the report and
# its readers quote, so the two must say the same thing about the same number.
#
# The mapping lives at this boundary on purpose. Renaming inside
# benchmarks_core would rename the columns of yearly_fixed.csv — the frozen
# reference validate_as_of checks the whole as-of path against — and breaking
# the proof to relabel the shop window is the wrong trade.
#
# One column per benchmark, and the family grows sideways: a model arriving
# later adds its own name beside these, it does not take one of them over.
PUBLISHED_NAME = {
    "action_pnl": "trailing_bias_value",
    "ideal": "extractable_value",
    "devil": "adverse_value",
    "always_buy": "passive_da_buy_value",
    "always_sell": "passive_da_sell_value",
}


# --------------------------------------------------------------------------- #
# Formatting — deterministic, because identical content must produce identical
# bytes for the immutability check to mean anything.
# --------------------------------------------------------------------------- #
def _num(v) -> str:
    return f"{float(v):.4f}"


def _ts(v) -> str:
    return pd.Timestamp(v).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(v) -> str:
    return pd.Timestamp(v).strftime("%Y%m%dT%H%M%SZ")


def _path(kind: str, area: str, day: pd.Timestamp, knownby) -> Path:
    return (OUT_DIR / area.lower() / f"{day:%Y}" / f"{day:%m}" /
            f"{kind}_{area}_{day:%Y-%m-%d}_{_stamp(knownby)}.xml")


def rel(path: Path) -> str:
    """A published path as a reader sees it — relative to the publication root.
    Lives here because this module owns the layout; anyone else computing it
    would be a second place to keep in step."""
    return str(path.relative_to(OUT_DIR))


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def _version(g: pd.DataFrame) -> dict:
    """
    The version the rows were COMPUTED under — read from the rows, not from
    today's settings.

    This layer computes nothing, and that has to include the version stamp. It
    used to call cfg.versions() at render time, which is the same value only
    until someone raises it: `publish` re-renders every historical group on
    every run, so the morning after the 2026-09-01 bump all sixty published
    files differed from their re-render and were refused, one ERROR each. The
    numbers had not moved a digit — only the label had.

    A frame without the columns predates the fix; falling back to the settings
    keeps such history renderable, and is exactly the behaviour that needs the
    one-off migration to stop mattering.
    """
    v = cfg.versions()
    for col, key in (("schema_version", "schema"),
                     ("methodology_version", "methodology")):
        if col in g.columns:
            seen = g[col].dropna().unique()
            if len(seen):
                v[key] = int(seen[0])
    return v


# Schema 4 renamed the five published values, renamed the trailing-bias window
# parameter and added <action_codes>. All three are RENDER-time changes, so
# without this gate every file already on disk would re-render differently and
# the frozen-file check would refuse the lot — the same storm the version stamp
# caused on 2026-09-01. A file renders in the vocabulary of its OWN schema.
def _v4(version: dict) -> bool:
    return int(version["schema"]) >= 4


def _v5(version: dict) -> bool:
    return int(version["schema"]) >= 5


def _v6(version: dict) -> bool:
    return int(version["schema"]) >= 6


def _value_names(version: dict) -> dict:
    return PUBLISHED_NAME if _v4(version) else {c: c for c in SCORE_VALUES}


def _params(g: pd.DataFrame) -> dict:
    """
    The tunables as they stood when the row was COMPUTED.

    The version stamp's twin, and it bit on 2026-09-17: moving `run_time_utc`
    from 09:30 to 09:40 would have made every published file re-render with the
    new value and be refused by the frozen-file check. A setting read at render
    time does not describe the run that produced the file, it describes today.
    """
    out = {k: str(v) for k, v in cfg.publication_parameters().items()}
    for col in ("n_days_trailing_bias_default", "final_day_of_month",
                "run_time_utc"):
        if col in g.columns:
            seen = g[col].dropna().unique()
            if len(seen):
                out[col] = str(seen[0])
    return out


def _publisher(g: pd.DataFrame) -> dict:
    """
    Who published these rows — read from the rows, not from today's settings.

    The version stamp's third twin, and the last one. Identity was deliberately
    kept out of the structure so that a rename would cost nothing: element names
    are frozen once published, a project name and a domain are not. True of the
    FORMAT — and false of the FILES, because `publish` re-renders every
    historical group on every run and compares bytes. Read at render time, the
    day `powertradinglab.org` went into settings every past file would have
    differed from itself and been refused. A file says who published it.
    """
    out = dict(cfg.publisher())
    if "publisher_name" in g.columns:
        seen = g["publisher_name"].dropna().unique()
        if len(seen):
            out = {"name": str(seen[0])}
            url = g["publisher_url"].dropna().unique() if "publisher_url" in g.columns else []
            if len(url) and str(url[0]).strip():
                out["url"] = str(url[0]).strip()
    return out


def _root(kind: str, area: str, day: pd.Timestamp, knownby, calc_iteration: int,
          version: dict, params: dict, g: pd.DataFrame) -> ET.Element:
    """The header every published file carries."""
    root = ET.Element("tradinglab", kind=kind, area=area,
                      delivery_day=f"{day:%Y-%m-%d}")
    ET.SubElement(root, "knownby_ts").text = _ts(knownby)
    ET.SubElement(root, "calc_iteration").text = str(calc_iteration)

    ET.SubElement(root, "version", schema=str(version["schema"]),
                  methodology=str(version["methodology"]))
    # The parameters that actually drive the numbers travel WITH the version:
    # the bumps are manual, so one will be forgotten, and then the difference
    # still shows here. The version is for comparison, these are the proof.
    params = dict(params)
    if not _v4(version):
        params = {("n_days_default" if k == "n_days_trailing_bias_default" else k): v
                  for k, v in params.items()}
    ET.SubElement(root, "parameters", **params)

    ET.SubElement(root, "publisher", **_publisher(g))
    ET.SubElement(root, "unit").text = UNIT
    if _v4(version):
        ET.SubElement(root, "action_codes").text = ACTION_CODES
    ET.SubElement(root, "attribution").text = ATTRIBUTION
    ET.SubElement(root, "notice").text = NOTICE
    return root


# Schema 5: a score file may carry no action at all. Four of the five values
# describe the market and are computed for every settled day; the fifth asks
# what our rule earned, and where no action was issued there is no such number.
# It is then ABSENT — attribute and all — never zero and never an empty string,
# for the same reason the forecast omits `action` rather than writing "NA": a
# rendered placeholder gets read as a value. `<no_action>` says why, so a reader
# does not have to fetch that day's forecast file to find out.
#
# Gated on the row's own schema like every other render-time change. A file
# written under schema 4 always had an action, so nothing on disk moves; and a
# pre-5 row without one is a bug in the writer, not a shape to render quietly.
def _no_action_attrs(g: pd.DataFrame) -> dict:
    reason = _skip_reason(g)
    if reason is not None:
        return parse_skip_reason(reason)
    # No forecast row for the day at all — the run never got that far. Distinct
    # from a refusal, which is us looking and declining.
    return {"reason": "no_forecast_issued"}


def _score_doc(area, day, knownby, calc_iteration, g) -> ET.Element:
    version = _version(g)
    names = _value_names(version)
    root = _root("score", area, day, knownby, calc_iteration, version,
                 _params(g), g)
    acted = g["action"].notna()
    if not acted.any() and not _v5(version):
        raise RuntimeError(
            f"score {area} {day:%Y-%m-%d}: no action on any PTU, but the row is "
            f"schema {version['schema']}. A file without an action is schema 5 "
            f"or later — raise publication.schema_version before writing it.")
    ET.SubElement(root, "totals",
                  **{names[c]: _num(g[c].sum())
                     for c in (SCORE_VALUES if acted.any() else MARKET_VALUES)})
    if not acted.any():
        ET.SubElement(root, "no_action", **_no_action_attrs(g))
    points = ET.SubElement(root, "points", count=str(len(g)))
    for _, r in g.sort_values("delivery_ts").iterrows():
        attrs = {"ts": _ts(r["delivery_ts"])}
        if pd.notna(r["action"]):
            attrs["action"] = str(int(r["action"]))
        attrs |= {names[c]: _num(r[c])
                  for c in (SCORE_VALUES if pd.notna(r["action"])
                            else MARKET_VALUES)}
        ET.SubElement(points, "p", **attrs)
    return root


def _skip_reason(g: pd.DataFrame) -> str | None:
    """
    Why this pass issued no action, if it was REFUSED rather than merely
    unformed. `None` for an ordinary pass — and for any row written before the
    input gate existed, whose table has no such column at all.
    """
    if "skip_reason" not in g.columns:
        return None
    seen = g["skip_reason"].dropna().unique()
    return str(seen[0]) if len(seen) else None


def parse_skip_reason(reason: str) -> dict:
    """
    `input_incomplete:2026-09-13:DA` -> the attributes of <no_forecast>.

    Everything after the kind is optional. The leg is the third part and rows
    written before 2026-09-14 carry only the kind and the day; `after_gate`
    carries neither, because no day is missing — the auction simply closed.
    """
    kind, _, rest = reason.partition(":")
    missing_day, _, leg = rest.partition(":")
    attrs = {"reason": kind}
    if missing_day:
        attrs["missing_day"] = missing_day
    if leg:
        attrs["leg"] = leg
    return attrs


def _forecast_doc(area, day, knownby, calc_iteration, g) -> ET.Element:
    root = _root("forecast", area, day, knownby, calc_iteration, _version(g),
                 _params(g), g)
    formed = g["action"].notna()
    ET.SubElement(root, "quality",
                  ptu=str(len(g)),
                  with_signal=str(int(formed.sum())),
                  data_through=_ts(g["data_through"].max()))
    reason = _skip_reason(g)
    if reason is not None and not formed.any():
        # No <points> at all. There is nothing to list: not one PTU of this day
        # was decided, and a grid of empty <p> elements would invite a reader to
        # think the decisions were merely missing from the file.
        ET.SubElement(root, "no_forecast", **parse_skip_reason(reason))
        return root

    if reason is not None:
        # PARTIAL: some hours decided, some not. Schema 6 — the shape begins
        # there, and a pre-6 row in this state is a bug in whatever wrote it,
        # not a file to render quietly.
        if not _v6(_version(g)):
            raise RuntimeError(
                f"forecast {area} {day:%Y-%m-%d}: {int((~formed).sum())} PTU "
                f"refused while others were decided, but the row is schema "
                f"{_version(g)['schema']}. A partial day is schema 6 or later — "
                f"raise publication.schema_version before writing it.")
        hours = sorted({int(t.hour) for t in
                        g.loc[g["skip_reason"].notna(), "delivery_ts"]})
        ET.SubElement(root, "partial_input",
                      **parse_skip_reason(reason),
                      hours=",".join(f"{h:02d}" for h in hours))

    points = ET.SubElement(root, "points", count=str(len(g)))
    for _, r in g.sort_values("delivery_ts").iterrows():
        attrs = {"ts": _ts(r["delivery_ts"])}
        # No action attribute at all when no signal was formed. An absent
        # decision is not the same as a decision to sit out, and writing
        # something like "NA" invites it to be read as a value.
        if pd.notna(r["action"]):
            attrs["action"] = str(int(r["action"]))
        ET.SubElement(points, "p", **attrs)
    return root


def _render(root: ET.Element) -> bytes:
    ET.indent(root, "  ")
    return (b'<?xml version="1.0" encoding="UTF-8"?>\n'
            + ET.tostring(root, encoding="utf-8", xml_declaration=False))


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def _write(path: Path, payload: bytes, force: bool) -> str:
    """-> 'written' | 'unchanged'. Raises rather than silently replacing."""
    if path.exists():
        if path.read_bytes() == payload:
            return "unchanged"
        if not force:
            raise RuntimeError(
                f"{path} already exists with different content. A published file "
                "is frozen: check whether the methodology or schema version "
                "should have been raised, then re-run with --force if the "
                "replacement is intended.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return "written"


def read_results(kind: str) -> pd.DataFrame:
    """Every zone's forecasts (or scores) — one table, already keyed by area."""
    table = {"forecast": FORECASTS, "score": SCORES}[kind]
    return pd.read_parquet(table) if os.path.exists(table) else pd.DataFrame()


def publish(kind: str, day_from: date | None = None, day_to: date | None = None,
            force: bool = False, log=None) -> pd.DataFrame:
    """Render every (area, delivery day, knownby_ts) group that has no file yet."""
    log = log or setup_logging()
    df = read_results(kind)
    if df.empty:
        log.info("publish %s: nothing to do, no zone has a table yet", kind)
        return pd.DataFrame()

    rows, counts = [], {"written": 0, "unchanged": 0, "empty": 0, "conflict": 0}

    for area, per_area in df.groupby("area"):
        per_area = per_area.assign(day=market_day(per_area["delivery_ts"], area))
        # The ordinal is computed over the WHOLE table, so restricting the date
        # range cannot renumber anything.
        order = (per_area[["day", "knownby_ts"]].drop_duplicates()
                 .sort_values(["day", "knownby_ts"]))
        order["calc_iteration"] = order.groupby("day").cumcount() + 1
        order = order.set_index(["day", "knownby_ts"])["calc_iteration"]

        for (day, knownby), g in per_area.groupby(["day", "knownby_ts"]):
            if day_from and day.date() < day_from:
                continue
            if day_to and day.date() > day_to:
                continue
            if (kind == "forecast" and not g["action"].notna().any()
                    and _skip_reason(g) is None):
                counts["empty"] += 1
                log.info("publish forecast %s %s: no signal on any PTU, no file",
                         area, day.date())
                continue

            n = int(order.loc[(day, knownby)])
            doc = (_score_doc if kind == "score" else _forecast_doc)(
                area, day, knownby, n, g)
            path = _path(kind, area, day, knownby)
            try:
                status = _write(path, _render(doc), force)
            except RuntimeError as e:
                # One frozen file that cannot be rewritten must NOT stop the
                # others: they are unrelated days, and refusing to publish today
                # because of a conflict in July is a far worse failure than the
                # conflict itself. Report every one, publish everything else,
                # and let the exit code carry the bad news.
                status = "conflict"
                log.error("%s", e)

            counts[status] += 1
            rows.append({"kind": kind, "area": area, "day": day.date(),
                         "knownby_ts": knownby, "calc_iteration": n,
                         "path": str(path), "status": status})
            if status == "written":
                log.info("publish %s %s %s (pass %d) -> %s", kind, area,
                         day.date(), n, rel(path))

    log.info("publish %s: %d written, %d already identical, %d skipped as empty,"
             " %d refused", kind, counts["written"], counts["unchanged"],
             counts["empty"], counts["conflict"])
    if counts["conflict"]:
        log.error("publish %s: %d file(s) would have changed. Either the schema "
                  "or methodology version needs raising, or something is wrong; "
                  "re-run with --force once you know which.",
                  kind, counts["conflict"])
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Render the result tables as published XML.")
    p.add_argument("--kind", choices=("forecast", "score", "both"), default="both")
    p.add_argument("--from", dest="day_from", type=date.fromisoformat, default=None)
    p.add_argument("--to", dest="day_to", type=date.fromisoformat, default=None)
    p.add_argument("--force", action="store_true",
                   help="replace a published file whose content differs")
    a = p.parse_args()

    log = setup_logging()
    kinds = ["forecast", "score"] if a.kind == "both" else [a.kind]
    refused = 0
    try:
        for kind in kinds:
            out = publish(kind, a.day_from, a.day_to, a.force, log)
            refused += int((out["status"] == "conflict").sum()) if len(out) else 0
    except Exception:
        # Through the LOGGER, not just stderr. An unhandled traceback goes to
        # whatever captured stdout — under launchd that is a separate file, so
        # workflow.log showed nothing at all and it looked like the step had
        # never run. It had; it had died.
        log.exception("publish failed")
        raise
    if refused:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
