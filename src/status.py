"""
The one mutable published file: outputs/status.xml.

Everything else we publish is frozen the moment it is written. This is the single
exception, and it lives in its own module for the same reason it lives outside
the dated folders — so that the boundary is visible without reading anything.
One exception is a rule; a second one would make it a category.

It exists to answer the question that costs the most time when a data pipeline
misbehaves: whose fault is it. Facts only, no self-assessment:

    generated_ts old                 -> the whole chain is down; nothing ran
    generated_ts fresh, source stale -> we are running, the source went quiet
    forecast status="no_forecast"    -> we ran and refused: the input day named
                                        in missing_day was not whole, so no
                                        action was issued for that delivery day
    both fresh, no new forecast      -> our side, but a different failure —
                                        the signal did not form

There is deliberately no ok="true/false". A process cannot report its own death,
so a status file that claims success is trustworthy only while nothing is really
wrong. Its own timestamp is the heartbeat, and that one cannot lie.

ONE file for all zones, not one per zone: a reader asking "is this alive?" should
not need to know a zone code first, and keeping the mutable set at exactly one
file is what stops the exception from spreading.

Run (from repo root):
    uv run python -m src.status
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

import src.config as cfg
from src.publish import (NOTICE_MUTABLE, OUT_DIR, _path, _render,
                            _skip_reason, _ts, market_day, parse_skip_reason,
                            rel)
from src.workflow_compute import FORECASTS, SCORES
from src.workflow import load_store, setup_logging, store_dir

STATUS = OUT_DIR / "status.xml"



def _latest(table: str, area: str) -> tuple[pd.Timestamp, pd.DataFrame] | None:
    """The most recent (by knownby_ts) group of rows for a zone, with its date."""
    if not os.path.exists(table):
        return None
    df = pd.read_parquet(table)
    df = df.loc[df["area"] == area]
    if df.empty:
        return None
    knownby = df["knownby_ts"].max()
    return knownby, df.loc[df["knownby_ts"] == knownby]


def _calc_iteration(table: str, area: str, day: pd.Timestamp,
                    knownby: pd.Timestamp) -> int:
    """Which pass over that delivery day this was — counted the same way the
    published file counts it, so the two never disagree."""
    df = pd.read_parquet(table)
    df = df.loc[df["area"] == area].assign(day=lambda d: market_day(d["delivery_ts"], area))
    seen = sorted(df.loc[df["day"] == day, "knownby_ts"].unique())
    return seen.index(knownby) + 1


def _area_element(root: ET.Element, area: str) -> None:
    da, im = load_store(area)

    # The mode belongs here: a reader seeing a zone with no forecast and no
    # score should be told it is being watched, not left to guess it is broken.
    src = ET.SubElement(root, "area", code=area, mode=cfg.zone_mode(area))
    attrs = {}
    if len(da):
        attrs["da_through"] = _ts(da["delivery_ts"].max())
    if len(im):
        attrs["im_through"] = _ts(im["delivery_ts"].max())
    polls_file = os.path.join(store_dir(area), "polls.parquet")
    if os.path.exists(polls_file):
        polls = pd.read_parquet(polls_file)
        polls = polls.loc[polls["area"] == area]
        if len(polls):
            attrs["last_poll"] = _ts(polls["fetched_ts"].max())
            # Present ONLY when the latest document of a leg disagrees with the
            # registry, so an ordinary status file keeps its ordinary shape.
            odd = []
            if "currency" in polls.columns:
                for doc, leg in (("A44", "DA"), ("A85", "IM")):
                    last = polls.loc[(polls["doc"] == doc) & polls["currency"].notna()]
                    if last.empty:
                        continue
                    got = last.sort_values("fetched_ts")["currency"].iloc[-1]
                    want = cfg.expected_currency(area, leg)
                    if want and got != want:
                        odd.append(f"{leg}:{got}(expected {want})")
            if odd:
                attrs["currency_unexpected"] = " ".join(odd)
    ET.SubElement(src, "source", **attrs)

    fc = _latest(FORECASTS, area)
    if fc:
        knownby, g = fc
        day = market_day(g["delivery_ts"], area).iloc[0]
        attrs = {"delivery_day": f"{day:%Y-%m-%d}",
                 "knownby_ts": _ts(knownby),
                 "data_through": _ts(g["data_through"].max()),
                 "with_signal": str(int(g["action"].notna().sum())),
                 "ptu": str(len(g))}
        # The question this file exists to answer is "whose fault". A refusal is
        # OURS and deliberate, and saying so here costs a reader nothing: the
        # alternative is with_signal="0" and a guess about what that means.
        reason = _skip_reason(g)
        if reason is not None:
            attrs |= {"status": "no_forecast", **parse_skip_reason(reason)}
        attrs["file"] = rel(_path("forecast", area, day, knownby))
        ET.SubElement(src, "forecast", **attrs)

    sc = _latest(SCORES, area)
    if sc:
        # ONE knownby_ts is not one delivery day. It was, until 2026-09-20, when
        # the first as-final pass graded twenty-six August days in a single
        # instant — and this element reported `delivery_day` of whichever row
        # came first with `ptu="2592"`, a number no delivery day can have. The
        # pass is described as a pass: the NEWEST day it produced, that day's
        # PTU, and how many days it covered when it covered more than one.
        knownby, g = sc
        days = market_day(g["delivery_ts"], area)
        day = days.max()
        of_day = g.loc[days == day]
        attrs = {"delivery_day": f"{day:%Y-%m-%d}",
                 "knownby_ts": _ts(knownby),
                 "calc_iteration": str(_calc_iteration(SCORES, area, day, knownby)),
                 "ptu": str(len(of_day))}
        if days.nunique() > 1:
            attrs["days_in_pass"] = str(days.nunique())
        attrs["file"] = rel(_path("score", area, day, knownby))
        ET.SubElement(src, "score", **attrs)


def write_status(log=None) -> Path:
    """Rewrite outputs/status.xml from whatever the tables currently say."""
    log = log or setup_logging()
    root = ET.Element("tradinglab", kind="status")
    ET.SubElement(root, "generated_ts").text = _ts(datetime.now(timezone.utc))
    v = cfg.versions()
    ET.SubElement(root, "version", schema=str(v["schema"]),
                  methodology=str(v["methodology"]))
    ET.SubElement(root, "publisher", **cfg.publisher())
    ET.SubElement(root, "notice").text = NOTICE_MUTABLE

    # One row per zone, and this is the point of keeping status in ONE file: a
    # reader asking "is this alive?" reads one document for twenty-five zones,
    # not twenty-five logs. A zone whose element cannot be built does not take
    # the file down with it — a status file that fails to be written is the one
    # failure a status file may not have.
    for area in cfg.AREAS:
        try:
            _area_element(root, area)
        except Exception:                    # noqa: BLE001 - logged, then skipped
            log.exception("status: %s FAILED", area)
            ET.SubElement(root, "area", code=area, status="unavailable")

    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_bytes(_render(root))          # NOT through _write: this one is
    log.info("status -> %s", STATUS)           # meant to change, every run
    return STATUS


if __name__ == "__main__":
    write_status()
