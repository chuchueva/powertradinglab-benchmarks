"""
Rebuild a zone's store from data/raw — and from nothing else.

    data/raw (every saved A44 / A85 response)  ->  the daily update's own code
                                               ->  zones/<AREA>/da.parquet, im.parquet

The store is a fold over the raw responses: replay them in the order they were
fetched, keep only what changed, and the store comes back. The parsing and the
diff are the same functions the morning run uses (parse_da_file, parse_im_file,
new_versions), so a rebuild cannot read a file differently from the day it
arrived.

What it does NOT bring back: the 2020-2026 history that was once poured into
the store from the File Library archive. The live service does not need it —
the trailing window is a few days, scoring starts on scoring.score_from, and
models are trained by hand from the archive. After a rebuild the store starts
where data/raw starts.

Not rebuilt either: polls.parquet. The next scheduled update starts it again.

Refuses to overwrite an existing store unless --force is given.

Run (from repo root):
    uv run python -m src.rebuild              # every zone in settings.json
    uv run python -m src.rebuild RO           # one zone
    uv run python -m src.rebuild --force NL   # replace an existing store
"""

import argparse
import json
import os

import pandas as pd

import src.config as cfg
from src.entsoe_ack import NoDataAvailable
from src.ingest import RAW_DIR, ROOT
from src.update import new_versions, parse_da_file, parse_im_file
from src.workflow import (DA_VALUES, IM_VALUES, da_store, for_each_area,
                          im_store, load_store, setup_logging, write_store)

DOCS = {"A44": (parse_da_file, DA_VALUES), "A85": (parse_im_file, IM_VALUES)}
# Where the zone's EIC sits in the request, for sidecars older than the `area`
# field (filenames used to carry the EIC instead of the zone code).
EIC_KEYS = ("in_Domain", "controlArea_Domain")
SIDECAR = ".meta.json"


def raw_responses(area: str) -> list[tuple[str, str, object]]:
    """Successful A44/A85 responses for one zone, oldest fetch first.

    Read from the sidecars, not from file names: the sidecar says whether the
    request succeeded, which document it was, and when it was fetched.
    ERROR_* files carry ok=false and are skipped here, as update skips them.
    """
    eic = cfg.eic(area)
    out = []
    for meta_path in RAW_DIR.rglob(f"*{SIDECAR}"):
        meta = json.loads(meta_path.read_text())
        if not meta.get("ok") or meta.get("doc_label") not in DOCS:
            continue
        if "area" in meta:
            if meta["area"] != area:
                continue
        else:
            params = meta.get("params_without_token", {})
            if not any(params.get(k) in (eic, cfg.eic_im(area)) for k in EIC_KEYS):
                continue
        raw = meta_path.with_name(meta_path.name[: -len(SIDECAR)])
        out.append((meta["fetched_ts_utc"], meta["doc_label"], raw))
    return sorted(out)


def rebuild(area: str, log=None, force: bool = False) -> None:
    """Replay data/raw for one zone into a fresh store."""
    log = log or setup_logging()
    if not force and (os.path.exists(da_store(area)) or os.path.exists(im_store(area))):
        raise FileExistsError(f"{area}: a store already exists; rerun with "
                              f"--force to replace it")

    tables = {"A44": None, "A85": None}
    used = missing = nodata = bad = 0
    for stamp, doc, raw in raw_responses(area):
        if not raw.exists():
            missing += 1
            log.warning("%s: sidecar without its file, skipped: %s",
                        area, raw.relative_to(ROOT))
            continue
        parse, values = DOCS[doc]
        try:
            fresh, _ = parse(raw, area)
        except NoDataAvailable:
            nodata += 1                   # a valid answer of "nothing here"
            continue
        except Exception as e:            # noqa: BLE001 - logged, and counted
            bad += 1
            log.error("%s: cannot parse %s: %s", area, raw.relative_to(ROOT), e)
            continue
        stored = tables[doc]
        if stored is None:
            tables[doc] = fresh.copy()
        else:
            keep, _, _ = new_versions(fresh, stored, values)
            tables[doc] = pd.concat([stored, keep], ignore_index=True)
        used += 1

    if tables["A44"] is None or tables["A85"] is None:
        raise ValueError(f"{area}: data/raw holds no usable "
                         f"{'A44' if tables['A44'] is None else 'A85'} response; "
                         f"nothing written")

    write_store(tables["A44"], tables["A85"], area)
    da, im = load_store(area)
    log.info("%s: store rebuilt from %d raw responses (%d without data, %d "
             "unreadable, %d sidecars without a file) | DA %d rows %s..%s | "
             "IM %d rows %s..%s", area, used, nodata, bad, missing,
             len(da), da["delivery_ts"].min(), da["delivery_ts"].max(),
             len(im), im["delivery_ts"].min(), im["delivery_ts"].max())


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild zone stores from data/raw.")
    p.add_argument("--force", action="store_true",
                   help="replace an existing store")
    p.add_argument("areas", nargs="*", metavar="AREA",
                   help="zones to rebuild (default: settings.json -> areas)")
    a = p.parse_args()
    log = setup_logging()
    failed = for_each_area(lambda area: rebuild(area, log, a.force), log,
                           "rebuild", a.areas or None)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
