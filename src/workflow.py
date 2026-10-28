"""
NL daily workflow (phase 1, simplest). One zone: NL. EUR (no FX). UTC only.

The workflow is the daily production loop:
    seed (once)  ->  update (each day)  ->  compute (as-of)  ->  daily output

This module owns the NL STORE and the shared LOGGING. Other steps
(update / compute / output) build on the store this file creates.

Store (parquet, UTC tz-aware, one growing table each):
    data/processed/nl/nl_da.parquet : delivery_ts, knownby_ts, price_da
    data/processed/nl/nl_im.parquet : delivery_ts, knownby_ts, currency,
                                       price_im_pos, price_im_neg

Design:
  - NL is a dual-price zone; 0 in an IM price is a REAL 0 EUR/MWh.
  - The store is VERSIONED: the key is (delivery_ts, knownby_ts), so a PTU may
    hold several rows — the value first seen, then each later revision. Nothing
    is ever overwritten. `as_of(current_ts)` is what collapses the versions down
    to one row per PTU, and it must do so at read time, not at write time: the
    whole point is to be able to ask what was known at a past instant.
    (It used to keep only the latest version per delivery_ts. Harmless while the
    File Library gave one version per PTU, but it would have silently discarded
    every as-known snapshot the moment update started polling.)
  - knownby_ts means "we could not have known this earlier". For seeded rows it
    is the archive's UpdateTime — a file stamp, sometimes even EARLIER than the
    delivery it describes, so treat it as provenance, not as publication time.
    For rows fetched live it is the instant of the request that first saw the
    value. See update.
  - Source-agnostic: `seed` bootstraps from the File Library history; `update`
    appends from the live API.

Run:
    uv run python src/workflow.py seed
    uv run python src/workflow.py backfill [ZONE ...]   # history BEFORE what was polled
    uv run python src/workflow.py migrate     # feather store -> parquet, once
"""


import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler

from pathlib import Path
import pandas as pd

# This module is run BOTH as a script (`python src/workflow.py seed`, which
# puts src/ on the path) and imported as `src.workflow` (which puts the repo
# root there). Only one of these imports can work at a time.
try:
    import src.config as cfg
except ImportError:                     # pragma: no cover - depends on entry point
    import config as cfg

# There is no "the zone" any more. Everything below takes an AREA, and the list
# of areas in play comes from settings. A default would have been convenient and
# wrong: it would let a caller that forgot to say which zone it meant keep
# working, silently, on whichever one happened to be first.
# Every path in this project is anchored to the REPOSITORY, never to the
# current directory. Relative paths do not fail when a script is run from
# elsewhere — they quietly build a parallel tree (a stray data/raw and logs/
# appeared under modelling/ exactly this way, and hid two fetched files long
# enough to send us looking for a network fault that was not there).
ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data/processed"
# One folder per zone, named by the AREA CODE — the project's identifier, the
# same one that appears in logs, raw filenames and published paths. The files
# inside carry no zone in their names: the folder already said it, and a name
# repeated is a name that can disagree with itself.
STORE_ROOT = os.path.join(ROOT, cfg.path("zone_store"))


def store_dir(area: str) -> str:
    return os.path.join(STORE_ROOT, area)


def da_store(area: str) -> str:
    return os.path.join(store_dir(area), "da.parquet")


def im_store(area: str) -> str:
    return os.path.join(store_dir(area), "im.parquet")


def _old_store(area: str, leg: str) -> str:
    """The pre-versioning feather, for migrate. NL only; nothing else has one."""
    return os.path.join(store_dir(area), f"{leg}.feather")
LOG_DIR = ROOT / "logs"
LOG_BACKUP_DAYS = 90       # one file per UTC day; ~90 days is a full season
LATE_START_MIN = 5         # beyond this, say so: the run is a catch-up, not punctual
SCHEDULED_ENV = "TRADINGLAB_SCHEDULED"   # set by scripts/run.sh, by nothing else
FIRST_STEP_ENV = "TRADINGLAB_FIRST_STEP"  # set for the chain's first command only

DA_COLS = ["delivery_ts", "knownby_ts", "currency", "price_da"]
IM_COLS = ["delivery_ts", "knownby_ts", "currency", "price_im_pos", "price_im_neg"]

# One row per (PTU, version). The value columns that a revision can change.
VERSION_KEY = ["delivery_ts", "knownby_ts"]
DA_VALUES = ["price_da"]
IM_VALUES = ["price_im_pos", "price_im_neg"]


# --------------------------------------------------------------------------- #
# Logging — shared by the whole workflow
# --------------------------------------------------------------------------- #
def scheduled_slot(now: datetime, times_utc: list[str] | None = None
                   ) -> tuple[str, float]:
    """
    Which scheduled slot this run belongs to, and how late it started (minutes).

    launchd does not tell a job which slot fired it, so we infer: the latest
    configured time at or before `now` (falling back to yesterday's last slot
    just after UTC midnight). Cheap, and it answers the only question that
    matters — did this run happen on time, or is it a catch-up after the Mac
    woke up? Without it, a punctual run and a run that fired hours late look
    identical in the log, and the only way to tell is reading file mtimes.
    """
    times = times_utc if times_utc is not None else cfg.scheduled_times_utc()
    today = now.date()
    slots = sorted(datetime.combine(today, datetime.strptime(t, "%H:%M").time(),
                                    tzinfo=timezone.utc) for t in times)
    past = [s for s in slots if s <= now]
    slot = past[-1] if past else slots[-1] - timedelta(days=1)
    return slot.strftime("%H:%MZ"), (now - slot).total_seconds() / 60


class _SlotFilter(logging.Filter):
    """Stamps every record with the slot this process was scheduled for."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label

    def filter(self, record: logging.LogRecord) -> bool:
        record.slot = self.label
        return True


def setup_logging(name: str = "tradinglab") -> logging.Logger:
    """
    Console + rotating file logger with UTC timestamps; safe to call repeatedly.

    Handlers go on the ROOT logger, not on a named one, so that any module can
    just do `logging.getLogger(__name__)` and land in the same file without
    importing anything from here. That is what lets the generic ingest layer log
    without depending on the NL workflow. (While ingest used `print`, its lines
    arrived out of order relative to the logged ones — different buffers.)

    Rotates at UTC midnight, keeping `LOG_BACKUP_DAYS` files. Not a size
    concern — the workflow writes a few kilobytes a day — but a file per day is
    far easier to read back than one endless one.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    root = logging.getLogger()
    if not root.handlers:              # configure once per process
        root.setLevel(logging.INFO)
        # Only a run STARTED BY THE SCHEDULER belongs to a slot. A run typed by
        # hand belongs to none, and inferring one for it produced a false
        # "machine asleep" warning every time — an alarm that cries wolf is
        # worse than no alarm. run.sh sets the variable; nothing else does.
        if os.getenv(SCHEDULED_ENV):
            slot, late_min = scheduled_slot(datetime.now(timezone.utc))
            slot = f"slot {slot}"
        else:
            slot, late_min = "manual", 0.0
        fmt = logging.Formatter(
            "%(asctime)sZ %(slot)s %(levelname)s %(name)s: %(message)s",
            "%Y-%m-%dT%H:%M:%S")
        logging.Formatter.converter = __import__("time").gmtime  # UTC in logs
        fh = TimedRotatingFileHandler(os.path.join(LOG_DIR, "workflow.log"),
                                      when="midnight", utc=True,
                                      backupCount=LOG_BACKUP_DAYS)
        fh.suffix = "%Y-%m-%d"
        ch = logging.StreamHandler()
        for h in (fh, ch):
            h.setFormatter(fmt)
            h.addFilter(_SlotFilter(slot))
            root.addHandler(h)
        # Once per CHAIN, not once per process: six identical warnings for one
        # late run is how a real signal gets tuned out. Labelling every line is
        # useful; raising the alarm belongs to the run as a whole.
        # The cause is left open on purpose — asleep, a catch-up after wake, and
        # a deliberate `launchctl start` all look identical from in here, and
        # naming only one of them would be a guess dressed as a diagnosis.
        if late_min > LATE_START_MIN and os.getenv(FIRST_STEP_ENV):
            logging.getLogger(name).warning(
                "started %.0f min after its %s — asleep, a catch-up after wake, "
                "or started by hand", late_min, slot)
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Store helpers
# --------------------------------------------------------------------------- #
def _assert_utc(df: pd.DataFrame, cols=("delivery_ts", "knownby_ts")) -> None:
    for c in cols:
        assert str(df[c].dt.tz) == "UTC", f"{c} is not UTC"


# Column types of an EMPTY store table. A new zone has no store yet, and an
# untyped empty frame is not harmless: concatenated with the first real rows it
# turns delivery_ts into plain objects, and the UTC check then fails. That is
# how four zones failed on their first run (2026-10-13) — NL and RO had never
# had an empty store, because they were filled from the archive first.
_STORE_DTYPES = {"delivery_ts": "datetime64[ns, UTC]",
                 "knownby_ts": "datetime64[ns, UTC]",
                 "currency": "object"}                 # prices: float64


def empty_table(cols: list[str]) -> pd.DataFrame:
    """An empty store table with the right column types."""
    return pd.DataFrame({c: pd.Series(dtype=_STORE_DTYPES.get(c, "float64"))
                         for c in cols})


def load_store(area: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one zone's DA and IM stores (empty frames if not seeded yet)."""
    if not os.path.exists(da_store(area)) and os.path.exists(_old_store(area, "da")):
        raise FileNotFoundError(
            f"The {area} store is still in the old single-version feather "
            f"format. Run: uv run python src/workflow.py migrate {area}")
    da = (pd.read_parquet(da_store(area)) if os.path.exists(da_store(area))
          else empty_table(DA_COLS))
    im = (pd.read_parquet(im_store(area)) if os.path.exists(im_store(area))
          else empty_table(IM_COLS))
    return da, im


def _dedup_versions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (delivery_ts, knownby_ts). Versions are KEPT, not collapsed."""
    return (df.drop_duplicates(VERSION_KEY, keep="last")
              .sort_values(VERSION_KEY)
              .reset_index(drop=True))


def write_store(da: pd.DataFrame, im: pd.DataFrame, area: str) -> None:
    """Persist one zone's tables (deduped on the version key, UTC-checked)."""
    os.makedirs(store_dir(area), exist_ok=True)
    da = _dedup_versions(da[DA_COLS]); _assert_utc(da)
    im = _dedup_versions(im[IM_COLS]); _assert_utc(im)
    da.to_parquet(da_store(area), index=False)
    im.to_parquet(im_store(area), index=False)


def migrate_store(area: str, log: logging.Logger | None = None) -> None:
    """
    Bring the store up to the current schema. Idempotent — safe to re-run.

    Two steps have been needed so far:
      1. feather -> parquet (single-version -> versioned). Format only; the old
         files held one version per PTU already, so values carry over untouched.
         The feathers are left on disk; delete them once the parquet looks right.
      2. add `currency` to the DA table. The seeded rows are NL, whose DA clears
         in EUR through SDAC, so EUR is the correct backfill — not a guess. (The
         column was always available in the history feather and simply dropped
         on the way in; the live parser now supplies it too.)
    """
    log = log or setup_logging()
    steps = []

    if os.path.exists(da_store(area)):
        da, im = pd.read_parquet(da_store(area)), pd.read_parquet(im_store(area))
    elif os.path.exists(_old_store(area, "da")):
        da = pd.read_feather(_old_store(area, "da"))
        im = pd.read_feather(_old_store(area, "im"))
        steps.append("feather -> parquet")
    else:
        raise FileNotFoundError(f"Nothing to migrate: no store at {store_dir(area)}")

    n_da, n_im = len(da), len(im)
    if "currency" not in da.columns:
        da = da.assign(currency="EUR")
        steps.append(f"DA currency=EUR on {n_da} rows")

    if not steps:
        log.info("store already current: DA %d rows, IM %d rows", n_da, n_im)
        return

    write_store(da, im, area)
    da_out, im_out = load_store(area)
    assert len(da_out) == n_da and len(im_out) == n_im, "row count changed"
    log.info("%s: migrated store (%s): DA %d rows, IM %d rows (values unchanged)",
             area, "; ".join(steps), len(da_out), len(im_out))


# --------------------------------------------------------------------------- #
# Seed — bootstrap the store from the File Library history feathers
# --------------------------------------------------------------------------- #
def seed_store(area: str, log: logging.Logger | None = None) -> None:
    """One-time bootstrap: one zone's slice of the history feathers -> its store."""
    log = log or setup_logging()
    da_hist = pd.read_feather(f"{PROCESSED}/history_da.feather")
    im_hist = pd.read_feather(f"{PROCESSED}/history_im.feather")

    da = da_hist.loc[da_hist["area"] == area, DA_COLS].copy()
    im = im_hist.loc[im_hist["area"] == area, IM_COLS].copy()
    if da.empty or im.empty:
        raise ValueError(f"{area}: nothing to seed — the history feathers hold "
                         f"DA {len(da)} rows and IM {len(im)} rows for it")

    # The currency is REPORTED, not demanded. This used to raise on anything but
    # EUR, which was true of NL and of nothing else: RO settles imbalance in RON
    # and is not a data error. Rule 7 — the currency is read from the document
    # and carried in the row; what a zone is EXPECTED to trade in belongs in the
    # registry, as an alarm, not here as a wall.
    for name, df in (("DA", da), ("IM", im)):
        log.info("%s %s currencies in the seed: %s", area, name,
                 ", ".join(sorted(df["currency"].dropna().unique())) or "none")

    write_store(da, im, area)
    da_out, im_out = load_store(area)
    log.info("%s: seeded store: DA rows=%d span=%s..%s | IM rows=%d span=%s..%s",
             area, len(da_out), da_out["delivery_ts"].min(), da_out["delivery_ts"].max(),
             len(im_out), im_out["delivery_ts"].min(), im_out["delivery_ts"].max())


def history_slice(area: str, da_hist: pd.DataFrame,
                  im_hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One zone's rows from the history feathers, imbalance under the same code
    mapping as the live path: Germany's imbalance is filed as "DE" in the
    archive from 2022-06-22 (benchmarks_core.IM_ALIAS) and belongs to DE_LU.
    """
    try:                                # script or module, see the imports above
        from src.benchmarks_core import IM_ALIAS
    except ImportError:                 # pragma: no cover - depends on entry point
        from benchmarks_core import IM_ALIAS
    im_names = [area] + [k for k, v in IM_ALIAS.items() if v == area]
    da = da_hist.loc[da_hist["area"] == area, DA_COLS].copy()
    im = im_hist.loc[im_hist["area"].isin(im_names), IM_COLS].copy()
    return da, im


def backfill(store: pd.DataFrame, hist: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    The store plus the history rows that come strictly BEFORE its first
    delivery_ts. Rows already in the store are never touched: they carry real
    knownby_ts from polling, which the archive's revision stamps cannot replace.
    An empty store takes the whole history (that is `seed`).
    """
    if store.empty:
        return hist.copy(), len(hist)
    older = hist.loc[hist["delivery_ts"] < store["delivery_ts"].min()]
    return pd.concat([older, store], ignore_index=True), len(older)


def backfill_store(area: str, log: logging.Logger | None = None,
                   da_hist: pd.DataFrame | None = None,
                   im_hist: pd.DataFrame | None = None) -> None:
    """
    Add the archive history BEFORE what the store already holds (2026-10-28).

    `seed` rewrites the store from the feathers and is right only for a zone
    that has never been polled. Ten zones were polled from 2026-08-31 before
    their history was poured in; for them this is the command. Idempotent: a
    second run finds nothing older than the store's first row and adds nothing.
    Scoring is unaffected: `score_from` keeps the history out of the passes.
    """
    log = log or setup_logging()
    if da_hist is None:
        da_hist = pd.read_feather(f"{PROCESSED}/history_da.feather")
    if im_hist is None:
        im_hist = pd.read_feather(f"{PROCESSED}/history_im.feather")
    h_da, h_im = history_slice(area, da_hist, im_hist)
    da, im = load_store(area)
    n_da0, n_im0 = len(da), len(im)

    da_new, add_da = backfill(da, h_da)
    im_new, add_im = backfill(im, h_im)
    if add_da == 0 and add_im == 0:
        log.info("%s: backfill — nothing older than the store, nothing added", area)
        return
    for name, df in (("DA", da_new), ("IM", im_new)):
        log.info("%s %s currencies after backfill: %s", area, name,
                 ", ".join(sorted(df["currency"].dropna().unique())) or "none")

    write_store(da_new, im_new, area)
    da_out, im_out = load_store(area)
    assert len(da_out) == n_da0 + add_da, f"{area} DA: {len(da_out)} != {n_da0} + {add_da}"
    assert len(im_out) == n_im0 + add_im, f"{area} IM: {len(im_out)} != {n_im0} + {add_im}"
    log.info("%s: backfill added DA %d rows (now %s..%s) and IM %d rows (now %s..%s)",
             area, add_da, da_out["delivery_ts"].min().date(), da_out["delivery_ts"].max().date(),
             add_im, im_out["delivery_ts"].min().date(), im_out["delivery_ts"].max().date())


# --------------------------------------------------------------------------- #
# Running every zone, and surviving one of them
# --------------------------------------------------------------------------- #
# ONE process loops over the zones; there is no job per zone. At the twenty-five
# zones this project is aimed at, a job each would mean twenty-five plists,
# twenty-five logs, twenty-five things to pause before a refactor and remember
# to bring back — and no single place showing the state of the whole. On a VM it
# would be twenty-five systemd units instead.
#
# What a single process owes in exchange is isolation, and this is it. A zone
# that raises is logged WITH ITS TRACEBACK (an unhandled one goes to stdout,
# which under launchd is a different file — that cost us a morning on
# 2026-08-09) and the loop carries on to the next zone. The step then reports a
# COUNT, and the exit code says whether anything failed rather than what failed:
# "2 of 25 zones failed" is the alarm, the log is the diagnosis.
def for_each_area(fn, log, step: str, areas: list[str] | None = None) -> int:
    """Run fn(area) for every configured zone. Returns how many FAILED."""
    areas = list(areas if areas is not None else cfg.AREAS)
    failed = 0
    for area in areas:
        try:
            fn(area)
        except Exception:                    # noqa: BLE001 - logged, then counted
            failed += 1
            log.exception("%s: %s FAILED", step, area)
    if failed:
        log.error("%s: %d of %d zones failed", step, failed, len(areas))
    return failed


def main() -> None:
    log = setup_logging()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "seed"
    # A zone may be named explicitly — seeding a NEW zone happens once, by hand,
    # and usually for one zone while the others are already running.
    # "NL RO" and "NL, RO" both work: a stray comma is not part of a zone code.
    areas = [a for a in (x.strip(", ") for x in sys.argv[2:]) if a] or None
    if cmd == "seed":
        failed = for_each_area(lambda a: seed_store(a, log), log, "seed", areas)
    elif cmd == "backfill":
        # The feathers are read once for all zones: ~450 MB, not per zone.
        da_hist = pd.read_feather(f"{PROCESSED}/history_da.feather")
        im_hist = pd.read_feather(f"{PROCESSED}/history_im.feather")
        failed = for_each_area(lambda a: backfill_store(a, log, da_hist, im_hist),
                               log, "backfill", areas)
    elif cmd == "migrate":
        failed = for_each_area(lambda a: migrate_store(a, log), log, "migrate", areas)
    else:
        log.error("unknown command %r (expected: seed | backfill | migrate)", cmd)
        raise SystemExit(1)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
