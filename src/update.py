"""
NL daily workflow, update step: poll ENTSO-E and append what is NEW.

    ingest.fetch_raw  ->  parse_da / parse_im  ->  diff  ->  store + poll log

Each run re-requests a WINDOW of the last few delivery days, not just yesterday.
Late arrivals and revisions are then picked up by the ordinary daily run, and
filling a gap is the same code with a longer range rather than a separate
backfill script that would never be exercised again.

What lands in the store
-----------------------
Only CHANGES. A row is written when the PTU is new, or when its value differs
from the latest version already stored. So `knownby_ts` reads as "the first time
we saw this value", and the as-known / as-final pair the methodology asks for is
simply the first and last row for that delivery_ts — no calc_iteration flag to
maintain, nothing that can disagree with the data.

`knownby_ts` is the instant of the REQUEST, taken from the raw file's sidecar.
Not <createdDateTime>: ENTSO-E regenerates that per request, so it
is our fetch time wearing a name that suggests publication time. Recording the
request instant claims exactly as much as we know — "we could not have seen this
earlier" — which is what as_of needs. <createdDateTime> is kept in the poll log
for cross-checking, where it cannot be mistaken for a property of the data.

The poll log
------------
data/processed/nl/polls.parquet — one row per request. Without it "no new rows"
is ambiguous between "the value did not change" and "we never looked", and the
publication-delay question stays unanswerable. This table IS the instrument for
that question: the gap between a delivery day and the first poll that returned
its imbalance is measured here, to a resolution of the poll spacing.

A range longer than the source allows
------------------------------------
Any range asked for here is cut into chunks of at most `max_query_days` before
it goes out, and each chunk is a request of its own. See `split_range`: the
chunks are disjoint, so the store ends up exactly where one long request would
have left it, and the poll log gains a row per request, which is what that
table has always counted.

Run (from repo root):
    uv run python -m src.update                      # last window_days
    uv run python -m src.update --from 2026-07-29 --to 2026-08-05
    uv run python -m src.update --days 7
    uv run python -m src.update --last-month         # the as-final pass
"""

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import src.config as cfg
from src.entsoe_ack import NoDataAvailable
from src.ingest import ROOT, _mask, fetch_raw
from src.parse_da import parse_da_xml
from src.parse_im import parse_im_bytes
from src.period import market_range_period, market_tz
from src.workflow import (DA_COLS, DA_VALUES, IM_COLS, IM_VALUES, for_each_area,
                          load_store, setup_logging, store_dir, write_store)


def polls_path(area: str) -> str:
    return os.path.join(store_dir(area), "polls.parquet")
POLL_COLS = ["fetched_ts", "doc", "area", "period_start", "period_end",
             "http_ok", "n_rows", "n_new", "n_changed", "created_datetime",
             "raw_file", "note", "currency"]
LEG = {"A44": "DA", "A85": "IM"}

# The live A85 parser names its columns after the ENTSO-E categories; the store
# (seeded from the File Library) uses the extract's own Positive/Negative names.
# VERIFIED on 2026-07-25, 40 of 96 PTU dual-priced: A04/long == Positive and
# A05/short == Negative, exactly. The mapping is the intuitive one.
IM_RENAME = {"price_im_long": "price_im_pos", "price_im_short": "price_im_neg"}


# --------------------------------------------------------------------------- #
# Fetch + parse
# --------------------------------------------------------------------------- #
def _fetched_ts(raw_path: Path) -> pd.Timestamp:
    """The request instant, read from the sidecar ingest wrote next to the file."""
    meta = json.loads(raw_path.with_suffix(raw_path.suffix + ".meta.json").read_text())
    return pd.Timestamp(datetime.strptime(meta["fetched_ts_utc"], "%Y%m%dT%H%M%SZ"),
                        tz="UTC")


def _check_currency(df: pd.DataFrame, doc: str, area: str) -> pd.DataFrame:
    """
    One document, one unit.

    Several currencies inside a single document make the unit of a row
    ambiguous, and that is a real data error whatever the zone. WHICH currency
    it is, is not our business here: rule 7 says read it from the document and
    carry it in the row. (This used to demand EUR, which was a fact about NL and
    about nothing else. RO settles imbalance in RON.)
    """
    seen = set(df["currency"].dropna().unique())
    if len(seen) > 1:
        raise ValueError(f"{doc} for {area} mixes currencies in one document: "
                         f"{sorted(seen)} — the unit of a row is then ambiguous")
    return df


def parse_da_file(path: Path, area: str) -> tuple[pd.DataFrame, dict]:
    """One saved A44 response -> the store's DA rows + the parser's meta.

    Shared by the daily update and by rebuild: a raw file is read the same way
    on the morning it arrives and on the day the store is rebuilt from it.
    """
    res = parse_da_xml(path.read_text(encoding="utf-8"))
    df = _check_currency(res.df.assign(knownby_ts=_fetched_ts(path)), "A44", area)
    return df[DA_COLS], res.meta


def parse_im_file(path: Path, area: str) -> tuple[pd.DataFrame, dict]:
    """One saved A85 response -> the store's IM rows + the parser's meta."""
    res = parse_im_bytes(path.read_bytes())
    df = res.df.rename(columns=IM_RENAME).assign(knownby_ts=_fetched_ts(path))
    df = _check_currency(df, "A85", area)
    return df[IM_COLS], res.meta


def fetch_da(ps: str, pe: str, log, area: str) -> tuple[pd.DataFrame, dict]:
    """A44 for the window -> [delivery_ts, knownby_ts, currency, price_da]."""
    eic = cfg.eic(area)
    params = {"documentType": "A44", "in_Domain": eic, "out_Domain": eic,
              "periodStart": ps, "periodEnd": pe}
    try:
        path = fetch_raw(params, "A44", area)
    except RuntimeError as e:
        # Known glitch (Jan 2026): A44 sometimes rejects a plain request and
        # needs businessType=A62. Retry once before giving up.
        if "HTTP 400" not in str(e):
            raise
        log.warning("%s: A44 returned HTTP 400, retrying with businessType=A62", area)
        path = fetch_raw({**params, "businessType": "A62"}, "A44", area)

    df, meta = parse_da_file(path, area)
    return df, {"created_datetime": meta.get("knownby_ts"),
                "raw_file": str(path.relative_to(ROOT))}


def fetch_im(ps: str, pe: str, log, area: str) -> tuple[pd.DataFrame, dict]:
    """A85 for the window -> the store's IM columns + poll record."""
    params = {"documentType": "A85", "controlArea_Domain": cfg.eic(area),
              "periodStart": ps, "periodEnd": pe}
    path = fetch_raw(params, "A85", area)

    df, meta = parse_im_file(path, area)
    return df, {"created_datetime": meta.get("knownby_ts"),
                "raw_file": str(path.relative_to(ROOT))}


# --------------------------------------------------------------------------- #
# Diff
# --------------------------------------------------------------------------- #
def new_versions(fresh: pd.DataFrame, stored: pd.DataFrame,
                 value_cols: list[str]) -> tuple[pd.DataFrame, int, int]:
    """
    Rows worth storing: unseen PTUs, plus PTUs whose value has changed.

    Compared against the LATEST stored version of each PTU, not against any
    earlier one — a value that reverts to what it was two revisions ago is a
    genuine change and gets its own row.

    Returns (rows_to_store, n_new, n_changed).
    """
    if stored.empty:
        return fresh.copy(), len(fresh), 0

    latest = (stored.sort_values("knownby_ts")
                    .drop_duplicates("delivery_ts", keep="last")
                    .set_index("delivery_ts")[value_cols])
    f = fresh.set_index("delivery_ts")
    seen = f.index.intersection(latest.index)

    a, b = f.loc[seen, value_cols], latest.loc[seen, value_cols]
    unchanged = (a.eq(b) | (a.isna() & b.isna())).all(axis=1)   # NaN == NaN here

    keep = f.loc[~f.index.isin(unchanged.index[unchanged.to_numpy()])]
    n_new = len(f) - len(seen)
    return keep.reset_index()[fresh.columns], n_new, int((~unchanged).sum())


REVISIONS_SHOWN = 10


def _log_revisions(log, doc: str, keep: pd.DataFrame, stored: pd.DataFrame,
                   value_cols: list[str]) -> None:
    """Print what actually changed, was -> is, so the log answers it alone."""
    latest = (stored.sort_values("knownby_ts")
                    .drop_duplicates("delivery_ts", keep="last")
                    .set_index("delivery_ts")[value_cols])
    changed = keep.loc[keep["delivery_ts"].isin(latest.index)].sort_values("delivery_ts")
    for _, r in changed.head(REVISIONS_SHOWN).iterrows():
        was = latest.loc[r["delivery_ts"]]
        moves = ", ".join(f"{c} {was[c]:g} -> {r[c]:g}"
                          for c in value_cols if was[c] != r[c])
        log.info("%s revised %s: %s", doc, r["delivery_ts"], moves)
    if len(changed) > REVISIONS_SHOWN:
        log.info("%s revised %d more PTU, not listed", doc,
                 len(changed) - REVISIONS_SHOWN)


def _append_polls(rows: list[dict], area: str) -> None:
    path = polls_path(area)
    df = pd.DataFrame(rows)[POLL_COLS]
    if os.path.exists(path):
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    os.makedirs(store_dir(area), exist_ok=True)
    df.sort_values("fetched_ts").reset_index(drop=True).to_parquet(path, index=False)


# --------------------------------------------------------------------------- #
# Update
# --------------------------------------------------------------------------- #
def split_range(day_from: date, day_to: date,
                max_days: int | None = None) -> list[tuple[date, date]]:
    """
    Cut [day_from, day_to] into consecutive chunks no longer than one request
    may cover. Disjoint and gapless: the chunks put back together are the range
    that was asked for, every day of it, exactly once.

    Split EVENLY, not "a full chunk and then the remainder": August under a
    30-day cap becomes 16 + 15, never 30 + 1. Both are legal today, but the even
    cut leaves every request clear of the limit, so a cap that tightens later
    does not send anyone back into this function.

    The split is invisible in the result. Chunks do not overlap and the store
    writes only changes, so two half-month passes leave precisely the store one
    whole-month pass would have left. What it does change is `polls.parquet`:
    two rows per document instead of one, because there were two requests. That
    table has always been one row per request, and it still is.
    """
    max_days = cfg.max_query_days() if max_days is None else max_days
    total = (day_to - day_from).days + 1
    if max_days <= 0 or total <= max_days:
        return [(day_from, day_to)]

    n = -(-total // max_days)               # fewest chunks that fit under the cap
    out, start = [], day_from
    for i in range(n):
        size = total // n + (1 if i < total % n else 0)
        end = start + timedelta(days=size - 1)
        out.append((start, end))
        start = end + timedelta(days=1)
    return out


def update(day_from: date, day_to: date, area: str, log=None) -> pd.DataFrame:
    """Poll both products for [day_from, day_to] and append what is new."""
    log = log or setup_logging()
    ps, pe = market_range_period(day_from, day_to, area)
    da_stored, im_stored = load_store(area)

    polls, updated = [], {}
    for doc, fetch, store, values, cols in (
            ("A44", fetch_da, da_stored, DA_VALUES, DA_COLS),
            ("A85", fetch_im, im_stored, IM_VALUES, IM_COLS)):
        rec = {"doc": doc, "area": area, "period_start": ps, "period_end": pe,
               "fetched_ts": pd.Timestamp.now(tz="UTC"), "http_ok": True,
               "n_rows": 0, "n_new": 0, "n_changed": 0,
               "created_datetime": None, "raw_file": None, "note": None,
               "currency": None}
        try:
            fresh, meta = fetch(ps, pe, log, area)
        except NoDataAvailable as e:
            # NOT a failure. The request was valid and the platform answered
            # that it holds nothing for this window — which is a fact about the
            # source, worth recording as such. Marking it http_ok=False would
            # make our own retries and the status file blame the wrong side.
            rec |= {"n_rows": 0, "note": str(e)}
            polls.append(rec)
            log.warning("%s %s %s..%s: %s", area, doc, day_from, day_to, e)
            continue
        except Exception as e:                    # noqa: BLE001 - logged, not hidden
            rec["http_ok"] = False
            polls.append(rec)
            # Masked again here, though ingest already masks what it raises:
            # a secret needs the belt and the braces, and this line logs text
            # that came from outside our control.
            log.error("%s %s fetch failed for %s..%s: %s", area, doc,
                      day_from, day_to, _mask(str(e)))
            continue

        # The currency is whatever the document says (one per document, checked
        # in fetch). It is recorded per request, so the poll log can show when a
        # zone changed unit, and compared with what the registry EXPECTS — a
        # warning when they differ, never a stop.
        seen = fresh["currency"].dropna().unique()
        rec["currency"] = str(seen[0]) if len(seen) else None
        expect = cfg.expected_currency(area, LEG[doc])
        if rec["currency"] and expect and rec["currency"] != expect:
            log.warning("%s %s: the document is in %s, the registry expects %s — "
                        "stored as published (rule 7); check whether the zone "
                        "changed currency, then update settings.json",
                        area, doc, rec["currency"], expect)

        keep, n_new, n_changed = new_versions(fresh, store, values)
        if n_changed:
            # A revision is rare and interesting, so it is worth spelling out:
            # otherwise "96 revised" sends you into the parquet to find out what
            # moved. Nothing is printed on an ordinary day, when nothing moved —
            # log volume should follow how surprising the event is.
            _log_revisions(log, f"{area} {doc}", keep, store, values)
        rec |= {"fetched_ts": fresh["knownby_ts"].iloc[0] if len(fresh) else rec["fetched_ts"],
                "n_rows": len(fresh), "n_new": n_new, "n_changed": n_changed, **meta}
        polls.append(rec)
        updated[doc] = pd.concat([store, keep], ignore_index=True)[cols]
        log.info("%s %s %s..%s: %d rows returned, %d new, %d revised",
                 area, doc, day_from, day_to, len(fresh), n_new, n_changed)

    write_store(updated.get("A44", da_stored), updated.get("A85", im_stored), area)
    _append_polls(polls, area)

    da_out, im_out = load_store(area)
    log.info("%s store now: DA %d rows (%d PTU), IM %d rows (%d PTU)",
             area, len(da_out), da_out["delivery_ts"].nunique(),
             len(im_out), im_out["delivery_ts"].nunique())
    return pd.DataFrame(polls)[POLL_COLS]


def _poll_with_retry(day_from: date, day_to: date, area: str, log) -> bool:
    """
    One chunk, retried. Returns whether it ended complete.

    Retry, because there is now only ONE poll a day and it sits on the decision
    itself. A single bad moment at the source used to be covered by the four
    other polls; observed on 2026-08-07, when both documents failed at exactly
    the 09:30 slot and the forecast ran on the previous day's data. A failed
    fetch does not raise — update() records it and returns — so the retry checks
    the poll log rather than catching anything.
    """
    attempts, delay = cfg.retry_attempts(), cfg.retry_delay_s()
    for n in range(1, attempts + 1):
        polls = update(day_from, day_to, area, log)
        if polls["http_ok"].all():
            return True
        if n < attempts:
            log.warning("%s: poll incomplete (attempt %d of %d), retrying in %d s",
                        area, n, attempts, delay)
            time.sleep(delay)
    log.error("%s: poll of %s..%s still incomplete after %d attempts — the run "
              "continues on the data already in the store, and data_through "
              "will say so", area, day_from, day_to, attempts)
    return False


def main() -> None:
    p = argparse.ArgumentParser(
        description="Poll ENTSO-E for every configured zone and append what is new.")
    p.add_argument("--days", type=int, default=None,
                   help="window length in delivery days (default: settings.json)")
    p.add_argument("--from", dest="day_from", type=date.fromisoformat, default=None)
    p.add_argument("--to", dest="day_to", type=date.fromisoformat, default=None)
    p.add_argument("--last-month", action="store_true",
                   help="re-request the whole previous calendar month")
    p.add_argument("areas", nargs="*", metavar="AREA",
                   help="zones to poll (default: settings.json -> areas)")
    a = p.parse_args()

    log = setup_logging()

    def one(area: str) -> None:
        # The delivery day is the zone's own: "today" in Bucharest is not
        # "today" in Amsterdam for the hour they disagree, and asking the source
        # for a day that has not started there yet is how a window quietly loses
        # its newest day.
        today = datetime.now(market_tz(area)).date()

        if a.last_month:
            # Before a day can be scored as-final, its values have to be LOOKED
            # at again. The ordinary window is three days, so a revision
            # arriving weeks later would never be seen, and the final pass would
            # faithfully confirm the as-known numbers — not because nothing
            # changed, but because nobody asked. This is that second look.
            day_to = today.replace(day=1) - timedelta(days=1)
            day_from = day_to.replace(day=1)
        else:
            day_to = a.day_to or today
            day_from = (a.day_from
                        or day_to - timedelta(days=(a.days or cfg.update_window_days()) - 1))
        if day_from > day_to:
            raise SystemExit(f"empty range: {day_from} .. {day_to}")

        chunks = split_range(day_from, day_to)
        if len(chunks) > 1:
            log.info("%s: %s..%s is %d days, over the %d-day limit at the source "
                     "— sent as %d requests: %s", area, day_from, day_to,
                     (day_to - day_from).days + 1, cfg.max_query_days(),
                     len(chunks), ", ".join(f"{c0}..{c1}" for c0, c1 in chunks))

        for c_from, c_to in chunks:
            # A chunk that never came good does not stop the ones behind it. The
            # store takes only changes, so a partial pass is worth exactly the
            # part that arrived — and the part that did not is still missing
            # afterwards, which is what the next run is for.
            _poll_with_retry(c_from, c_to, area, log)

    areas = a.areas or None
    raise SystemExit(1 if for_each_area(one, log, "update", areas) else 0)


if __name__ == "__main__":
    main()
