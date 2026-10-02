"""
ECB reference rates in the live path: fetch, keep the raw response, rebuild the
daily FX table.

Every run requests the ECB's LAST-90-DAYS file, not the one-day file. The one-day
file carries a single publication, so one failed morning loses that day's rate
for good: the next morning's file already carries the next day, and an open day
with no rate is — by rule 9 — a day nothing non-euro can be valued on. The
90-day file costs a few dozen kilobytes and turns any gap shorter than three
months into something the next successful run closes by itself. Same move as
the three-day poll window for ENTSO-E: filling a gap is the ordinary command,
not a procedure of its own. (The one-day file was the owner's first choice, to
avoid downloading the whole history every time; the 90-day file keeps that
property and removes the hole.)

What this does NOT do is invent. The table is extended past the last
publication only over days on which the ECB was CLOSED (see
`parse_fx.ecb_closed`). If the rate for an open day has not arrived, the table
simply ends before it, and the zones that need a rate for that day are refused
downstream with `leg="FX"` — and scored by themselves once it lands, because
scoring is driven by the store and the day stays a candidate.

Euro zones never need any of this: EUR is a unit, set to 1.0, never looked up.
So a failure here is logged and reported in the exit code, and nothing else in
the chain waits for it.

Run:
    uv run python -m src.fx
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.ingest import ROOT, _month_dir
from src.parse_fx import FX_DIR, OUT, build_fx_daily
from src.workflow import setup_logging

URL_90D = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist-90d.xml"
# The working copy the table builder reads, next to the full history. The raw
# tree keeps every response, dated; this one is simply the latest.
WORKING_COPY = FX_DIR / "eurofxref-hist-90d.xml"
TIMEOUT_S = 60


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def fetch_90d(log) -> Path:
    """
    One request to the ECB, saved like any other source: the body exactly as
    received plus a provenance sidecar, success or failure alike. `data/raw` is
    what the source said, `logs/` is what we did — and a second source in the
    live path does not get to be the exception.
    """
    ts = _stamp()
    out_dir = _month_dir(ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL_90D, timeout=TIMEOUT_S)
    ok = resp.status_code == 200 and b"Envelope" in resp.content[:500]
    name = f"{'' if ok else 'ERROR_'}ECB_eurofxref-hist-90d_{ts}.{'xml' if ok else 'txt'}"
    path = out_dir / name
    path.write_bytes(resp.content)
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps({
        "ok": ok,
        "source": "ECB",
        "url": URL_90D,
        "fetched_ts_utc": ts,
        "http_status": resp.status_code,
        "content_type": resp.headers.get("Content-Type"),
        "n_bytes": len(resp.content),
        "raw_file": name,
    }, indent=2), encoding="utf-8")
    if not ok:
        raise RuntimeError(f"ECB answered HTTP {resp.status_code} "
                           f"({len(resp.content)} bytes), saved as "
                           f"{path.relative_to(ROOT)}")
    log.info("fx: saved raw %s (%d bytes)", path.relative_to(ROOT), len(resp.content))
    return path


def rebuild(log) -> None:
    """Rebuild data/processed/fx_daily.feather from everything under data/fx."""
    fx = build_fx_daily()
    assert str(fx["rate_date"].dt.tz) == "UTC", "rate_date is not UTC"
    assert not fx.duplicated(["currency", "rate_date"]).any(), "duplicate (currency, date)"
    fx.to_feather(OUT)
    ecb = fx.loc[fx["source"] == "ecb", "rate_date"]
    log.info("fx: table rebuilt — last ECB publication %s, covered through %s "
             "(closed days carried), %d currencies",
             ecb.max().date(), fx["rate_date"].max().date(), fx["currency"].nunique())


def main() -> None:
    log = setup_logging()
    failed = False
    try:
        path = fetch_90d(log)
        shutil.copyfile(path, WORKING_COPY)
    except Exception as e:                   # noqa: BLE001 - logged, reported, not fatal
        failed = True
        log.error("fx: fetch failed — %s. The table is rebuilt from what is "
                  "already on disk; zones that need a missing rate will be "
                  "refused with leg=\"FX\", euro zones are unaffected.", e)
    # Rebuilt either way: it is cheap, it is deterministic, and a table left
    # from an older run is the one state that could disagree with its inputs.
    rebuild(log)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
