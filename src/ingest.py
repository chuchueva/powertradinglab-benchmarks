"""
Ingestion layer: fetch from ENTSO-E and save the RAW response to disk.

Responsibilities (and ONLY these):
  - make the API request,
  - save the raw response bytes exactly as received (XML for DA, zip for IM),
  - record a small sidecar with request metadata (for provenance / frozen raw).

It does NOT parse, unzip, or interpret. Parsing is a separate layer that reads
these saved files. This keeps a faithful, immutable copy of the source: if the
parser changes later, we re-run it on saved raw files instead of re-hitting the API.

Raw files are named so the request is reconstructable from the filename:
    {doc}_{zone}_{periodStart}_{periodEnd}_{fetched_ts}.{ext}

and filed by the MONTH THEY WERE FETCHED:
    data/raw/YYYY/MM/{filename}

Five polls a day put 20 files (payloads + sidecars) into the tree daily — about
7300 a year. The bytes are nothing; a single flat directory with that many
entries is the problem, for git, for Finder and for finding anything by hand.
Filing by fetch month keeps each directory small and makes "archive last year"
a matter of moving one folder. Fetch month, not delivery month, because a single
request may span a month boundary while it can only be made at one instant.

Nothing reads this tree by scanning it — the filename carries the whole request
— so files written under the old flat layout can stay exactly where they are.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_URL = "https://web-api.tp.entsoe.eu/api"
# Every path in this project is anchored to the REPOSITORY, never to the
# current directory. Relative paths do not fail when a script is run from
# elsewhere — they quietly build a parallel tree (a stray data/raw and logs/
# appeared under modelling/ exactly this way, and hid two fetched files long
# enough to send us looking for a network fault that was not there).
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data/raw"          # frozen raw lives here

# Plain module logger: no handlers configured here. Whoever runs the workflow
# calls setup_logging(), which puts handlers on the root logger, and these lines
# land in the same file, in the right order, with UTC timestamps. A library
# layer should not decide where logs go.
log = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("ENTSOE_API_KEY")
if not API_KEY:
    raise RuntimeError("ENTSOE_API_KEY not found in .env")


def _mask(text: str) -> str:
    return text.replace(API_KEY, "***") if API_KEY else text


def _month_dir(fetched_ts: str) -> Path:
    """data/raw/YYYY/MM for a "%Y%m%dT%H%M%SZ" fetch stamp."""
    return RAW_DIR / fetched_ts[:4] / fetched_ts[4:6]


def _write_sidecar(out_path: Path, resp, params: dict, doc_label: str,
                   area: str, fetched_ts: str, n_bytes: int, ext: str,
                   ok: bool) -> None:
    """
    Provenance for ONE request, written next to its body — success or failure.

    A failed request is an event of exactly the same kind as a successful one:
    it happened, at a known instant, with known parameters, and the server said
    something. Writing the sidecar only on success left refusals as a bare HTML
    page with no record of what was even asked for.

    Never contains the API key (it travels as a URL parameter, so `params` here
    is the caller's dict, before the token is added).
    """
    sidecar = {
        "ok": ok,
        "doc_label": doc_label,
        "area": area,
        "params_without_token": params,
        "fetched_ts_utc": fetched_ts,
        "http_status": resp.status_code,
        "content_type": resp.headers.get("Content-Type"),
        "n_bytes": n_bytes,
        "detected_ext": ext,
        "raw_file": out_path.name,
    }
    out_path.with_suffix(out_path.suffix + ".meta.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8")


def _why(body: str) -> str:
    """
    The human sentence out of an HTML refusal page, for the log line.

    ENTSO-E answers an outage with a full styled page — 18 KB, nearly all of it
    base64 logos — carrying one line that actually says anything ("Service
    Temporarily Unavailable"). Worth lifting into the log so a failure can be
    read without opening the file. Falls back to the first line of a plain-text
    error, which is what the API returns for ordinary rejections.
    """
    for pattern in (r'class="main-heading"[^>]*>([^<]+)<', r"<title>([^<]+)</title>",
                    r"<text[^>]*>([^<]+)</text>"):
        m = re.search(pattern, body, re.IGNORECASE)
        if m and m.group(1).strip():
            return m.group(1).strip()
    first = body.strip().splitlines()[0] if body.strip() else ""
    return first[:200]


def fetch_raw(params: dict, doc_label: str, area: str) -> Path:
    """
    Make one request and save the raw bytes. Returns the saved file path.

    params     : query params WITHOUT securityToken (added here) — the EIC goes
                 in here, because only the request needs it
    doc_label  : short tag for the filename, e.g. 'A44' or 'A85'
    area       : our zone code, e.g. 'NL'. A LABEL for the filename and the
                 sidecar, nothing more. Filenames used to carry the raw EIC
                 (A85_10YCS-CG-TSO---S_...), which no one can read at a glance.
    """
    full_params = {"securityToken": API_KEY, **params}
    try:
        resp = requests.get(BASE_URL, params=full_params, timeout=60)
    except requests.RequestException as e:
        # The token travels as a URL PARAMETER, so requests' own exception text
        # quotes the whole URL, key and all — which is how it ended up in the log
        # on 2026-08-05 when DNS failed. Masking has to happen here, in the only
        # place that knows the key: then no caller can leak it by logging the
        # exception, which is precisely what update does.
        # `from None` matters too — chaining would print the original, unmasked
        # message in the traceback.
        raise RuntimeError(f"request failed: {_mask(str(e))}") from None

    fetched_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _month_dir(fetched_ts)

    if resp.status_code != 200:
        # Save the failed body too - it explains WHY (ENTSO-E returns XML errors,
        # or a whole styled maintenance page).
        out_dir.mkdir(parents=True, exist_ok=True)
        body = _mask(resp.text)
        err_path = out_dir / f"ERROR_{doc_label}_{area}_{fetched_ts}.txt"
        err_path.write_text(body, encoding="utf-8")
        _write_sidecar(err_path, resp, params, doc_label, area, fetched_ts,
                       len(body), "txt", ok=False)
        log.error("%s HTTP %d (%s) for %s..%s, body saved to %s", doc_label,
                  resp.status_code, _why(body), params.get("periodStart"),
                  params.get("periodEnd"), err_path)
        raise RuntimeError(
            f"HTTP {resp.status_code} ({_why(body)}). Body saved to {err_path}"
        )

    raw = resp.content  # bytes - keep exactly as received

    # Choose extension by actual content, not by assumption:
    # zip responses start with 'PK'.
    ext = "zip" if raw[:2] == b"PK" else "xml"

    ps = params.get("periodStart", "NA")
    pe = params.get("periodEnd", "NA")
    fname = f"{doc_label}_{area}_{ps}_{pe}_{fetched_ts}.{ext}"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    out_path.write_bytes(raw)  # write BYTES, not text - preserves zip integrity

    _write_sidecar(out_path, resp, params, doc_label, area, fetched_ts,
                   len(raw), ext, ok=True)

    log.info("saved raw %s (%d bytes, %s)", out_path, len(raw), ext)
    return out_path


# --- manual test ------------------------------------------------------------
# Fetches one NL day and saves it. Run:
#     uv run python ingest.py
if __name__ == "__main__":
    from datetime import timedelta

    from period import market_day_period

    zone = "10YNL----------L"
    day = (datetime.now(timezone.utc) - timedelta(days=2)).date()
    # Period bounds come from the methodology helper: local-midnight to next
    # local-midnight, DST-aware. This is the ONLY place market-local time enters.
    ps, pe = market_day_period(day, zone)

    # DA example
    fetch_raw(
        params={
            "documentType": "A44",
            "in_Domain": zone,
            "out_Domain": zone,
            "periodStart": ps,
            "periodEnd": pe,
        },
        doc_label="A44",
        zone=zone,
    )

    # IM example
    fetch_raw(
        params={
            "documentType": "A85",
            "controlArea_Domain": zone,
            "periodStart": ps,
            "periodEnd": pe,
        },
        doc_label="A85",
        zone=zone,
    )