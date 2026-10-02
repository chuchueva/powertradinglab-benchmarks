"""
Day-ahead (A44) price parser: raw XML text -> clean pandas DataFrame.

Input:  the XML string returned by the ENTSO-E A44 endpoint (bare XML).
Output: a DataFrame with one row per 15-minute delivery MTU:
            delivery_ts  (UTC, tz-aware, start of the MTU)
            knownby_ts   (UTC, tz-aware, the document's <createdDateTime>)
            currency     (ISO code from <currency_Unit.name>)
            price_da     (in that currency per MWh, NOT converted here)
        plus a small metadata dict describing what was parsed and how many
        points were explicit vs forward-filled (for the quality layer).

  knownby_ts is the document's <createdDateTime>. ENTSO-E assembles the response
  on the fly and stamps this with the generation time, so it is REGENERATED on
  every request and effectively tracks our fetch time (two pulls minutes apart
  carry different createdDateTime even when the prices are identical). We keep it
  as the snapshot timestamp so each pull is tagged in the data itself.

  It is NOT a finality flag: a moving createdDateTime does not mean the values
  changed. ENTSO-E exposes no as-known/as-final marker, so the calc_iteration
  0 (as-known, ~T+1) vs 1 (as-final) distinction comes from WHEN we snapshot on
  our own schedule and from comparing the values across snapshots - never from
  this field alone.

Design notes (see Notion / entsoe_notes.md):
  - Timestamps are already UTC in the XML (suffix Z). No CET conversion.
  - curveType A03 = variable-sized block: a position is published only when the
    value changes; a missing position repeats the previous value; the last
    published point holds until the end of the period grid.
  - Do NOT hardcode 96 points. Expected count is derived from the Period's
    timeInterval and resolution.
  - A44 returns bare XML rooted at Publication_MarketDocument; there may be
    several TimeSeries (e.g. two calendar days in one response), each with its
    own Period. We parse them all and concatenate.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd
from lxml import etree

try:
    from src.entsoe_ack import raise_if_acknowledgement
except ImportError:          # run as a plain script from src/
    from entsoe_ack import raise_if_acknowledgement

# ISO 8601 duration -> minutes, for the resolutions ENTSO-E actually uses.
_RESOLUTION_MINUTES = {
    "PT15M": 15,
    "PT30M": 30,
    "PT60M": 60,
}


@dataclass
class ParseResult:
    """Parsed data plus quality metadata."""
    df: pd.DataFrame
    meta: dict = field(default_factory=dict)


def _strip_ns(tag: str) -> str:
    """'{namespace}Point' -> 'Point'. ENTSO-E documents are namespaced."""
    return tag.rsplit("}", 1)[-1]


def _parse_utc(ts: str) -> datetime:
    """'2026-07-24T22:00Z' -> tz-aware UTC datetime."""
    # Python's fromisoformat handles the trailing Z from 3.11+.
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def _expected_point_count(start: datetime, end: datetime, minutes: int) -> int:
    """Number of MTUs in [start, end) at the given resolution."""
    total_minutes = int((end - start).total_seconds() // 60)
    return total_minutes // minutes


def _parse_period(period_el, ns_strip) -> tuple[list[dict], int, int]:
    """
    Parse one <Period> block.

    Returns (rows, n_explicit, n_expected) where rows is a list of
    {delivery_ts, price} dicts on the full grid (A03 forward-filled).
    """
    # --- read interval + resolution ---
    ti = period_el.find("{*}timeInterval")
    start = _parse_utc(ti.find("{*}start").text)
    end = _parse_utc(ti.find("{*}end").text)
    resolution = period_el.find("{*}resolution").text
    if resolution not in _RESOLUTION_MINUTES:
        raise ValueError(f"Unexpected resolution: {resolution!r}")
    minutes = _RESOLUTION_MINUTES[resolution]

    n_expected = _expected_point_count(start, end, minutes)

    # --- read explicit points into {position: price} ---
    explicit: dict[int, float] = {}
    for pt in period_el.findall("{*}Point"):
        pos = int(pt.find("{*}position").text)
        price = float(pt.find("{*}price.amount").text)
        explicit[pos] = price
    n_explicit = len(explicit)

    # --- forward-fill the full 1..n_expected grid (A03) ---
    rows: list[dict] = []
    last_value: float | None = None
    for pos in range(1, n_expected + 1):
        if pos in explicit:
            last_value = explicit[pos]
        if last_value is None:
            # Position 1 must exist in a valid A03 curve. If it doesn't, we
            # have a real gap at the very start - do not invent a value.
            raise ValueError(
                f"No value available for position {pos}: "
                "curve does not start at position 1 (real gap, not A03 compression)."
            )
        delivery_ts = start + timedelta(minutes=minutes * (pos - 1))
        rows.append({"delivery_ts": delivery_ts, "price_da": last_value})

    return rows, n_explicit, n_expected


def _single_currency(currencies: set[str], doc: str) -> str | None:
    """
    The document's currency, read rather than assumed.

    DA is NOT always EUR: most zones price through SDAC in EUR, but some publish
    in local currency (UA_IPS in UAH, ~2000/MWh — assuming EUR there made the
    numbers ~40x too large). The same holds for A85. Conversion is a later,
    derived step; the parser's job is to carry the label truthfully.

    Several currencies in one document would make each row's unit ambiguous, so
    that is refused rather than silently resolved.
    """
    if len(currencies) > 1:
        raise ValueError(f"{doc} carries several currencies: {sorted(currencies)}")
    return next(iter(currencies)) if currencies else None


def parse_da_xml(xml_text: str) -> ParseResult:
    """Parse an A44 (day-ahead) XML string into a ParseResult."""
    root = etree.fromstring(xml_text.encode("utf-8"))
    # Checked BEFORE the type: an acknowledgement has no <type> at all, so the
    # type check would report a missing field and hide the actual answer.
    raise_if_acknowledgement(root, "A44")

    doc_type = root.findtext("{*}type")
    if doc_type != "A44":
        raise ValueError(f"Expected documentType A44, got {doc_type!r}")

    sender = root.find("{*}sender_MarketParticipant.mRID")
    sender_eic = sender.text if sender is not None else None

    # Document generation time: regenerated per request, so it tags THIS
    # snapshot (~ our fetch time), not a data-republication event.
    created_text = root.findtext("{*}createdDateTime")
    if not created_text:
        raise ValueError("Missing <createdDateTime> in A44 document.")
    knownby_ts = _parse_utc(created_text)

    all_rows: list[dict] = []
    total_explicit = 0
    total_expected = 0
    n_series = 0
    currencies: set[str] = set()

    for ts_el in root.findall("{*}TimeSeries"):
        n_series += 1
        cur = ts_el.findtext("{*}currency_Unit.name")
        if cur:
            currencies.add(cur)
        for period_el in ts_el.findall("{*}Period"):
            rows, n_explicit, n_expected = _parse_period(period_el, _strip_ns)
            all_rows.extend(rows)
            total_explicit += n_explicit
            total_expected += n_expected

    currency = _single_currency(currencies, "A44")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        # Guard against overlapping TimeSeries covering the same MTU.
        df = df.drop_duplicates(subset="delivery_ts").sort_values("delivery_ts")
        df = df.reset_index(drop=True)
        # Stamp the whole snapshot with its publisher creation time, and keep a
        # tidy column order: what/when-delivered, when-known, unit, value.
        df["knownby_ts"] = knownby_ts
        df["currency"] = currency
        df = df[["delivery_ts", "knownby_ts", "currency", "price_da"]]

    meta = {
        "document_type": "A44",
        "sender_eic": sender_eic,
        "currency": currency,
        "knownby_ts": knownby_ts.isoformat(),
        "n_timeseries": n_series,
        "n_expected_points": total_expected,
        "n_explicit_points": total_explicit,
        "n_filled_points": total_expected - total_explicit,
        "rows_out": len(df),
    }
    return ParseResult(df=df, meta=meta)


# --- manual test run --------------------------------------------------------
# Reads a saved XML file (so we don't hit the API here) and prints the result.
# Usage:
#     uv run python parse_da.py path/to/saved_da.xml
if __name__ == "__main__":
    import sys

    # if len(sys.argv) != 2:
    #     print("Usage: uv run python parse_da.py data/raw/A44_10YNL----------L_202607252200_202607262200_20260728T082449Z.xml")
    #     raise SystemExit(1)
    #
    xml_file = "data/raw/A44_10YNL----------L_202607252200_202607262200_20260728T084245Z.xml"
    with open(xml_file, "r", encoding="utf-8") as f:
        xml_text = f.read()

    result = parse_da_xml(xml_text)
    print("META:")
    for k, v in result.meta.items():
        print(f"  {k}: {v}")
    print("-" * 60)
    print(result.df.head(10).to_string(index=False))
    print("...")
    print(result.df.tail(5).to_string(index=False))
    print(f"\nTotal rows: {len(result.df)}")
    print(f"Price range: {result.df['price_da'].min()} .. {result.df['price_da'].max()}")
