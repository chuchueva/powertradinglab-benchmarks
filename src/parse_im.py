"""
Imbalance (A85) price parser: raw ZIP bytes -> clean pandas DataFrame.

Input:  the ZIP the ENTSO-E A85 endpoint returns (ingest saves it verbatim).
        Inside is a single Balancing_MarketDocument XML.
Output: a DataFrame with one row per 15-minute settlement MTU:
            delivery_ts    (UTC, tz-aware, start of the MTU)
            knownby_ts     (UTC, tz-aware, the document's <createdDateTime>)
            currency       (ISO code from <currency_Unit.name>)
            price_im_long  (category A04 = long / surplus, per MWh in `currency`)
            price_im_short (category A05 = short / deficit, per MWh in `currency`)
        plus a metadata dict with per-category quality counts.

Design notes (see docs/ENTSOE.md):
  - Response is a ZIP (starts with 'PK'); unzip in memory, do NOT touch disk.
  - Root is Balancing_MarketDocument; price field is imbalance_Price.amount.
  - DUAL PRICE: NL publishes two prices per MTU, tagged imbalance_Price.category
    A04 (long/surplus) and A05 (short/deficit). In practice each category sits in
    its OWN TimeSeries (one all-A04, one all-A05). We store both, never average.
  - curveType A03 = variable-sized block: a position is published only when the
    value changes; missing position repeats the previous; last point holds to the
    end of the grid. Each category skips DIFFERENT positions, so we forward-fill
    EACH category INDEPENDENTLY.
  - Do NOT hardcode 96 points. Expected count is derived from the Period's
    timeInterval and resolution (DST days give 92/100).
  - knownby_ts is <createdDateTime>: regenerated per request, so it tags THIS
    snapshot (~ our fetch time), not a republication event. It is NOT a finality
    flag; as-known vs as-final is decided by our snapshot schedule + comparison.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import io
import zipfile

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


# Every row in the store is one quarter-hour. A price published per hour (AL,
# 2026) or per half-hour applies to each quarter-hour inside it, so it is
# written once per quarter-hour with the same value. Nothing is invented: it is
# the same product, and the sum over 1 MW is unchanged. Without this an hourly
# zone has 24 rows a day, every day fails the 96-quarter completeness check,
# and the zone never gets a decision.
QUARTER_MIN = 15

# imbalance_Price.category -> our column name.
_CATEGORY_COLUMN = {
    "A04": "price_im_long",   # long / surplus
    "A05": "price_im_short",  # short / deficit
}


@dataclass
class ParseResult:
    """Parsed data plus quality metadata."""
    df: pd.DataFrame
    meta: dict = field(default_factory=dict)


def _parse_utc(ts: str) -> datetime:
    """'2026-07-25T22:00Z' -> tz-aware UTC datetime."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def _expected_point_count(start: datetime, end: datetime, minutes: int) -> int:
    """Number of MTUs in [start, end) at the given resolution."""
    total_minutes = int((end - start).total_seconds() // 60)
    return total_minutes // minutes


def _unzip_to_xml(raw: bytes) -> str:
    """
    Take the raw A85 response bytes and return the inner XML text.

    ENTSO-E wraps A85 in a ZIP (magic 'PK'). If we ever get bare XML instead,
    pass it straight through so the parser still works.
    """
    if raw[:2] != b"PK":
        return raw.decode("utf-8")
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("A85 ZIP is empty.")
        # There is one member per response; guard if that ever changes.
        xml_members = [n for n in names if n.lower().endswith(".xml")]
        member = xml_members[0] if xml_members else names[0]
        return zf.read(member).decode("utf-8")


def _forward_fill_period(period_el) -> tuple[dict[str, dict[datetime, float]],
                                             dict[str, int], int]:
    """
    Forward-fill one <Period>, splitting by imbalance_Price.category.

    Returns (series_by_cat, explicit_by_cat, n_expected) where
      series_by_cat[cat]  = {delivery_ts: price} on the full A03-filled grid,
      explicit_by_cat[cat] = number of explicitly published points for that cat.
    """
    ti = period_el.find("{*}timeInterval")
    start = _parse_utc(ti.find("{*}start").text)
    end = _parse_utc(ti.find("{*}end").text)
    resolution = period_el.find("{*}resolution").text
    if resolution not in _RESOLUTION_MINUTES:
        raise ValueError(f"Unexpected resolution: {resolution!r}")
    minutes = _RESOLUTION_MINUTES[resolution]
    n_expected = _expected_point_count(start, end, minutes)

    # Read explicit points into {category: {position: price}}.
    explicit: dict[str, dict[int, float]] = {}
    for pt in period_el.findall("{*}Point"):
        pos = int(pt.find("{*}position").text)
        price = float(pt.find("{*}imbalance_Price.amount").text)
        cat = pt.findtext("{*}imbalance_Price.category")
        if cat is None:
            raise ValueError(f"Point at position {pos} has no imbalance_Price.category.")
        explicit.setdefault(cat, {})[pos] = price

    # Forward-fill each category independently over the full grid.
    series_by_cat: dict[str, dict[datetime, float]] = {}
    explicit_counts: dict[str, int] = {}
    for cat, pos_price in explicit.items():
        explicit_counts[cat] = len(pos_price)
        series: dict[datetime, float] = {}
        last_value: float | None = None
        for pos in range(1, n_expected + 1):
            if pos in pos_price:
                last_value = pos_price[pos]
            if last_value is None:
                raise ValueError(
                    f"Category {cat}: no value for position {pos} "
                    "(curve does not start at position 1 - real gap, not A03)."
                )
            mtu_start = start + timedelta(minutes=minutes * (pos - 1))
            for q in range(minutes // QUARTER_MIN):
                series[mtu_start + timedelta(minutes=QUARTER_MIN * q)] = last_value
        series_by_cat[cat] = series

    return series_by_cat, explicit_counts, n_expected


def _single_currency(currencies: set[str], doc: str) -> str | None:
    """
    The document's currency, read rather than assumed (see parse_da for why).

    Many non-euro zones settle imbalance in local currency (PLN, CZK, HUF, RON,
    SEK, NOK, DKK, CHF, GBP, BGN, UAH...). Several currencies in one document
    would make each row's unit ambiguous, so that is refused.
    """
    if len(currencies) > 1:
        raise ValueError(f"{doc} carries several currencies: {sorted(currencies)}")
    return next(iter(currencies)) if currencies else None


def parse_im_bytes(raw: bytes) -> ParseResult:
    """Parse raw A85 response bytes (ZIP or bare XML) into a ParseResult."""
    xml_text = _unzip_to_xml(raw)
    root = etree.fromstring(xml_text.encode("utf-8"))
    # Checked BEFORE the type: an acknowledgement has no <type> at all, so the
    # type check would report a missing field and hide the actual answer.
    raise_if_acknowledgement(root, "A85")

    doc_type = root.findtext("{*}type")
    if doc_type != "A85":
        raise ValueError(f"Expected documentType A85, got {doc_type!r}")

    sender = root.find("{*}sender_MarketParticipant.mRID")
    sender_eic = sender.text if sender is not None else None
    area = root.findtext("{*}area_Domain.mRID")

    created_text = root.findtext("{*}createdDateTime")
    if not created_text:
        raise ValueError("Missing <createdDateTime> in A85 document.")
    knownby_ts = _parse_utc(created_text)

    # Accumulate {category: {delivery_ts: price}} across all TimeSeries/Periods.
    merged: dict[str, dict[datetime, float]] = {}
    explicit_total: dict[str, int] = {}
    expected_total = 0
    n_series = 0

    currencies: set[str] = set()

    for ts_el in root.findall("{*}TimeSeries"):
        n_series += 1
        cur = ts_el.findtext("{*}currency_Unit.name")
        if cur:
            currencies.add(cur)
        for period_el in ts_el.findall("{*}Period"):
            series_by_cat, explicit_counts, n_expected = _forward_fill_period(period_el)
            expected_total += n_expected
            for cat, series in series_by_cat.items():
                merged.setdefault(cat, {}).update(series)
                explicit_total[cat] = explicit_total.get(cat, 0) + explicit_counts[cat]

    # Build wide DataFrame: one row per delivery_ts, one column per category.
    frames = []
    for cat, series in merged.items():
        col = _CATEGORY_COLUMN.get(cat, f"price_im_{cat}")
        s = pd.Series(series, name=col)
        s.index.name = "delivery_ts"
        frames.append(s)

    currency = _single_currency(currencies, "A85")

    if frames:
        df = pd.concat(frames, axis=1).sort_index().reset_index()
        df.insert(1, "knownby_ts", knownby_ts)
        df.insert(2, "currency", currency)
        # Stable, predictable column order.
        cols = ["delivery_ts", "knownby_ts", "currency"]
        for cat in ("A04", "A05"):
            c = _CATEGORY_COLUMN[cat]
            if c in df.columns:
                cols.append(c)
        cols += [c for c in df.columns if c not in cols]
        df = df[cols]
    else:
        df = pd.DataFrame(columns=["delivery_ts", "knownby_ts", "currency"])

    meta = {
        "document_type": "A85",
        "sender_eic": sender_eic,
        "area_domain": area,
        "currency": currency,
        "knownby_ts": knownby_ts.isoformat(),
        "n_timeseries": n_series,
        "n_expected_points_per_category": expected_total // max(len(merged), 1),
        "categories": sorted(merged.keys()),
        "n_explicit_by_category": {k: explicit_total[k] for k in sorted(explicit_total)},
        "rows_out": len(df),
    }
    return ParseResult(df=df, meta=meta)


# --- manual test run --------------------------------------------------------
# Reads a saved A85 zip (so we don't hit the API here) and prints the result.
# Usage:
#     uv run python parse_im.py data/raw/A85_..._....zip
if __name__ == "__main__":
    import sys

    # if len(sys.argv) != 2:
    #     print("Usage: uv run python parse_im.py <path_to_A85_zip>")
    #     raise SystemExit(1)
    zip_file = "data/raw/A85_10YNL----------L_202607252200_202607262200_20260728T084246Z.zip"
    with open(zip_file, "rb") as f:
        raw = f.read()

    result = parse_im_bytes(raw)
    print("META:")
    for k, v in result.meta.items():
        print(f"  {k}: {v}")
    print("-" * 70)
    print(result.df.head(10).to_string(index=False))
    print("...")
    print(result.df.tail(5).to_string(index=False))
    print(f"\nTotal rows: {len(result.df)}")
    for col in ("price_im_long", "price_im_short"):
        if col in result.df.columns:
            print(f"{col} range: {result.df[col].min()} .. {result.df[col].max()}")
