"""
Tests for the update step's diff — what is worth writing to the store.

Synthetic frames only: no network, no store, no API key. The diff is the part
that decides what history the project ends up with, so it is tested on its own.

Run:
    uv run pytest tests/test_update.py -q
    uv run python tests/test_update.py
"""

import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entsoe_ack import NoDataAvailable  # noqa: E402
from src.parse_da import parse_da_xml  # noqa: E402
from src.parse_im import parse_im_bytes  # noqa: E402
from src.update import IM_RENAME, new_versions, split_range  # noqa: E402

VALUES = ["price_im_pos", "price_im_neg"]
T0 = pd.Timestamp("2026-08-01 00:00", tz="UTC")


def _frame(prices, knownby_hours) -> pd.DataFrame:
    """One row per PTU: prices as (pos, neg) pairs, knownby as hours after T0."""
    return pd.DataFrame({
        "delivery_ts": [T0 + pd.Timedelta(minutes=15 * i) for i in range(len(prices))],
        "knownby_ts": [T0 + pd.Timedelta(hours=h) for h in knownby_hours],
        "price_im_pos": [p[0] for p in prices],
        "price_im_neg": [p[1] for p in prices],
    })


def test_empty_store_takes_everything():
    fresh = _frame([(10.0, 12.0), (11.0, 11.0)], [5, 5])
    keep, n_new, n_changed = new_versions(fresh, fresh.iloc[:0], VALUES)
    assert len(keep) == 2 and n_new == 2 and n_changed == 0


def test_identical_poll_writes_nothing():
    """The common case: we poll, nothing moved, the store must not grow."""
    stored = _frame([(10.0, 12.0), (11.0, 11.0)], [5, 5])
    fresh = _frame([(10.0, 12.0), (11.0, 11.0)], [9, 9])   # same values, later poll
    keep, n_new, n_changed = new_versions(fresh, stored, VALUES)
    assert keep.empty and n_new == 0 and n_changed == 0


def test_one_revision_writes_one_row():
    stored = _frame([(10.0, 12.0), (11.0, 11.0)], [5, 5])
    fresh = _frame([(10.0, 12.0), (11.0, 99.0)], [9, 9])   # second PTU revised
    keep, n_new, n_changed = new_versions(fresh, stored, VALUES)
    assert n_changed == 1 and n_new == 0
    assert len(keep) == 1
    assert keep.loc[0, "price_im_neg"] == 99.0
    assert keep.loc[0, "knownby_ts"] == T0 + pd.Timedelta(hours=9)
    assert list(keep.columns) == list(fresh.columns)


def test_new_ptu_counts_as_new_not_revised():
    stored = _frame([(10.0, 12.0)], [5])
    fresh = _frame([(10.0, 12.0), (11.0, 11.0)], [9, 9])
    keep, n_new, n_changed = new_versions(fresh, stored, VALUES)
    assert n_new == 1 and n_changed == 0 and len(keep) == 1


def test_compared_against_the_latest_version_only():
    """
    A value that returns to what it was two revisions ago is a real change.

    The store holds versions, so the naive check 'have we ever seen this value?'
    would swallow it and leave a stretch of history where the store claims a
    value that was not in force.
    """
    stored = pd.concat([_frame([(10.0, 12.0)], [5]),      # first seen
                        _frame([(20.0, 22.0)], [7])],     # revised
                       ignore_index=True)
    fresh = _frame([(10.0, 12.0)], [9])                   # back to the old value
    keep, n_new, n_changed = new_versions(fresh, stored, VALUES)
    assert n_changed == 1 and len(keep) == 1


def test_missing_stays_missing():
    """NaN vs NaN is not a change — otherwise every poll of a gap writes a row."""
    stored = _frame([(np.nan, np.nan)], [5])
    fresh = _frame([(np.nan, np.nan)], [9])
    keep, n_new, n_changed = new_versions(fresh, stored, VALUES)
    assert keep.empty and n_changed == 0

    filled = _frame([(10.0, 12.0)], [9])                  # the gap gets filled
    keep, _, n_changed = new_versions(filled, stored, VALUES)
    assert n_changed == 1 and len(keep) == 1


def test_im_rename_matches_the_verified_mapping():
    """
    A04/long -> Positive, A05/short -> Negative.

    Checked against real files on 2026-07-25: 40 of the 96 PTU were dual-priced
    and every one of them matched exactly. Guarded here because the two names
    are plausible either way round, and getting it backwards would mirror the
    columns without changing any benchmark (min/max is label-agnostic) — a
    corruption that no downstream number would reveal.
    """
    assert IM_RENAME == {"price_im_long": "price_im_pos",
                         "price_im_short": "price_im_neg"}


def _a44(*currencies: str) -> str:
    """A minimal A44 document, one TimeSeries per currency given."""
    series = "".join(f"""
      <TimeSeries>
        <currency_Unit.name>{c}</currency_Unit.name>
        <Period>
          <timeInterval><start>2026-08-04T22:00Z</start><end>2026-08-04T23:00Z</end></timeInterval>
          <resolution>PT15M</resolution>
          <Point><position>1</position><price.amount>10.5</price.amount></Point>
        </Period>
      </TimeSeries>""" for c in currencies)
    return f"""<Publication_MarketDocument>
      <type>A44</type>
      <sender_MarketParticipant.mRID>X</sender_MarketParticipant.mRID>
      <createdDateTime>2026-08-05T08:00:00Z</createdDateTime>{series}
    </Publication_MarketDocument>"""


def test_currency_is_read_from_the_document():
    """DA is not always EUR — UA_IPS prices in UAH, and assuming EUR there made
    the numbers ~40x too big. So the label is read, never defaulted."""
    res = parse_da_xml(_a44("UAH"))
    assert res.meta["currency"] == "UAH"
    assert (res.df["currency"] == "UAH").all()
    assert list(res.df.columns) == ["delivery_ts", "knownby_ts", "currency", "price_da"]


def test_mixed_currency_document_is_refused():
    """Two currencies in one document make each row's unit ambiguous — stop, don't guess."""
    try:
        parse_da_xml(_a44("EUR", "UAH"))
    except ValueError as e:
        assert "several currencies" in str(e)
    else:
        raise AssertionError("a mixed-currency document must be refused")


_ACK = """<?xml version="1.0" encoding="UTF-8"?>
<Acknowledgement_MarketDocument
  xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0">
  <mRID>8f4d06eb</mRID>
  <createdDateTime>2026-08-12T10:00:36Z</createdDateTime>
  <Reason><code>999</code><text>No matching data found for Data item Imbalance Prices [17.1.G].</text></Reason>
</Acknowledgement_MarketDocument>"""


def test_no_data_is_reported_as_the_source_saying_no():
    """
    Asked for a period it has nothing for, ENTSO-E answers HTTP 200 with an
    acknowledgement. Left unrecognised, the parser said "Expected documentType
    A85, got None" — which reads as OUR bug and cost a wrong diagnosis: we
    concluded the request had never reached the network.

    "The source has no data" and "we are broken" must not look alike.
    """
    for parse, payload in ((parse_da_xml, _ACK), (parse_im_bytes, _ACK.encode())):
        try:
            parse(payload)
        except NoDataAvailable as e:
            assert e.code == "999", e.code
            assert "No matching data" in e.text
            assert "no data" in str(e)
        else:
            raise AssertionError(f"{parse.__name__} accepted an acknowledgement")


# --------------------------------------------------------------------------- #
# Splitting a range the source will not take whole
# --------------------------------------------------------------------------- #
def _days(chunks):
    """Every delivery day the chunks would actually request, in order."""
    out = []
    for a, b in chunks:
        d = a
        while d <= b:
            out.append(d)
            d += timedelta(days=1)
    return out


def test_a_range_under_the_cap_is_left_alone():
    """The three-day daily poll must not become three requests. Nothing about
    the ordinary run changes because of a limit it never reaches."""
    a, b = date(2026, 9, 14), date(2026, 9, 16)
    assert split_range(a, b, 30) == [(a, b)]
    # and exactly at the cap is still one request — the limit is inclusive
    assert split_range(date(2026, 6, 1), date(2026, 6, 30), 30) == \
           [(date(2026, 6, 1), date(2026, 6, 30))]


def test_a_31_day_month_is_cut_in_half_not_into_30_plus_1():
    """August is the case that forced this: 31 days against a 30-day cap. A
    remainder cut would send 30 days — legal today, and one tightening of the
    limit away from failing. The even cut leaves both halves clear of it."""
    chunks = split_range(date(2026, 8, 1), date(2026, 8, 31), 30)
    assert chunks == [(date(2026, 8, 1), date(2026, 8, 16)),
                      (date(2026, 8, 17), date(2026, 8, 31))], chunks


def test_a_long_backfill_covers_every_day_exactly_once():
    """
    The real reason this exists is not the monthly pass — that one can be missed
    and comes round again — but the backfill after an outage, which cannot.
    Ninety days must arrive as ninety days: no gap quietly left unrequested, no
    day fetched twice.
    """
    a, b = date(2026, 6, 1), date(2026, 8, 29)          # 90 days
    chunks = split_range(a, b, 30)
    assert len(chunks) == 3, chunks
    days = _days(chunks)
    assert days == [a + timedelta(days=i) for i in range(90)]
    assert len(set(days)) == len(days)                  # nothing twice
    assert all(x[1] < y[0] for x, y in zip(chunks, chunks[1:]))   # no overlap
    assert max((y - x).days + 1 for x, y in chunks) <= 30


def test_no_cap_means_one_request_however_long():
    """0 is how the limit gets lifted: one number in settings, no code touched.
    A year in a single request is what the platform used to accept, and what it
    should accept again."""
    a, b = date(2020, 1, 1), date(2020, 12, 31)
    assert split_range(a, b, 0) == [(a, b)]


def test_a_single_day_survives_the_cut():
    """The degenerate range is the one that breaks arithmetic written for the
    general case. Recovering one missing day is a common enough errand."""
    d = date(2026, 9, 13)
    assert split_range(d, d, 30) == [(d, d)]
    assert split_range(d, d, 1) == [(d, d)]


def test_the_cap_comes_from_settings_when_not_given():
    """
    The default argument is for the tests; the live path reads settings, so that
    changing the limit is one edit in one file.

    Asserted as an EQUIVALENCE, not as a chunk count. The first version of this
    test hard-coded "one day over the cap must give two chunks" and went red the
    same day the cap was lifted to 0 — it was testing the value in settings, not
    the wiring. The wiring is: called without a number, split_range behaves
    exactly as if it had been handed the number settings holds.
    """
    import src.config as cfg
    a, b = date(2020, 1, 1), date(2020, 12, 31)
    assert split_range(a, b) == split_range(a, b, cfg.max_query_days())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
