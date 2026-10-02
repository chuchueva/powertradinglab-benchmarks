"""
Tests for the publication layer. No parquet, no files — just naming and bytes.

Run:
    uv run pytest tests/test_publish.py -q
    uv run python tests/test_publish.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.publish as P  # noqa: E402

DAY = pd.Timestamp("2026-08-06")
K1 = pd.Timestamp("2026-08-07 09:30:12", tz="UTC")
K2 = pd.Timestamp("2026-09-20 09:30:08", tz="UTC")


def _score_rows(knownby, action=-1):
    ptus = pd.date_range("2026-08-05 22:00", periods=4, freq="15min", tz="UTC")
    return pd.DataFrame({"delivery_ts": ptus, "knownby_ts": knownby, "area": "NL",
                         "action": action, "action_pnl": -8.79, "always_buy": 1.5,
                         "always_sell": -8.79, "ideal": 9.15, "devil": -15.58})


def test_two_passes_never_collide():
    """
    The filename carries the computation instant, so a second pass over the same
    delivery day lands beside the first instead of on top of it — which is the
    whole reason for not numbering files by a counter derived from the folder.
    """
    p1 = P._path("score", "NL", DAY, K1)
    p2 = P._path("score", "NL", DAY, K2)
    assert p1 != p2
    assert p1.name == "score_NL_2026-08-06_20260807T093012Z.xml"
    # filed by DELIVERY month, so both passes sit together even though the
    # second was computed in September
    assert p1.parent == p2.parent
    assert p1.parent.as_posix().endswith("nl/2026/08")


def test_rendering_is_byte_stable():
    """Identical input must give identical bytes, or the 'already published and
    unchanged' check would report differences that are not there."""
    g = _score_rows(K1)
    a = P._render(P._score_doc("NL", DAY, K1, 1, g))
    b = P._render(P._score_doc("NL", DAY, K1, 1, g))
    assert a == b
    assert a.startswith(b'<?xml version="1.0" encoding="UTF-8"?>')
    assert b'delivery_day="2026-08-06"' in a
    assert b'<calc_iteration>1</calc_iteration>' in a


def test_no_signal_means_no_action_attribute():
    """
    An absent decision is not a decision to sit out.

    Writing action="NA" (or 0) would invite a reader to treat it as a value;
    leaving the attribute out says plainly that no signal was formed.
    """
    ptus = pd.date_range("2026-08-05 22:00", periods=2, freq="15min", tz="UTC")
    g = pd.DataFrame({"delivery_ts": ptus, "knownby_ts": K1, "area": "NL",
                      "action": pd.array([1, None], dtype="Int8"),
                      "data_through": K1})
    xml = P._render(P._forecast_doc("NL", DAY, K1, 1, g)).decode()
    assert xml.count("action=") == 1, xml
    assert 'with_signal="1"' in xml
    assert 'ptu="2"' in xml


def test_a_refusal_is_published_while_an_unformed_signal_is_not():
    """
    The one case where nothing computed still produces a file.

    An unformed signal is the method saying "not enough history yet" — no file,
    because an empty file looks like a result. A REFUSAL is us saying "we ran,
    the input day was not whole, and this delivery day will never have an
    action". A reader who finds a hole in the series cannot tell those apart,
    and it is the second one they need to be told about.
    """
    ptus = pd.date_range("2026-08-05 22:00", periods=2, freq="15min", tz="UTC")
    base = {"delivery_ts": ptus, "knownby_ts": K1, "area": "NL",
            "action": pd.array([None, None], dtype="Int8"), "data_through": K1}

    unformed = pd.DataFrame({**base, "skip_reason": [None, None]})
    assert P._skip_reason(unformed) is None

    refused = pd.DataFrame({**base, "skip_reason": "input_incomplete:2026-08-29"})
    assert P._skip_reason(refused) == "input_incomplete:2026-08-29"

    xml = P._render(P._forecast_doc("NL", DAY, K1, 1, refused)).decode()
    assert '<no_forecast reason="input_incomplete" missing_day="2026-08-29" />' in xml, xml
    assert "<points" not in xml, "nothing was decided, so there is nothing to list"
    assert 'with_signal="0"' in xml


def test_a_table_written_before_the_gate_existed_still_publishes():
    """The column is new; every forecast already in the table predates it. Its
    absence must read as 'ordinary pass', not crash the publication of history."""
    ptus = pd.date_range("2026-08-05 22:00", periods=2, freq="15min", tz="UTC")
    old = pd.DataFrame({"delivery_ts": ptus, "knownby_ts": K1, "area": "NL",
                        "action": pd.array([1, -1], dtype="Int8"),
                        "data_through": K1})
    assert P._skip_reason(old) is None
    xml = P._render(P._forecast_doc("NL", DAY, K1, 1, old)).decode()
    assert xml.count("action=") == 2, xml


def test_the_version_comes_from_the_row_not_from_the_settings():
    """
    The bug of 2026-09-01, pinned.

    publish re-renders every historical group on every run. While the version
    was read from settings at render time, raising it by hand made all sixty
    already-published files differ from their own re-render — sixty refusals a
    day, for numbers that had not moved a digit. And a --force at that moment
    would have relabelled July's results with September's methodology, which is
    the one claim these numbers exist to make impossible.
    """
    import src.config as cfg

    g = _score_rows(K1).assign(schema_version=2, methodology_version=1)
    xml = P._render(P._score_doc("NL", DAY, K1, 1, g)).decode()
    assert '<version schema="2" methodology="1" />' in xml, xml

    # ... and today's setting, whatever it is, does not leak into an old file
    now = cfg.versions()
    assert f'schema="{now["schema"]}"' not in xml or now["schema"] == 2, (now, xml)

    fresh = _score_rows(K2).assign(schema_version=9, methodology_version=7)
    xml = P._render(P._score_doc("NL", DAY, K2, 2, fresh)).decode()
    assert '<version schema="9" methodology="7" />' in xml, xml


def test_a_table_without_version_columns_falls_back_to_the_settings():
    """Rows written before the columns existed must still render — that is what
    the one-off migration is for, and until it has run this is the behaviour."""
    import src.config as cfg

    v = cfg.versions()
    g = _score_rows(K1)
    assert "schema_version" not in g.columns
    xml = P._render(P._score_doc("NL", DAY, K1, 1, g)).decode()
    assert f'<version schema="{v["schema"]}" methodology="{v["methodology"]}" />' in xml


def test_schema_4_renames_the_five_values_and_schema_3_does_not():
    """
    The public vocabulary is a property of the ROW's schema, not of today's code.

    Renaming at render time is the same shape of bug the version stamp was:
    publish re-renders every historical group on every run, so a vocabulary
    applied unconditionally would make all 86 files on disk differ from
    themselves and be refused. A file speaks the language of its own schema.
    """
    old = _score_rows(K1).assign(schema_version=3, methodology_version=2)
    xml = P._render(P._score_doc("NL", DAY, K1, 1, old)).decode()
    assert 'ideal=' in xml and 'action_pnl=' in xml, xml
    assert 'extractable_value' not in xml
    assert 'n_days_default=' in xml and 'n_days_trailing_bias' not in xml
    assert '<action_codes>' not in xml

    new = _score_rows(K1).assign(schema_version=4, methodology_version=2)
    xml = P._render(P._score_doc("NL", DAY, K1, 1, new)).decode()
    for name in ("trailing_bias_value", "extractable_value", "adverse_value",
                 "passive_da_buy_value", "passive_da_sell_value"):
        assert f'{name}=' in xml, (name, xml)
    for gone in ("action_pnl=", "ideal=", "devil=", "always_buy=", "always_sell="):
        assert gone not in xml, (gone, xml)
    # the direction stays `action`; what +1 and -1 mean is said once, in prose
    assert 'action="-1"' in xml
    assert "+1 = buy on the day-ahead auction" in xml
    assert 'n_days_trailing_bias_default=' in xml


def test_a_refusal_can_name_the_leg_and_survives_without_one():
    """`leg` is the optional third part of skip_reason. Rows written before it
    existed carry two parts and must still render."""
    ptus = pd.date_range("2026-08-05 22:00", periods=2, freq="15min", tz="UTC")
    base = {"delivery_ts": ptus, "knownby_ts": K1, "area": "NL",
            "action": pd.array([None, None], dtype="Int8"), "data_through": K1,
            "schema_version": 4, "methodology_version": 2}

    with_leg = pd.DataFrame({**base, "skip_reason": "input_incomplete:2026-09-13:DA"})
    xml = P._render(P._forecast_doc("NL", DAY, K1, 1, with_leg)).decode()
    assert 'missing_day="2026-09-13"' in xml and 'leg="DA"' in xml, xml

    without = pd.DataFrame({**base, "skip_reason": "input_incomplete:2026-08-29"})
    xml = P._render(P._forecast_doc("NL", DAY, K1, 1, without)).decode()
    assert 'missing_day="2026-08-29"' in xml and "leg=" not in xml, xml

    assert P.parse_skip_reason("input_incomplete:2026-09-13:DA,IM")["leg"] == "DA,IM"


def test_a_day_without_an_action_still_publishes_the_four_market_values():
    """
    The defect this schema exists for (found 2026-09-18).

    ideal, devil, always_buy and always_sell are functions of the day's prices.
    They are published for a settled day whether or not we managed to issue an
    action for it. The fifth value asks what OUR rule earned, so where there is
    no action there is no number — and it is absent, not zero, not empty.
    """
    g = _score_rows(K1, action=pd.array([None] * 4, dtype="Int8")).assign(
        action_pnl=float("nan"), schema_version=5, methodology_version=2,
        skip_reason="input_incomplete:2026-09-13:DA")
    xml = P._render(P._score_doc("NL", DAY, K1, 1, g)).decode()

    assert "trailing_bias_value" not in xml, xml
    assert "action=" not in xml, xml
    for name in ("extractable_value", "adverse_value", "passive_da_buy_value",
                 "passive_da_sell_value"):
        assert name in xml, (name, xml)
    assert '<no_action reason="input_incomplete" missing_day="2026-09-13" leg="DA"' in xml, xml


def test_a_day_nobody_even_forecast_says_so():
    """A refusal is us looking and declining; this is the run never getting
    there at all. A reader can tell the two apart without our logs."""
    g = _score_rows(K1, action=pd.array([None] * 4, dtype="Int8")).assign(
        action_pnl=float("nan"), schema_version=5, methodology_version=2)
    xml = P._render(P._score_doc("NL", DAY, K1, 1, g)).decode()
    assert '<no_action reason="no_forecast_issued"' in xml, xml


def test_an_actionless_row_is_refused_under_an_older_schema():
    """The shape begins at schema 5. A pre-5 row without an action is a bug in
    whatever wrote it, and must not be rendered into a file that claims to speak
    schema 4 — that vocabulary has no way to say "no action"."""
    g = _score_rows(K1, action=pd.array([None] * 4, dtype="Int8")).assign(
        action_pnl=float("nan"), schema_version=4, methodology_version=2)
    try:
        P._score_doc("NL", DAY, K1, 1, g)
    except RuntimeError as e:
        assert "schema 5" in str(e), e
    else:
        raise AssertionError("an actionless schema-4 row must not render")


def test_a_scored_day_with_an_action_renders_exactly_as_before():
    """The new branch must not touch the old one: an ordinary day still carries
    all five values and its action attribute, in the same order."""
    g = _score_rows(K1).assign(schema_version=5, methodology_version=2)
    xml = P._render(P._score_doc("NL", DAY, K1, 1, g)).decode()
    assert 'action="-1" trailing_bias_value=' in xml, xml
    assert "no_action" not in xml, xml


def test_a_partial_day_publishes_its_points_and_says_which_hours_were_refused():
    """
    Schema 6: some hours decided, some not. The refused hours carry no action —
    absent, never "NA" — and one element says why and which. Before this, a
    single missing quarter-hour anywhere in the window turned the whole day into
    a <no_forecast> with no points at all.
    """
    ptus = pd.date_range("2026-09-26 22:00", periods=8, freq="15min", tz="UTC")
    g = pd.DataFrame({
        "delivery_ts": ptus, "knownby_ts": K1, "area": "NL",
        "action": pd.array([1, 1, 1, 1, None, None, None, None], dtype="Int8"),
        "data_through": K1, "schema_version": 6, "methodology_version": 3,
        "skip_reason": [None] * 4 + ["hour_incomplete:2026-09-25:IM"] * 4,
    })
    xml = P._render(P._forecast_doc("NL", DAY, K1, 1, g)).decode()

    assert '<partial_input reason="hour_incomplete" missing_day="2026-09-25" leg="IM" hours="23"' in xml, xml
    assert "<points count=\"8\"" in xml, xml
    assert xml.count('action="1"') == 4, xml
    assert "no_forecast" not in xml, xml


def test_a_partial_day_is_refused_under_an_older_schema():
    """The shape begins at schema 6; a pre-6 row in this state is a bug in
    whatever wrote it, not a file to render quietly."""
    ptus = pd.date_range("2026-09-26 22:00", periods=2, freq="15min", tz="UTC")
    g = pd.DataFrame({
        "delivery_ts": ptus, "knownby_ts": K1, "area": "NL",
        "action": pd.array([1, None], dtype="Int8"), "data_through": K1,
        "schema_version": 5, "methodology_version": 2,
        "skip_reason": [None, "hour_incomplete:2026-09-25:IM"],
    })
    try:
        P._forecast_doc("NL", DAY, K1, 1, g)
    except RuntimeError as e:
        assert "schema 6" in str(e), e
    else:
        raise AssertionError("a partial day must not render under schema 5")


def test_the_publisher_comes_from_the_row_not_from_the_settings():
    """
    The version stamp's third twin, and the last one. Identity was kept out of
    the structure so a rename would cost nothing — true of the FORMAT, false of
    the FILES, because publish re-renders every historical group and compares
    bytes. The day the domain went into settings, every past file would have
    differed from itself and been refused.
    """
    old = _score_rows(K1).assign(schema_version=6, methodology_version=3,
                                 publisher_name="TradingLab", publisher_url="")
    xml = P._render(P._score_doc("NL", DAY, K1, 1, old)).decode()
    assert '<publisher name="TradingLab" />' in xml, xml

    new = _score_rows(K2).assign(schema_version=6, methodology_version=3,
                                 publisher_name="PowerTradingLab",
                                 publisher_url="https://powertradinglab.org")
    xml = P._render(P._score_doc("NL", DAY, K2, 2, new)).decode()
    assert ('<publisher name="PowerTradingLab" '
            'url="https://powertradinglab.org" />') in xml, xml


def test_a_row_written_before_the_publisher_moved_still_renders():
    """No columns at all: the fallback is the settings, which is what keeps
    history renderable until the migration has stamped it."""
    g = _score_rows(K1).assign(schema_version=5, methodology_version=2)
    xml = P._render(P._score_doc("NL", DAY, K1, 1, g)).decode()
    assert "<publisher name=" in xml, xml


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
