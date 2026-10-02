"""
Tests for the archive layer: which months get read, and from where.

The File Library offers the same data two ways — one zip per month, or one
bundle holding the whole history. They differ in nothing but the number of
members, so the reader must not care; and the month window must be applied by
NAME, so that a month outside it is never decompressed at all.

Run:
    uv run pytest tests/test_archive.py -q
    uv run python tests/test_archive.py
"""

import os
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.parse_filelibrary import (iter_zip_frames,  # noqa: E402
                                   member_month, read_zip_frame)

HEADER = "DateTime(UTC)\tResolutionCode\tAreaCode\tCurrency\tUpdateTime(UTC)"
ROW = "2020-01-01 00:00:00\tPT60M\t10YNL----------L\tEUR\t2020-01-02 00:00:00"


def _zip(tmp: str, name: str, members: list[str]) -> str:
    path = os.path.join(tmp, name)
    with zipfile.ZipFile(path, "w") as zf:
        for m in members:
            zf.writestr(m, f"{HEADER}\n{ROW}\n")
    return path


def test_a_monthly_zip_is_a_bundle_of_one():
    """One reader for both downloads. The per-month zip and the full-history
    bundle hold the same files under the same names; only the count differs, so
    a branch on 'which kind of zip is this' would be a branch on nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        one = _zip(tmp, "2020_01_EnergyPrices_12.1.D_r3.1.zip",
                   ["2020_01_EnergyPrices_12.1.D_r3.1.csv"])
        many = _zip(tmp, "EnergyPrices_12.1.D_r3.1.zip",
                    ["2020_01_EnergyPrices_12.1.D_r3.1.csv",
                     "2020_02_EnergyPrices_12.1.D_r3.1.csv"])

        assert [n for n, _ in iter_zip_frames(one)] == \
               ["2020_01_EnergyPrices_12.1.D_r3.1.csv"]
        assert len(list(iter_zip_frames(many))) == 2
        # and the single-member helper still answers for the single-member case
        assert len(read_zip_frame(one)) == 1


def test_the_window_is_applied_by_name_and_is_inclusive():
    """
    Months outside the window are never decompressed — that is the point of
    filtering on the filename. The bundles reach back to 2013/2014 and forward
    into the month in progress; only 2020_01 .. last closed month is ours.
    """
    with tempfile.TemporaryDirectory() as tmp:
        z = _zip(tmp, "ImbalancePrices_17.1.G_r3.1.zip",
                 [f"{m}_ImbalancePrices_17.1.G_r3.1.csv" for m in
                  ("2013_12", "2019_12", "2020_01", "2026_08", "2026_09")])

        got = [member_month(n) for n, _ in
               iter_zip_frames(z, "2020_01", "2026_08")]
        assert got == ["2020_01", "2026_08"], got

        # no bounds at all means everything, in month order
        assert len(list(iter_zip_frames(z))) == 5


def test_a_member_without_a_month_is_kept_not_dropped():
    """A name we cannot judge is not a name we may discard. Dropping it would be
    a silent loss; keeping it fails loudly later if it is not really data."""
    with tempfile.TemporaryDirectory() as tmp:
        z = _zip(tmp, "bundle.zip", ["readme_notes.csv",
                                     "2019_01_ImbalancePrices.csv"])
        names = [n for n, _ in iter_zip_frames(z, "2020_01", "2026_08")]
        assert names == ["readme_notes.csv"], names
        assert member_month("readme_notes.csv") is None
        assert member_month("2020_01_EnergyPrices_12.1.D_r3.1.csv") == "2020_01"


def test_the_glob_picks_the_bundle_and_leaves_the_monthly_snapshot_alone():
    """
    The only thing separating the two names is the leading "YYYY_MM_", so the
    DA/IM patterns carry no leading "*_". Get this wrong and the July per-month
    snapshot is silently folded into the refreshed history — two sources for one
    product, and no way to tell afterwards which row came from where.
    """
    import fnmatch

    sys.path.insert(0, os.path.join(ROOT, "src"))   # build_history is a script
    from build_history import PRODUCTS

    pattern = PRODUCTS["da"]["glob"]
    assert fnmatch.fnmatch("EnergyPrices_12.1.D_r3.1.zip", pattern)
    assert not fnmatch.fnmatch("2020_01_EnergyPrices_12.1.D_r3.1.zip", pattern)

    # ... and the same holds for LOAD, which now has a bundle too
    assert fnmatch.fnmatch("ActualTotalLoad_6.1.A_r3.zip",
                           PRODUCTS["load"]["glob"])
    assert not fnmatch.fnmatch("2020_01_ActualTotalLoad_6.1.A_r3.zip",
                               PRODUCTS["load"]["glob"])


def test_month_to_defaults_to_the_last_closed_month():
    """History is made of finished months. Left empty in settings, the end of
    the window is computed, so it cannot go stale — and a three-day tail of the
    month in progress never sneaks into a dated snapshot."""
    from datetime import datetime, timedelta, timezone

    import src.config as cfg

    m_from, m_to = cfg.history_months()
    today = datetime.now(timezone.utc).date()
    expected = f"{today.replace(day=1) - timedelta(days=1):%Y_%m}"
    assert m_from == "2020_01", m_from
    assert m_to == expected, (m_to, expected)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
