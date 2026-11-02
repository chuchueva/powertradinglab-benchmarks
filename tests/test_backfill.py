"""backfill: history poured in BEFORE what the store already holds (2026-10-28)."""
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.workflow as W              # noqa: E402

LOG = logging.getLogger("test")


def _da(start, n, price, knownby, area="DE_LU"):
    ts = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"delivery_ts": ts, "knownby_ts": pd.Timestamp(knownby, tz="UTC"),
                         "currency": "EUR", "price_da": price, "area": area})


def _im(start, n, price, knownby, area="DE_LU"):
    ts = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"delivery_ts": ts, "knownby_ts": pd.Timestamp(knownby, tz="UTC"),
                         "currency": "EUR", "price_im_pos": price, "price_im_neg": price,
                         "area": area})


def test_backfill_keeps_polled_rows_and_adds_only_older_history(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "STORE_ROOT", str(tmp_path))
    polled_da = _da("2026-08-31 22:00", 8, 50.0, "2026-09-01 09:40")
    polled_im = _im("2026-08-31 22:00", 8, 60.0, "2026-09-02 09:40")
    W.write_store(polled_da, polled_im, "DE_LU")

    # history overlaps the polled rows with DIFFERENT values; imbalance is
    # filed as "DE" for part of it, as the archive does after 2022-06-22
    da_hist = _da("2026-08-31 20:00", 16, 1.0, "2026-10-28 08:00")
    im_hist = pd.concat([_im("2026-08-31 20:00", 4, 2.0, "2026-10-28 08:00"),
                         _im("2026-08-31 21:00", 12, 3.0, "2026-10-28 08:00", area="DE")])

    W.backfill_store("DE_LU", LOG, da_hist, im_hist)
    da, im = W.load_store("DE_LU")

    assert len(da) == 8 + 8 and len(im) == 8 + 8
    first = pd.Timestamp("2026-08-31 22:00", tz="UTC")
    assert (da.loc[da["delivery_ts"] >= first, "price_da"] == 50.0).all(), "polled rows untouched"
    assert (im.loc[im["delivery_ts"] >= first, "price_im_pos"] == 60.0).all()
    assert (da.loc[da["delivery_ts"] < first, "price_da"] == 1.0).all()
    # the "DE" imbalance rows before the first polled one are DE_LU's
    assert (im.loc[(im["delivery_ts"] >= pd.Timestamp("2026-08-31 21:00", tz="UTC"))
                   & (im["delivery_ts"] < first), "price_im_pos"] == 3.0).all()

    W.backfill_store("DE_LU", LOG, da_hist, im_hist)       # idempotent
    da2, im2 = W.load_store("DE_LU")
    assert len(da2) == len(da) and len(im2) == len(im)


def test_backfill_of_an_empty_store_is_the_whole_history(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "STORE_ROOT", str(tmp_path))
    W.backfill_store("DE_LU", LOG, _da("2026-01-01", 8, 1.0, "2026-10-28"),
                     _im("2026-01-01", 8, 2.0, "2026-10-28"))
    da, im = W.load_store("DE_LU")
    assert len(da) == 8 and len(im) == 8
