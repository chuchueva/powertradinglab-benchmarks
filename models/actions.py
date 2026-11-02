"""
From a model's output to an action — the ONE place this happens.

A model returns a score p in [0, 1]: the probability that BUY on the
day-ahead is the better side for that hour (always_buy > always_sell).

    to_action(p)                    -> +1 / -1          (no sensitivity)
    to_action(p, sensitivity=s)     -> +1 / 0 / -1      0 = do nothing

With sensitivity s (0 < s < 0.5):
    p > 0.5 + s   -> +1 (buy)
    p < 0.5 - s   -> -1 (sell)
    otherwise     ->  0 (no position)

p = <NA> (the model could not form a score) -> action <NA>, which is NOT 0:
<NA> = no decision was possible, 0 = a decision to stay out.

What an action earned is benchmarks_core.action_pnl: 0 earns 0.0, <NA> NaN.
Before a 0 goes live, the publication and the site must also know what a 0 is.
"""
import numpy as np
import pandas as pd


def to_action(p: pd.Series, sensitivity: float | None = None) -> pd.Series:
    index = p.index if isinstance(p, pd.Series) else None
    p = pd.Series(p, dtype="Float64")
    na = p.isna().to_numpy()
    p = p.to_numpy(dtype="float64", na_value=np.nan)
    if sensitivity is None:
        a = np.where(p > 0.5, 1, -1)
    else:
        assert 0 < sensitivity < 0.5, "sensitivity must be in (0, 0.5)"
        a = np.where(p > 0.5 + sensitivity, 1,
                     np.where(p < 0.5 - sensitivity, -1, 0))
    out = pd.Series(pd.array(a, dtype="Int8"), index=index)
    out[na] = pd.NA
    return out
