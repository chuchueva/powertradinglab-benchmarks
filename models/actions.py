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


def filled_action(action, price_da, buy_limit: float | None = None,
                  sell_limit: float | None = None) -> pd.Series:
    """
    The action after the day-ahead auction, for a bid with a price limit
    (2026-11-24). A strategy bids +1 MW @ buy_limit (buy only if the auction
    clears at or below it) and -1 MW @ sell_limit (sell only at or above it):

        +1 and price_da >  buy_limit    -> 0   (not filled)
        -1 and price_da <  sell_limit   -> 0   (not filled)
        otherwise                       -> the action as it was
        <NA>                            -> <NA> (no decision stays no decision)

    A limit of None means no limit on that side. What a filled action earned
    is still benchmarks_core.action_pnl — this only says what was filled.
    The trailing-bias BENCHMARK bids without limits by definition; limits
    belong to strategies (the Portfolio models).
    """
    index = action.index if isinstance(action, pd.Series) else None
    a = pd.Series(action, dtype="Int8").reset_index(drop=True)
    p = pd.Series(price_da, dtype="float64").reset_index(drop=True)
    out = a.copy()
    if buy_limit is not None:
        unfilled_buy = (a == 1).fillna(False) & (p > buy_limit)
        out[unfilled_buy.to_numpy(dtype=bool)] = 0
    if sell_limit is not None:
        unfilled_sell = (a == -1).fillna(False) & (p < sell_limit)
        out[unfilled_sell.to_numpy(dtype=bool)] = 0
    if index is not None:
        out.index = index
    return out

