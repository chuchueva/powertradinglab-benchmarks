"""
Trailing bias — the rule.

    sig_buy  = mean(always_buy_d1 .. always_buy_dN)
    sig_sell = mean(always_sell_d1 .. always_sell_dN)
    action   = +1 (buy DA) if sig_buy > sig_sell, else -1 (sell DA)

An incomplete window (any dk missing) gives sig <NA> and action <NA>:
an unformed signal is not a -1. Definitions: docs/METHOD.md.
"""
import numpy as np
import pandas as pd


def decide(x: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Add sig_buy, sig_sell, n_obs, data_through, action to the matrix rows."""
    x = x.copy()
    buy_cols = [f"always_buy_d{k}" for k in range(1, n_days + 1)]
    sell_cols = [f"always_sell_d{k}" for k in range(1, n_days + 1)]

    x["sig_buy"] = x[buy_cols].mean(axis=1, skipna=False)
    x["sig_sell"] = x[sell_cols].mean(axis=1, skipna=False)
    x["n_obs"] = x[buy_cols].notna().sum(axis=1).astype("Int64")
    x["data_through"] = x["delivery_ts_d1"]

    formed = x["sig_buy"].notna() & x["sig_sell"].notna()
    action = pd.array(np.where(x["sig_buy"] > x["sig_sell"], 1, -1), dtype="Int8")
    action[~formed.to_numpy()] = pd.NA
    x["action"] = action
    return x
