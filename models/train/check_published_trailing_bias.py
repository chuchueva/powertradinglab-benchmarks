"""
Manual check: replay the trailing-bias rule at the instants the service
really decided, compare with what it published (forecasts.parquet).
Hours the service refused at the gate (skip_reason set) are left out.
"""
import numpy as np
import pandas as pd

import src.config as cfg
import src.workflow_compute as wc
from models.features import build_matrix
from models.trailing_bias import decide

area = "RO"
tol = 1e-6                       # EUR per MW; float sums in a different order


f = pd.read_parquet(wc.FORECASTS)
f = f[(f["area"] == area) & f["skip_reason"].isna()].copy()
f["delivery_ts"] = f["delivery_ts"].dt.floor("h")
pub = (f.groupby(["knownby_ts", "delivery_ts"], as_index=False)
        [["action", "sig_buy", "sig_sell", "n_obs", "data_through", "methodology_version"]]
        .first())
pub = pub.rename(columns={c: f"pub_{c}" for c in
                          ["action", "sig_buy", "sig_sell", "n_obs", "data_through"]})

n_days = cfg.n_days(area)
knownby_list = sorted(pub["knownby_ts"].unique())
matrix = build_matrix(area, knownby_list)
x = decide(matrix, n_days)
x = x.merge(pub, how="inner", on=["knownby_ts", "delivery_ts"])

x["same_action"] = (x["action"] == x["pub_action"]).fillna(False) | \
                   (x["action"].isna() & x["pub_action"].isna())
formed = x["action"].notna() & x["pub_action"].notna()
x["d_sig_buy"] = (x["sig_buy"] - x["pub_sig_buy"]).abs()
x["d_sig_sell"] = (x["sig_sell"] - x["pub_sig_sell"]).abs()
x["match"] = x["same_action"] & (~formed | ((x["d_sig_buy"] <= tol) & (x["d_sig_sell"] <= tol)
                                            & (x["data_through"] == x["pub_data_through"])))

summary = x.groupby("methodology_version")["match"].agg(["size", "sum"])
bad = x[~x["match"]]

print(summary)
print(f"{area}: {len(bad)} of {len(x)} hours differ, action changed in "
      f"{int((~x['same_action']).sum())}")
