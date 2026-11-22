"""
Manual check of a model in SHADOW: replay the logistic regression at the
instants the service really decided, compare with what it wrote to
data/processed/shadow/logistic_regression_<ZONE>.parquet, then see what both
variants earned on the days that have settled, next to trailing bias.

Expected: 0 slots differ. A difference means the live path and the replay do
not build the same matrix — that is the thing this check exists to catch.
"""
import numpy as np
import pandas as pd

import src.benchmarks_core as C
import src.workflow_compute as wc
from models.features import build_matrix, outcome
from models.logreg import decide
from models.model_utils import load_model

area = "EE"
kind = "logistic_regression"
tol = 1e-12                      # p_buy, computed by the same formula

shadow = pd.read_parquet(wc.shadow_path(kind, area))
model_name = shadow["model_name"].iloc[-1]
meta, scaler, model = load_model(model_name)
features = meta["features"]
sensitivity = float(meta["sensitivity"])
step = meta["step"]
n_lags = int(meta["n_lags"])

# --- 1. Replay: the same instants, the same functions ------------------------
knownby_list = sorted(shadow["knownby_ts"].unique())
matrix = build_matrix(area, knownby_list, n_lags, True, step)
plain = decide(matrix, model, features, None)
sens = decide(matrix, model, features, sensitivity)
replay = plain[["knownby_ts", "delivery_ts", "p_buy", "action"]].copy()
replay["action_sens"] = sens["action"]
replay = replay.rename(columns={"p_buy": "re_p_buy", "action": "re_action",
                                "action_sens": "re_action_sens"})

pub = shadow[["knownby_ts", "delivery_ts", "p_buy", "action", "action_sens"]]
x = pub.merge(replay, how="inner", on=["knownby_ts", "delivery_ts"])

both_na = x["action"].isna() & x["re_action"].isna()
x["same_action"] = (x["action"] == x["re_action"]).fillna(False) | both_na
both_na_sens = x["action_sens"].isna() & x["re_action_sens"].isna()
x["same_action_sens"] = (x["action_sens"] == x["re_action_sens"]).fillna(False) | both_na_sens
p_pub = x["p_buy"].to_numpy(dtype="float64", na_value=np.nan)
p_re = x["re_p_buy"].to_numpy(dtype="float64", na_value=np.nan)
d_p = np.abs(p_pub - p_re)
same_p = (d_p <= tol) | (np.isnan(p_pub) & np.isnan(p_re))
x["same_p"] = same_p
x["match"] = x["same_action"] & x["same_action_sens"] & x["same_p"]
bad = x[~x["match"]]

print(f"{area} {model_name}: {len(knownby_list)} decisions, {len(x)} slots, "
      f"{len(bad)} differ")

# --- 2. What it earned, on settled slots --------------------------------------
out = outcome(area, step)
y = shadow.merge(out, how="inner", on=["area", "delivery_ts"])
y["pnl_lr"] = C.action_pnl(y["action"], y["always_buy"], y["always_sell"])
y["pnl_lr_sens"] = C.action_pnl(y["action_sens"], y["always_buy"], y["always_sell"])

scores = pd.read_parquet(wc.SCORES)
tb = scores[scores["area"] == area].sort_values("knownby_ts")
tb = tb.drop_duplicates("delivery_ts", keep="last")[["delivery_ts", "action_pnl"]]
tb = tb.rename(columns={"action_pnl": "pnl_tb"})
y = y.merge(tb, how="left", on="delivery_ts")

y["day"] = y["delivery_ts"].dt.tz_convert("UTC").dt.date
daily = y.groupby("day")[["pnl_lr", "pnl_lr_sens", "pnl_tb", "ideal"]].sum()
print(daily)
print(daily.sum())
