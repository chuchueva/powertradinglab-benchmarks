"""
Parallel run: this machine (portfolio, methodology 4) against the server
(main, methodology 3), for one decision day. Forecasts per zone and hour,
scores per zone and day.

Server results are copied beforehand to data/compare/server/results/.
"""
import pandas as pd

day = "2026-11-01"                     # the UTC date of the decision

local_f = pd.read_parquet("data/processed/results/forecasts.parquet")
server_f = pd.read_parquet("data/compare/server/results/forecasts.parquet")
local_s = pd.read_parquet("data/processed/results/scores.parquet")
server_s = pd.read_parquet("data/compare/server/results/scores.parquet")

local_f = local_f[local_f["knownby_ts"].dt.strftime("%Y-%m-%d") == day].copy()
server_f = server_f[server_f["knownby_ts"].dt.strftime("%Y-%m-%d") == day].copy()
local_f["hour"] = local_f["delivery_ts"].dt.floor("h")
server_f["hour"] = server_f["delivery_ts"].dt.floor("h")

cols = ["action", "sig_buy", "sig_sell", "data_through", "skip_reason", "methodology_version"]
lf = local_f.groupby(["area", "hour"], as_index=False)[cols].first()
sf = server_f.groupby(["area", "hour"], as_index=False)[cols].first()
f = lf.merge(sf, how="outer", on=["area", "hour"], suffixes=("_local", "_server"))

f["same_action"] = (f["action_local"] == f["action_server"]).fillna(False) | \
                   (f["action_local"].isna() & f["action_server"].isna())
f["d1_today_local"] = f["data_through_local"].dt.strftime("%Y-%m-%d") == day
f["d1_today_server"] = f["data_through_server"].dt.strftime("%Y-%m-%d") == day

forecast_summary = f.groupby("area").agg(
    hours=("hour", "size"),
    action_differs=("same_action", lambda s: int((~s).sum())),
    d1_today_local=("d1_today_local", "sum"),
    d1_today_server=("d1_today_server", "sum"),
    meth_local=("methodology_version_local", "first"),
    meth_server=("methodology_version_server", "first"),
)
diff = f[~f["same_action"]]

# scores: the day scored today on both machines
sday = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
local_s = local_s[local_s["knownby_ts"].dt.strftime("%Y-%m-%d") == day]
server_s = server_s[server_s["knownby_ts"].dt.strftime("%Y-%m-%d") == day]
vals = ["action_pnl", "ideal", "devil", "always_buy", "always_sell"]
ls = local_s.groupby("area")[vals].sum()
ss = server_s.groupby("area")[vals].sum()
score_diff = (ls - ss).round(4)

print(forecast_summary)
print(score_diff)
