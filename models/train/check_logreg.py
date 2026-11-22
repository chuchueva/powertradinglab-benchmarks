"""
Logistic regression: check a saved model on the matrix. PLACEHOLDER.
Uses models.logreg.decide — the same function the live forecast calls.
"""
from datetime import datetime, UTC

import src.benchmarks_core as C
from models.features import backtest_matrix, outcome
from models.logreg import decide
from models.model_utils import load_model

model_name = "logreg_EE_v1"
start = datetime(2026, 10, 1, tzinfo=UTC)
end = datetime(2026, 10, 31, tzinfo=UTC)

meta, scaler, model = load_model(model_name)
areas = meta["areas"]
step = meta["step"]
n_lags = meta["n_lags"]
sensitivity = meta["sensitivity"]
m = backtest_matrix(areas, start, end, n_lags, step)
x = decide(m, model, meta["features"], sensitivity=sensitivity)
out = outcome(areas, step)
x = x.merge(out, how="left", on=["area", "delivery_ts"])
x["action_pnl"] = C.action_pnl(x["action"], x["always_buy"], x["always_sell"])

print(x.groupby("area")[["action_pnl", "ideal", "devil"]].sum())
