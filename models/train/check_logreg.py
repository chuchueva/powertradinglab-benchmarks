"""
Logistic regression: check a saved model on the matrix. PLACEHOLDER.
Uses models.logreg.decide — the same function the live forecast calls.
"""
from datetime import datetime, UTC

import src.benchmarks_core as C
from models.features import area_columns, backtest_matrix, outcome
from models.logreg import decide
from models.model_utils import load_model

model_name = "logreg_v0"
sensitivity = None            # None -> +1/-1; e.g. 0.05 -> +1/0/-1
start = datetime(2025, 1, 1, tzinfo=UTC)
end = datetime(2025, 12, 31, tzinfo=UTC)

meta, scaler, model = load_model(model_name)
areas = meta["areas"]

m = backtest_matrix(areas, start, end)
matrix = area_columns(m, areas)
x = decide(matrix, model, meta["features"], sensitivity)
out = outcome(areas)
x = x.merge(out, how="left", on=["area", "delivery_ts"])
x["action_pnl"] = C.action_pnl(x["action"], x["always_buy"], x["always_sell"])

print(x.groupby("area")[["action_pnl", "ideal", "devil"]].sum())
