"""
Neural network: check a saved model on the matrix. PLACEHOLDER.
Uses models.nn.decide — the same function the live forecast will call.
"""
from datetime import datetime, UTC

from models.features import area_columns, backtest_matrix, outcome, scale
from models.model_utils import load_model
from models.nn import decide

model_name = "nn_portfolio_v0"
sensitivity = None
start = datetime(2025, 1, 1, tzinfo=UTC)
end = datetime(2025, 12, 31, tzinfo=UTC)

meta, scaler, state = load_model(model_name)
# net = Net(**meta["architecture"]); net.load_state_dict(state)       # models/nn.py
model = None
areas = meta["areas"]
m = backtest_matrix(areas, start, end)
m = area_columns(m, areas)
matrix = scale(m, scaler)
x = decide(matrix, model, meta["features"], sensitivity)
out = outcome(areas)
x = x.merge(out, how="left", on=["area", "delivery_ts"])
