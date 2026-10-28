"""
Neural network: train. PLACEHOLDER.

One network for all portfolio zones: zones stacked in one matrix, the zone
as one-hot input. Same target and the same time split as train_logreg.py.

Needs torch (or another framework; not yet in pyproject.toml).
"""
from datetime import datetime, UTC

from models.features import (CALENDAR_COLUMNS, area_columns, backtest_matrix,
                             fit_scaler, lag_columns, outcome, scale)
# import torch
# from models.model_utils import save_model

areas = ["NL", "RO", "EE", "SI", "SE1", "AL", "DE_LU"]
n_lags = 14
start = datetime(2022, 1, 1, tzinfo=UTC)
end = datetime(2025, 12, 31, tzinfo=UTC)
split = datetime(2025, 1, 1, tzinfo=UTC)
model_name = "nn_portfolio_v0"

matrix = backtest_matrix(areas, start, end, n_lags)
out = outcome(areas)
dd = matrix.merge(out, how="left", on=["area", "delivery_ts"])
dd = area_columns(dd, areas)
dd["y"] = (dd["always_buy"] > dd["always_sell"]).astype(int)

features = lag_columns(n_lags) + CALENDAR_COLUMNS + [f"area_{a}" for a in areas]
dd = dd.dropna(subset=features + ["always_buy", "always_sell"])

train = dd[dd["delivery_ts"] < split]
test = dd[dd["delivery_ts"] >= split]

scaler = fit_scaler(train, lag_columns(n_lags))      # from TRAINING rows only
train_s = scale(train, scaler)
test_s = scale(test, scaler)

# architecture = {"n_inputs": len(features), "hidden": [64, 32]}
# net = Net(**architecture)                          # models/nn.py
# ... training loop on train_s[features], train_s["y"]: sigmoid output, BCE loss
# save_model(model_name, features, dict(areas=areas, n_lags=n_lags, start=start,
#            end=end, split=split, architecture=architecture),
#            state_dict=net.state_dict(), scaler=scaler)

print(train.groupby("area").size())
