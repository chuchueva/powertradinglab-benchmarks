"""
Logistic regression: train. PLACEHOLDER.

Target: y = 1 if always_buy > always_sell for the hour (buy on the day-ahead
was the better side), else 0. Inputs: lags + calendar (+ zone one-hot when
several zones). Split by TIME, never shuffled: the test period is after
the training period.

Needs scikit-learn (not yet in pyproject.toml).
"""
from datetime import datetime, UTC

from models.features import (CALENDAR_COLUMNS, area_columns, backtest_matrix,
                             lag_columns, outcome)
# from sklearn.linear_model import LogisticRegression
# from models.model_utils import save_model

areas = ["NL"]
n_lags = 14
start = datetime(2022, 1, 1, tzinfo=UTC)
end = datetime(2025, 12, 31, tzinfo=UTC)
split = datetime(2025, 1, 1, tzinfo=UTC)          # train < split <= test
model_name = "logreg_v0"

matrix = backtest_matrix(areas, start, end, n_lags)
out = outcome(areas)
dd = matrix.merge(out, how="left", on=["area", "delivery_ts"])
dd = area_columns(dd, areas)
dd["y"] = (dd["always_buy"] > dd["always_sell"]).astype(int)

features = lag_columns(n_lags) + CALENDAR_COLUMNS + ([f"area_{a}" for a in areas] if len(areas) > 1 else [])
dd = dd.dropna(subset=features + ["always_buy", "always_sell"])

train = dd[dd["delivery_ts"] < split]
test = dd[dd["delivery_ts"] >= split]

# clf = LogisticRegression(max_iter=1000)
# clf.fit(train[features].to_numpy(float), train["y"].to_numpy())

#

# save_model(model_name, features, dict(areas=areas, n_lags=n_lags, start=start, end=end, split=split),
#            model=clf)

print(f"train {len(train)} rows, test {len(test)} rows, {len(features)} features")
