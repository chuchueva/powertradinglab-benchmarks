"""
Logistic regression: train.
"""
from datetime import datetime, UTC

from models.features import CALENDAR_COLUMNS, backtest_matrix, lag_columns, outcome
from models.actions import to_action
import src.benchmarks_core as C
import pandas as pd

import numpy as np
np.random.seed(123)

from sklearn.linear_model import LogisticRegression
from models.model_utils import save_model

model_name = "logreg_EE_v1"

areas = ["EE"]
n_lags = 14
step = "15min"
start = datetime(2025, 1, 1, tzinfo=UTC)
end = datetime(2026, 10, 31, tzinfo=UTC)

matrix = backtest_matrix(areas, start, end, n_lags, step)
out = outcome(areas, step)
dd = matrix.merge(out, how="left", on=["area", "delivery_ts"])
dd["y"] = (dd["always_buy"] > dd["always_sell"]).astype(int)
features = lag_columns(n_lags) + CALENDAR_COLUMNS + ([f"area_{a}" for a in areas] if len(areas) > 1 else [])

dd = dd.dropna(subset=features + ["always_buy", "always_sell"]).reset_index(drop=True)
# dd['trail_bias'] = (dd[["always_buy_d1", "always_buy_d2", "always_buy_d3"]].sum(axis=1) >
#                     dd[["always_sell_d1", "always_sell_d2", "always_sell_d3"]].sum(axis=1)).astype(int)
# dd['trail_diff'] = (dd[["always_buy_d1", "always_buy_d2", "always_buy_d3"]].sum(axis=1) -
#                     dd[["always_sell_d1", "always_sell_d2", "always_sell_d3"]].sum(axis=1)).astype(float)

feature_list = [
'trail_bias',
      # 'trail_diff',
# 'always_buy_d1',
      'price_da_d1',
# 'always_buy_d2',
      'price_da_d2',
# 'always_buy_d3',
      'price_da_d3',
# 'always_buy_d4', 'price_da_d4',
# 'always_buy_d5', 'price_da_d5',
# 'always_buy_d6', 'price_da_d6',
# 'always_buy_d7', 'price_da_d7',
# 'always_buy_d8', 'price_da_d8',
# 'always_buy_d9', 'price_da_d9',
# 'always_buy_d10', 'price_da_d10',
# 'hour_utc',
      # 'hour_local',
# 'quarter_of_hour',
'weekday_local',
      'month_local',
      'is_weekend',
# 'is_holiday'
]

#  Shuffle
idx  = list(dd.index)
np.random.shuffle(idx)
idx_train = idx[:int(len(idx) * 0.99)]
idx_test = idx[int(len(idx) * 0.99):]

# split = datetime(2026, 6, 1, tzinfo=UTC)
# idx_train = list(dd.index[dd["delivery_ts"] < split])
# idx_test = list(dd.index[dd["delivery_ts"] >= split])

dd_train = dd.loc[idx_train, feature_list]
dd_test = dd.loc[idx_test, feature_list]
y_train = dd.loc[idx_train, "y"]
y_test = dd.loc[idx_test, "y"]

buy_train = dd.loc[idx_train, "always_buy"].to_numpy(float)
sell_train = dd.loc[idx_train, "always_sell"].to_numpy(float)
buy_test = dd.loc[idx_test, "always_buy"].to_numpy(float)
sell_test = dd.loc[idx_test, "always_sell"].to_numpy(float)

tb_train = dd.loc[idx_train, "trail_bias"]
tb_test = dd.loc[idx_test, "trail_bias"]

# Train
stake_train = np.abs(buy_train - sell_train)
clf = LogisticRegression(max_iter=10000)
clf.fit(dd_train, y_train, sample_weight=stake_train)

# Output
y_model_train = clf.predict_proba(dd_train)[:,1]
y_model_test = clf.predict_proba(dd_test)[:,1]
y_model_train_int = (y_model_train > 0.5).astype(int)
y_model_test_int = (y_model_test > 0.5).astype(int)

# Assess
corr_train = (y_model_train_int == y_train).astype(int)
corr_test = (y_model_test_int == y_test).astype(int)
corr_tb_train = (tb_train == y_train).astype(int)
corr_tb_test = (tb_test == y_test).astype(int)

# PnL
action_train = to_action(y_model_train)
action_test = to_action(y_model_test)
tb_action_train = to_action(tb_train)
tb_action_test = to_action(tb_test)

pnl_train = C.action_pnl(action_train, buy_train, sell_train)
pnl_test = C.action_pnl(action_test, buy_test, sell_test)
pnl_tb_train = C.action_pnl(tb_action_train, buy_train, sell_train)
pnl_tb_test = C.action_pnl(tb_action_test, buy_test, sell_test)

print(f"Train {len(dd_train)} rows, match {np.mean(corr_train):.4f}, PnL {np.sum(pnl_train):.0f}, "
      f"TB PnL {np.sum(pnl_tb_train):.0f}, TB match {np.mean(corr_tb_train):.4f} \n"
      f"Test {len(dd_test)} rows, match {np.mean(corr_test):.4f}, PnL {np.sum(pnl_test):.0f}, "
      f"TB PnL {np.sum(pnl_tb_test):.0f}, TB match {np.mean(corr_tb_test):.4f}  \n")

for sens in np.arange(0.01, 0.2, 0.01):
      action_train_sens = to_action(y_model_train, sensitivity=sens)
      action_test_sens = to_action(y_model_test, sensitivity=sens)
      pnl_train_sens = C.action_pnl(action_train_sens, buy_train, sell_train)
      pnl_test_sens = C.action_pnl(action_test_sens, buy_test, sell_test)
      sh_train = (sum(action_train_sens.isin([-1,1])) / len(action_train_sens))
      sh_test = (sum(action_test_sens.isin([-1,1])) / len(action_test_sens))

      print(f"Sens {sens:.2f}, share train {sh_train:.2f}, share test {sh_test:.2f} "
            f"PnL train {np.sum(pnl_train_sens):.0f}, PnL test {np.sum(pnl_test_sens):.0f}")


coef = clf.coef_[0]
intercept = clf.intercept_[0]
std_train = dd.loc[idx_train, feature_list].std().to_numpy()
coef_table = pd.DataFrame({"feature": feature_list, "coef": coef, "std": std_train})
coef_table["coef_x_std"] = coef_table["coef"] * coef_table["std"]
coef_table["abs_effect"] = coef_table["coef_x_std"].abs()
coef_table = coef_table.sort_values("abs_effect", ascending=False)
print(intercept)
print(coef_table.to_string(index=False))

# Save model
coefficients = dict(zip(feature_list, clf.coef_[0].tolist()))
intercept = float(clf.intercept_[0])
meta = dict(areas=areas,
            step=step,
            n_lags=3,
            start=start,
            end=end,
            weights="stake = |always_buy - always_sell|",
            sensitivity=0.05,
            sensitivity_rule="drop approx 30% least confident on training")

# save_model(model_name, feature_list, meta, coefficients={"intercept": intercept, "coefficients": coefficients})
