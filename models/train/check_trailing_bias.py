"""
Trailing bias on the matrix: decision -> what it earned. No training.
"""
from datetime import datetime, UTC

import src.benchmarks_core as C
from models.features import backtest_matrix, outcome
from models.trailing_bias import decide

areas = ["EE"]
n_days = 3
start = datetime(2026, 1, 1, tzinfo=UTC)
end = datetime(2026, 10, 31, 23,45, tzinfo=UTC)

matrix = backtest_matrix(areas, start, end)         # Rerun matrix collection

out = outcome(areas)

x = decide(matrix, n_days)
x = x.merge(out, how="left", on=["area", "delivery_ts"])
x["action_pnl"] = C.action_pnl(x["action"], x["always_buy"], x["always_sell"])

# Can arrange the way I need
x['year'] = x['delivery_ts'].dt.year
x['month'] = x['delivery_ts'].dt.month
summary = x.groupby(["area", "year", "month"])[["action_pnl", "ideal", "devil", "always_buy", "always_sell"]].sum()
print(summary)
