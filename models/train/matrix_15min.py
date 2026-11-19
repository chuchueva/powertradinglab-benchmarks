"""
The input matrix at 15 minutes: one row per (area, delivery quarter-hour,
knownby_ts), lags d1..d14 (always_buy, always_sell and the bare prices) and
calendar columns. Inputs only — no decision, no outcome.
Several zones stack into one matrix (`area` column).
"""
import os
from datetime import datetime, UTC

from models.features import backtest_matrix
from src.workflow import ROOT

OUT_DIR = os.path.join(ROOT, "data", "matrix")

areas = ["EE"]
n_lags = 14
step = "15min"
start = datetime(2025, 1, 1, tzinfo=UTC)
end = datetime(2025, 12, 31, tzinfo=UTC)

matrix = backtest_matrix(areas, start, end, n_lags, step)

name = f"15min_{'-'.join(areas)}_{start:%Y%m%d}_{end:%Y%m%d}"
os.makedirs(OUT_DIR, exist_ok=True)
matrix.to_parquet(os.path.join(OUT_DIR, name + ".parquet"), index=False)

print(f"{name}: {len(matrix)} rows, {matrix['area'].nunique()} zones")
