"""
The input matrix: one row per (area, delivery hour, knownby_ts), lags
d1..d14 and calendar columns. Inputs only — no decision, no outcome.
Several zones stack into one matrix (`area` column).
"""
import os
from datetime import datetime, UTC

from models.features import backtest_matrix
from src.workflow import ROOT

OUT_DIR = os.path.join(ROOT, "data", "matrix")

areas = ["NL", "RO"]
n_lags = 14
start = datetime(2025, 1, 1, tzinfo=UTC)
end = datetime(2025, 12, 31, tzinfo=UTC)

matrix = backtest_matrix(areas, start, end, n_lags)     # Check what method produces

name = f"hourly_{'-'.join(areas)}_{start:%Y%m%d}_{end:%Y%m%d}"
os.makedirs(OUT_DIR, exist_ok=True)
matrix.to_parquet(os.path.join(OUT_DIR, name + ".parquet"), index=False)
# matrix.to_csv(os.path.join(OUT_DIR, name + ".csv"), index=False)

print(f"{name}: {len(matrix)} rows, {matrix['area'].nunique()} zones")
