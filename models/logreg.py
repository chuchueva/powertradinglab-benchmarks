"""
Logistic regression — the decision. PLACEHOLDER.

    decide(x, model, features, sensitivity=None) -> x with p_buy, action

x is the matrix (features.py). The same function is called by the live
forecast and by train/check_logreg.py; nothing else computes this action.
"""
import pandas as pd

from models.actions import to_action


def decide(x: pd.DataFrame, model, features: list[str],
           sensitivity: float | None = None) -> pd.DataFrame:
    x = x.copy()
    ok = x[features].notna().all(axis=1)          # a hole in the inputs -> <NA>
    p = pd.Series(pd.NA, index=x.index, dtype="Float64")
    if ok.any():
        p[ok] = model.predict_proba(x.loc[ok, features].to_numpy(float))[:, 1]
    x["p_buy"] = p
    x["action"] = to_action(p, sensitivity)
    return x
