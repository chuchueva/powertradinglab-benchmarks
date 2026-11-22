"""
Logistic regression — the decision, from saved coefficients.

    decide(x, model, features, sensitivity=None) -> x with the terms, logit,
                                                     p_buy, action

x is the matrix (features.py). `model` is the coefficients dict from
coefficients.json (model_utils.load_model):
    {"intercept": b0, "coefficients": {feature: b, ...}}

    logit = b0 + sum(b_i * x_i)
    p_buy = 1 / (1 + exp(-logit))

This is exactly what scikit-learn's predict_proba does for two classes; the
output is computed here, so no library is needed to run a saved model.
Every term b_i * x_i is left in the row as its own column, term_<feature>,
so the decision can be read off the row.

The same function is called by the live forecast and by train/check_logreg.py;
nothing else computes this action.
"""
import numpy as np
import pandas as pd

from models.actions import to_action


def decide(x: pd.DataFrame, model: dict, features: list[str],
           sensitivity: float | None = None) -> pd.DataFrame:
    x = x.copy()
    coefficients = model["coefficients"]
    if list(coefficients.keys()) != list(features):
        raise ValueError(f"model features {list(coefficients.keys())} "
                         f"do not match {list(features)}")
    ok = x[features].notna().all(axis=1)          # a hole in the inputs -> <NA>

    logit = pd.Series(float(model["intercept"]), index=x.index)
    for f in features:
        term = coefficients[f] * x[f].astype(float)
        x[f"term_{f}"] = term
        logit = logit + term
    x["logit"] = logit.where(ok)

    p = 1.0 / (1.0 + np.exp(-x["logit"].astype(float)))
    x["p_buy"] = pd.array(p.where(ok), dtype="Float64")
    x["action"] = to_action(x["p_buy"], sensitivity)
    return x
