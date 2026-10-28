"""
Neural network — the decision. PLACEHOLDER.

One network for all zones of the portfolio is possible: the matrix stacks
zones with an `area` column; the zone enters the network as a one-hot
input (area_<ZONE> columns, see train/train_nn.py).

The network class (class Net(torch.nn.Module)) goes here, built from
meta["architecture"]; the saved file holds its weights only (model_utils).
Inputs are scaled with features.scale(x, scaler) BEFORE decide().

    decide(x, model, features, sensitivity=None) -> x with p_buy, action
"""
import pandas as pd

from models.actions import to_action


def decide(x: pd.DataFrame, model, features: list[str],
           sensitivity: float | None = None) -> pd.DataFrame:
    x = x.copy()
    ok = x[features].notna().all(axis=1)
    p = pd.Series(pd.NA, index=x.index, dtype="Float64")
    if ok.any():
        raise NotImplementedError("nn.decide: predict with the trained network")
    x["p_buy"] = p
    x["action"] = to_action(p, sensitivity)
    return x
