"""
Save / load a trained model. One folder per version, the same for every kind:

    models/_model_storage/<name>/
        meta.json      features (order matters), areas, n_lags, period, split,
                       sensitivity, architecture (for a network), versions
        scaler.json    {feature: {"mean": .., "std": ..}} — or absent
        coefficients.json
                       logistic regression: {"intercept": b0,
                       "coefficients": {feature: b, ...}} in training order.
                       Readable by hand, publishable, no library needed to
                       use it (models/logreg.py computes the output itself).
        model.joblib   a scikit-learn object, if one is ever saved (not used
                       for the logistic regression since 2026-11-21)
        model.pt       torch network: WEIGHTS only (state_dict)

<name> carries the version ("logreg_v1", "nn_portfolio_v1"). A version is
never overwritten: a new model is a new folder.

Why this way:
  - the scaler is two numbers per feature; as JSON it can be read by hand and
    does not depend on the scikit-learn version;
  - joblib is pickle underneath, so a model may not load under another
    scikit-learn version — the version is written into meta.json;
  - torch keeps weights, not the network: the class lives in models/nn.py and
    is rebuilt from meta["architecture"] before load_state_dict. Changing the
    class means a new version.
"""
import json
import os
from typing import Any

from src.workflow import ROOT

STORAGE = os.path.join(ROOT, "models", "_model_storage")


def _versions() -> dict:
    out = {}
    for lib in ("sklearn", "torch", "numpy", "pandas"):
        try:
            out[lib] = __import__(lib).__version__
        except ImportError:
            pass
    return out


def save_model(name: str, features: list[str], meta: dict,
               model=None, state_dict=None, scaler: dict | None = None,
               coefficients: dict | None = None) -> str:
    """
    coefficients -> coefficients.json (logistic regression);
    model -> model.joblib (scikit-learn); state_dict -> model.pt (torch).

    coefficients = {"intercept": float, "coefficients": {feature: float}};
    the features must be exactly `features`, nothing missing, nothing extra.
    """
    if coefficients is not None:
        names = list(coefficients["coefficients"].keys())
        if names != list(features):
            raise ValueError(f"coefficients {names} do not match features {list(features)}")
    folder = os.path.join(STORAGE, name)
    os.makedirs(folder, exist_ok=False)          # never overwrite a version
    if coefficients is not None:
        with open(os.path.join(folder, "coefficients.json"), "w") as f:
            json.dump(coefficients, f, indent=2)
    if model is not None:
        import joblib
        joblib.dump(model, os.path.join(folder, "model.joblib"))
    if state_dict is not None:
        import torch
        torch.save(state_dict, os.path.join(folder, "model.pt"))
    if scaler is not None:
        with open(os.path.join(folder, "scaler.json"), "w") as f:
            json.dump(scaler, f, indent=2)
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump({"name": name, "features": features, **meta,
                   "versions": _versions()}, f, indent=2, default=str)
    files = sorted(os.listdir(folder))
    print(f"model {name} saved: {folder} ({', '.join(files)}), "
          f"{len(features)} features")
    return folder


def load_model(name: str) -> tuple[dict, dict | None, Any]:
    """
    (meta, scaler, model). `model` is the coefficients dict for
    coefficients.json (logistic regression), the scikit-learn object for
    model.joblib, or the state_dict for model.pt — build the network from
    meta["architecture"] and call load_state_dict on it (models/nn.py).
    """
    folder = os.path.join(STORAGE, name)
    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)
    scaler = None
    if os.path.exists(os.path.join(folder, "scaler.json")):
        with open(os.path.join(folder, "scaler.json")) as f:
            scaler = json.load(f)
    model = None
    if os.path.exists(os.path.join(folder, "coefficients.json")):
        with open(os.path.join(folder, "coefficients.json")) as f:
            model = json.load(f)
    elif os.path.exists(os.path.join(folder, "model.joblib")):
        import joblib
        model = joblib.load(os.path.join(folder, "model.joblib"))
    elif os.path.exists(os.path.join(folder, "model.pt")):
        import torch
        model = torch.load(os.path.join(folder, "model.pt"), weights_only=True)
    return meta, scaler, model
