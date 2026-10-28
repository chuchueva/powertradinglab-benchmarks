"""
Models — what the live forecast calls (like algorithms/ in the old framework).

    features.py       the matrix: one row per (area, delivery hour, knownby_ts),
                      lags d1..d14, calendar; outcome() from the benchmarks;
                      area_columns(), fit_scaler(), scale(). The ONLY place
                      inputs are prepared, for live, backtest and training.
    actions.py        to_action(p, sensitivity): probability -> +1 / 0 / -1.
    model_utils.py    save_model / load_model, models/_model_storage/<version>/.
    trailing_bias.py  decide() — no training.
    logreg.py         decide() — logistic regression.
    nn.py             decide() — neural network.

    train/            flat scripts for the Python Console (like model-train/):
                      matrix_hourly, check_trailing_bias,
                      check_published_trailing_bias, train_/check_logreg,
                      train_/check_nn.

What an action earned: src/benchmarks_core.action_pnl — one function for the
live score, the report and the checks.
"""
