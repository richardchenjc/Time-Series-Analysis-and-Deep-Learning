"""
Forecasting metrics: MAE and MASE.

Both are computed per series, then averaged across all series.
"""

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season_length: int,
) -> float:
    """Mean Absolute Scaled Error.

    Scales MAE by the in-sample naive seasonal forecast error.
    MASE < 1 means better than seasonal naive; MASE > 1 means worse.
    """
    # In-sample naive seasonal errors
    naive_errors = np.abs(y_train[season_length:] - y_train[:-season_length])
    scale = np.mean(naive_errors)
    if scale == 0 or np.isnan(scale):
        return np.inf
    return mae(y_true, y_pred) / scale


def compute_metrics_per_series(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_train: pd.DataFrame,
    season_length: int,
    model_col: str,
) -> pd.DataFrame:
    """Compute MAE and MASE per series for a given model.

    Parameters
    ----------
    df_true : DataFrame
        Test actuals with columns ['unique_id', 'ds', 'y'].
    df_pred : DataFrame
        Predictions with columns ['unique_id', 'ds', model_col].
    df_train : DataFrame
        Training data with columns ['unique_id', 'ds', 'y'].
    season_length : int
        Seasonal period for MASE calculation.
    model_col : str
        Column name containing predictions in df_pred.

    Returns
    -------
    DataFrame with columns ['unique_id', 'mae', 'mase'].
    """
    # Pre-build O(1) lookup dicts via groupby — avoids O(n_series × n_rows)
    # boolean masking inside the loop.
    true_by_uid = {uid: g["y"].values for uid, g in df_true.groupby("unique_id")}
    pred_by_uid = {uid: g[model_col].values for uid, g in df_pred.groupby("unique_id")}
    train_by_uid = {uid: g["y"].values for uid, g in df_train.groupby("unique_id")}

    results = []
    for uid in true_by_uid:
        if uid not in pred_by_uid:
            continue
        y_true_s = true_by_uid[uid]
        y_pred_s = pred_by_uid[uid]
        y_train_s = train_by_uid.get(uid, np.array([]))

        # Handle length mismatches (truncate to shorter)
        min_len = min(len(y_true_s), len(y_pred_s))
        if min_len == 0:
            continue
        y_true_s = y_true_s[:min_len]
        y_pred_s = y_pred_s[:min_len]

        mae_val = mae(y_true_s, y_pred_s)
        mase_val = mase(y_true_s, y_pred_s, y_train_s, season_length)

        results.append({
            "unique_id": uid,
            "mae": mae_val,
            "mase": mase_val,
        })

    return pd.DataFrame(results)
