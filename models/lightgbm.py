"""LightGBM model with lag features (and optional exogenous calendar features) via mlforecast.

Supports three kinds of configuration overrides:
  - objective           : LightGBM loss function (e.g. 'tweedie' for M5)
  - exog_cols           : dynamic exogenous feature column names (e.g. SNAP days)
  - lags, date_features,
    lag_transforms      : full per-dataset feature set (expanded beyond defaults)

The pipeline in run_lightgbm.py composes a per-dataset feature config and
passes it here, giving each dataset a principled (but modest) feature set
tailored to its frequency and the assignment's "LightGBM with lag features"
requirement.
"""

from typing import Sequence

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from lightgbm import LGBMRegressor

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import LGBM_PARAMS
from models import ModelSpec


def _default_lags(season_length: int) -> list[int]:
    """Fallback lag set if a caller doesn't override: [1..season, 2*season]."""
    lags = list(range(1, season_length + 1))
    if 2 * season_length not in lags:
        lags.append(2 * season_length)
    return lags


def _default_lag_transforms(season_length: int) -> dict:
    """Fallback rolling stats at the seasonal window."""
    return {
        season_length: [
            RollingMean(window_size=season_length),
            RollingStd(window_size=season_length),
        ],
    }


def _default_date_features(freq: str) -> list[str]:
    """Fallback date features keyed by pandas frequency string."""
    if freq in ("D", "d"):
        return ["dayofweek", "month"]
    elif freq in ("h", "H"):
        return ["hour", "dayofweek"]
    elif freq in ("ME", "MS", "M"):
        return ["month"]
    return []


def build(
    freq: str,
    season_length: int,
    seed: int,
    lags: Sequence[int] | None = None,
    lag_transforms: dict | None = None,
    date_features: list[str] | None = None,
    exog_cols: list[str] | None = None,
    objective: str | None = None,
    tweedie_variance_power: float | None = None,
) -> ModelSpec:
    """Build an MLForecast object with LightGBM + configurable features.

    Parameters
    ----------
    freq, season_length, seed : standard fields.
    lags : Sequence[int] or None
        Explicit lag values. None → default of [1..season, 2*season].
    lag_transforms : dict or None
        Rolling / other lag transforms (mlforecast dict format).
        None → default of (RollingMean + RollingStd at the seasonal window).
    date_features : list[str] or None
        Calendar features (e.g. 'dayofweek', 'month', 'hour').
        None → default keyed by `freq`.
    exog_cols : list[str] or None
        Time-varying exogenous feature column names already attached to
        df_train by the loader (e.g. M5 SNAP days). Walk-forward handles
        the X_df plumbing at predict time.
    objective : str or None
        LightGBM objective override. Pass 'tweedie' for M5 intermittent data.
    tweedie_variance_power : float or None
        Only used when objective='tweedie'. Defaults to 1.5 (standard retail).

    Returns
    -------
    ModelSpec wrapping an unfitted MLForecast object.
    """
    # Resolve each feature family: explicit override > default
    effective_lags = list(lags) if lags is not None else _default_lags(season_length)
    effective_lag_transforms = (
        lag_transforms if lag_transforms is not None
        else _default_lag_transforms(season_length)
    )
    effective_date_features = (
        date_features if date_features is not None
        else _default_date_features(freq)
    )

    # Build LightGBM kwargs, conditionally adding Tweedie loss
    lgb_kwargs = dict(
        n_estimators=LGBM_PARAMS["n_estimators"],
        learning_rate=LGBM_PARAMS["learning_rate"],
        num_leaves=LGBM_PARAMS["num_leaves"],
        random_state=seed,
        verbosity=-1,
    )
    if objective is not None:
        lgb_kwargs["objective"] = objective
        if objective == "tweedie":
            lgb_kwargs["tweedie_variance_power"] = (
                tweedie_variance_power if tweedie_variance_power is not None else 1.5
            )

    model = LGBMRegressor(**lgb_kwargs)

    mlf = MLForecast(
        models={"LightGBM": model},
        freq=freq,
        lags=effective_lags,
        lag_transforms=effective_lag_transforms,
        date_features=effective_date_features if effective_date_features else None,
    )
    return ModelSpec(
        name="LightGBM",
        model_type="ml",
        forecaster=mlf,
        needs_seed=True,
        exog_cols=exog_cols,
    )
