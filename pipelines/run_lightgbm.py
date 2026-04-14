"""
Pipeline: LightGBM across all 3 datasets (M4, M5, Traffic).

Per-dataset feature engineering
-------------------------------
LightGBM uses dataset-appropriate feature sets. The assignment requires
"LightGBM with lag features"; we expand modestly on that minimum to
produce a principled classical-ML baseline that reflects reasonable
practitioner effort without competition-level tuning.

  M4 (monthly)
    - Lags: [1..12, 24, 36]  (seasonal period + 2× + 3×)
    - Rolling stats at windows 3 and 12 (quarterly + annual)
    - Date features: month

  M5 (daily, intermittent retail)
    - Lags: [1..7, 14, 28]  (week + 2×, 4× = assignment horizon)
    - Rolling stats at windows 7 and 28
    - Date features: dayofweek, month
    - Exog: SNAP days (3 states) + event flags (5 cols)
    - Objective: Tweedie (variance_power=1.5) for intermittent count data

  Traffic (hourly, strong daily+weekly periodicity)
    - Lags: [1..24, 48, 168]  (day + 2 days + week)
    - Rolling stats at windows 24 and 168 (daily + weekly)
    - Date features: hour, dayofweek
    - (168-hour-of-week would be cleaner but requires a custom transform;
      hour + dayofweek together span the same state space for LGBM.)

All feature choices are frequency-justified (no blind grid search) and
symmetric across models — every dataset gets the same quality of effort.

Usage:
    python pipelines/run_lightgbm.py                # Full run
    python pipelines/run_lightgbm.py --smoke-test   # Quick validation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlforecast.lag_transforms import RollingMean, RollingStd

from models.lightgbm import build
from pipelines.run_model import run_pipeline
from data_prep.m5_prep import M5_EXOG_COLS


# Per-dataset feature configuration. Each entry is a kwargs dict passed
# straight into models.lightgbm.build(). Keeping this as data rather than
# code makes it easy to scan and modify for the report description.
_DATASET_FEATURE_CONFIGS = {
    "M4": {
        # Monthly: capture annual seasonality + a slow-moving trend signal.
        "lags": list(range(1, 13)) + [24, 36],
        "lag_transforms": {
            3:  [RollingMean(window_size=3),  RollingStd(window_size=3)],
            12: [RollingMean(window_size=12), RollingStd(window_size=12)],
        },
        "date_features": ["month"],
        "exog_cols": None,
        "objective": None,
    },
    "M5": {
        # Daily retail: weekly rhythm + SNAP/event calendar + Tweedie loss.
        "lags": list(range(1, 8)) + [14, 28],
        "lag_transforms": {
            7:  [RollingMean(window_size=7),  RollingStd(window_size=7)],
            28: [RollingMean(window_size=28), RollingStd(window_size=28)],
        },
        "date_features": ["dayofweek", "month"],
        "exog_cols": M5_EXOG_COLS,
        "objective": "tweedie",
    },
    "Traffic": {
        # Hourly: daily + weekly periodicity, no calendar features available.
        "lags": list(range(1, 25)) + [48, 168],
        "lag_transforms": {
            24:  [RollingMean(window_size=24),  RollingStd(window_size=24)],
            168: [RollingMean(window_size=168), RollingStd(window_size=168)],
        },
        "date_features": ["hour", "dayofweek"],
        "exog_cols": None,
        "objective": None,
    },
}


def _factory(cfg):
    """Return a factory closure for the given dataset config."""
    feat = _DATASET_FEATURE_CONFIGS[cfg["name"]]

    def _build(seed, max_steps=None):
        return build(
            freq=cfg["freq"],
            season_length=cfg["season_length"],
            seed=seed,
            lags=feat["lags"],
            lag_transforms=feat["lag_transforms"],
            date_features=feat["date_features"],
            exog_cols=feat["exog_cols"],
            objective=feat["objective"],
        )
    return _build


def main(smoke_test: bool = False):
    run_pipeline(
        model_name="LightGBM",
        build_fn_for_cfg=_factory,
        needs_seed=True,
        smoke_test=smoke_test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LightGBM on all datasets.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run with minimal settings for quick validation")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
