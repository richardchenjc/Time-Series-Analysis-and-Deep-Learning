"""
LightGBM Traffic bare-bones check.

Context
-------
The insurance checks earlier verified that LightGBM on M5 still loses to
DL models without its calendar exogenous features. A similar concern
applies to Traffic: LightGBM is statistically significantly better than
every DL model except DLinear on Traffic (Nemenyi CD test). Part of that
performance may come from the date features (hour, dayofweek) that the
DL models don't have.

Does the DL-beats-LightGBM-on-M5 finding mirror here as
LightGBM-wins-Traffic-only-because-of-date-features?

This script runs LightGBM on Traffic with NO date features and compares
to the main-run LightGBM Traffic result (MAE 0.0101). If the bare-bones
version is meaningfully worse (say, > 5% MAE increase), the Traffic
ranking may not be as clean as it looks.

If the bare-bones version is about the same, LightGBM's Traffic performance
is robust to feature engineering — it's the lag structure that does the
work, not the date features.

Output
------
  results/insurance_checks/traffic_lightgbm_no_date/   (full walk_forward)
  Plus a verdict printed to stdout.

Usage:
    python analysis/traffic_lightgbm_bare_check.py
"""

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import pandas as pd
from mlforecast.lag_transforms import RollingMean, RollingStd

from config import TRAFFIC_CONFIG, SEEDS, RESULTS_DIR
from data_prep.traffic_prep import load_traffic
from evaluation.walk_forward import run_walk_forward
from models.lightgbm import build as build_lightgbm


OUT_DIR = RESULTS_DIR / "insurance_checks" / "traffic_lightgbm_no_date"


def _factory(_cfg):
    # Bare-bones config: only seasonal lags + rolling stats.
    # No date_features, no exog_cols.
    bare_lags = [1, 2, 3, 6, 12, 24, 48, 168]  # hourly lags up to 1 week
    bare_lag_transforms = {
        24:  [RollingMean(window_size=24),  RollingStd(window_size=24)],
        168: [RollingMean(window_size=168), RollingStd(window_size=168)],
    }

    def _build(seed, max_steps=None):
        return build_lightgbm(
            freq=_cfg["freq"],
            season_length=_cfg["season_length"],
            seed=seed,
            lags=bare_lags,
            lag_transforms=bare_lag_transforms,
            date_features=None,   # ← no hour/dayofweek features
            exog_cols=None,
            objective=None,
        )
    return _build


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#'*60}")
    print(f"# LightGBM Traffic bare-bones check")
    print(f"# (lag features only, no hour/dayofweek date features)")
    print(f"{'#'*60}")
    t0 = time.time()

    cfg = TRAFFIC_CONFIG
    df_train, df_test = load_traffic(n_series=cfg["n_series_sample"])
    df_full = pd.concat([df_train, df_test], ignore_index=True)

    try:
        results = run_walk_forward(
            df_full=df_full,
            dataset_name="Traffic",
            horizon=cfg["horizon"],
            input_size=cfg["input_size"],
            freq=cfg["freq"],
            season_length=cfg["season_length"],
            n_windows=cfg["walk_forward_windows"],
            seeds=SEEDS,
            results_dir=OUT_DIR,
            build_model_fn=_factory(cfg),
            needs_seed=True,
            max_steps=None,
            max_train_size=cfg.get("max_train_size"),
            save_predictions=False,
        )
    except Exception as e:
        print(f"\n  ✗ Experiment FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    elapsed = time.time() - t0
    print(f"\n  Experiment complete in {elapsed:.1f}s")

    if results.empty:
        print("  ✗ No results")
        return

    bare_mae = results["mae_mean"].mean()
    main_summary = pd.read_csv(RESULTS_DIR / "summary_table.csv")
    full_row = main_summary[
        (main_summary["dataset"] == "Traffic") &
        (main_summary["model"] == "LightGBM")
    ]
    full_mae = float(full_row["mae_mean"].iloc[0]) if not full_row.empty else None

    print(f"\n  ── RESULTS ──")
    print(f"  LightGBM Traffic (lag features + date features):  MAE = {full_mae:.4f}")
    print(f"  LightGBM Traffic (lag features only, no dates):   MAE = {bare_mae:.4f}")
    if full_mae:
        delta_pct = 100 * (bare_mae - full_mae) / full_mae
        print(f"  Delta: {delta_pct:+.1f}% — date features {'helped' if delta_pct > 0 else 'hurt'}")

        # DLinear is 0.0103 — does LightGBM still beat DLinear without dates?
        dlinear_row = main_summary[
            (main_summary["dataset"] == "Traffic") &
            (main_summary["model"] == "DLinear")
        ]
        if not dlinear_row.empty:
            dlinear_mae = float(dlinear_row["mae_mean"].iloc[0])
            print(f"\n  Reference: DLinear Traffic MAE = {dlinear_mae:.4f}")
            if bare_mae < dlinear_mae:
                print(f"  ✓ Bare-bones LightGBM ({bare_mae:.4f}) STILL beats DLinear ({dlinear_mae:.4f}).")
                print(f"    LightGBM's Traffic ranking is robust to date-feature removal.")
            else:
                print(f"  ✗ Bare-bones LightGBM ({bare_mae:.4f}) LOSES to DLinear ({dlinear_mae:.4f}).")
                print(f"    The LightGBM > DLinear ranking on Traffic depends on date features.")
                print(f"    This is worth flagging in the report's fairness discussion.")


if __name__ == "__main__":
    main()
