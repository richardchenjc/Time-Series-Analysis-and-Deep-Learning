"""
Sampling sanity check: does stratified sampling bias cross-model comparisons?

Runs LightGBM on M4 twice:
  1. With stratified sampling (main-run default)
  2. With uniform random sampling

If both produce similar MAE / MASE, then stratified sampling is not
biasing the comparison and our main-run methodology is defensible.
If they diverge substantially, we need to rethink the sampling strategy
before committing to the main run.

Why LightGBM / M4:
  - LightGBM is the cheapest model (~5 min total across all 3 datasets),
    so running it twice costs almost nothing.
  - M4 is the only dataset where stratification shifts (EDA showed M4
    Fs 0.21 full → 0.31 random → 0.25 stratified). So any sampling
    method effect is most likely to show up here.
  - The sanity check focuses on the one dataset where the question
    actually matters, rather than re-running everywhere.

Output:
  - results/LightGBM_M4.csv and the normal parquet prediction dump
    (the stratified run — this is what the main run would have produced)
  - results/sampling_sanity/LightGBM_M4_random.csv
  - Console-printed comparison table at the end

Usage:
  python pipelines/run_sampling_sanity_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from mlforecast.lag_transforms import RollingMean, RollingStd

from config import M4_CONFIG, SEEDS, RESULTS_DIR
from data_prep.m4_prep import load_m4_monthly
from models.lightgbm import build as build_lgbm
from evaluation.walk_forward import run_walk_forward


# Match the main-run LightGBM feature config for M4 so the comparison is apples-to-apples
_M4_FEATURE_CONFIG = {
    "lags": list(range(1, 13)) + [24, 36],
    "lag_transforms": {
        3:  [RollingMean(window_size=3),  RollingStd(window_size=3)],
        12: [RollingMean(window_size=12), RollingStd(window_size=12)],
    },
    "date_features": ["month"],
}


def _build_lgbm_factory(cfg):
    """Factory that produces a LightGBM ModelSpec matching main-run M4 config."""
    def _build(seed, max_steps=None):
        return build_lgbm(
            freq=cfg["freq"],
            season_length=cfg["season_length"],
            seed=seed,
            lags=_M4_FEATURE_CONFIG["lags"],
            lag_transforms=_M4_FEATURE_CONFIG["lag_transforms"],
            date_features=_M4_FEATURE_CONFIG["date_features"],
        )
    return _build


def _run_one_pass(label: str, stratified: bool, results_subdir: Path) -> pd.DataFrame:
    """Run LightGBM on M4 once with the given sampling strategy."""
    print(f"\n{'#'*60}")
    print(f"# Sampling sanity check — {label}")
    print(f"# stratified={stratified}")
    print(f"{'#'*60}\n")

    cfg = M4_CONFIG
    df_train, df_test = load_m4_monthly(
        n_series=cfg["n_series_sample"],
        random_state=42,
        stratified=stratified,
    )
    df_full = pd.concat([df_train, df_test], ignore_index=True)

    build_fn = _build_lgbm_factory(cfg)

    results = run_walk_forward(
        df_full=df_full,
        dataset_name="M4",
        horizon=cfg["horizon"],
        input_size=cfg["input_size"],
        freq=cfg["freq"],
        season_length=cfg["season_length"],
        n_windows=cfg["walk_forward_windows"],
        seeds=SEEDS,
        results_dir=results_subdir,
        build_model_fn=build_fn,
        needs_seed=True,
        max_steps=None,
        max_train_size=cfg.get("max_train_size"),
        save_predictions=False,  # no need to dump parquet for sanity check
    )
    return results


def main():
    sanity_dir = RESULTS_DIR / "sampling_sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: stratified
    strat_dir = sanity_dir / "stratified"
    df_strat = _run_one_pass("PASS 1: STRATIFIED", stratified=True, results_subdir=strat_dir)

    # Pass 2: random
    random_dir = sanity_dir / "random"
    df_rand = _run_one_pass("PASS 2: RANDOM", stratified=False, results_subdir=random_dir)

    # ── Comparison ───────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print("# Sampling sanity check — comparison")
    print(f"{'#'*60}\n")

    def _agg(df, label):
        return {
            "sampling": label,
            "mae_mean":        df["mae_mean"].mean(),
            "mae_std":         df["mae_mean"].std(),
            "mase_mean":       df["mase_mean"].mean(),
            "mase_std":        df["mase_mean"].std(),
            "mase_median":     df["mase_median"].mean(),
            "mase_median_std": df["mase_median"].std(),
            "n_runs":          int(len(df)),
        }

    rows = [_agg(df_strat, "stratified"), _agg(df_rand, "random")]
    comparison = pd.DataFrame(rows)
    print(comparison.to_string(index=False))

    out_path = sanity_dir / "comparison.csv"
    comparison.to_csv(out_path, index=False)
    print(f"\nSaved comparison to {out_path}")

    # Quick verdict: is the gap larger than seed-to-seed variance?
    mae_gap = abs(rows[0]["mae_mean"] - rows[1]["mae_mean"])
    noise_floor = max(rows[0]["mae_std"], rows[1]["mae_std"])
    if noise_floor == 0 or pd.isna(noise_floor):
        noise_floor = 0.01 * rows[0]["mae_mean"]  # fallback 1%
    ratio = mae_gap / noise_floor
    print(f"\nMAE gap between strategies: {mae_gap:.4f}")
    print(f"Seed-to-seed noise floor:   {noise_floor:.4f}")
    print(f"Ratio (gap / noise):        {ratio:.2f}")
    if ratio < 2:
        print("VERDICT: gap is within noise — stratification does not bias "
              "cross-model comparisons on M4. Main run is methodologically safe.")
    else:
        print("VERDICT: gap exceeds noise — stratification may bias comparisons. "
              "Review before committing to main run.")


if __name__ == "__main__":
    main()
