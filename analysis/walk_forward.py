"""
Walk-forward (sliding-window) evaluation driver — single-model version.

Implements the core evaluation loop for one model at a time:
  For each sliding window:
    For each seed (or once if no seed):
      Build model → Fit → Predict → Compute metrics → Record timing

This generic engine works with any model type (stats / ml / neural) via
the ModelSpec returned by each model's build() function.
"""

import pandas as pd
import numpy as np
import traceback
from pathlib import Path

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from evaluation.metrics import compute_metrics_per_series
from evaluation.timing import Timer


def _sliding_window_splits(
    df: pd.DataFrame,
    horizon: int,
    input_size: int,
    n_windows: int,
    freq: str,
    max_train_size: int | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate sliding-window train/test splits.

    Each window holds out `horizon` time steps for testing. Windows move
    forward so that test sets do not overlap.

    Parameters
    ----------
    df : DataFrame
        Full dataset in long format ['unique_id', 'ds', 'y'].
    horizon : int
        Number of time steps to forecast.
    input_size : int
        Minimum number of time steps needed for model input.
    n_windows : int
        Number of sliding windows.
    freq : str
        Pandas frequency string (e.g. 'ME', 'D', 'h'). Used to compute
        exact date offsets so that window boundaries land precisely on
        calendar-aligned timestamps (critical for M4 monthly whose months
        vary from 28–31 days — a median-diff estimate drifts by ~0.5 day
        per step, misaligning windows by several days over 18-step horizons).
    max_train_size : int or None
        If set, caps training history to this many steps per series.

    Returns
    -------
    List of (df_train, df_test) tuples.
    """
    splits = []

    # Determine the global max date across all series
    max_date = df["ds"].max()

    # Use exact calendar-aware offset instead of a median diff estimate.
    # pd.tseries.frequencies.to_offset("ME") → MonthEnd (exact calendar month),
    # to_offset("D") → Day, to_offset("h") → Hour — all do proper arithmetic.
    offset = pd.tseries.frequencies.to_offset(freq)

    for w in range(n_windows):
        # Test window: ends at max_date - w*horizon steps back
        test_end    = max_date - (w * horizon) * offset
        test_start  = test_end - (horizon - 1) * offset

        # Train: everything strictly before the test window
        train_cutoff = test_start - offset

        if max_train_size is not None:
            train_start = train_cutoff - max_train_size * offset
            df_train_w = df[(df["ds"] > train_start) & (df["ds"] <= train_cutoff)].copy()
        else:
            df_train_w = df[df["ds"] <= train_cutoff].copy()

        df_test_w = df[(df["ds"] >= test_start) & (df["ds"] <= test_end)].copy()

        # Only keep series that have enough history. We log the count of
        # filtered-out series so the report can quote how many M4 series
        # were dropped per window — required for honest reporting.
        n_before = df_train_w["unique_id"].nunique()
        series_lens = df_train_w.groupby("unique_id").size()
        valid_series = series_lens[series_lens >= input_size].index
        df_train_w = df_train_w[df_train_w["unique_id"].isin(valid_series)]
        df_test_w = df_test_w[df_test_w["unique_id"].isin(valid_series)]
        n_after = df_train_w["unique_id"].nunique()
        n_dropped_short = n_before - n_after

        if len(df_test_w) == 0 or len(df_train_w) == 0:
            print(f"  [Walk-forward] Window {w+1}: skipped (insufficient data)")
            continue

        print(f"  [Walk-forward] Window {w+1}: {n_after} series "
              f"(filtered {n_dropped_short} with < {input_size} obs), "
              f"train up to {train_cutoff}, test {test_start} → {test_end}")
        splits.append((df_train_w, df_test_w))

    # Reverse so earliest window comes first
    splits.reverse()
    return splits


def run_walk_forward(
    df_full: pd.DataFrame,
    dataset_name: str,
    horizon: int,
    input_size: int,
    freq: str,
    season_length: int,
    n_windows: int,
    seeds: list[int],
    results_dir: Path,
    build_model_fn,
    needs_seed: bool,
    max_steps: int | None = None,
    max_train_size: int | None = None,
) -> pd.DataFrame:
    """Run walk-forward evaluation for a single model on one dataset.

    Parameters
    ----------
    df_full : DataFrame
        Complete dataset in long format ['unique_id', 'ds', 'y'].
    dataset_name : str
        Name for logging/saving (e.g. 'M4', 'M5', 'Traffic').
    horizon, input_size : int
        Forecast horizon and lookback window.
    freq : str
        Pandas frequency string.
    season_length : int
        Seasonal period.
    n_windows : int
        Number of sliding windows.
    seeds : list[int]
        Random seeds to iterate over (used only when needs_seed=True).
    results_dir : Path
        Directory to save the result CSV.
    build_model_fn : callable
        Signature: (seed, max_steps) -> ModelSpec.
        Called once per (window, seed) combination to get a fresh model.
    needs_seed : bool
        If False, the model is run once per window (seed=None).
        If True, the model is run once per (window, seed) pair.
    max_steps : int or None
        Override training steps (for smoke tests).
    max_train_size : int or None
        Cap training history length.

    Returns
    -------
    DataFrame with per-seed, per-window results for this model.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Generate sliding-window splits
    print(f"\n{'='*60}")
    print(f"[{dataset_name}] Generating {n_windows} sliding-window splits...")
    print(f"{'='*60}")
    splits = _sliding_window_splits(df_full, horizon, input_size, n_windows, freq, max_train_size)

    if not splits:
        print(f"[{dataset_name}] WARNING: No valid splits generated!")
        return pd.DataFrame()

    for w_idx, (df_train, df_test) in enumerate(splits):
        window_id = w_idx + 1
        print(f"\n{'─'*50}")
        print(f"[{dataset_name}] Window {window_id}/{len(splits)}")
        print(f"{'─'*50}")

        # Iterate over seeds (or run once if deterministic)
        seed_list = seeds if needs_seed else [None]

        for seed in seed_list:
            seed_label = f"seed={seed}" if seed is not None else "no seed"
            print(f"\n  ▸ [{seed_label}]")

            try:
                spec = build_model_fn(seed=seed, max_steps=max_steps)

                with Timer() as t:
                    if spec.model_type == "neural":
                        spec.forecaster.fit(df=df_train, val_size=horizon)
                        preds = spec.forecaster.predict().reset_index()
                    else:  # "stats" or "ml"
                        spec.forecaster.fit(df_train)
                        preds = spec.forecaster.predict(h=horizon).reset_index()

                # Drop spurious 'index' column occasionally added by reset_index()
                if "index" in preds.columns:
                    preds = preds.drop(columns=["index"])

                # The model prediction column matches spec.name
                model_col = spec.name
                if model_col not in preds.columns:
                    # Fallback: use first non-id column
                    id_cols = {"unique_id", "ds"}
                    model_col = next(c for c in preds.columns if c not in id_cols)

                metrics_df = compute_metrics_per_series(
                    df_test, preds, df_train, season_length, model_col
                )
                avg_mae = metrics_df["mae"].mean()

                # MASE: distinguish well-defined per-series values from
                # undefined ones. MASE diverges when the in-sample naive
                # seasonal denominator mean(|y_t − y_{t−s}|) is ~0, which
                # happens often on M5 (median 73 % zeros per series — see
                # results/eda/full/M5/). We track:
                #   • mase_mean      — required by spec, but unstable on M5
                #   • mase_median    — robust alternative; gap to mean is
                #                      itself an intermittency diagnostic
                #   • mase_n_dropped — series with undefined MASE in this run
                #   • mase_n_total   — total per-series MASE values for context
                # The previous implementation silently filtered inf via
                # .replace().mean(), hiding the M5 problem.
                mase_series = metrics_df["mase"].replace([np.inf, -np.inf], np.nan)
                n_total = int(len(mase_series))
                n_dropped = int(mase_series.isna().sum())
                mase_valid = mase_series.dropna()
                avg_mase = float(mase_valid.mean()) if len(mase_valid) else np.nan
                median_mase = float(mase_valid.median()) if len(mase_valid) else np.nan

                all_results.append({
                    "dataset": dataset_name,
                    "model": spec.name,
                    "seed": seed if seed is not None else "N/A",
                    "window": window_id,
                    "mae_mean": avg_mae,
                    "mase_mean": avg_mase,
                    "mase_median": median_mase,
                    "mase_n_total": n_total,
                    "mase_n_dropped": n_dropped,
                    "train_time_sec": t.elapsed,
                    "peak_gpu_mb": t.peak_gpu_mb,
                })
                print(f"    {spec.name}: MAE={avg_mae:.4f}, "
                      f"MASE={avg_mase:.4f} (med={median_mase:.4f}, "
                      f"dropped={n_dropped}/{n_total}), Time={t.elapsed:.1f}s")

            except Exception as e:
                print(f"    ✗ Model failed ({seed_label}): {e}")
                traceback.print_exc()

    # ── Save results ──
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        # Derive model name from first result row for filename
        model_name = results_df["model"].iloc[0]
        out_path = results_dir / f"{model_name}_{dataset_name}.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\n[{dataset_name}] Results saved to {out_path}")
        print(f"[{dataset_name}] Total rows: {len(results_df)}")

    return results_df
