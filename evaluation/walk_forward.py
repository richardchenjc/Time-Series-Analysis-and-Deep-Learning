"""
Walk-forward (sliding-window) evaluation driver — single-model version.

Implements the core evaluation loop for one model at a time:
  For each sliding window:
    For each seed (or once if no seed):
      Build model → Fit → Predict → Compute metrics → Record timing

This generic engine works with any model type (stats / ml / neural) via
the ModelSpec returned by each model's build() function.

Exogenous feature handling
--------------------------
Loaders may attach extra columns to df_train / df_test (e.g. M5's SNAP
day flags). The convention is:

  - Models that DECLARE `exog_cols` on their ModelSpec see those columns
    in df_train at fit time and receive a future X_df at predict time.
    Only ml-type models support this currently.
  - Models that DON'T declare exog_cols have *all* extra columns stripped
    from df_train and df_test before fit, so they never accidentally
    receive features they weren't designed for.

Raw prediction saving
---------------------
Alongside the aggregated metrics CSV, we also save one Parquet file per
(model, dataset) with the raw per-step predictions:

  results/predictions/<Model>_<Dataset>.parquet
    columns: dataset, model, seed, window, unique_id, ds, horizon_step,
             y_true, y_pred

This enables post-hoc analysis that doesn't require retraining:
  - Per-horizon MAE decomposition (h=1, h=2, ..., h=H)
  - Metrics stratified by series characteristic (zero fraction, length,
    category, volume quintile)
  - Per-series residual distributions / outlier inspection
  - Significance testing via bootstrap over the prediction pool

Parquet is chosen over CSV for ~10× smaller files on this numeric data
and much faster read-back. Requires `pyarrow` (already a Pandas optional
dep on modern Pandas but may need explicit install on some systems).
"""

import pandas as pd
import numpy as np
import traceback
from pathlib import Path

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

# Force UTF-8 output on Windows consoles (cp1252 can't encode → ✗ ✓ etc.).
# Must run before any print() in this module or downstream modules.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # non-Windows platforms or non-reconfigurable streams

from evaluation.metrics import compute_metrics_per_series
from evaluation.timing import Timer

# Columns every model expects — anything else is treated as a candidate exog
_BASE_COLS = {"unique_id", "ds", "y"}


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
        Full dataset in long format ['unique_id', 'ds', 'y', ...].
        Extra columns (exog) are preserved across splits.
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

        # Only keep series that have enough history. We must filter on
        # input_size + horizon, not just input_size: neuralforecast's
        # .fit(val_size=horizon) carves out `horizon` timestamps from
        # the END of each training series for internal validation, so
        # a series needs input_size points for the model AND horizon
        # extra points for val_size, otherwise PatchTST/N-BEATS/etc
        # raise ValueError("shortest series has only X timestamps").
        # Filtering up-front in walk_forward (rather than relying on
        # neuralforecast's runtime check) means stats/ml models also
        # see the same series set, keeping comparisons apples-to-apples.
        min_required = input_size + horizon
        n_before = df_train_w["unique_id"].nunique()
        series_lens = df_train_w.groupby("unique_id").size()
        valid_series = series_lens[series_lens >= min_required].index
        df_train_w = df_train_w[df_train_w["unique_id"].isin(valid_series)]
        df_test_w = df_test_w[df_test_w["unique_id"].isin(valid_series)]
        n_after = df_train_w["unique_id"].nunique()
        n_dropped_short = n_before - n_after

        if len(df_test_w) == 0 or len(df_train_w) == 0:
            print(f"  [Walk-forward] Window {w+1}: skipped (insufficient data)")
            continue

        print(f"  [Walk-forward] Window {w+1}: {n_after} series "
              f"(filtered {n_dropped_short} with < {min_required} obs "
              f"= input_size {input_size} + val horizon {horizon}), "
              f"train up to {train_cutoff}, test {test_start} → {test_end}")
        
        splits.append((df_train_w, df_test_w))

    # Reverse so earliest window comes first
    splits.reverse()
    return splits


def _prepare_inputs(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    exog_cols: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Strip / preserve exogenous columns based on the model's declaration.

    Returns
    -------
    df_train_in : DataFrame
        Training frame with appropriate columns for this model.
    df_test_in : DataFrame
        Test frame with the same columns as df_train_in (used for
        actuals lookup; not what's passed to predict).
    X_df : DataFrame or None
        Future exogenous regressors with columns [unique_id, ds, *exog_cols],
        ready to pass to mlforecast.MLForecast.predict(X_df=...).
        None if the model doesn't use exogenous features.
    """
    extra = [c for c in df_train.columns if c not in _BASE_COLS]

    if exog_cols:
        # Verify all declared columns are present — fail loudly if not
        missing = [c for c in exog_cols if c not in df_train.columns]
        if missing:
            raise ValueError(
                f"Model declared exog_cols {exog_cols} but df_train is missing {missing}"
            )
        # Keep only base + declared exog (drop any other extras a loader may
        # have attached). Model sees exactly what it asked for.
        keep = ["unique_id", "ds", "y"] + list(exog_cols)
        df_train_in = df_train[keep].copy()
        df_test_in = df_test[keep].copy()
        X_df = df_test_in[["unique_id", "ds"] + list(exog_cols)].copy()
        return df_train_in, df_test_in, X_df

    # No exog declared — strip any attached extras so models that don't
    # know about them never see them.
    if extra:
        df_train_in = df_train.drop(columns=extra)
        df_test_in = df_test.drop(columns=extra)
        return df_train_in, df_test_in, None

    return df_train, df_test, None


def _build_prediction_records(
    df_test_in: pd.DataFrame,
    preds: pd.DataFrame,
    model_col: str,
    dataset_name: str,
    model_name: str,
    seed,
    window_id: int,
) -> pd.DataFrame:
    """Join per-step predictions with ground truth into a long-format
    DataFrame suitable for Parquet dump.

    Output columns:
      dataset, model, seed, window, unique_id, ds,
      horizon_step (1-indexed), y_true, y_pred

    horizon_step is computed per (unique_id) by counting the order of
    timestamps within the test window. This is robust to ragged
    forecasting (different series having different test dates).
    """
    # Inner-join test actuals with model predictions on (unique_id, ds)
    merged = df_test_in[["unique_id", "ds", "y"]].merge(
        preds[["unique_id", "ds", model_col]],
        on=["unique_id", "ds"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    # Compute horizon_step = 1..H per series, by sorted ds order within uid
    merged = merged.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    merged["horizon_step"] = merged.groupby("unique_id").cumcount() + 1

    merged = merged.rename(columns={"y": "y_true", model_col: "y_pred"})
    merged["dataset"] = dataset_name
    merged["model"] = model_name
    merged["seed"] = str(seed) if seed is not None else "N/A"
    merged["window"] = window_id

    # Reorder columns for readability
    return merged[[
        "dataset", "model", "seed", "window",
        "unique_id", "ds", "horizon_step", "y_true", "y_pred",
    ]]


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
    save_predictions: bool = True,
) -> pd.DataFrame:
    """Run walk-forward evaluation for a single model on one dataset.

    Parameters
    ----------
    df_full : DataFrame
        Complete dataset in long format ['unique_id', 'ds', 'y', ...].
        May contain extra (exog) columns; handled per-model.
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
    save_predictions : bool
        If True (default), dump raw per-step predictions to
        results/predictions/<Model>_<Dataset>.parquet alongside the
        aggregated metrics CSV.

    Returns
    -------
    DataFrame with per-seed, per-window results for this model.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = results_dir / "predictions"
    if save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_predictions = []  # accumulates per-run prediction frames

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
                exog_cols = getattr(spec, "exog_cols", None)

                # Prepare per-model inputs (strip or preserve exog)
                df_train_in, df_test_in, X_df = _prepare_inputs(
                    df_train, df_test, exog_cols
                )
                if exog_cols:
                    print(f"    using exog: {exog_cols}")

                with Timer() as t:
                    if spec.model_type == "neural":
                        spec.forecaster.fit(df=df_train_in, val_size=horizon)
                        preds = spec.forecaster.predict().reset_index()
                    elif spec.model_type == "ml":
                        # mlforecast treats non-target columns as static by
                        # default. When exog is declared, tell it they're
                        # dynamic via static_features=[].
                        if X_df is not None:
                            spec.forecaster.fit(df_train_in, static_features=[])
                            preds = spec.forecaster.predict(
                                h=horizon, X_df=X_df
                            ).reset_index()
                        else:
                            spec.forecaster.fit(df_train_in)
                            preds = spec.forecaster.predict(h=horizon).reset_index()
                    else:  # "stats"
                        spec.forecaster.fit(df_train_in)
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
                    df_test_in, preds, df_train_in, season_length, model_col
                )
                avg_mae = metrics_df["mae"].mean()

                # MASE: distinguish well-defined per-series values from
                # undefined ones. MASE diverges when the in-sample naive
                # seasonal denominator mean(|y_t − y_{t−s}|) is ~0, which
                # in our datasets turns out never to happen (M5 series have
                # ~73% zeros but the non-zero 27% keeps the denominator
                # finite). We still track:
                #   • mase_mean      — required by spec
                #   • mase_median    — robust alternative
                #   • mase_n_dropped — series with undefined MASE in this run
                #   • mase_n_total   — total per-series MASE values for context
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

                # ── Raw prediction capture ──────────────────────────────
                # Build the long-format (uid, ds, step, y_true, y_pred)
                # frame for this run and stash it. Dumped to Parquet after
                # all runs for this (model, dataset) complete.
                if save_predictions:
                    pred_records = _build_prediction_records(
                        df_test_in=df_test_in,
                        preds=preds,
                        model_col=model_col,
                        dataset_name=dataset_name,
                        model_name=spec.name,
                        seed=seed,
                        window_id=window_id,
                    )
                    if not pred_records.empty:
                        all_predictions.append(pred_records)

            except Exception as e:
                print(f"    ✗ Model failed ({seed_label}): {e}")
                traceback.print_exc()

    # ── Save aggregated metrics ──
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        # Derive model name from first result row for filename
        model_name = results_df["model"].iloc[0]
        out_path = results_dir / f"{model_name}_{dataset_name}.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\n[{dataset_name}] Results saved to {out_path}")
        print(f"[{dataset_name}] Total rows: {len(results_df)}")

        # ── Save raw predictions as Parquet ──
        if save_predictions and all_predictions:
            preds_df = pd.concat(all_predictions, ignore_index=True)
            preds_path = predictions_dir / f"{model_name}_{dataset_name}.parquet"
            try:
                preds_df.to_parquet(preds_path, index=False)
                print(f"[{dataset_name}] Predictions saved to {preds_path} "
                      f"({len(preds_df):,} rows)")
            except Exception as e:
                # Fall back to CSV if pyarrow isn't available
                fallback = predictions_dir / f"{model_name}_{dataset_name}.csv"
                preds_df.to_csv(fallback, index=False)
                print(f"[{dataset_name}] Parquet failed ({e}); "
                      f"predictions saved as CSV to {fallback}")

    return results_df
