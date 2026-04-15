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

Series filtering — the input_size + horizon + offset rule
---------------------------------------------------------
Each series must have at least `input_size + horizon + input_size_offset`
training timestamps to participate in a window:

  - input_size           : the model's lookback window length
  - horizon              : neuralforecast carves out this many points at the
                           END of each series for internal validation, so
                           they are unavailable for forming input windows
  - input_size_offset    : per-model adjustment, defaults to 0
                           DeepAR specifically passes 1 because its
                           autoregressive structure requires one additional
                           timestamp beyond the standard input_size
                           (neuralforecast reports input_size=37 when we
                           pass input_size=36, hence the +1 fix)

This means a series of length T must satisfy:
    T >= input_size + horizon + input_size_offset

Below that, the series is filtered out and logged in the per-window
"filtered N with < X obs" message. Filtering happens BEFORE the model
ever sees the data, which is essential for two reasons:

  1. Crash prevention. neuralforecast raises ValueError mid-fit if any
     series is too short, killing the entire (window, seed) run.

  2. Apples-to-apples comparison. Every model in a given run sees the
     same series subset. Without the filter, classical baselines would
     happily train on shorter series that DL models can't handle, and
     the cross-model MAE comparison would be skewed by the long-tail of
     hard-to-forecast short series being scored against baselines only.

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
    input_size_offset: int = 0,
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
        Model lookback window length.
    n_windows : int
        Number of sliding windows.
    freq : str
        Pandas frequency string (e.g. 'ME', 'D', 'h').
    max_train_size : int or None
        If set, caps training history to this many steps per series.
    input_size_offset : int, default 0
        Extra timestamps required beyond the standard input_size + horizon.
        DeepAR passes 1 here because its autoregressive structure requires
        one additional timestamp; all other models pass 0.

    Returns
    -------
    List of (df_train, df_test) tuples.
    """
    splits = []

    max_date = df["ds"].max()
    offset = pd.tseries.frequencies.to_offset(freq)

    for w in range(n_windows):
        test_end    = max_date - (w * horizon) * offset
        test_start  = test_end - (horizon - 1) * offset
        train_cutoff = test_start - offset

        if max_train_size is not None:
            train_start = train_cutoff - max_train_size * offset
            df_train_w = df[(df["ds"] > train_start) & (df["ds"] <= train_cutoff)].copy()
        else:
            df_train_w = df[df["ds"] <= train_cutoff].copy()

        df_test_w = df[(df["ds"] >= test_start) & (df["ds"] <= test_end)].copy()

        # ── Filter: keep only series long enough to produce a training pair ──
        # See module docstring for the full rationale of this rule.
        min_required = input_size + horizon + input_size_offset
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

        offset_note = f" (+{input_size_offset} model offset)" if input_size_offset else ""
        print(f"  [Walk-forward] Window {w+1}: {n_after} series "
              f"(filtered {n_dropped_short} with < {min_required} obs "
              f"= input {input_size} + horizon {horizon}{offset_note}), "
              f"train up to {train_cutoff}, test {test_start} → {test_end}")
        splits.append((df_train_w, df_test_w))

    splits.reverse()
    return splits


def _prepare_inputs(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    exog_cols: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Strip / preserve exogenous columns based on the model's declaration."""
    extra = [c for c in df_train.columns if c not in _BASE_COLS]

    if exog_cols:
        missing = [c for c in exog_cols if c not in df_train.columns]
        if missing:
            raise ValueError(
                f"Model declared exog_cols {exog_cols} but df_train is missing {missing}"
            )
        keep = ["unique_id", "ds", "y"] + list(exog_cols)
        df_train_in = df_train[keep].copy()
        df_test_in = df_test[keep].copy()
        X_df = df_test_in[["unique_id", "ds"] + list(exog_cols)].copy()
        return df_train_in, df_test_in, X_df

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
    """
    merged = df_test_in[["unique_id", "ds", "y"]].merge(
        preds[["unique_id", "ds", model_col]],
        on=["unique_id", "ds"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    merged = merged.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    merged["horizon_step"] = merged.groupby("unique_id").cumcount() + 1

    merged = merged.rename(columns={"y": "y_true", model_col: "y_pred"})
    merged["dataset"] = dataset_name
    merged["model"] = model_name
    merged["seed"] = str(seed) if seed is not None else "N/A"
    merged["window"] = window_id

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
    input_size_offset: int = 0,
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
    input_size_offset : int, default 0
        Extra series-length headroom required beyond input_size + horizon.
        Pass 1 for DeepAR; 0 for everything else.

    Returns
    -------
    DataFrame with per-seed, per-window results for this model.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = results_dir / "predictions"
    if save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_predictions = []

    print(f"\n{'='*60}")
    print(f"[{dataset_name}] Generating {n_windows} sliding-window splits...")
    print(f"{'='*60}")
    splits = _sliding_window_splits(
        df_full, horizon, input_size, n_windows, freq, max_train_size,
        input_size_offset=input_size_offset,
    )

    if not splits:
        print(f"[{dataset_name}] WARNING: No valid splits generated!")
        return pd.DataFrame()

    for w_idx, (df_train, df_test) in enumerate(splits):
        window_id = w_idx + 1
        print(f"\n{'─'*50}")
        print(f"[{dataset_name}] Window {window_id}/{len(splits)}")
        print(f"{'─'*50}")

        seed_list = seeds if needs_seed else [None]

        for seed in seed_list:
            seed_label = f"seed={seed}" if seed is not None else "no seed"
            print(f"\n  ▸ [{seed_label}]")

            try:
                spec = build_model_fn(seed=seed, max_steps=max_steps)
                exog_cols = getattr(spec, "exog_cols", None)

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

                if "index" in preds.columns:
                    preds = preds.drop(columns=["index"])

                model_col = spec.name
                if model_col not in preds.columns:
                    id_cols = {"unique_id", "ds"}
                    model_col = next(c for c in preds.columns if c not in id_cols)

                metrics_df = compute_metrics_per_series(
                    df_test_in, preds, df_train_in, season_length, model_col
                )
                avg_mae = metrics_df["mae"].mean()

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
        model_name = results_df["model"].iloc[0]
        out_path = results_dir / f"{model_name}_{dataset_name}.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\n[{dataset_name}] Results saved to {out_path}")
        print(f"[{dataset_name}] Total rows: {len(results_df)}")

        if save_predictions and all_predictions:
            preds_df = pd.concat(all_predictions, ignore_index=True)
            preds_path = predictions_dir / f"{model_name}_{dataset_name}.parquet"
            try:
                preds_df.to_parquet(preds_path, index=False)
                print(f"[{dataset_name}] Predictions saved to {preds_path} "
                      f"({len(preds_df):,} rows)")
            except Exception as e:
                fallback = predictions_dir / f"{model_name}_{dataset_name}.csv"
                preds_df.to_csv(fallback, index=False)
                print(f"[{dataset_name}] Parquet failed ({e}); "
                      f"predictions saved as CSV to {fallback}")

    return results_df
