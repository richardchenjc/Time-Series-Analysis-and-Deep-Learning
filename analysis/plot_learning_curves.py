"""
Generate training/validation learning curves for all iteratively-trained models.

Re-runs one representative fit per (model, dataset) with explicit loss logging:
  - 6 neural models via NeuralForecast (CSVLogger captures per-step metrics)
  - LightGBM via direct LGBMRegressor fit with eval_set callback

Outputs:
  - results/learning_curves/<Model>_<Dataset>.csv   (raw per-step metrics)
  - results/plots/learning_curve_<Model>_<Dataset>.png  (individual plots)
  - results/plots/learning_curves_grid.png  (7×3 combined grid)

Models WITHOUT iterative training (SeasonalNaive, AutoARIMA) are excluded —
their generalization is assessed via the data-volume curve instead.

Runtime: ~50-60 min total (TimesNet on Traffic dominates).
"""

import sys
import time
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
# Handle case where script is in analysis/ or project root
project_root = Path(__file__).resolve().parent
if (project_root / "config.py").exists():
    pass
elif (project_root.parent / "config.py").exists():
    project_root = project_root.parent
    sys.path.insert(0, str(project_root))

from config import (
    M4_CONFIG, M5_CONFIG, TRAFFIC_CONFIG,
    SEEDS, MAX_STEPS, BATCH_SIZE, EARLY_STOP_PATIENCE, VAL_CHECK_STEPS,
    LR_TRANSFORMER, LR_MLP, LR_RNN,
    PATCHTST_PARAMS, NBEATS_PARAMS, TIDE_PARAMS, DEEPAR_PARAMS,
    TIMESNET_PARAMS, LGBM_PARAMS, RESULTS_DIR,
)
from data_prep.m4_prep import load_m4_monthly
from data_prep.m5_prep import load_m5
from data_prep.traffic_prep import load_traffic

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Force UTF-8 on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Output directories ─────────────────────────────────────────────────
LC_DIR = RESULTS_DIR / "learning_curves"
PLOTS_DIR = RESULTS_DIR / "plots"
LC_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Datasets ────────────────────────────────────────────────────────────
DATASETS = [
    ("M4",      M4_CONFIG,      load_m4_monthly),
    ("M5",      M5_CONFIG,      load_m5),
    ("Traffic", TRAFFIC_CONFIG, load_traffic),
]

# Use first seed only — one representative curve per (model, dataset)
SEED = SEEDS[0]

# ── PatchTST per-dataset patch configs (from run_patchtst.py) ──────────
_PATCH_CONFIGS = {
    "M4":      {"patch_len": 6,  "stride": 3},
    "M5":      {"patch_len": 7,  "stride": 7},
    "Traffic": {"patch_len": 24, "stride": 12},
}

# ── DeepAR per-dataset hidden sizes (from run_deepar.py) ───────────────
_DEEPAR_HIDDEN = {"M4": 128, "M5": 128, "Traffic": 256}

# ── TimesNet per-dataset encoder layers (from run_timesnet.py) ──────────
_TIMESNET_LAYERS = {"M4": 2, "M5": 2, "Traffic": 3}


# ═══════════════════════════════════════════════════════════════════════
#  Neural model fitting with loss capture
# ═══════════════════════════════════════════════════════════════════════

def _fit_neural_model_with_logging(model_name, nf, df_train, horizon, log_dir):
    """Fit a NeuralForecast object and capture per-step loss.

    Strategy:
    1. Configure a CSVLogger pointing to log_dir
    2. Inject it into the NF model before fitting
    3. After fitting, read back the metrics.csv

    Returns a DataFrame with columns [step, train_loss, val_loss] or None.
    """
    try:
        from pytorch_lightning.loggers import CSVLogger
    except ImportError:
        from lightning.pytorch.loggers import CSVLogger

    # Clean up any prior logs for this run
    run_log_dir = Path(log_dir) / model_name
    if run_log_dir.exists():
        shutil.rmtree(run_log_dir)

    logger = CSVLogger(save_dir=str(log_dir), name=model_name, version=0)

    # Try to inject the logger into the NF model.
    # NeuralForecast's BaseModel stores trainer config internally.
    # Recent versions: model has a `logger` attribute or `trainer_kwargs`.
    injected = False
    for m in nf.models:
        # Approach 1: direct logger attribute
        if hasattr(m, "logger"):
            m.logger = logger
            injected = True
        # Approach 2: trainer_kwargs dict
        if hasattr(m, "trainer_kwargs") and isinstance(m.trainer_kwargs, dict):
            m.trainer_kwargs["logger"] = logger
            injected = True

    if not injected:
        print(f"    [WARN] Could not inject CSVLogger for {model_name}")

    # Fit
    nf.fit(df=df_train, val_size=horizon)

    # Read back metrics
    metrics_csv = run_log_dir / "version_0" / "metrics.csv"
    if not metrics_csv.exists():
        # Try alternative paths
        for p in run_log_dir.rglob("metrics.csv"):
            metrics_csv = p
            break

    if metrics_csv.exists():
        df_metrics = pd.read_csv(metrics_csv)
        print(f"    [OK] Captured {len(df_metrics)} metric rows from CSVLogger")
        return df_metrics
    else:
        print(f"    [WARN] No metrics.csv found at {run_log_dir}")
        # Fallback: check default lightning_logs
        default_logs = Path("lightning_logs")
        if default_logs.exists():
            versions = sorted(default_logs.glob("version_*"), key=lambda x: int(x.name.split("_")[1]))
            if versions:
                latest = versions[-1] / "metrics.csv"
                if latest.exists():
                    df_metrics = pd.read_csv(latest)
                    if len(df_metrics) > 0:
                        print(f"    [OK] Found {len(df_metrics)} rows in default lightning_logs")
                        return df_metrics
        print(f"    [FAIL] No training metrics captured for {model_name}")
        return None


def _build_and_fit_neural(model_name, dataset_name, cfg, df_train, log_dir):
    """Build one neural model, fit it, return metrics DataFrame."""

    horizon = cfg["horizon"]
    input_size = cfg["input_size"]
    freq = cfg["freq"]
    max_steps = MAX_STEPS  # 800 from config

    if model_name == "PatchTST":
        from neuralforecast import NeuralForecast
        from neuralforecast.models import PatchTST
        pc = _PATCH_CONFIGS[dataset_name]
        m = PatchTST(
            h=horizon, input_size=input_size,
            patch_len=pc["patch_len"], stride=pc["stride"],
            n_heads=PATCHTST_PARAMS["n_heads"],
            hidden_size=PATCHTST_PARAMS["hidden_size"],
            encoder_layers=PATCHTST_PARAMS["encoder_layers"],
            max_steps=max_steps, batch_size=BATCH_SIZE,
            learning_rate=LR_TRANSFORMER, random_seed=SEED,
            early_stop_patience_steps=EARLY_STOP_PATIENCE,
            val_check_steps=VAL_CHECK_STEPS, scaler_type="standard",
        )
        nf = NeuralForecast(models=[m], freq=freq)

    elif model_name == "NBEATS":
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS
        m = NBEATS(
            h=horizon, input_size=input_size,
            stack_types=NBEATS_PARAMS["stack_types"],
            n_blocks=NBEATS_PARAMS["n_blocks"],
            mlp_units=NBEATS_PARAMS["mlp_units"],
            max_steps=max_steps, batch_size=BATCH_SIZE,
            learning_rate=LR_MLP, random_seed=SEED,
            early_stop_patience_steps=EARLY_STOP_PATIENCE,
            val_check_steps=VAL_CHECK_STEPS, scaler_type="standard",
        )
        nf = NeuralForecast(models=[m], freq=freq)

    elif model_name == "TiDE":
        from neuralforecast import NeuralForecast
        from neuralforecast.models import TiDE
        m = TiDE(
            h=horizon, input_size=input_size,
            hidden_size=TIDE_PARAMS["hidden_size"],
            decoder_output_dim=TIDE_PARAMS["decoder_output_dim"],
            num_encoder_layers=TIDE_PARAMS["num_encoder_layers"],
            num_decoder_layers=TIDE_PARAMS["num_decoder_layers"],
            max_steps=max_steps, batch_size=BATCH_SIZE,
            learning_rate=LR_TRANSFORMER, random_seed=SEED,
            early_stop_patience_steps=EARLY_STOP_PATIENCE,
            val_check_steps=VAL_CHECK_STEPS, scaler_type="standard",
        )
        nf = NeuralForecast(models=[m], freq=freq)

    elif model_name == "DeepAR":
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        from neuralforecast import NeuralForecast
        from neuralforecast.models import DeepAR
        hs = _DEEPAR_HIDDEN[dataset_name]
        m = DeepAR(
            h=horizon, input_size=input_size,
            lstm_hidden_size=hs,
            lstm_n_layers=DEEPAR_PARAMS["n_layers"],
            max_steps=max_steps, batch_size=BATCH_SIZE,
            learning_rate=LR_RNN, random_seed=SEED,
            early_stop_patience_steps=EARLY_STOP_PATIENCE,
            val_check_steps=VAL_CHECK_STEPS, scaler_type="standard",
        )
        nf = NeuralForecast(models=[m], freq=freq)

    elif model_name == "DLinear":
        from neuralforecast import NeuralForecast
        from neuralforecast.models import DLinear
        m = DLinear(
            h=horizon, input_size=input_size,
            max_steps=max_steps, batch_size=BATCH_SIZE,
            learning_rate=LR_MLP, random_seed=SEED,
            early_stop_patience_steps=EARLY_STOP_PATIENCE,
            val_check_steps=VAL_CHECK_STEPS, scaler_type="standard",
        )
        nf = NeuralForecast(models=[m], freq=freq)

    elif model_name == "TimesNet":
        from neuralforecast import NeuralForecast
        from neuralforecast.models import TimesNet
        el = _TIMESNET_LAYERS[dataset_name]
        m = TimesNet(
            h=horizon, input_size=input_size,
            top_k=TIMESNET_PARAMS["top_k"],
            num_kernels=TIMESNET_PARAMS["num_kernels"],
            hidden_size=TIMESNET_PARAMS["d_model"],
            conv_hidden_size=TIMESNET_PARAMS["d_ff"],
            encoder_layers=el,
            max_steps=max_steps, batch_size=BATCH_SIZE,
            learning_rate=LR_TRANSFORMER, random_seed=SEED,
            early_stop_patience_steps=EARLY_STOP_PATIENCE,
            val_check_steps=VAL_CHECK_STEPS, scaler_type="standard",
        )
        nf = NeuralForecast(models=[m], freq=freq)
    else:
        raise ValueError(f"Unknown neural model: {model_name}")

    return _fit_neural_model_with_logging(model_name, nf, df_train, horizon, str(log_dir))


# ═══════════════════════════════════════════════════════════════════════
#  LightGBM fitting with boosting-round loss capture
# ═══════════════════════════════════════════════════════════════════════

def _fit_lgbm_with_logging(dataset_name, cfg, df_train):
    """Fit LightGBM directly (bypassing mlforecast) with eval_set logging.

    Strategy:
    1. Use mlforecast to compute lag features (preprocess)
    2. Split preprocessed data into train/val
    3. Fit LGBMRegressor with eval_set + callbacks

    Returns DataFrame with columns [step, train_loss, val_loss] or None.
    """
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean, RollingStd
    from lightgbm import LGBMRegressor, log_evaluation, record_evaluation
    import lightgbm as lgb

    season_length = cfg["season_length"]
    freq = cfg["freq"]
    horizon = cfg["horizon"]

    # Build lags matching run_lightgbm.py config
    lags = list(range(1, season_length + 1))
    if 2 * season_length not in lags:
        lags.append(2 * season_length)

    lag_transforms = {
        season_length: [
            RollingMean(window_size=season_length),
            RollingStd(window_size=season_length),
        ],
    }

    date_features = []
    if freq in ("D", "d"):
        date_features = ["dayofweek", "month"]
    elif freq in ("h", "H"):
        date_features = ["hour", "dayofweek"]
    elif freq in ("ME", "MS", "M"):
        date_features = ["month"]

    # Create MLForecast for preprocessing only
    lgbm_model = LGBMRegressor(
        n_estimators=LGBM_PARAMS["n_estimators"],
        learning_rate=LGBM_PARAMS["learning_rate"],
        num_leaves=LGBM_PARAMS["num_leaves"],
        random_state=SEED,
        verbosity=-1,
    )

    mlf = MLForecast(
        models={"LightGBM": lgbm_model},
        freq=freq,
        lags=lags,
        lag_transforms=lag_transforms,
        date_features=date_features if date_features else None,
    )

    # Preprocess to get lag features
    try:
        prep_df = mlf.preprocess(df_train)
    except AttributeError:
        # Older mlforecast: fit and then extract the internal data
        print("    [WARN] mlforecast.preprocess() not available, using fit approach")
        try:
            mlf.fit(df_train)
            # After fit, the internal models have been trained.
            # We can't capture per-round loss retroactively.
            # Fall back to re-fitting manually.
            return None
        except Exception as e:
            print(f"    [FAIL] LightGBM preprocessing failed: {e}")
            return None

    if prep_df is None or len(prep_df) == 0:
        print("    [FAIL] Preprocessing returned empty DataFrame")
        return None

    # prep_df has columns: unique_id, ds, y, lag1, lag2, ..., rolling features, date features
    target_col = "y"
    id_col = "unique_id"
    ds_col = "ds"

    feature_cols = [c for c in prep_df.columns if c not in {id_col, ds_col, target_col}]
    prep_df = prep_df.dropna(subset=feature_cols)

    if len(prep_df) == 0:
        print("    [FAIL] No rows after dropping NaN lag features")
        return None

    # Time-based split: last `horizon` steps per series as validation
    max_dates = prep_df.groupby(id_col)[ds_col].max().reset_index()
    max_dates["cutoff"] = max_dates[ds_col] - pd.Timedelta(days=horizon if freq in ("D","d") else 1)
    # For monthly/hourly, use a different offset
    if freq in ("ME", "MS", "M"):
        max_dates["cutoff"] = max_dates[ds_col] - pd.DateOffset(months=horizon)
    elif freq in ("h", "H"):
        max_dates["cutoff"] = max_dates[ds_col] - pd.Timedelta(hours=horizon)

    prep_df = prep_df.merge(max_dates[[id_col, "cutoff"]], on=id_col)
    train_mask = prep_df[ds_col] <= prep_df["cutoff"]
    val_mask = prep_df[ds_col] > prep_df["cutoff"]

    X_train = prep_df.loc[train_mask, feature_cols].values
    y_train = prep_df.loc[train_mask, target_col].values
    X_val = prep_df.loc[val_mask, feature_cols].values
    y_val = prep_df.loc[val_mask, target_col].values

    if len(X_val) == 0 or len(X_train) == 0:
        print(f"    [FAIL] Train/val split produced empty set: train={len(X_train)}, val={len(X_val)}")
        return None

    print(f"    LightGBM: {len(X_train)} train rows, {len(X_val)} val rows, {len(feature_cols)} features")

    # Fit with eval logging
    eval_result = {}
    model = LGBMRegressor(
        n_estimators=LGBM_PARAMS["n_estimators"],
        learning_rate=LGBM_PARAMS["learning_rate"],
        num_leaves=LGBM_PARAMS["num_leaves"],
        random_state=SEED,
        verbosity=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        eval_metric="l1",  # MAE
        callbacks=[
            record_evaluation(eval_result),
            log_evaluation(period=0),  # suppress console output
        ],
    )

    # Extract per-round losses
    if "train" in eval_result and "l1" in eval_result["train"]:
        train_losses = eval_result["train"]["l1"]
        val_losses = eval_result["val"]["l1"]
        df_metrics = pd.DataFrame({
            "step": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses,
        })
        print(f"    [OK] Captured {len(df_metrics)} boosting rounds")
        return df_metrics
    else:
        print(f"    [FAIL] No eval_result captured. Keys: {list(eval_result.keys())}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════

def _normalize_neural_metrics(df_metrics):
    """Extract train_loss and val_loss from NeuralForecast CSVLogger output.

    CSVLogger produces a CSV with columns like:
      train_loss_step, train_loss_epoch, valid_loss, ptl/train_loss, etc.
    Column names vary by version. This function tries common patterns.
    """
    if df_metrics is None or len(df_metrics) == 0:
        return None

    cols = list(df_metrics.columns)
    print(f"    [DEBUG] Available columns: {cols}")

    result = pd.DataFrame()

    # Find step column
    for c in ["step", "epoch", "global_step"]:
        if c in cols:
            result["step"] = df_metrics[c]
            break
    if "step" not in result.columns:
        result["step"] = range(len(df_metrics))

    # Find train loss
    train_candidates = [
        "train_loss_step", "train_loss", "train_loss_epoch",
        "ptl/train_loss", "loss",
    ]
    for c in train_candidates:
        if c in cols:
            result["train_loss"] = df_metrics[c]
            break

    # Find val loss
    val_candidates = [
        "valid_loss", "val_loss", "valid_loss_epoch",
        "ptl/val_loss", "validation_loss",
    ]
    for c in val_candidates:
        if c in cols:
            result["val_loss"] = df_metrics[c]
            break

    # Drop rows where both train and val are NaN
    has_train = "train_loss" in result.columns
    has_val = "val_loss" in result.columns

    if not has_train and not has_val:
        print(f"    [FAIL] No recognized loss columns in: {cols}")
        return None

    # For NeuralForecast CSVLogger, train_loss and val_loss are logged at
    # different rows (train every step, val every val_check_steps).
    # We need to handle them separately.
    if has_train and has_val:
        # Forward-fill val_loss so it persists between val checks
        result["val_loss"] = result["val_loss"].ffill()
        # Drop rows where train_loss is NaN (val-only rows at the start)
        result = result.dropna(subset=["train_loss"])

    return result


def _plot_single_curve(df_metrics, model_name, dataset_name, is_lgbm=False):
    """Plot one learning curve and save to disk."""
    if df_metrics is None or len(df_metrics) == 0:
        return False

    fig, ax = plt.subplots(figsize=(6, 4))

    has_train = "train_loss" in df_metrics.columns and df_metrics["train_loss"].notna().any()
    has_val = "val_loss" in df_metrics.columns and df_metrics["val_loss"].notna().any()

    x_label = "Boosting Round" if is_lgbm else "Training Step"
    steps = df_metrics["step"].values

    if has_train:
        train_vals = df_metrics["train_loss"].values
        # Smooth train loss for neural models (very noisy per-step)
        if not is_lgbm and len(train_vals) > 20:
            window = max(5, len(train_vals) // 50)
            smoothed = pd.Series(train_vals).rolling(window, min_periods=1, center=True).mean().values
            ax.plot(steps, train_vals, alpha=0.15, color="tab:blue", linewidth=0.5)
            ax.plot(steps, smoothed, color="tab:blue", linewidth=2, label="Train Loss (smoothed)")
        else:
            ax.plot(steps, train_vals, color="tab:blue", linewidth=1.5, label="Train Loss")

    if has_val:
        val_vals = df_metrics["val_loss"].values
        if is_lgbm:
            ax.plot(steps, val_vals, color="tab:orange", linewidth=1.5, label="Val Loss")
        else:
            # Val loss is sparse (every val_check_steps) — plot with markers
            val_valid = df_metrics[df_metrics["val_loss"].notna()]
            if len(val_valid) > 0:
                # Don't forward-fill for plotting — show actual check points
                raw_val = df_metrics["val_loss"].values.copy()
                # Find actual val check points (where val changes)
                val_changes = np.where(~np.isnan(df_metrics["val_loss"].values))[0]
                if len(val_changes) > 1:
                    ax.plot(steps, df_metrics["val_loss"].ffill().values,
                            color="tab:orange", linewidth=1.5, alpha=0.7, label="Val Loss")
                    ax.scatter(steps[val_changes],
                               df_metrics["val_loss"].values[val_changes],
                               color="tab:orange", s=30, zorder=5)
                else:
                    ax.plot(steps, df_metrics["val_loss"].ffill().values,
                            color="tab:orange", linewidth=1.5, label="Val Loss")

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Loss (MAE)", fontsize=11)
    ax.set_title(f"{model_name} — {dataset_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Mark early stopping point if val loss was captured
    if has_val and not is_lgbm:
        val_series = df_metrics["val_loss"].dropna()
        if len(val_series) > 2:
            last_step = steps[-1]
            max_step = MAX_STEPS
            if last_step < max_step * 0.9:
                ax.axvline(last_step, color="red", linestyle="--", alpha=0.5, linewidth=1)
                ax.annotate("Early stop", xy=(last_step, ax.get_ylim()[1] * 0.95),
                            fontsize=8, color="red", alpha=0.7, ha="right")

    plt.tight_layout()
    out_path = PLOTS_DIR / f"learning_curve_{model_name}_{dataset_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_path}")
    return True


def _plot_combined_grid(all_results):
    """Plot 7×3 grid of learning curves (rows=models, cols=datasets)."""
    model_order = ["DLinear", "LightGBM", "TiDE", "NBEATS", "PatchTST", "DeepAR", "TimesNet"]
    dataset_order = ["M4", "M5", "Traffic"]

    n_rows = len(model_order)
    n_cols = len(dataset_order)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 24), squeeze=False)
    fig.suptitle("Learning Curves: Training vs Validation Loss",
                 fontsize=16, fontweight="bold", y=0.995)

    for i, model_name in enumerate(model_order):
        for j, dataset_name in enumerate(dataset_order):
            ax = axes[i][j]
            key = (model_name, dataset_name)

            if key in all_results and all_results[key] is not None:
                df = all_results[key]
                is_lgbm = (model_name == "LightGBM")
                steps = df["step"].values

                has_train = "train_loss" in df.columns and df["train_loss"].notna().any()
                has_val = "val_loss" in df.columns and df["val_loss"].notna().any()

                if has_train:
                    train_vals = df["train_loss"].values
                    if not is_lgbm and len(train_vals) > 20:
                        window = max(5, len(train_vals) // 50)
                        smoothed = pd.Series(train_vals).rolling(window, min_periods=1, center=True).mean().values
                        ax.plot(steps, train_vals, alpha=0.1, color="tab:blue", linewidth=0.3)
                        ax.plot(steps, smoothed, color="tab:blue", linewidth=1.5, label="Train")
                    else:
                        ax.plot(steps, train_vals, color="tab:blue", linewidth=1, label="Train")

                if has_val:
                    ax.plot(steps, df["val_loss"].ffill().values,
                            color="tab:orange", linewidth=1.5, label="Val")

                # Mark early stopping
                if has_val and not is_lgbm:
                    last_step = steps[-1]
                    if last_step < MAX_STEPS * 0.9:
                        ax.axvline(last_step, color="red", linestyle="--",
                                   alpha=0.4, linewidth=0.8)
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="gray")

            # Labels
            if i == 0:
                ax.set_title(dataset_name, fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(model_name, fontsize=11, fontweight="bold")
            if i == n_rows - 1:
                x_label = "Boosting Round" if model_name == "LightGBM" else "Step"
                ax.set_xlabel(x_label, fontsize=9)
            if i == 0 and j == n_cols - 1:
                ax.legend(fontsize=7, loc="upper right")

            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = PLOTS_DIR / "learning_curves_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved combined grid: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

NEURAL_MODELS = ["PatchTST", "NBEATS", "TiDE", "DeepAR", "DLinear", "TimesNet"]


def main():
    print("=" * 60)
    print("  Learning Curves — 7 models × 3 datasets")
    print("  (1 seed, window 1 only, for visualization)")
    print("=" * 60)

    all_results = {}
    total_start = time.time()

    for dataset_name, cfg, loader in DATASETS:
        n_series = cfg.get("n_series_sample")
        print(f"\n{'─' * 60}")
        print(f"  Dataset: {dataset_name}  (n_series={n_series})")
        print(f"{'─' * 60}")

        # Load data
        t0 = time.time()
        df_train, df_test = loader(n_series=n_series)
        df_full = pd.concat([df_train, df_test], ignore_index=True)

        # For walk-forward window 1: use everything up to the last `horizon` steps
        max_date = df_full["ds"].max()
        offset = pd.tseries.frequencies.to_offset(cfg["freq"])
        test_start = max_date - (cfg["horizon"] - 1) * offset
        train_cutoff = test_start - offset
        df_train_w1 = df_full[df_full["ds"] <= train_cutoff].copy()

        # Filter short series
        min_req = cfg["input_size"] + cfg["horizon"] + 1
        series_lens = df_train_w1.groupby("unique_id").size()
        valid = series_lens[series_lens >= min_req].index
        df_train_w1 = df_train_w1[df_train_w1["unique_id"].isin(valid)]

        print(f"  Loaded in {time.time()-t0:.1f}s — {df_train_w1['unique_id'].nunique()} series for training")

        # ── Neural models ──────────────────────────────────────────
        for model_name in NEURAL_MODELS:
            print(f"\n  [{model_name}] on {dataset_name}...")
            t1 = time.time()
            try:
                log_dir = LC_DIR / "logs" / dataset_name
                log_dir.mkdir(parents=True, exist_ok=True)

                df_metrics = _build_and_fit_neural(
                    model_name, dataset_name, cfg, df_train_w1, log_dir
                )

                if df_metrics is not None:
                    df_metrics = _normalize_neural_metrics(df_metrics)

                if df_metrics is not None and len(df_metrics) > 0:
                    # Save raw CSV
                    csv_path = LC_DIR / f"{model_name}_{dataset_name}.csv"
                    df_metrics.to_csv(csv_path, index=False)
                    print(f"    Saved metrics: {csv_path}")

                    # Plot individual curve
                    _plot_single_curve(df_metrics, model_name, dataset_name, is_lgbm=False)
                    all_results[(model_name, dataset_name)] = df_metrics
                else:
                    all_results[(model_name, dataset_name)] = None

            except Exception as e:
                print(f"    [ERROR] {model_name} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[(model_name, dataset_name)] = None

            print(f"    Time: {time.time()-t1:.1f}s")

        # ── LightGBM ───────────────────────────────────────────────
        print(f"\n  [LightGBM] on {dataset_name}...")
        t1 = time.time()
        try:
            df_metrics = _fit_lgbm_with_logging(dataset_name, cfg, df_train_w1)
            if df_metrics is not None and len(df_metrics) > 0:
                csv_path = LC_DIR / f"LightGBM_{dataset_name}.csv"
                df_metrics.to_csv(csv_path, index=False)
                print(f"    Saved metrics: {csv_path}")
                _plot_single_curve(df_metrics, "LightGBM", dataset_name, is_lgbm=True)
                all_results[("LightGBM", dataset_name)] = df_metrics
            else:
                all_results[("LightGBM", dataset_name)] = None
        except Exception as e:
            print(f"    [ERROR] LightGBM on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[("LightGBM", dataset_name)] = None
        print(f"    Time: {time.time()-t1:.1f}s")

    # ── Combined grid ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Generating combined 7×3 grid...")
    _plot_combined_grid(all_results)

    # ── Summary ────────────────────────────────────────────────────
    total_time = time.time() - total_start
    n_ok = sum(1 for v in all_results.values() if v is not None)
    n_total = len(all_results)
    print(f"\n{'=' * 60}")
    print(f"  Learning curves complete: {n_ok}/{n_total} successful")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Individual plots: results/plots/learning_curve_*.png")
    print(f"  Combined grid:    results/plots/learning_curves_grid.png")
    print(f"  Raw CSVs:         results/learning_curves/*.csv")
    print(f"{'=' * 60}")

    if n_ok < n_total:
        print(f"\n  [NOTE] {n_total - n_ok} models failed to capture metrics.")
        print("  This usually means the CSVLogger injection didn't work")
        print("  for that NeuralForecast version. Failed cells show 'No data'")
        print("  in the grid plot. The loss capture mechanism may need")
        print("  adjustment for your specific neuralforecast version.")


if __name__ == "__main__":
    main()
