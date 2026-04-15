"""
Plot example predictions: actual vs predicted for 3 sample series per dataset.

Reads the parquet prediction files and produces intuitive line charts showing
how each model's forecast compares to ground truth on representative series.

Outputs:
  - results/plots/example_preds_<Dataset>.png  (one per dataset)
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent if (_THIS.parent.parent / "config.py").exists() else _THIS.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PREDS_DIR = RESULTS_DIR / "predictions"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["M4", "M5", "Traffic"]

# Highlight models (bold color) vs context (grey)
HIGHLIGHT = {
    "M4": ["NBEATS", "AutoARIMA", "DLinear"],
    "M5": ["NBEATS", "PatchTST", "SeasonalNaive"],
    "Traffic": ["DLinear", "LightGBM", "DeepAR"],
}

COLORS = {
    "NBEATS": "#2980b9", "AutoARIMA": "#27ae60", "DLinear": "#e67e22",
    "PatchTST": "#c0392b", "LightGBM": "#27ae60", "SeasonalNaive": "#7f8c8d",
    "DeepAR": "#16a085", "TiDE": "#8e44ad", "TimesNet": "#d35400",
}


def _load_preds(dataset):
    """Load all model predictions for a dataset, return combined DataFrame."""
    dfs = []
    for pq in PREDS_DIR.glob(f"*_{dataset}.parquet"):
        try:
            df = pd.read_parquet(pq)
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        # Try CSV fallback
        for csv in PREDS_DIR.glob(f"*_{dataset}_preds.csv"):
            try:
                df = pd.read_csv(csv)
                dfs.append(df)
            except Exception:
                pass
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def _pick_series(df, n=3):
    """Pick n representative series: one easy, one medium, one hard."""
    # Compute per-series MAE across all models for "difficulty"
    per_series = df.groupby("unique_id").apply(
        lambda g: (g["y_true"] - g["y_pred"]).abs().mean()
    ).reset_index(name="avg_mae")

    per_series = per_series.sort_values("avg_mae")
    n_total = len(per_series)
    if n_total < 3:
        return per_series["unique_id"].tolist()

    indices = [
        int(n_total * 0.25),  # easy (25th percentile)
        int(n_total * 0.50),  # medium (median)
        int(n_total * 0.75),  # hard (75th percentile)
    ]
    return [per_series.iloc[i]["unique_id"] for i in indices]


def plot_example_preds(dataset):
    """Plot 3×1 grid of example predictions for one dataset."""
    df = _load_preds(dataset)
    if df is None or len(df) == 0:
        print(f"  {dataset}: no prediction data found")
        return

    # Use seed=42, window=1 for consistency
    if "seed" in df.columns:
        seeds = df["seed"].unique()
        df = df[df["seed"] == seeds[0]]
    if "window" in df.columns:
        windows = df["window"].unique()
        df = df[df["window"] == windows[0]]

    models = df["model"].unique()
    highlights = HIGHLIGHT.get(dataset, list(models)[:3])
    series_ids = _pick_series(df, n=3)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    difficulty_labels = ["Easy Series", "Medium Series", "Hard Series"]

    for idx, (uid, label) in enumerate(zip(series_ids, difficulty_labels)):
        ax = axes[idx]
        sub = df[df["unique_id"] == uid].copy()

        if len(sub) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        # Plot actual values
        # Get one model's actual (they're all the same)
        first_model = sub["model"].iloc[0]
        actual = sub[sub["model"] == first_model].sort_values("horizon_step")
        ax.plot(actual["horizon_step"].values, actual["y_true"].values,
                color="black", linewidth=2.5, label="Actual", zorder=10,
                marker="s", markersize=4)

        # Plot each model's predictions
        for model in models:
            msub = sub[sub["model"] == model].sort_values("horizon_step")
            if len(msub) == 0:
                continue

            is_highlight = model in highlights
            color = COLORS.get(model, "gray")
            lw = 1.8 if is_highlight else 0.6
            alpha = 1.0 if is_highlight else 0.3
            zorder = 5 if is_highlight else 1

            ax.plot(msub["horizon_step"].values, msub["y_pred"].values,
                    color=color if is_highlight else "lightgray",
                    linewidth=lw, alpha=alpha, zorder=zorder,
                    label=model if is_highlight else None)

        ax.set_ylabel("Value", fontsize=10)
        ax.set_title(f"{label} (series: {uid})", fontsize=11)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=8, ncol=2, loc="upper right",
                      framealpha=0.9)

    axes[-1].set_xlabel("Forecast Horizon Step", fontsize=11)
    fig.suptitle(f"{dataset} — Example Predictions (Actual vs Forecast)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = PLOTS_DIR / f"example_preds_{dataset}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    print("Generating example prediction plots...")
    for ds in DATASETS:
        plot_example_preds(ds)
    print("Done.")


if __name__ == "__main__":
    main()
