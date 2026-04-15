"""
Cost-vs-accuracy frontier plot — with GPU memory subplot.

UPDATE FROM PREVIOUS VERSION:
Previous version had a single "training time vs MAE" scatter per dataset.
This version produces TWO scatters per dataset:
  - Training time vs MAE (all 9 models)
  - Peak GPU memory vs MAE (DL-only, since classical/ML models use ~0 GPU)

The GPU memory subplot excludes SeasonalNaive, AutoARIMA, and LightGBM
because those models run on CPU and their peak_gpu_mb is either 0 or
artifactually small (just from the PyTorch backend being initialised).
Including them would falsely suggest they have "zero GPU cost" when
really they have "different cost units" — we'd be comparing apples to
oranges on the same axis.

For each dataset we produce:
  - results/plots/cost_time_<dataset>.png       (time axis, all models)
  - results/plots/cost_gpu_<dataset>.png        (GPU axis, DL-only)
  - results/plots/cost_vs_accuracy_<dataset>.png (combined 2-panel)
  - results/plots/cost_vs_accuracy_all.png      (3x2 grid: all datasets × both cost axes)

Usage:
    python analysis/plot_cost_vs_accuracy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})
sns.set_theme(style="whitegrid", palette="Set2")

PLOTS_DIR = RESULTS_DIR / "plots"

_MODEL_CATEGORY = {
    "SeasonalNaive": "Classical",
    "AutoARIMA":     "Classical",
    "LightGBM":      "Classical",
    "DLinear":       "Linear",
    "NBEATS":        "Deep Learning",
    "PatchTST":      "Deep Learning",
    "TiDE":          "Deep Learning",
    "DeepAR":        "Deep Learning",
    "TimesNet":      "Deep Learning",
}
_CATEGORY_COLORS = {
    "Classical":     "#2e7d32",
    "Linear":        "#ef6c00",
    "Deep Learning": "#1565c0",
}

# Models to EXCLUDE from the GPU memory plot. These run on CPU
# so their recorded peak_gpu_mb doesn't represent a GPU cost.
_CPU_ONLY_MODELS = {"SeasonalNaive", "AutoARIMA", "LightGBM"}


def _load_joined() -> pd.DataFrame:
    summary = pd.read_csv(RESULTS_DIR / "summary_table.csv")
    costs = pd.read_csv(RESULTS_DIR / "computational_costs.csv")
    df = summary.merge(costs, on=["dataset", "model"])
    df["category"] = df["model"].map(_MODEL_CATEGORY).fillna("Other")
    return df


def _plot_one_axis(
    df: pd.DataFrame,
    dataset_name: str,
    cost_col: str,
    cost_label: str,
    ax,
    exclude_cpu_only: bool = False,
):
    """Draw one cost-vs-accuracy scatter on the given axis."""
    sub = df[df["dataset"] == dataset_name].copy()
    if exclude_cpu_only:
        sub = sub[~sub["model"].isin(_CPU_ONLY_MODELS)]
    if sub.empty:
        return

    for cat, color in _CATEGORY_COLORS.items():
        cat_sub = sub[sub["category"] == cat]
        if cat_sub.empty:
            continue
        ax.scatter(
            cat_sub[cost_col], cat_sub["mae_mean"],
            s=140, color=color, label=cat, edgecolor="black", linewidth=0.7,
            alpha=0.85, zorder=3,
        )

    # Annotate each point with its model name
    for _, row in sub.iterrows():
        ax.annotate(
            row["model"],
            (row[cost_col], row["mae_mean"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )

    # Log scale on cost axis unless all values are very close
    cost_vals = sub[cost_col].replace(0, np.nan).dropna()
    if not cost_vals.empty and cost_vals.max() / cost_vals.min() > 10:
        ax.set_xscale("log")
    ax.set_xlabel(cost_label)
    ax.set_ylabel("MAE")
    title_suffix = " (DL-only)" if exclude_cpu_only else ""
    ax.set_title(f"{dataset_name}{title_suffix}")
    ax.legend(title="Category", loc="best", fontsize=8)
    ax.grid(True, alpha=0.4)


def _plot_two_panel(df: pd.DataFrame, dataset_name: str):
    """Side-by-side time/GPU scatter plots for one dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    _plot_one_axis(
        df, dataset_name,
        cost_col="avg_train_time_sec",
        cost_label="Avg training time per run (s, log scale)",
        ax=axes[0],
        exclude_cpu_only=False,
    )
    _plot_one_axis(
        df, dataset_name,
        cost_col="peak_gpu_mb",
        cost_label="Peak GPU memory (MB)",
        ax=axes[1],
        exclude_cpu_only=True,
    )
    fig.suptitle(f"Cost vs Accuracy: {dataset_name}", fontsize=14, y=1.02)
    plt.tight_layout()

    out = PLOTS_DIR / f"cost_vs_accuracy_{dataset_name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _plot_single_axis(df: pd.DataFrame, dataset_name: str, axis: str):
    """Standalone 1-panel plot for one cost axis."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    if axis == "time":
        _plot_one_axis(
            df, dataset_name,
            cost_col="avg_train_time_sec",
            cost_label="Avg training time per run (s, log scale)",
            ax=ax,
            exclude_cpu_only=False,
        )
        out = PLOTS_DIR / f"cost_time_{dataset_name}.png"
    elif axis == "gpu":
        _plot_one_axis(
            df, dataset_name,
            cost_col="peak_gpu_mb",
            cost_label="Peak GPU memory (MB)",
            ax=ax,
            exclude_cpu_only=True,
        )
        out = PLOTS_DIR / f"cost_gpu_{dataset_name}.png"
    else:
        return
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _plot_grand_overview(df: pd.DataFrame):
    """3x2 grid: rows = datasets, columns = (time, GPU)."""
    datasets = ["M4", "M5", "Traffic"]
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    for row_idx, ds in enumerate(datasets):
        _plot_one_axis(
            df, ds,
            cost_col="avg_train_time_sec",
            cost_label="Avg training time (s, log scale)",
            ax=axes[row_idx, 0],
            exclude_cpu_only=False,
        )
        _plot_one_axis(
            df, ds,
            cost_col="peak_gpu_mb",
            cost_label="Peak GPU memory (MB)",
            ax=axes[row_idx, 1],
            exclude_cpu_only=True,
        )
    fig.suptitle("Cost vs Accuracy — all datasets", fontsize=16, y=1.00)
    plt.tight_layout()

    out = PLOTS_DIR / "cost_vs_accuracy_all.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Plotting cost-vs-accuracy (time + GPU) into {PLOTS_DIR}\n")

    df = _load_joined()

    for ds in ["M4", "M5", "Traffic"]:
        _plot_single_axis(df, ds, axis="time")
        _plot_single_axis(df, ds, axis="gpu")
        _plot_two_panel(df, ds)

    _plot_grand_overview(df)

    print("\nDone. Two cost axes per dataset:")
    print("  - time  : training time vs MAE, all 9 models")
    print("  - gpu   : peak GPU memory vs MAE, DL-only")
    print("Classical and LightGBM are excluded from the GPU plot because")
    print("they run on CPU and their recorded GPU memory is ~0.")


if __name__ == "__main__":
    main()
