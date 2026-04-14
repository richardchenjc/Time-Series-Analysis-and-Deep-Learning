"""
Generate all figures for the report.

Reads aggregated results and produces publication-quality plots:
  1. MAE comparison bar chart (grouped by dataset, with error bars)
  2. MASE comparison bar chart
  3. Computational cost bar chart
  4. Model ranking heatmap
  5. Per-dataset detailed comparison

All saved as PNG to results/plots/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR

# ── Style ──
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (12, 6),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})
sns.set_theme(style="whitegrid", palette="Set2")

PLOTS_DIR = RESULTS_DIR / "plots"


def _load_summary() -> pd.DataFrame:
    path = RESULTS_DIR / "summary_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run aggregate_results.py first! Missing: {path}")
    return pd.read_csv(path)


def _load_costs() -> pd.DataFrame:
    path = RESULTS_DIR / "computational_costs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run aggregate_results.py first! Missing: {path}")
    return pd.read_csv(path)


def plot_mae_comparison(df: pd.DataFrame):
    """Bar chart: MAE per model, grouped by dataset, with error bars."""
    fig, ax = plt.subplots(figsize=(14, 7))
    datasets = df["dataset"].unique()
    models = df["model"].unique()
    n_datasets = len(datasets)
    n_models = len(models)
    bar_width = 0.8 / n_datasets
    x = np.arange(n_models)

    for i, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset].set_index("model").reindex(models)
        ax.bar(
            x + i * bar_width,
            subset["mae_mean"],
            bar_width,
            yerr=subset["mae_std"],
            label=dataset,
            capsize=3,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("MAE")
    ax.set_title("MAE Comparison Across Models and Datasets")
    ax.set_xticks(x + bar_width * (n_datasets - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(title="Dataset")
    plt.tight_layout()

    out = PLOTS_DIR / "mae_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_mase_comparison(df: pd.DataFrame):
    """Bar chart: MASE per model, grouped by dataset, with error bars."""
    fig, ax = plt.subplots(figsize=(14, 7))
    datasets = df["dataset"].unique()
    models = df["model"].unique()
    n_datasets = len(datasets)
    bar_width = 0.8 / n_datasets
    x = np.arange(len(models))

    for i, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset].set_index("model").reindex(models)
        ax.bar(
            x + i * bar_width,
            subset["mase_mean"],
            bar_width,
            yerr=subset["mase_std"],
            label=dataset,
            capsize=3,
        )

    # Reference line at MASE=1 (seasonal naive performance)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="MASE=1 (Naive)")
    ax.set_xlabel("Model")
    ax.set_ylabel("MASE")
    ax.set_title("MASE Comparison Across Models and Datasets")
    ax.set_xticks(x + bar_width * (n_datasets - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(title="Dataset")
    plt.tight_layout()

    out = PLOTS_DIR / "mase_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_computational_costs(costs: pd.DataFrame):
    """Bar chart: training time per model per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Training time
    pivot_time = costs.pivot(
        index="model", columns="dataset", values="avg_train_time_sec"
    )
    pivot_time.plot(kind="bar", ax=axes[0], rot=45)
    axes[0].set_ylabel("Avg Training Time (seconds)")
    axes[0].set_title("Training Time per Model")
    axes[0].legend(title="Dataset")

    # GPU memory
    pivot_gpu = costs.pivot(
        index="model", columns="dataset", values="peak_gpu_mb"
    )
    pivot_gpu.plot(kind="bar", ax=axes[1], rot=45)
    axes[1].set_ylabel("Peak GPU Memory (MB)")
    axes[1].set_title("Peak GPU Memory per Model")
    axes[1].legend(title="Dataset")

    plt.tight_layout()
    out = PLOTS_DIR / "computational_costs.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_model_ranking_heatmap(df: pd.DataFrame):
    """Heatmap: model ranks per dataset (lower rank = better)."""
    # Rank models within each dataset by MAE
    df_ranked = df.copy()
    df_ranked["rank"] = df_ranked.groupby("dataset")["mae_mean"].rank()

    pivot = df_ranked.pivot(index="model", columns="dataset", values="rank")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn_r",
        ax=ax,
        cbar_kws={"label": "Rank (1 = best)"},
    )
    ax.set_title("Model Rankings by MAE (lower = better)")
    plt.tight_layout()

    out = PLOTS_DIR / "model_ranking_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_dl_vs_baselines(df: pd.DataFrame):
    """Grouped comparison: DL models vs. baselines per dataset."""
    classical_baselines = {"SeasonalNaive", "AutoARIMA", "LightGBM"}
    linear_baseline = {"DLinear"}
    def _categorize(m):
        if m in classical_baselines:
            return "Classical Baseline"
        elif m in linear_baseline:
            return "Linear Baseline"
        return "Deep Learning"
    df["category"] = df["model"].apply(_categorize)

    fig, axes = plt.subplots(1, len(df["dataset"].unique()), figsize=(18, 6),
                              sharey=False)
    if len(df["dataset"].unique()) == 1:
        axes = [axes]

    color_map = {
        "Classical Baseline": "#2ecc71",
        "Linear Baseline": "#f39c12",
        "Deep Learning": "#3498db",
    }

    for ax, dataset in zip(axes, df["dataset"].unique()):
        subset = df[df["dataset"] == dataset].sort_values("mae_mean")
        colors = [color_map[cat] for cat in subset["category"]]
        ax.barh(subset["model"], subset["mae_mean"],
                xerr=subset["mae_std"], color=colors, capsize=3)
        ax.set_xlabel("MAE")
        ax.set_title(f"{dataset}")
        ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Classical Baseline"),
        Patch(facecolor="#f39c12", label="Linear Baseline"),
        Patch(facecolor="#3498db", label="Deep Learning"),
    ]
    fig.legend(handles=legend_elements, loc="upper right")
    fig.suptitle("MAE: Baselines vs. Deep Learning", fontsize=16, y=1.02)
    plt.tight_layout()

    out = PLOTS_DIR / "dl_vs_baselines.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating plots in {PLOTS_DIR}\n")

    summary = _load_summary()
    costs = _load_costs()

    plot_mae_comparison(summary)
    plot_mase_comparison(summary)
    plot_computational_costs(costs)
    plot_model_ranking_heatmap(summary)
    plot_dl_vs_baselines(summary)

    print(f"\n✓ All plots generated in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
