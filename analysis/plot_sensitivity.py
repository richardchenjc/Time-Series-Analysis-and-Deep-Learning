"""
Plot HP sensitivity results from Night 2.

Reads results/hp_sensitivity/<study>/_combined.csv for each of the four
sensitivity studies and produces one plot per study showing how MAE
changes as the swept hyperparameter varies.

Each plot is a simple bar/line chart with error bars across (seed, window).

Outputs to results/plots/sensitivity_<study>.png

Studies:
  1. patchtst_patch_len_m4   — does patch_len ∈ {3,6,12} matter?
  2. patchtst_lookback_m4    — does lookback ∈ {18,36,72,144} matter?
  3. nbeats_n_blocks_m4      — does depth ∈ {[1,1],[3,3],[5,5]} matter?
  4. dlinear_lookback_traffic — does lookback ∈ {24,48,96,168} matter?

Also generates a single combined "robustness summary" plot showing
the relative MAE change (% from baseline) for each sweep, as a quick
visual answer to "how sensitive is each model to its hyperparameter?"

Usage:
    python analysis/plot_sensitivity.py
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
HP_DIR = RESULTS_DIR / "hp_sensitivity"

# Study metadata: (study folder name, x-axis label, x-axis sort key extractor)
STUDIES = [
    {
        "name": "patchtst_patch_len_m4",
        "title": "PatchTST × patch_len on M4",
        "xlabel": "patch_len (with stride)",
        "model": "PatchTST",
        "sort_key": lambda v: int(v.split("=")[1].split(" ")[0]),  # "patch_len=3 stride=2" → 3
    },
    {
        "name": "patchtst_lookback_m4",
        "title": "PatchTST × lookback on M4",
        "xlabel": "input_size (lookback length)",
        "model": "PatchTST",
        "sort_key": lambda v: int(v.split("=")[1]),  # "lookback=72" → 72
    },
    {
        "name": "nbeats_n_blocks_m4",
        "title": "N-BEATS × n_blocks on M4",
        "xlabel": "n_blocks per stack",
        "model": "NBEATS",
        "sort_key": lambda v: sum(eval(v.split("=")[1])),  # "n_blocks=[3,3]" → 6
    },
    {
        "name": "dlinear_lookback_traffic",
        "title": "DLinear × lookback on Traffic",
        "xlabel": "input_size (lookback length)",
        "model": "DLinear",
        "sort_key": lambda v: int(v.split("=")[1]),  # "lookback=96" → 96
    },
]


def _aggregate_study(study_path: Path) -> pd.DataFrame:
    """Aggregate one study's _combined.csv to per-value mean ± std."""
    if not study_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(study_path)
    if df.empty:
        return df
    agg = (
        df.groupby("value")
          .agg(mae_mean=("mae_mean", "mean"),
               mae_std=("mae_mean", "std"),
               mase_median=("mase_median", "mean"),
               n_runs=("mae_mean", "count"))
          .reset_index()
    )
    agg["mae_std"] = agg["mae_std"].fillna(0)
    return agg


def _plot_one_study(study_meta: dict) -> Path | None:
    study_path = HP_DIR / study_meta["name"] / "_combined.csv"
    agg = _aggregate_study(study_path)
    if agg.empty:
        print(f"Skipping {study_meta['name']}: no data")
        return None

    # Sort by the swept hyperparameter value
    try:
        agg["_sort_key"] = agg["value"].apply(study_meta["sort_key"])
        agg = agg.sort_values("_sort_key").reset_index(drop=True)
    except Exception as e:
        print(f"Sort failed for {study_meta['name']}: {e} — using as-is")

    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(len(agg))
    bar_color = sns.color_palette("Set2")[0]

    bars = ax.bar(
        x_positions,
        agg["mae_mean"],
        yerr=agg["mae_std"],
        color=bar_color,
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
    )

    # Annotate each bar with the actual MAE
    for x, mae in zip(x_positions, agg["mae_mean"]):
        ax.annotate(
            f"{mae:.2f}",
            (x, mae),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )

    # Compute relative spread for the title — "robustness" indicator
    mae_min, mae_max = agg["mae_mean"].min(), agg["mae_mean"].max()
    spread_pct = 100 * (mae_max - mae_min) / mae_min
    robustness = "robust" if spread_pct < 5 else "moderate" if spread_pct < 15 else "sensitive"

    ax.set_xticks(x_positions)
    ax.set_xticklabels(agg["value"], rotation=20, ha="right", fontsize=9)
    ax.set_xlabel(study_meta["xlabel"])
    ax.set_ylabel("MAE")
    ax.set_title(f"{study_meta['title']}\n(spread: {spread_pct:.1f}% — {robustness})")
    ax.grid(True, alpha=0.4, axis="y")

    plt.tight_layout()
    out = PLOTS_DIR / f"sensitivity_{study_meta['name']}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def _plot_robustness_summary():
    """One combined bar chart: % MAE spread for each sweep."""
    spreads = []
    for study in STUDIES:
        agg = _aggregate_study(HP_DIR / study["name"] / "_combined.csv")
        if agg.empty:
            continue
        mae_min, mae_max = agg["mae_mean"].min(), agg["mae_mean"].max()
        spread_pct = 100 * (mae_max - mae_min) / mae_min
        spreads.append({
            "study": study["title"],
            "spread_pct": spread_pct,
        })

    if not spreads:
        print("No data for robustness summary")
        return None

    summary = pd.DataFrame(spreads).sort_values("spread_pct")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2e7d32" if s < 5 else "#fb8c00" if s < 15 else "#c62828"
              for s in summary["spread_pct"]]
    bars = ax.barh(summary["study"], summary["spread_pct"], color=colors,
                   edgecolor="black", linewidth=0.5)

    for bar, spread in zip(bars, summary["spread_pct"]):
        ax.annotate(
            f"{spread:.1f}%",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            ha="left",
            va="center",
            fontsize=10,
            xytext=(3, 0),
            textcoords="offset points",
        )

    ax.axvline(x=5, color="green", linestyle="--", alpha=0.5, label="robust < 5%")
    ax.axvline(x=15, color="orange", linestyle="--", alpha=0.5, label="moderate < 15%")
    ax.set_xlabel("MAE spread across sweep values (%)")
    ax.set_title("Hyperparameter Sensitivity Summary\n(lower = more robust to HP choice)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.4, axis="x")
    plt.tight_layout()

    out = PLOTS_DIR / "sensitivity_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Plotting HP sensitivity into {PLOTS_DIR}\n")

    for study in STUDIES:
        _plot_one_study(study)

    print()
    _plot_robustness_summary()

    print("\nDone. Read the bars like this:")
    print("  - Tall bars with similar heights = model robust to that HP")
    print("  - Wildly different bar heights = model sensitive to that HP")
    print("  - PatchTST sweeps should be flat (paper claim: insensitive to patch_len)")
    print("  - DLinear lookback sweep should DROP with longer lookback (Zeng et al. 2023)")


if __name__ == "__main__":
    main()
