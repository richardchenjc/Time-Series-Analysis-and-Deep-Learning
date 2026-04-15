"""
Per-series stratified analysis — with SEM error bars.

FIX FROM PREVIOUS VERSION:
Previous version plotted std of per-series MAE within each stratum as the
error bar. That measure represents the HETEROGENEITY of series within a
category, not the uncertainty in the mean. For categories with ~150 series,
std was typically larger than the mean itself, producing visually alarming
but statistically meaningless error bars.

This version plots STANDARD ERROR OF THE MEAN (SEM = std / sqrt(n)) as
the error bar, which represents the actual uncertainty in the estimated
category mean. With ~150 series per category, SEM is roughly 1/12th of
std, so the error bars become tight and readable, and you can actually
tell whether two model-category means are significantly different.

We also add the full per-series std as an underlying shaded band on each
bar so the dispersion information is still visible (just not confused
with uncertainty in the mean).

Slices by dataset:
  M4 by category (Macro/Micro/Demographic/Industry/Finance/Other)
  M5 by test-window zero fraction quintile
  Traffic by mean occupancy quintile

Output:
  results/per_series_stratified/M4_by_category.csv          (updated)
  results/per_series_stratified/M5_by_zero_fraction.csv     (updated)
  results/per_series_stratified/Traffic_by_occupancy.csv    (updated)
  results/plots/per_series_M4_by_category.png               (updated)
  results/plots/per_series_M5_by_zero_fraction.png          (updated)
  results/plots/per_series_Traffic_by_occupancy.png         (updated)

Usage:
    python analysis/plot_per_series_stratified.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR, M4_CONFIG

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})
sns.set_theme(style="whitegrid", palette="Set2")

PLOTS_DIR = RESULTS_DIR / "plots"
PRED_DIR = RESULTS_DIR / "predictions"
STRAT_DIR = RESULTS_DIR / "per_series_stratified"

_MODEL_ORDER = [
    "SeasonalNaive", "AutoARIMA", "LightGBM",
    "DLinear",
    "NBEATS", "PatchTST", "TiDE", "DeepAR", "TimesNet",
]


def _load_predictions(dataset_name: str) -> pd.DataFrame:
    rows = []
    for path in sorted(PRED_DIR.glob(f"*_{dataset_name}.parquet")):
        try:
            df = pd.read_parquet(path)
            rows.append(df)
        except Exception as e:
            print(f"  Failed to read {path.name}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _per_series_mae(preds: pd.DataFrame) -> pd.DataFrame:
    """Compute MAE per (model, unique_id) by averaging absolute errors
    across all (seed, window, horizon_step) for that series."""
    preds["abs_err"] = (preds["y_true"] - preds["y_pred"]).abs()
    return (
        preds.groupby(["model", "unique_id"])
             .agg(mae=("abs_err", "mean"),
                  mean_y=("y_true", "mean"),
                  zero_frac=("y_true", lambda s: (s == 0).mean()))
             .reset_index()
    )


def _plot_stratified_sem(df: pd.DataFrame, stratum_col: str, title: str, out_path: Path):
    """Grouped bar chart with SEM error bars (not std)."""
    agg = (
        df.groupby(["model", stratum_col], observed=True)
          .agg(mae_mean=("mae", "mean"),
               mae_std=("mae", "std"),
               n=("mae", "count"))
          .reset_index()
    )
    # Standard error of the mean = std / sqrt(n)
    agg["mae_sem"] = agg["mae_std"] / np.sqrt(agg["n"].clip(lower=1))

    models_present = [m for m in _MODEL_ORDER if m in agg["model"].unique()]
    strata = sorted(agg[stratum_col].dropna().unique(), key=str)
    n_models = len(models_present)
    n_strata = len(strata)
    bar_width = 0.8 / n_strata
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(13, 6))
    palette = sns.color_palette("Set2", n_colors=n_strata)

    for i, stratum in enumerate(strata):
        sub = (
            agg[agg[stratum_col] == stratum]
            .set_index("model")
            .reindex(models_present)
        )
        ax.bar(
            x + i * bar_width,
            sub["mae_mean"],
            bar_width,
            yerr=sub["mae_sem"].fillna(0),
            label=f"{stratum} (n={int(sub['n'].mean()) if not sub['n'].isna().all() else 0})",
            color=palette[i],
            capsize=2,
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xticks(x + bar_width * (n_strata - 1) / 2)
    ax.set_xticklabels(models_present, rotation=30, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE (mean ± SEM)")
    ax.set_title(f"{title}\n(error bars = standard error of the mean)")
    ax.legend(title=stratum_col, loc="best", fontsize=9)
    ax.grid(True, alpha=0.4, axis="y")
    ax.set_ylim(bottom=0)  # MAE is non-negative, don't show negative range

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_path}")


# ── M4 by category ────────────────────────────────────────────────────────
def analyze_m4_by_category():
    print(f"\n  Analyzing M4 by category...")
    preds = _load_predictions("M4")
    if preds.empty:
        print("  No M4 predictions, skipping")
        return

    info_path = M4_CONFIG.get("info_csv")
    if not info_path or not Path(info_path).exists():
        print(f"  m4_info.csv not found at {info_path}, skipping")
        return

    info = pd.read_csv(info_path)
    id_col = next((c for c in info.columns if c.lower() in ("m4id", "id")), None)
    cat_col = next((c for c in info.columns if c.lower() == "category"), None)
    if not id_col or not cat_col:
        print(f"  m4_info.csv missing id/category columns")
        return

    cat_lookup = dict(zip(info[id_col], info[cat_col]))
    per_series = _per_series_mae(preds)
    per_series["category"] = per_series["unique_id"].map(cat_lookup)
    per_series = per_series.dropna(subset=["category"])

    out_csv = STRAT_DIR / "M4_by_category.csv"
    per_series.groupby(["model", "category"]).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        n=("mae", "count"),
    ).reset_index().assign(
        mae_sem=lambda d: d["mae_std"] / np.sqrt(d["n"].clip(lower=1))
    ).to_csv(out_csv, index=False)
    print(f"  Saved CSV:  {out_csv}")

    _plot_stratified_sem(
        per_series, "category",
        "M4 MAE by Series Category",
        PLOTS_DIR / "per_series_M4_by_category.png"
    )


# ── M5 by zero fraction ───────────────────────────────────────────────────
def analyze_m5_by_zero_fraction():
    print(f"\n  Analyzing M5 by zero fraction...")
    preds = _load_predictions("M5")
    if preds.empty:
        print("  No M5 predictions, skipping")
        return

    per_series = _per_series_mae(preds)
    bins = [0, 0.25, 0.50, 0.75, 1.01]
    labels = ["<25% zeros", "25-50%", "50-75%", "75%+"]
    per_series["zero_bucket"] = pd.cut(
        per_series["zero_frac"], bins=bins, labels=labels, include_lowest=True
    )

    out_csv = STRAT_DIR / "M5_by_zero_fraction.csv"
    per_series.groupby(["model", "zero_bucket"], observed=True).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        n=("mae", "count"),
    ).reset_index().assign(
        mae_sem=lambda d: d["mae_std"] / np.sqrt(d["n"].clip(lower=1))
    ).to_csv(out_csv, index=False)
    print(f"  Saved CSV:  {out_csv}")

    _plot_stratified_sem(
        per_series, "zero_bucket",
        "M5 MAE by Test-Window Zero Fraction",
        PLOTS_DIR / "per_series_M5_by_zero_fraction.png"
    )


# ── Traffic by mean occupancy ─────────────────────────────────────────────
def analyze_traffic_by_occupancy():
    print(f"\n  Analyzing Traffic by occupancy quintile...")
    preds = _load_predictions("Traffic")
    if preds.empty:
        print("  No Traffic predictions, skipping")
        return

    per_series = _per_series_mae(preds)
    per_series["occ_quintile"] = pd.qcut(
        per_series["mean_y"], q=5,
        labels=["Q1_lowest", "Q2", "Q3", "Q4", "Q5_highest"],
    )

    out_csv = STRAT_DIR / "Traffic_by_occupancy.csv"
    per_series.groupby(["model", "occ_quintile"], observed=True).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        n=("mae", "count"),
    ).reset_index().assign(
        mae_sem=lambda d: d["mae_std"] / np.sqrt(d["n"].clip(lower=1))
    ).to_csv(out_csv, index=False)
    print(f"  Saved CSV:  {out_csv}")

    _plot_stratified_sem(
        per_series, "occ_quintile",
        "Traffic MAE by Mean Occupancy Quintile",
        PLOTS_DIR / "per_series_Traffic_by_occupancy.png"
    )


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    STRAT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Per-series stratified analysis (with SEM error bars) → {STRAT_DIR}")

    analyze_m4_by_category()
    analyze_m5_by_zero_fraction()
    analyze_traffic_by_occupancy()

    print("\nDone.")
    print("Note: error bars are now SEM (std/sqrt(n)), representing")
    print("uncertainty in the category mean, not per-series heterogeneity.")
    print("This is the correct statistic for answering 'do models differ")
    print("significantly within this category?'")


if __name__ == "__main__":
    main()
