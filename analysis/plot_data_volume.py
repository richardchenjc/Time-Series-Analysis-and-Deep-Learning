"""
Plot data-volume curves from the Night 2 + Night 3 sweep results.

Reads results/data_volume/{M4,M5,Traffic}_combined.csv and produces:
  - results/plots/data_volume_M4.png
  - results/plots/data_volume_M5.png
  - results/plots/data_volume_Traffic.png

Each plot has:
  - x-axis: n_series (log-scale)
  - y-axis: MAE
  - one line per model (auto-detected from the CSV), with ±1 std band
  - annotated n values at each marker

The model list is auto-detected from whatever's in the CSV, so this
script works for any subset of models in any combination. The style
dictionary provides canonical colors/markers for known models, and
unknown models fall back to a default grey-cross style.

Usage:
    python analysis/plot_data_volume.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
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
SWEEP_DIR = RESULTS_DIR / "data_volume"

# Visual style — distinct color and marker per model.
# Any model not listed here gets a default grey style at plot time.
_MODEL_STYLE = {
    "NBEATS":        {"color": "#2e7d32", "marker": "o", "label": "N-BEATS (DL)"},
    "PatchTST":      {"color": "#1565c0", "marker": "s", "label": "PatchTST (DL)"},
    "DLinear":       {"color": "#ef6c00", "marker": "^", "label": "DLinear (Linear)"},
    "AutoARIMA":     {"color": "#6a1b9a", "marker": "D", "label": "AutoARIMA (Classical)"},
    "SeasonalNaive": {"color": "#546e7a", "marker": "x", "label": "SeasonalNaive (Classical)"},
    "TimesNet":      {"color": "#3949ab", "marker": "v", "label": "TimesNet (DL)"},
    "DeepAR":        {"color": "#7b1fa2", "marker": "P", "label": "DeepAR (DL)"},
    "TiDE":          {"color": "#5e35b1", "marker": "h", "label": "TiDE (DL)"},
    "LightGBM":      {"color": "#66bb6a", "marker": "*", "label": "LightGBM (Classical)"},
}

# Preferred ordering for legend: classical first, linear second, DL third.
# Any model not in this list appears at the end in alphabetical order.
_PREFERRED_ORDER = [
    "SeasonalNaive", "AutoARIMA", "LightGBM",      # Classical
    "DLinear",                                       # Linear
    "NBEATS", "PatchTST", "TiDE", "DeepAR", "TimesNet",  # DL
]


def _order_models(available: list[str]) -> list[str]:
    """Return available models sorted by the preferred order."""
    in_order = [m for m in _PREFERRED_ORDER if m in available]
    extras = sorted([m for m in available if m not in _PREFERRED_ORDER])
    return in_order + extras


def _plot_one_dataset(dataset_name: str, csv_path: Path) -> Path | None:
    if not csv_path.exists():
        print(f"  Skipping {dataset_name}: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"  Skipping {dataset_name}: empty CSV")
        return None

    # Aggregate: mean and std across (seed, window) per (model, n_series)
    agg = (
        df.groupby(["model", "n_series"])
          .agg(mae_mean=("mae_mean", "mean"),
               mae_std=("mae_mean", "std"))
          .reset_index()
    )
    agg["mae_std"] = agg["mae_std"].fillna(0)

    # Auto-detect models — works for any combination in the CSV
    available_models = agg["model"].unique().tolist()
    model_order = _order_models(available_models)
    print(f"  {dataset_name}: plotting {len(model_order)} models: {model_order}")

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Plot one line per model in the preferred order
    for model in model_order:
        sub = agg[agg["model"] == model].sort_values("n_series")
        if sub.empty:
            continue
        style = _MODEL_STYLE.get(
            model,
            {"color": "gray", "marker": "x", "label": f"{model}"},
        )
        ax.plot(
            sub["n_series"], sub["mae_mean"],
            color=style["color"],
            marker=style["marker"],
            markersize=8,
            linewidth=2,
            label=style["label"],
        )
        # Shaded ±1 std band — only if std > 0 (deterministic models have 0 std)
        if (sub["mae_std"] > 0).any():
            ax.fill_between(
                sub["n_series"],
                sub["mae_mean"] - sub["mae_std"],
                sub["mae_mean"] + sub["mae_std"],
                color=style["color"],
                alpha=0.15,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Number of training series (log scale)")
    ax.set_ylabel("MAE")
    ax.set_title(f"Data-Volume Curve: {dataset_name}")
    ax.legend(title="Model", loc="best", fontsize=9, ncol=2 if len(model_order) > 4 else 1)
    ax.grid(True, alpha=0.4)

    # Annotate each marker with n value — but only for the first model
    # to avoid visual clutter (the x-axis ticks carry the same info)
    first_model = model_order[0] if model_order else None
    if first_model:
        sub_first = agg[agg["model"] == first_model].sort_values("n_series")
        for _, row in sub_first.iterrows():
            ax.annotate(
                f"n={int(row['n_series'])}",
                (row["n_series"], row["mae_mean"]),
                textcoords="offset points",
                xytext=(5, 7),
                fontsize=8,
                alpha=0.7,
            )

    plt.tight_layout()
    out = PLOTS_DIR / f"data_volume_{dataset_name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Plotting data-volume curves into {PLOTS_DIR}\n")

    # All three datasets — auto-detect models in each CSV
    for ds in ["M4", "M5", "Traffic"]:
        csv_path = SWEEP_DIR / f"{ds}_combined.csv"
        _plot_one_dataset(ds, csv_path)

    print("\nDone. Read the curves like this:")
    print("  - Steeper downward slope = model benefits more from extra data")
    print("  - Flat curve = model performance is data-volume-insensitive")
    print("  - Lines crossing at small n = a known 'linear beats DL at small n' signal")
    print("  - All trained models worsening at large n = likely training-budget effect")
    print("    (compare against deterministic SeasonalNaive curve to diagnose)")


if __name__ == "__main__":
    main()
