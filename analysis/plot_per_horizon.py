"""
Per-horizon MAE decomposition — with highlight/fade design.

Previous version drew all 9 models as fully-colored lines, producing a
densely overlapping plot that was hard to read. This version highlights
2-3 "story" models per dataset in bold colors with markers, and draws
the remaining 6-7 models in faded grey for visual context.

Highlights are chosen to match each dataset's report narrative:
  M4:      NBEATS + AutoARIMA (the tied top pair) + TiDE (clear worst)
  M5:      NBEATS (unique winner) + LightGBM (worst classical) + SeasonalNaive (worst overall)
  Traffic: DLinear (unique winner) + LightGBM (second) + DeepAR (worst)

The greyed-out models still appear in the legend (so readers know what's
there) but take visual background, making the bold lines dominant.

Also adds an annotation for the "best model at h=1" to call out the
short-horizon finding directly on the plot.

Output:
  results/plots/per_horizon_M4.png       (updated)
  results/plots/per_horizon_M5.png       (updated)
  results/plots/per_horizon_Traffic.png  (updated)

Usage:
    python analysis/plot_per_horizon.py
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
PRED_DIR = RESULTS_DIR / "predictions"

# Per-dataset highlighted models and their colors.
# Everything else fades to a light grey background.
_DATASET_HIGHLIGHTS = {
    "M4": {
        "NBEATS":     {"color": "#1565c0", "marker": "o", "label": "N-BEATS (DL, best)"},
        "AutoARIMA":  {"color": "#2e7d32", "marker": "s", "label": "AutoARIMA (classical, tied best)"},
        "TiDE":       {"color": "#c62828", "marker": "v", "label": "TiDE (DL, worst)"},
    },
    "M5": {
        "NBEATS":        {"color": "#1565c0", "marker": "o", "label": "N-BEATS (DL, unique best)"},
        "LightGBM":      {"color": "#ef6c00", "marker": "D", "label": "LightGBM (ML baseline)"},
        "SeasonalNaive": {"color": "#c62828", "marker": "v", "label": "SeasonalNaive (worst)"},
    },
    "Traffic": {
        "DLinear":  {"color": "#ef6c00", "marker": "^", "label": "DLinear (linear, unique best)"},
        "LightGBM": {"color": "#2e7d32", "marker": "D", "label": "LightGBM (ML, second)"},
        "DeepAR":   {"color": "#c62828", "marker": "v", "label": "DeepAR (worst)"},
    },
}

# Stable model order used for legend consistency
_ALL_MODELS = [
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


def _plot_one_dataset(dataset_name: str, horizon: int):
    print(f"\n  Plotting {dataset_name}...")
    df = _load_predictions(dataset_name)
    if df.empty:
        print(f"    No prediction data, skipping")
        return

    df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
    per_horizon = (
        df.groupby(["model", "horizon_step"])["abs_err"]
          .mean()
          .reset_index()
          .rename(columns={"abs_err": "mae"})
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    highlighted = _DATASET_HIGHLIGHTS.get(dataset_name, {})
    highlight_names = set(highlighted.keys())

    # ── Draw faded context lines first (behind) ────────────────────────
    for model in _ALL_MODELS:
        if model in highlight_names:
            continue
        sub = per_horizon[per_horizon["model"] == model].sort_values("horizon_step")
        if sub.empty:
            continue
        ax.plot(
            sub["horizon_step"], sub["mae"],
            color="#b0b0b0",
            linewidth=1.2,
            alpha=0.6,
            zorder=1,
            label=f"{model} (context)",
        )

    # ── Draw highlighted lines on top ──────────────────────────────────
    for model, style in highlighted.items():
        sub = per_horizon[per_horizon["model"] == model].sort_values("horizon_step")
        if sub.empty:
            continue
        ax.plot(
            sub["horizon_step"], sub["mae"],
            color=style["color"],
            linewidth=2.5,
            marker=style["marker"],
            markersize=6,
            zorder=3,
            label=style["label"],
        )

    # ── Annotate best model at h=1 ────────────────────────────────────
    at_h1 = per_horizon[per_horizon["horizon_step"] == 1].sort_values("mae")
    if not at_h1.empty:
        best_h1 = at_h1.iloc[0]
        best_h1_model = best_h1["model"]
        best_h1_mae = best_h1["mae"]
        # Check if there's a near-tie at h=1 (within 5% of the best)
        at_h1_tied = at_h1[at_h1["mae"] <= best_h1_mae * 1.05]
        if len(at_h1_tied) >= 2:
            tied_names = ", ".join(at_h1_tied["model"].tolist()[:2])
            annotation = f"h=1 best: {tied_names}\nMAE ≈ {best_h1_mae:.3f}"
        else:
            annotation = f"h=1 best: {best_h1_model}\nMAE = {best_h1_mae:.3f}"
        ax.annotate(
            annotation,
            xy=(1, best_h1_mae),
            xytext=(horizon * 0.15, best_h1_mae),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="black", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )

    # ── Best model at h=H annotation ──────────────────────────────────
    at_hH = per_horizon[per_horizon["horizon_step"] == horizon].sort_values("mae")
    if not at_hH.empty:
        best_hH = at_hH.iloc[0]
        best_hH_model = best_hH["model"]
        best_hH_mae = best_hH["mae"]
        at_hH_tied = at_hH[at_hH["mae"] <= best_hH_mae * 1.05]
        if len(at_hH_tied) >= 2:
            tied_names = ", ".join(at_hH_tied["model"].tolist()[:2])
            annotation_hH = f"h={horizon} best: {tied_names}\nMAE ≈ {best_hH_mae:.3f}"
        else:
            annotation_hH = f"h={horizon} best: {best_hH_model}\nMAE = {best_hH_mae:.3f}"
        ax.annotate(
            annotation_hH,
            xy=(horizon, best_hH_mae),
            xytext=(horizon * 0.55, best_hH_mae * 1.1),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="black", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )

    ax.set_xlabel("Forecast horizon step (h)")
    ax.set_ylabel("MAE")
    ax.set_title(
        f"Per-Horizon MAE Decomposition: {dataset_name}\n"
        f"(Highlighted: key story models. Grey: all other models for context.)"
    )
    # Two-column legend; put highlighted ones first
    handles, labels = ax.get_legend_handles_labels()
    # Reorder: highlighted first (bold colored), context after (grey)
    highlight_entries = [
        (h, l) for h, l in zip(handles, labels)
        if "(context)" not in l
    ]
    context_entries = [
        (h, l) for h, l in zip(handles, labels)
        if "(context)" in l
    ]
    ordered = highlight_entries + context_entries
    if ordered:
        h_ordered, l_ordered = zip(*ordered)
        ax.legend(
            h_ordered, l_ordered,
            loc="upper left", fontsize=9, ncol=2, framealpha=0.85,
        )
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0.5, horizon + 0.5)

    plt.tight_layout()
    out = PLOTS_DIR / f"per_horizon_{dataset_name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Plotting per-horizon MAE (highlighted) into {PLOTS_DIR}")

    DATASETS_HORIZONS = {
        "M4":      18,
        "M5":      28,
        "Traffic": 24,
    }

    for ds, h in DATASETS_HORIZONS.items():
        _plot_one_dataset(ds, h)

    print("\nDone.")
    print("Visual guide:")
    print("  - Bold colored lines = story models for that dataset")
    print("  - Grey background lines = all other models for context")
    print("  - White annotations = best model at h=1 and h=H")


if __name__ == "__main__":
    main()
