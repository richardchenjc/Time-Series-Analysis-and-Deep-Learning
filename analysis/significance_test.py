"""
Statistical significance testing for cross-model comparison.

Computes the Friedman test and Nemenyi post-hoc test from per-series
prediction errors, and produces a critical-difference (CD) diagram —
the standard visualization for comparing many forecasting methods
across many series in the literature.

How to read the CD diagram
--------------------------
The horizontal axis is "mean rank" (lower = better). Each model is
plotted at its mean rank position. Models whose mean ranks differ by
LESS than the critical difference (CD) value are statistically
indistinguishable at α=0.05; they are connected by a thick horizontal
bar to indicate they form a "clique."

  - Models in the same clique : NOT significantly different from each
    other. You cannot statistically claim one is better.
  - Models in DIFFERENT cliques: significantly different from each other.
  - A single model alone (no clique bar) is significantly different
    from every other model on the diagram.

The CD bar at the top of the diagram shows the width of the critical
difference threshold for visual reference.

Methodology
-----------
The Friedman test asks: "Are the model rankings across series
significantly different from random?" If p < 0.05 (it is, very
strongly, on all our datasets), the Nemenyi post-hoc test gives
pairwise comparisons via a critical difference threshold:

  CD = q_α × √(k(k+1) / (6n))

where k = number of models, n = number of series, and q_α is from
the studentized range distribution. We use Demšar (2006) Table 5
critical values.

Three CD diagrams produced, one per dataset:
  - results/plots/cd_diagram_M4.png
  - results/plots/cd_diagram_M5.png
  - results/plots/cd_diagram_Traffic.png

Plus a CSV per dataset with the rank table:
  - results/significance/<dataset>_ranks.csv

Usage:
    python analysis/significance_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from config import RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / "plots"
PRED_DIR = RESULTS_DIR / "predictions"
SIG_DIR = RESULTS_DIR / "significance"

# Studentized range distribution critical values for Nemenyi test at α=0.05
# Source: Demšar (2006), Table 5
_NEMENYI_Q_05 = {
    2: 1.960,  3: 2.343,  4: 2.569,  5: 2.728,  6: 2.850,
    7: 2.949,  8: 3.031,  9: 3.102, 10: 3.164,
}


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


def _build_rank_matrix(preds: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build a [n_series × n_models] matrix of MAE values, then rank
    within each row (series). Lower MAE = lower (better) rank.
    """
    preds["abs_err"] = (preds["y_true"] - preds["y_pred"]).abs()
    per_series = (
        preds.groupby(["unique_id", "model"])["abs_err"]
             .mean()
             .reset_index(name="mae")
    )
    matrix = per_series.pivot(index="unique_id", columns="model", values="mae")
    matrix = matrix.dropna()
    if matrix.empty:
        return matrix, []
    models = list(matrix.columns)
    rank_matrix = matrix.rank(axis=1, method="average")
    return rank_matrix, models


def _critical_difference(n_models: int, n_series: int, alpha: float = 0.05) -> float:
    if n_models not in _NEMENYI_Q_05:
        raise ValueError(f"No critical value for k={n_models}, only have {list(_NEMENYI_Q_05.keys())}")
    q = _NEMENYI_Q_05[n_models]
    return q * np.sqrt(n_models * (n_models + 1) / (6 * n_series))


def _find_cliques(sorted_models: list[str], sorted_ranks: list[float], cd: float) -> list[tuple[int, int]]:
    """Find maximal cliques of models whose pairwise rank distance ≤ CD.

    Standard Demšar cliques: a clique is a maximal set of consecutive
    (in rank order) models such that the FIRST and LAST member of the
    set are within CD of each other. We return (start_idx, end_idx)
    inclusive index pairs into sorted_models.

    A "clique" of size 1 (a model alone) is NOT returned — those models
    are significantly different from all others and don't get a bar drawn.
    """
    cliques = []
    n = len(sorted_ranks)
    i = 0
    while i < n:
        # Find the longest j such that sorted_ranks[j] - sorted_ranks[i] <= cd
        j = i
        while j + 1 < n and sorted_ranks[j + 1] - sorted_ranks[i] <= cd:
            j += 1
        if j > i:
            cliques.append((i, j))
        i += 1
    # Remove cliques that are subsets of larger cliques to keep only maximal
    maximal = []
    for c in cliques:
        if not any(other != c and other[0] <= c[0] and other[1] >= c[1] for other in cliques):
            maximal.append(c)
    return maximal


def _plot_cd_diagram(
    mean_ranks: pd.Series,
    cd: float,
    n_series: int,
    n_models: int,
    p_value: float,
    dataset_name: str,
):
    """Render a Demšar-style CD diagram.

    Layout:
        ┌────────────────────────────────────┐
        │  CD = X.XXX  ────                  │  ← CD reference bar
        │                                    │
        │  1 ─────┬───┬───┬───┬───┬─── k     │  ← rank axis with ticks
        │         |   |   |   |   |          │
        │       ──┴───┴──                    │  ← clique bars below axis
        │       Model A  Model B   ...       │  ← model labels below cliques
        └────────────────────────────────────┘
    """
    sorted_pairs = sorted(mean_ranks.items(), key=lambda x: x[1])
    sorted_models = [m for m, _ in sorted_pairs]
    sorted_ranks = [r for _, r in sorted_pairs]

    cliques = _find_cliques(sorted_models, sorted_ranks, cd)

    fig, ax = plt.subplots(figsize=(13, 4.5))

    # ── Rank axis at y=0 ──
    rank_min, rank_max = 1, n_models
    pad = 0.3
    ax.set_xlim(rank_min - pad, rank_max + pad)
    ax.set_ylim(-2.5, 2.0)

    # Draw the main axis line
    ax.plot([rank_min, rank_max], [0, 0], "k-", linewidth=1.5)
    # Tick marks
    for r in range(rank_min, rank_max + 1):
        ax.plot([r, r], [-0.05, 0.05], "k-", linewidth=1.2)
        ax.text(r, 0.18, str(r), ha="center", va="bottom", fontsize=10)

    # Axis label
    ax.text(
        (rank_min + rank_max) / 2, 0.7,
        "Mean rank (lower = better)",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )

    # ── CD reference bar in the upper-left area ──
    cd_y = 1.4
    cd_left = rank_min
    cd_right = rank_min + cd
    ax.plot([cd_left, cd_right], [cd_y, cd_y], "k-", linewidth=3)
    # Tick marks at the ends
    ax.plot([cd_left, cd_left], [cd_y - 0.1, cd_y + 0.1], "k-", linewidth=2)
    ax.plot([cd_right, cd_right], [cd_y - 0.1, cd_y + 0.1], "k-", linewidth=2)
    ax.text(
        (cd_left + cd_right) / 2, cd_y + 0.2,
        f"CD = {cd:.3f}",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

    # ── Model markers and labels ──
    # Models below the rank axis, alternating left/right tail to avoid label overlap
    half = (n_models + 1) // 2

    label_y_step = 0.32
    label_y_start = -0.6

    for idx, (model, r) in enumerate(zip(sorted_models, sorted_ranks)):
        if idx < half:
            label_y = label_y_start - idx * label_y_step
            label_x = rank_min - 0.05
            ax.plot([r, r], [0, label_y], "k-", linewidth=0.8)
            ax.plot([r, label_x], [label_y, label_y], "k-", linewidth=0.8)
            ax.text(
                label_x - 0.05, label_y, f"{model}  ({r:.2f})",
                ha="right", va="center", fontsize=10,
            )
        else:
            label_y = label_y_start - (n_models - 1 - idx) * label_y_step
            label_x = rank_max + 0.05
            ax.plot([r, r], [0, label_y], "k-", linewidth=0.8)
            ax.plot([r, label_x], [label_y, label_y], "k-", linewidth=0.8)
            ax.text(
                label_x + 0.05, label_y, f"({r:.2f})  {model}",
                ha="left", va="center", fontsize=10,
            )

    # ── Clique bars (horizontal lines connecting indistinguishable models) ──
    clique_y_base = -0.18
    clique_y_step = -0.10
    for c_idx, (start, end) in enumerate(cliques):
        clique_y = clique_y_base + c_idx * clique_y_step
        x_left = sorted_ranks[start] - 0.07
        x_right = sorted_ranks[end] + 0.07
        ax.plot([x_left, x_right], [clique_y, clique_y],
                "k-", linewidth=4.5, solid_capstyle="round")

    # ── Title with stats ──
    sig_label = "highly significant" if p_value < 0.001 else "significant" if p_value < 0.05 else "NOT significant"
    p_str = "p < 1e-30" if p_value < 1e-30 else f"p = {p_value:.3g}"
    title = (
        f"Critical Difference Diagram — {dataset_name}\n"
        f"Friedman {p_str}, n = {n_series} series, k = {n_models} models  ({sig_label})"
    )
    ax.set_title(title, fontsize=12, pad=15)

    # Hide all spines and ticks (we drew our own axis)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / f"cd_diagram_{dataset_name}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def analyze_dataset(dataset_name: str):
    print(f"\n  {dataset_name}: loading predictions...")
    preds = _load_predictions(dataset_name)
    if preds.empty:
        print(f"    No predictions for {dataset_name}, skipping")
        return

    rank_matrix, models = _build_rank_matrix(preds)
    if rank_matrix.empty or len(models) < 3:
        print(f"    Insufficient data for Friedman test on {dataset_name}")
        return

    n_series = len(rank_matrix)
    n_models = len(models)
    print(f"    {n_series} series × {n_models} models in rank matrix")

    statistic, p_value = stats.friedmanchisquare(*[rank_matrix[m] for m in models])
    mean_ranks = rank_matrix.mean(axis=0)

    print(f"    Friedman χ² = {statistic:.4f}, p = {p_value:.4g}")
    print(f"    Model ranks (lower = better):")
    for m, r in mean_ranks.sort_values().items():
        print(f"      {m:<14} {r:.3f}")

    SIG_DIR.mkdir(parents=True, exist_ok=True)
    rank_table = pd.DataFrame({
        "model": mean_ranks.index,
        "mean_rank": mean_ranks.values,
    }).sort_values("mean_rank")
    rank_table["friedman_p"] = p_value
    rank_table["friedman_stat"] = statistic
    rank_table["n_series"] = n_series
    out_csv = SIG_DIR / f"{dataset_name}_ranks.csv"
    rank_table.to_csv(out_csv, index=False)
    print(f"    Saved: {out_csv}")

    if n_models <= 10:
        cd = _critical_difference(n_models, n_series, alpha=0.05)
        print(f"    Critical difference (α=0.05): {cd:.4f}")

        # Print clique structure as text so you can verify the plot
        sorted_pairs = sorted(mean_ranks.items(), key=lambda x: x[1])
        sorted_models = [m for m, _ in sorted_pairs]
        sorted_ranks_list = [r for _, r in sorted_pairs]
        cliques = _find_cliques(sorted_models, sorted_ranks_list, cd)
        print(f"    Cliques (groups of statistically indistinguishable models):")
        if not cliques:
            print(f"      (no cliques — every model is significantly different)")
        else:
            for start, end in cliques:
                members = sorted_models[start : end + 1]
                ranks_in = [f"{r:.2f}" for r in sorted_ranks_list[start : end + 1]]
                print(f"      {{ {', '.join(members)} }}  ranks = [{', '.join(ranks_in)}]")
        # Also print models that are alone (not in any clique)
        in_any_clique = set()
        for s, e in cliques:
            for i in range(s, e + 1):
                in_any_clique.add(i)
        loners = [
            (sorted_models[i], sorted_ranks_list[i])
            for i in range(len(sorted_models))
            if i not in in_any_clique
        ]
        if loners:
            print(f"    Models significantly different from ALL others:")
            for m, r in loners:
                print(f"      {m} (rank {r:.2f})")

        _plot_cd_diagram(mean_ranks, cd, n_series, n_models, p_value, dataset_name)
    else:
        print(f"    Too many models ({n_models}) for our CD lookup table")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Statistical significance analysis → {SIG_DIR}")

    for dataset_name in ["M4", "M5", "Traffic"]:
        analyze_dataset(dataset_name)

    print("\nDone. Read the CD diagrams like this:")
    print("  - Each model is plotted at its mean rank on the horizontal axis")
    print("  - The 'CD = X.XXX' bar at the top shows the critical difference width")
    print("  - Models connected by a thick horizontal bar BELOW the axis are")
    print("    statistically indistinguishable (their ranks differ by < CD)")
    print("  - Models without a clique bar are significantly different from")
    print("    every other model in the diagram")
    print("  - Models on the LEFT have lower (better) ranks")


if __name__ == "__main__":
    main()
