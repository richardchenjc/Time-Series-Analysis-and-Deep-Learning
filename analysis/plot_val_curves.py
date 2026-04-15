"""
Learning curves — VALIDATION LOSS ONLY version.

Reads the CSVs already produced by plot_learning_curves.py and replots
showing only the validation loss curve per (model, dataset) cell.
This removes the train/val scale mismatch confusion caused by
NeuralForecast's internal per-series normalization.

Outputs (separate from the train+val originals):
  - results/plots/val_curve_<Model>_<Dataset>.png   (individual)
  - results/plots/val_curves_grid.png               (7×3 combined)

No retraining needed — pure re-plotting from existing CSVs.
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
if (_THIS.parent / "config.py").exists():
    PROJECT_ROOT = _THIS.parent
elif (_THIS.parent.parent / "config.py").exists():
    PROJECT_ROOT = _THIS.parent.parent
else:
    PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR, MAX_STEPS

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

LC_DIR = RESULTS_DIR / "learning_curves"
PLOTS_DIR = RESULTS_DIR / "plots"

MODEL_ORDER = ["DLinear", "LightGBM", "TiDE", "NBEATS", "PatchTST", "DeepAR", "TimesNet"]
DS_ORDER = ["M4", "M5", "Traffic"]

# Colors per model for individual plots
_COLORS = {
    "DLinear": "#e67e22", "LightGBM": "#27ae60", "TiDE": "#8e44ad",
    "NBEATS": "#2980b9", "PatchTST": "#c0392b", "DeepAR": "#16a085",
    "TimesNet": "#d35400",
}

# ── Convergence diagnosis ───────────────────────────────────────────

def _diagnose(val_steps, val_vals, max_steps, is_deepar=False):
    """Return a short diagnosis string for the val curve shape."""
    if len(val_vals) < 2:
        return "insufficient data"

    last_step = val_steps[-1]
    early_stopped = last_step < max_steps * 0.9

    # Trend: compare first third vs last third of val checks
    n = len(val_vals)
    first_third = np.mean(val_vals[:max(1, n // 3)])
    last_third = np.mean(val_vals[-max(1, n // 3):])

    if is_deepar:
        # For NLL: more negative = better, so "improving" means val going down
        change = (last_third - first_third) / (abs(first_third) + 1e-8)
    else:
        change = (last_third - first_third) / (abs(first_third) + 1e-8)

    if early_stopped:
        if change > 0.05:
            return f"Early stop @ {last_step} (rising val = overfitting)"
        else:
            return f"Early stop @ {last_step} (converged)"
    else:
        if change > 0.1:
            return "Ran full budget, val rising (overfit risk)"
        elif change < -0.05:
            return "Ran full budget, still improving"
        else:
            return "Ran full budget, val plateaued"


# ── Individual val-only plots ───────────────────────────────────────

def _plot_single_val(df, model_name, ds_name):
    """Plot validation loss only for one (model, dataset)."""
    if df is None or len(df) == 0:
        return False

    is_lgbm = model_name == "LightGBM"
    is_deepar = model_name == "DeepAR"

    # Extract val points
    val_rows = df[df["val_loss"].notna()]
    if len(val_rows) == 0:
        return False

    steps = val_rows["step"].values
    vals = val_rows["val_loss"].values
    color = _COLORS.get(model_name, "tab:blue")

    fig, ax = plt.subplots(figsize=(6, 4))

    if is_lgbm:
        # LightGBM has val at every boosting round — plot as line
        ax.plot(steps, vals, color=color, linewidth=1.5)
        ax.set_xlabel("Boosting Round", fontsize=11)
    else:
        # Neural models have val at every val_check_steps — plot with markers
        ax.plot(steps, vals, color=color, linewidth=2, marker="o",
                markersize=5, markeredgecolor="white", markeredgewidth=0.5)
        ax.set_xlabel("Training Step", fontsize=11)

        # Mark early stopping
        last_step = int(df["step"].max())
        if last_step < MAX_STEPS * 0.9:
            ax.axvline(last_step, color="red", linestyle="--", alpha=0.6, linewidth=1)
            y_pos = ax.get_ylim()[0] + 0.85 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.annotate(f"Early stop\n(step {last_step})",
                        xy=(last_step, y_pos), fontsize=8, color="red",
                        alpha=0.8, ha="right",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.5))

    # Diagnosis annotation
    diag = _diagnose(steps, vals, MAX_STEPS, is_deepar)
    ax.annotate(diag, xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=8, color="gray", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.7))

    loss_label = "Val Loss (NLL)" if is_deepar else "Val Loss (MAE)"
    ax.set_ylabel(loss_label, fontsize=11)
    ax.set_title(f"{model_name} — {ds_name}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = PLOTS_DIR / f"val_curve_{model_name}_{ds_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


# ── Combined 7×3 grid (val only) ───────────────────────────────────

def _plot_val_grid(all_data):
    """7×3 grid showing only validation loss curves."""

    n_rows = len(MODEL_ORDER)
    n_cols = len(DS_ORDER)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 22), squeeze=False)
    fig.suptitle("Validation Loss Curves by Model and Dataset",
                 fontsize=16, fontweight="bold", y=0.998)

    for i, mn in enumerate(MODEL_ORDER):
        for j, dn in enumerate(DS_ORDER):
            ax = axes[i][j]
            key = (mn, dn)
            df = all_data.get(key)
            is_lgbm = mn == "LightGBM"
            is_deepar = mn == "DeepAR"
            color = _COLORS.get(mn, "tab:blue")

            if df is not None and len(df) > 0:
                val_rows = df[df["val_loss"].notna()]
                if len(val_rows) > 0:
                    steps = val_rows["step"].values
                    vals = val_rows["val_loss"].values

                    if is_lgbm:
                        ax.plot(steps, vals, color=color, linewidth=1.2)
                    else:
                        ax.plot(steps, vals, color=color, linewidth=1.8,
                                marker="o", markersize=3,
                                markeredgecolor="white", markeredgewidth=0.3)

                        # Early stopping marker
                        last_step = int(df["step"].max())
                        if last_step < MAX_STEPS * 0.9:
                            ax.axvline(last_step, color="red", linestyle="--",
                                       alpha=0.5, linewidth=0.8)

                    # Diagnosis text
                    diag = _diagnose(steps, vals, MAX_STEPS, is_deepar)
                    ax.annotate(diag, xy=(0.03, 0.05), xycoords="axes fraction",
                                fontsize=6, color="gray", style="italic",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                          ec="gray", alpha=0.6))
                else:
                    ax.text(0.5, 0.5, "No val data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=9, color="gray")
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="gray")

            # Labels
            if i == 0:
                ax.set_title(dn, fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(mn, fontsize=10, fontweight="bold")
            if i == n_rows - 1:
                ax.set_xlabel("Round" if is_lgbm else "Step", fontsize=8)

            # Y-axis label for DeepAR
            if is_deepar and j == 0:
                ax.set_ylabel(f"{mn}\n(NLL)", fontsize=10, fontweight="bold")

            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    out = PLOTS_DIR / "val_curves_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Val-only learning curves (re-plot from existing CSVs)")
    print("=" * 60)

    all_data = {}
    found = 0

    for mn in MODEL_ORDER:
        for dn in DS_ORDER:
            csv_path = LC_DIR / f"{mn}_{dn}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_data[(mn, dn)] = df
                found += 1

                # Individual plot
                ok = _plot_single_val(df, mn, dn)
                if ok:
                    print(f"  {mn} {dn}: plotted ({len(df[df['val_loss'].notna()])} val points)")
                else:
                    print(f"  {mn} {dn}: no val data to plot")
            else:
                all_data[(mn, dn)] = None
                print(f"  {mn} {dn}: CSV not found at {csv_path}")

    print(f"\nFound {found}/{len(MODEL_ORDER) * len(DS_ORDER)} CSVs")

    # Grid
    print("\nGenerating val-only grid...")
    _plot_val_grid(all_data)

    print(f"\nIndividual plots: results/plots/val_curve_*.png")
    print(f"Combined grid:    results/plots/val_curves_grid.png")
    print(f"(Original train+val plots preserved as learning_curve_*.png)")


if __name__ == "__main__":
    main()
