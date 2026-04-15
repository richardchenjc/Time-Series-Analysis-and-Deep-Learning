"""
Per-horizon MAE exporter.

Reads the per-model prediction parquet files and exports a flat CSV
with per-horizon MAE per (model, dataset, horizon_step). The CSV is
useful for the report when we want to cite specific numbers like
"at h=1 the best model is SeasonalNaive with MAE 261; at h=18 the best
model is NBEATS with MAE 839."

Output
------
  results/per_horizon_mae.csv

Columns:
  dataset, model, horizon_step, mae, n_obs

Aggregation:
  MAE is computed per (dataset, model, horizon_step) by averaging
  absolute errors across all (seed, window, series) for that cell.
  n_obs is the count of prediction rows contributing to that cell.

Usage:
    python analysis/export_per_horizon_csv.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import RESULTS_DIR

PRED_DIR = RESULTS_DIR / "predictions"


def main():
    rows = []
    for parquet_path in sorted(PRED_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"  Failed to read {parquet_path.name}: {e}")
            continue

        df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
        # Group by the natural per-horizon axis
        agg = (
            df.groupby(["dataset", "model", "horizon_step"])
              .agg(mae=("abs_err", "mean"),
                   n_obs=("abs_err", "count"))
              .reset_index()
        )
        rows.append(agg)
        print(f"  Processed {parquet_path.name}")

    if not rows:
        print("  No parquet files found")
        return

    combined = pd.concat(rows, ignore_index=True)
    # Collapse duplicates if multiple parquet files had overlapping data
    combined = (
        combined.groupby(["dataset", "model", "horizon_step"])
                .agg(mae=("mae", "mean"), n_obs=("n_obs", "sum"))
                .reset_index()
    )

    out_path = RESULTS_DIR / "per_horizon_mae.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n  Saved {len(combined)} rows to {out_path}")

    # Print a "best model at h=1 vs h=H" summary per dataset for quick reference
    print(f"\n  ── Best model at h=1 and h=H per dataset ──")
    for ds in combined["dataset"].unique():
        sub = combined[combined["dataset"] == ds]
        h_min, h_max = int(sub["horizon_step"].min()), int(sub["horizon_step"].max())
        at_h_min = sub[sub["horizon_step"] == h_min].sort_values("mae")
        at_h_max = sub[sub["horizon_step"] == h_max].sort_values("mae")
        if not at_h_min.empty and not at_h_max.empty:
            b_min = at_h_min.iloc[0]
            b_max = at_h_max.iloc[0]
            print(f"  {ds}:")
            print(f"    h={h_min:2d}: best = {b_min['model']:<14} (MAE {b_min['mae']:.4f})")
            print(f"    h={h_max:2d}: best = {b_max['model']:<14} (MAE {b_max['mae']:.4f})")


if __name__ == "__main__":
    main()
