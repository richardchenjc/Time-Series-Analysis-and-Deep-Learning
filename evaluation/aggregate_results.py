"""
Aggregate raw results from all model pipelines into summary tables.

Reads all per-model CSVs from results/ (e.g. PatchTST_M4.csv),
computes mean ± std across seeds and windows, and outputs:
  - summary_table.csv          (main comparison table)
  - computational_costs.csv    (training time & hardware)
  - pivot_mae.csv              (Model × Dataset pivot)
  - pivot_mase.csv             (Model × Dataset pivot, mean MASE)
  - pivot_mase_median.csv      (Model × Dataset pivot, median MASE)

Run after one or more model pipelines have completed:
    python analysis/aggregate_results.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR

# Output files produced by this script (excluded from CSV discovery)
_SUMMARY_FILES = {
    "summary_table",
    "computational_costs",
    "pivot_mae",
    "pivot_mase",
    "pivot_mase_median",
}


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill columns added in the metrics-upgrade patch.

    Old per-model CSVs (produced before mase_median + dropped tracking
    was added) won't have these columns. Fill them with NaN / 0 so the
    aggregation still runs against a mixed pool of old and new CSVs.
    A column-existence check makes the upgrade strictly additive.
    """
    for col in ("mase_median", "mase_n_total", "mase_n_dropped"):
        if col not in df.columns:
            df[col] = np.nan if col == "mase_median" else 0
    return df


def aggregate():
    """Load all per-model CSVs and produce summary tables."""
    results_dir = RESULTS_DIR
    all_dfs = []

    # Discover all per-model result CSVs, excluding summary files
    csv_files = sorted(results_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if f.stem not in _SUMMARY_FILES]

    if not csv_files:
        print(f"No result CSVs found in {results_dir}")
        print("Run one or more model pipelines first, e.g.:")
        print("    python pipelines/run_seasonal_naive.py")
        return

    for csv_path in csv_files:
        df = _ensure_columns(pd.read_csv(csv_path))
        all_dfs.append(df)
        print(f"Loaded {csv_path.name} ({len(df)} rows)")

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df_all)}")

    # ── Summary table: mean ± std of MAE and MASE per (dataset, model) ──
    # mae_*       : aggregate of per-window/per-seed MAE values (always defined)
    # mase_*      : aggregate of per-window/per-seed mean MASE (M5 unstable)
    # mase_med_*  : aggregate of per-window/per-seed median MASE (robust on M5)
    # drop_*      : how many per-series MASEs were undefined per run
    summary = (
        df_all
        .groupby(["dataset", "model"])
        .agg(
            mae_mean=("mae_mean", "mean"),
            mae_std=("mae_mean", "std"),
            mase_mean=("mase_mean", "mean"),
            mase_std=("mase_mean", "std"),
            mase_median_mean=("mase_median", "mean"),
            mase_median_std=("mase_median", "std"),
            mase_dropped_mean=("mase_n_dropped", "mean"),
            mase_total_mean=("mase_n_total", "mean"),
            n_runs=("mae_mean", "count"),
        )
        .reset_index()
    )
    # Fill NaN std (single-run models like baselines)
    for col in ("mae_std", "mase_std", "mase_median_std"):
        summary[col] = summary[col].fillna(0)

    # Average % of series with undefined MASE per run.
    # Critical M5 diagnostic: if a model "wins" on M5 mean MASE only by
    # having more series silently dropped, the comparison is unfair.
    summary["mase_drop_pct"] = np.where(
        summary["mase_total_mean"] > 0,
        100 * summary["mase_dropped_mean"] / summary["mase_total_mean"],
        0.0,
    )

    # Format display columns
    summary["MAE"] = summary.apply(
        lambda r: f"{r['mae_mean']:.4f} ± {r['mae_std']:.4f}", axis=1
    )
    summary["MASE_mean"] = summary.apply(
        lambda r: f"{r['mase_mean']:.4f} ± {r['mase_std']:.4f}", axis=1
    )
    summary["MASE_median"] = summary.apply(
        lambda r: f"{r['mase_median_mean']:.4f} ± {r['mase_median_std']:.4f}", axis=1
    )
    summary["MASE_drop%"] = summary["mase_drop_pct"].map(lambda v: f"{v:.1f}%")

    summary_path = results_dir / "summary_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary table saved to {summary_path}")
    display_cols = ["dataset", "model", "MAE", "MASE_mean", "MASE_median", "MASE_drop%", "n_runs"]
    print(summary[display_cols].to_string(index=False))

    # ── Computational costs ──
    costs = (
        df_all
        .groupby(["dataset", "model"])
        .agg(
            avg_train_time_sec=("train_time_sec", "mean"),
            total_train_time_sec=("train_time_sec", "sum"),
            peak_gpu_mb=("peak_gpu_mb", "max"),
        )
        .reset_index()
    )

    costs_path = results_dir / "computational_costs.csv"
    costs.to_csv(costs_path, index=False)
    print(f"\nComputational costs saved to {costs_path}")
    print(costs.to_string(index=False))

    # ── Pivot tables for report (Model × Dataset) ──
    pivot_mae = summary.pivot(index="model", columns="dataset", values="MAE")
    pivot_mase = summary.pivot(index="model", columns="dataset", values="MASE_mean")
    pivot_mase_median = summary.pivot(index="model", columns="dataset", values="MASE_median")

    for name, p in [("pivot_mae", pivot_mae),
                    ("pivot_mase", pivot_mase),
                    ("pivot_mase_median", pivot_mase_median)]:
        out = results_dir / f"{name}.csv"
        p.to_csv(out)
        print(f"Pivot table saved to {out}")

    return summary, costs


if __name__ == "__main__":
    aggregate()
