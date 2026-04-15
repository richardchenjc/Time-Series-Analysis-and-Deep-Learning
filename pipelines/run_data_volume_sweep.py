"""
Data-volume sensitivity sweep.

Directly addresses Key Question 3: "Do deep models overfit on smaller
datasets?" by retraining 3 representative models at multiple sample
sizes on M4 and M5, then plotting MAE vs n_series.

Models chosen for contrast:
  - NBEATS    : the DL winner at full sample — does it still win small?
  - PatchTST  : the Transformer — does the "Transformers are data-hungry"
                hypothesis show up as a worse small-data slope?
  - DLinear   : the linear baseline — the "is simple enough at small
                scales" question

Sample sizes (geometric, NOT linear, to span orders of magnitude):
  M4   : [100, 300, 1000, 2000]   — main-run sample is 1000 (mid-anchor)
  M5   : [100, 250, 500]          — main-run sample is 500 (top-anchor)
  Traffic skipped : 862 series total, compressed range, less informative

Total fits:
  M4   : 3 models × 4 sizes × 3 seeds × 2 windows = 72
  M5   : 3 models × 3 sizes × 3 seeds × 2 windows = 54
  Total: 126 fits

Output:
  results/data_volume/<model>_<dataset>_n<size>.csv
  results/data_volume/<dataset>_combined.csv

Usage:
    python pipelines/run_data_volume_sweep.py
"""

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import pandas as pd

from config import M4_CONFIG, M5_CONFIG, SEEDS, RESULTS_DIR
from data_prep.m4_prep import load_m4_monthly
from data_prep.m5_prep import load_m5
from evaluation.walk_forward import run_walk_forward
from models.nbeats import build as build_nbeats
from models.patchtst import build as build_patchtst
from models.dlinear import build as build_dlinear


SWEEP_DIR = RESULTS_DIR / "data_volume"


# Per-dataset sample-size grids
_SAMPLE_GRIDS = {
    "M4": [100, 300, 1000, 2000],
    "M5": [100, 250, 500],
}


# Model factories — same patch_len/stride logic as the main run for PatchTST
_PATCH_CONFIGS = {
    "M4": {"patch_len": 6,  "stride": 3},
    "M5": {"patch_len": 7,  "stride": 7},
}


def _build_factory(model_name: str, cfg: dict):
    """Build a (seed, max_steps) → ModelSpec factory for the given model."""
    if model_name == "NBEATS":
        def _f(seed, max_steps=None):
            return build_nbeats(
                horizon=cfg["horizon"],
                input_size=cfg["input_size"],
                freq=cfg["freq"],
                seed=seed,
                max_steps=max_steps,
            )
    elif model_name == "PatchTST":
        patch = _PATCH_CONFIGS[cfg["name"]]
        def _f(seed, max_steps=None):
            return build_patchtst(
                horizon=cfg["horizon"],
                input_size=cfg["input_size"],
                freq=cfg["freq"],
                seed=seed,
                max_steps=max_steps,
                patch_len=patch["patch_len"],
                stride=patch["stride"],
            )
    elif model_name == "DLinear":
        def _f(seed, max_steps=None):
            return build_dlinear(
                horizon=cfg["horizon"],
                input_size=cfg["input_size"],
                freq=cfg["freq"],
                seed=seed,
                max_steps=max_steps,
            )
    else:
        raise ValueError(f"Unknown model {model_name}")
    return _f


def _sweep_one_dataset(dataset_name: str):
    print(f"\n{'#'*60}")
    print(f"# Sweeping {dataset_name}")
    print(f"{'#'*60}")

    if dataset_name == "M4":
        cfg = M4_CONFIG
        loader = load_m4_monthly
    elif dataset_name == "M5":
        cfg = M5_CONFIG
        loader = load_m5
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    sizes = _SAMPLE_GRIDS[dataset_name]
    models = ["NBEATS", "PatchTST", "DLinear"]

    all_results = []

    for n_series in sizes:
        print(f"\n{'='*60}")
        print(f"  {dataset_name} @ n_series = {n_series}")
        print(f"{'='*60}")

        # Load ONCE per sample size (3 models will share this dataframe)
        df_train, df_test = loader(n_series=n_series)
        df_full = pd.concat([df_train, df_test], ignore_index=True)

        for model_name in models:
            print(f"\n  ── {model_name} @ n={n_series} ──")
            factory = _build_factory(model_name, cfg)
            out_subdir = SWEEP_DIR / dataset_name / f"n_{n_series}"
            out_subdir.mkdir(parents=True, exist_ok=True)

            try:
                results = run_walk_forward(
                    df_full=df_full,
                    dataset_name=dataset_name,
                    horizon=cfg["horizon"],
                    input_size=cfg["input_size"],
                    freq=cfg["freq"],
                    season_length=cfg["season_length"],
                    n_windows=cfg["walk_forward_windows"],
                    seeds=SEEDS,
                    results_dir=out_subdir,
                    build_model_fn=factory,
                    needs_seed=True,
                    max_steps=None,
                    max_train_size=cfg.get("max_train_size"),
                    save_predictions=False,
                )
                if not results.empty:
                    results["n_series"] = n_series
                    # Rewrite tagged CSV in place so the per-cell file already
                    # has the n_series column for downstream plotting
                    out_path = out_subdir / f"{model_name}_{dataset_name}.csv"
                    results.to_csv(out_path, index=False)
                    all_results.append(results)
                    print(f"  ✓ {model_name} @ n={n_series} done")
            except Exception as e:
                print(f"  ✗ {model_name} @ n={n_series} FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Combined per-dataset table for plotting
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = SWEEP_DIR / f"{dataset_name}_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n  Combined results saved to {combined_path}")


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# Data-Volume Sweep")
    print(f"# M4: 3 models × 4 sizes × 6 runs = 72 fits")
    print(f"# M5: 3 models × 3 sizes × 6 runs = 54 fits")
    print(f"# Total: ~126 fits")
    print(f"{'#'*60}")
    t0 = time.time()

    _sweep_one_dataset("M4")
    _sweep_one_dataset("M5")

    elapsed = time.time() - t0
    print(f"\n{'#'*60}")
    print(f"# Sweep complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
