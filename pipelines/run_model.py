"""
Shared pipeline utility — runs a single model across all three datasets.

This module is the core engine used by every per-model pipeline script.
It loads each dataset, applies smoke-test overrides when requested, and
calls the walk-forward evaluation for the given model.

Usage (from a per-model pipeline):
    from pipelines.run_model import run_pipeline
    run_pipeline("PatchTST", factory_fn, needs_seed=True, smoke_test=False)
"""

# Force UTF-8 output on Windows consoles (cp1252 can't encode → ✗ ✓ etc.).
# Must run before any print() in this module or downstream modules.
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import pandas as pd
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    M4_CONFIG, M5_CONFIG, TRAFFIC_CONFIG,
    SEEDS, RESULTS_DIR,
)
from data_prep.m4_prep import load_m4_monthly
from data_prep.m5_prep import load_m5
from data_prep.traffic_prep import load_traffic
from evaluation.walk_forward import run_walk_forward

# Smoke-test series counts (small enough for a quick validation run)
_SMOKE_SERIES = {"M4": 50, "M5": 30, "Traffic": 10}

# Ordered list of (dataset_name, config_dict, loader_fn)
DATASETS = [
    ("M4",      M4_CONFIG,      load_m4_monthly),
    ("M5",      M5_CONFIG,      load_m5),
    ("Traffic", TRAFFIC_CONFIG, load_traffic),
]


def run_pipeline(
    model_name: str,
    build_fn_for_cfg: Callable[[dict], Callable],
    needs_seed: bool,
    smoke_test: bool = False,
) -> list[pd.DataFrame]:
    """Run one model across all 3 datasets and save per-dataset result CSVs.

    Parameters
    ----------
    model_name : str
        Human-readable model name used in headers and filenames.
    build_fn_for_cfg : callable
        Given a dataset config dict, returns a factory closure with signature
        ``(seed, max_steps=None) -> ModelSpec``.
    needs_seed : bool
        Whether to iterate over multiple random seeds.
        True for ML/DL models, False for deterministic baselines.
    smoke_test : bool
        If True, use minimal settings (fewer series, 1 seed, 1 window,
        10 training steps) for quick end-to-end validation.

    Returns
    -------
    list of DataFrames
        One DataFrame per dataset (may be empty if evaluation failed).
    """
    mode = "SMOKE TEST" if smoke_test else "FULL RUN"
    print(f"\n{'#'*60}")
    print(f"# {model_name} — {mode}")
    print(f"{'#'*60}")

    seeds = [SEEDS[0]] if smoke_test else SEEDS
    max_steps = 10 if smoke_test else None

    all_results = []
    total_start = time.time()

    for dataset_name, cfg, loader in DATASETS:
        n_series = _SMOKE_SERIES[dataset_name] if smoke_test else cfg["n_series_sample"]
        n_windows = 1 if smoke_test else cfg["walk_forward_windows"]

        print(f"\n{'='*60}")
        print(f"  Dataset : {dataset_name}")
        print(f"  Series  : {n_series}  |  Seeds: {len(seeds)}  |  Windows: {n_windows}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            # Load and concatenate train+test into one full DataFrame
            df_train, df_test = loader(n_series=n_series)
            df_full = pd.concat([df_train, df_test], ignore_index=True)

            # Build a dataset-specific factory closure
            build_model_fn = build_fn_for_cfg(cfg)

            # Run walk-forward evaluation
            results = run_walk_forward(
                df_full=df_full,
                dataset_name=dataset_name,
                horizon=cfg["horizon"],
                input_size=cfg["input_size"],
                freq=cfg["freq"],
                season_length=cfg["season_length"],
                n_windows=n_windows,
                seeds=seeds,
                results_dir=RESULTS_DIR,
                build_model_fn=build_model_fn,
                needs_seed=needs_seed,
                max_steps=max_steps,
                max_train_size=cfg.get("max_train_size"),
            )
            all_results.append(results)
            print(f"\n  ✓ {dataset_name} done in {time.time() - t0:.1f}s")

        except Exception as e:
            print(f"\n  ✗ {dataset_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append(pd.DataFrame())

    total_elapsed = time.time() - total_start
    print(f"\n{'#'*60}")
    print(f"# {model_name} complete — Total: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")
    print(f"{'#'*60}")

    return all_results
