"""
Extended data-volume sweep — additions to the Night 2 sweep.

Fills in two gaps from the original sweep:

  1. AutoARIMA × M4 at 4 sample sizes (100, 300, 1000, 2000)
     AutoARIMA statistically TIES with NBEATS on M4 (per Nemenyi CD
     analysis). To characterise this tie across sample sizes, we need
     to know how AutoARIMA's MAE scales with n_series. If the tie
     holds at all sizes, the report claim "classical methods match
     DL on M4" is robust. If AutoARIMA only catches up at large n,
     that's a different (also interesting) story.

  2. Traffic × {NBEATS, PatchTST, DLinear} at 4 sample sizes
     (50, 150, 500, 862). Originally skipped as "compressed range",
     but worth running to verify that DLinear's dominance on Traffic
     holds at small sample sizes — the "linear is enough" thesis
     should be strongest at small n.

Total fits:
  AutoARIMA × M4:         4 sizes × 2 windows (deterministic) = 8 fits
  Traffic × 3 models:     4 sizes × 3 × 3 seeds × 2 windows  = 72 fits

  Total: ~80 fits, ~30-45 minutes of compute.

Output:
  results/data_volume/AutoARIMA_M4_n*.csv        (per size)
  results/data_volume/AutoARIMA_M4_combined.csv  (all sizes stacked)
  results/data_volume/Traffic/n_*/                (per size/model)
  results/data_volume/Traffic_combined.csv

Usage:
    python pipelines/run_data_volume_sweep_extension.py
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

from config import M4_CONFIG, TRAFFIC_CONFIG, SEEDS, RESULTS_DIR
from data_prep.m4_prep import load_m4_monthly
from data_prep.traffic_prep import load_traffic
from evaluation.walk_forward import run_walk_forward

from models.auto_arima import build as build_autoarima
from models.nbeats import build as build_nbeats
from models.patchtst import build as build_patchtst
from models.dlinear import build as build_dlinear


SWEEP_DIR = RESULTS_DIR / "data_volume"


# ── AutoARIMA factory for M4 (no seed needed) ─────────────────────────────
def _build_autoarima_m4(_cfg):
    def _build(seed=None, max_steps=None):
        return build_autoarima(
            season_length=_cfg["season_length"],
            freq=_cfg["freq"],
        )
    return _build


# ── Traffic model factories (same as Night 2 sweep but for Traffic) ───────
_TRAFFIC_PATCH_CONFIG = {"patch_len": 24, "stride": 12}


def _build_nbeats_traffic(_cfg):
    def _build(seed, max_steps=None):
        return build_nbeats(
            horizon=_cfg["horizon"],
            input_size=_cfg["input_size"],
            freq=_cfg["freq"],
            seed=seed,
            max_steps=max_steps,
        )
    return _build


def _build_patchtst_traffic(_cfg):
    def _build(seed, max_steps=None):
        return build_patchtst(
            horizon=_cfg["horizon"],
            input_size=_cfg["input_size"],
            freq=_cfg["freq"],
            seed=seed,
            max_steps=max_steps,
            patch_len=_TRAFFIC_PATCH_CONFIG["patch_len"],
            stride=_TRAFFIC_PATCH_CONFIG["stride"],
        )
    return _build


def _build_dlinear_traffic(_cfg):
    def _build(seed, max_steps=None):
        return build_dlinear(
            horizon=_cfg["horizon"],
            input_size=_cfg["input_size"],
            freq=_cfg["freq"],
            seed=seed,
            max_steps=max_steps,
        )
    return _build


# ── AutoARIMA M4 sweep ─────────────────────────────────────────────────────
def sweep_autoarima_m4():
    print(f"\n{'#'*60}")
    print(f"# AutoARIMA × M4 data-volume sweep")
    print(f"{'#'*60}")

    cfg = M4_CONFIG
    sizes = [100, 300, 1000, 2000]
    factory = _build_autoarima_m4(cfg)

    all_results = []
    for n_series in sizes:
        print(f"\n  === AutoARIMA @ M4 n={n_series} ===")
        df_train, df_test = load_m4_monthly(n_series=n_series)
        df_full = pd.concat([df_train, df_test], ignore_index=True)

        out_subdir = SWEEP_DIR / "M4" / f"n_{n_series}"
        out_subdir.mkdir(parents=True, exist_ok=True)

        try:
            results = run_walk_forward(
                df_full=df_full,
                dataset_name="M4",
                horizon=cfg["horizon"],
                input_size=cfg["input_size"],
                freq=cfg["freq"],
                season_length=cfg["season_length"],
                n_windows=cfg["walk_forward_windows"],
                seeds=SEEDS,
                results_dir=out_subdir,
                build_model_fn=factory,
                needs_seed=False,  # AutoARIMA is deterministic
                max_steps=None,
                max_train_size=cfg.get("max_train_size"),
                save_predictions=False,
            )
            if not results.empty:
                results["n_series"] = n_series
                out_path = out_subdir / f"AutoARIMA_M4.csv"
                results.to_csv(out_path, index=False)
                all_results.append(results)
        except Exception as e:
            print(f"  ✗ AutoARIMA @ n={n_series} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Append to the existing M4_combined.csv so the plot script can pick it up
    # in a single read
    m4_combined_path = SWEEP_DIR / "M4_combined.csv"
    if all_results:
        new_rows = pd.concat(all_results, ignore_index=True)
        if m4_combined_path.exists():
            existing = pd.read_csv(m4_combined_path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows
        combined.to_csv(m4_combined_path, index=False)
        print(f"\n  Appended to {m4_combined_path}")


# ── Traffic data-volume sweep ──────────────────────────────────────────────
def sweep_traffic():
    print(f"\n{'#'*60}")
    print(f"# Traffic data-volume sweep (3 models × 4 sizes)")
    print(f"{'#'*60}")

    cfg = TRAFFIC_CONFIG
    # Sizes chosen to span the Traffic population: 50 → 150 → 500 → full 862.
    # 50 is our original main-run size; 862 is the full population.
    sizes = [50, 150, 500, 862]
    model_specs = [
        ("NBEATS",   _build_nbeats_traffic),
        ("PatchTST", _build_patchtst_traffic),
        ("DLinear",  _build_dlinear_traffic),
    ]

    all_results = []
    for n_series in sizes:
        print(f"\n  === Traffic @ n={n_series} ===")
        # Pass None for "use all" when requesting the full population
        loader_n = None if n_series >= 862 else n_series
        df_train, df_test = load_traffic(n_series=loader_n)
        df_full = pd.concat([df_train, df_test], ignore_index=True)

        for model_name, factory_fn in model_specs:
            print(f"\n    ── {model_name} @ n={n_series} ──")
            factory = factory_fn(cfg)
            out_subdir = SWEEP_DIR / "Traffic" / f"n_{n_series}"
            out_subdir.mkdir(parents=True, exist_ok=True)
            try:
                results = run_walk_forward(
                    df_full=df_full,
                    dataset_name="Traffic",
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
                    out_path = out_subdir / f"{model_name}_Traffic.csv"
                    results.to_csv(out_path, index=False)
                    all_results.append(results)
                    print(f"    ✓ {model_name} @ n={n_series} done")
            except Exception as e:
                print(f"    ✗ {model_name} @ n={n_series} FAILED: {e}")
                import traceback
                traceback.print_exc()

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = SWEEP_DIR / "Traffic_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n  Saved {combined_path}")


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#'*60}")
    print(f"# Data-Volume Sweep — Extensions")
    print(f"# 1. AutoARIMA × M4 @ 4 sizes")
    print(f"# 2. Traffic × 3 models @ 4 sizes")
    print(f"{'#'*60}")
    t0 = time.time()

    sweep_autoarima_m4()
    sweep_traffic()

    elapsed = time.time() - t0
    print(f"\n{'#'*60}")
    print(f"# Extensions complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
