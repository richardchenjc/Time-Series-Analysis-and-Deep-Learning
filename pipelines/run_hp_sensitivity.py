"""
Hyperparameter sensitivity analysis.

Runs four narrow sensitivity studies, each varying ONE hyperparameter on
ONE model on ONE dataset, with all other settings held fixed:

  1. PatchTST × patch_len ∈ {3, 6, 12} on M4
       — does aligning patches to half-year vs quarter vs annual matter?
  2. PatchTST × input_size (lookback) ∈ {18, 36, 72, 144} on M4
       — replicates the Zeng et al. (2023) "Transformers don't benefit
         from longer lookbacks" critique on our M4 data
  3. NBEATS × n_blocks per stack ∈ {[1,1], [3,3], [5,5]} on M4
       — characterises the sensitivity of our winning model to depth
  4. DLinear × input_size (lookback) ∈ {24, 48, 96, 168} on Traffic
       — replicates the OTHER half of Zeng et al.: "linear models DO
         benefit from longer lookbacks". If we see this on Traffic,
         it's a clean replication of the paper's headline finding.

Why sensitivity not grid search
-------------------------------
The aim is to characterise how stable each model is to reasonable
hyperparameter choices, NOT to find the single best configuration.
Grid search would bias the cross-model comparison in favor of models
with larger tunable search spaces. We use published paper defaults
as the centre of each sensitivity sweep and verify that small
deviations do not flip the model rankings.

Each study runs 3 values × 3 seeds × 2 windows = 18 fits.
Four studies = 72 fits total. Results saved to:

  results/hp_sensitivity/<study_name>/<value_label>.csv
  results/hp_sensitivity/<study_name>/_combined.csv

Usage:
    python pipelines/run_hp_sensitivity.py
"""

import sys
from pathlib import Path

# UTF-8 for Windows console — must run before any prints
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
from models.patchtst import build as build_patchtst
from models.nbeats import build as build_nbeats
from models.dlinear import build as build_dlinear


HP_DIR = RESULTS_DIR / "hp_sensitivity"


def _load_dataset(name: str) -> tuple[pd.DataFrame, dict]:
    """Load one of the bumped-sample datasets and return (df_full, cfg)."""
    if name == "M4":
        df_train, df_test = load_m4_monthly(n_series=M4_CONFIG["n_series_sample"])
        return pd.concat([df_train, df_test], ignore_index=True), M4_CONFIG
    elif name == "Traffic":
        df_train, df_test = load_traffic(n_series=TRAFFIC_CONFIG["n_series_sample"])
        return pd.concat([df_train, df_test], ignore_index=True), TRAFFIC_CONFIG
    raise ValueError(f"Unknown dataset {name}")


def _run_one_config(
    study_name: str,
    value_label: str,
    df_full: pd.DataFrame,
    cfg: dict,
    factory,
    out_subdir: Path,
) -> pd.DataFrame:
    """Run walk-forward eval once for a single HP value."""
    out_subdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  [{study_name}] value = {value_label}")
    print(f"{'─'*60}")

    results = run_walk_forward(
        df_full=df_full,
        dataset_name=cfg["name"],
        horizon=cfg["horizon"],
        input_size=cfg["input_size"],   # walk-forward filter uses this; per-call
                                         # input_size is set inside the factory
        freq=cfg["freq"],
        season_length=cfg["season_length"],
        n_windows=cfg["walk_forward_windows"],
        seeds=SEEDS,
        results_dir=out_subdir,
        build_model_fn=factory,
        needs_seed=True,
        max_steps=None,
        max_train_size=cfg.get("max_train_size"),
        save_predictions=False,  # post-hoc per-horizon analysis isn't the goal here
    )
    if not results.empty:
        # Tag each row with the study name and value for downstream aggregation
        results["study"] = study_name
        results["value"] = value_label
        # Rewrite the CSV with the tagged version
        model_name = results["model"].iloc[0]
        out_path = out_subdir / f"{model_name}_{cfg['name']}.csv"
        results.to_csv(out_path, index=False)
    return results


# ─── Study 1: PatchTST × patch_len on M4 ──────────────────────────────────
def study_patchtst_patch_len(df_m4: pd.DataFrame):
    study_name = "patchtst_patch_len_m4"
    print(f"\n{'='*60}")
    print(f"# Study 1: {study_name}")
    print(f"{'='*60}")

    cfg = M4_CONFIG
    sweep = [
        ("3",  3,  2),    # quarter-year patches with overlap
        ("6",  6,  3),    # half-year patches (default)
        ("12", 12, 6),    # annual patches
    ]
    all_results = []
    for label, patch_len, stride in sweep:
        def _factory(seed, max_steps=None, _pl=patch_len, _st=stride):
            return build_patchtst(
                horizon=cfg["horizon"],
                input_size=cfg["input_size"],
                freq=cfg["freq"],
                seed=seed,
                max_steps=max_steps,
                patch_len=_pl,
                stride=_st,
            )
        out_subdir = HP_DIR / study_name / f"patch_len_{label}"
        results = _run_one_config(
            study_name, f"patch_len={label} stride={stride}",
            df_m4, cfg, _factory, out_subdir,
        )
        all_results.append(results)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(HP_DIR / study_name / "_combined.csv", index=False)
        print(f"\n  Combined results saved to {HP_DIR / study_name / '_combined.csv'}")


# ─── Study 2: PatchTST × input_size (lookback) on M4 ──────────────────────
def study_patchtst_lookback(df_m4: pd.DataFrame):
    study_name = "patchtst_lookback_m4"
    print(f"\n{'='*60}")
    print(f"# Study 2: {study_name}")
    print(f"{'='*60}")

    cfg = M4_CONFIG
    # Sweep lookback at 1×, 2×, 4×, 8× horizon
    # M4 horizon = 18 months
    sweep = [18, 36, 72, 144]
    all_results = []
    for lookback in sweep:
        # patch_len/stride scaled to lookback so we always get ~6-12 patches
        patch_len = max(3, lookback // 6)
        stride = max(1, patch_len // 2)

        def _factory(seed, max_steps=None, _is=lookback, _pl=patch_len, _st=stride):
            return build_patchtst(
                horizon=cfg["horizon"],
                input_size=_is,
                freq=cfg["freq"],
                seed=seed,
                max_steps=max_steps,
                patch_len=_pl,
                stride=_st,
            )
        # IMPORTANT: walk-forward filter must allow series this long
        # — pass a per-config input_size into run_walk_forward via a temp cfg
        cfg_local = {**cfg, "input_size": lookback}
        out_subdir = HP_DIR / study_name / f"lookback_{lookback}"
        results = _run_one_config(
            study_name, f"lookback={lookback}",
            df_m4, cfg_local, _factory, out_subdir,
        )
        all_results.append(results)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(HP_DIR / study_name / "_combined.csv", index=False)
        print(f"\n  Combined results saved to {HP_DIR / study_name / '_combined.csv'}")


# ─── Study 3: NBEATS × n_blocks on M4 ─────────────────────────────────────
def study_nbeats_n_blocks(df_m4: pd.DataFrame):
    study_name = "nbeats_n_blocks_m4"
    print(f"\n{'='*60}")
    print(f"# Study 3: {study_name}")
    print(f"{'='*60}")

    cfg = M4_CONFIG
    sweep = [
        ("[1,1]", [1, 1]),    # shallow
        ("[3,3]", [3, 3]),    # paper default
        ("[5,5]", [5, 5]),    # deep
    ]
    all_results = []
    for label, n_blocks in sweep:
        def _factory(seed, max_steps=None, _nb=n_blocks):
            return build_nbeats(
                horizon=cfg["horizon"],
                input_size=cfg["input_size"],
                freq=cfg["freq"],
                seed=seed,
                max_steps=max_steps,
                n_blocks=_nb,
            )
        out_subdir = HP_DIR / study_name / f"n_blocks_{label.replace(',', '_').replace('[','').replace(']','')}"
        results = _run_one_config(
            study_name, f"n_blocks={label}",
            df_m4, cfg, _factory, out_subdir,
        )
        all_results.append(results)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(HP_DIR / study_name / "_combined.csv", index=False)
        print(f"\n  Combined results saved to {HP_DIR / study_name / '_combined.csv'}")


# ─── Study 4: DLinear × lookback on Traffic ───────────────────────────────
def study_dlinear_lookback(df_traffic: pd.DataFrame):
    study_name = "dlinear_lookback_traffic"
    print(f"\n{'='*60}")
    print(f"# Study 4: {study_name}")
    print(f"{'='*60}")

    cfg = TRAFFIC_CONFIG
    # Traffic horizon = 24 hours; sweep lookback from 1 day → 7 days
    sweep = [24, 48, 96, 168]
    all_results = []
    for lookback in sweep:
        def _factory(seed, max_steps=None, _is=lookback):
            return build_dlinear(
                horizon=cfg["horizon"],
                input_size=_is,
                freq=cfg["freq"],
                seed=seed,
                max_steps=max_steps,
            )
        cfg_local = {**cfg, "input_size": lookback}
        out_subdir = HP_DIR / study_name / f"lookback_{lookback}"
        results = _run_one_config(
            study_name, f"lookback={lookback}",
            df_traffic, cfg_local, _factory, out_subdir,
        )
        all_results.append(results)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(HP_DIR / study_name / "_combined.csv", index=False)
        print(f"\n  Combined results saved to {HP_DIR / study_name / '_combined.csv'}")


def main():
    HP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# HP Sensitivity Pipeline")
    print(f"# 4 studies × ~3 values × 3 seeds × 2 windows ≈ 72 fits")
    print(f"{'#'*60}")
    t0 = time.time()

    # Load datasets ONCE — each loader is expensive (M4 = ~30s, Traffic = ~10s)
    print("\nLoading M4 (will be reused across studies 1-3)...")
    df_m4, _ = _load_dataset("M4")
    print(f"M4 loaded: {df_m4.shape}")

    print("\nLoading Traffic (will be used by study 4)...")
    df_traffic, _ = _load_dataset("Traffic")
    print(f"Traffic loaded: {df_traffic.shape}")

    # Run studies in order
    study_patchtst_patch_len(df_m4)
    study_patchtst_lookback(df_m4)
    study_nbeats_n_blocks(df_m4)
    study_dlinear_lookback(df_traffic)

    elapsed = time.time() - t0
    print(f"\n{'#'*60}")
    print(f"# HP Sensitivity complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
