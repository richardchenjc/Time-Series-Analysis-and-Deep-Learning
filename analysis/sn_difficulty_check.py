"""
SeasonalNaive sweep on M4 — verifies the "harder at n=2000" hypothesis.

Background
----------
The Night 2 data-volume sweep showed all three models (NBEATS, PatchTST,
DLinear) getting WORSE when going from n=1000 to n=2000 series on M4.
This is unexpected — usually more data = better performance.

The hypothesis is that this is a SAMPLING artifact, not a model artifact:
  - Stratified sampling allocates n / 6 series per M4 category
  - At n=1000, each category gets ~167 series (well below population size)
  - At n=2000, each category requests ~333, but Other only has 277
    available in the population. Other saturates, the surplus is
    redistributed to the 5 other categories, drawing deeper into their
    long tails.
  - Deeper tails = harder series = higher MAE
  - So the n=2000 sample is intrinsically harder than the n=1000 sample,
    regardless of which model evaluates it.

How to verify
-------------
Run a model whose performance is purely a function of sample difficulty
(no training, no hyperparameters, no seeds). SeasonalNaive fits this:
its prediction at horizon h is y[t-h+season_length], a deterministic
lookup. Its MAE is entirely determined by the series in the sample.

If SeasonalNaive ALSO shows a jump at n=2000, the cause is sample
difficulty (Hypothesis A confirmed).
If SeasonalNaive is flat across all sample sizes and only the trained
models jump at n=2000, the cause is something model-specific
(maybe overfitting, maybe optimizer issues — investigate further).

What this script does
---------------------
Runs SeasonalNaive at sample sizes [100, 300, 1000, 2000] on M4,
matching the existing data-volume sweep grid. SeasonalNaive is
deterministic so we only need 1 "seed" but we still get 2 windows
per size = 8 fits total. Should run in under 2 minutes.

Output:
  results/data_volume/SeasonalNaive_M4_n*.csv  (per size)
  results/data_volume/M4_combined.csv          (appended with new rows)

Usage:
    python analysis/sn_difficulty_check.py
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

from config import M4_CONFIG, SEEDS, RESULTS_DIR
from data_prep.m4_prep import load_m4_monthly
from evaluation.walk_forward import run_walk_forward
from models.seasonal_naive import build as build_seasonal_naive

SWEEP_DIR = RESULTS_DIR / "data_volume"


def _build_sn(_cfg):
    def _build(seed=None, max_steps=None):
        return build_seasonal_naive(
            season_length=_cfg["season_length"],
            freq=_cfg["freq"],
        )
    return _build


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#'*60}")
    print(f"# SeasonalNaive M4 difficulty check across sample sizes")
    print(f"{'#'*60}")
    t0 = time.time()

    cfg = M4_CONFIG
    sizes = [100, 300, 1000, 2000]
    factory = _build_sn(cfg)

    all_results = []
    for n_series in sizes:
        print(f"\n  === SeasonalNaive @ M4 n={n_series} ===")
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
                needs_seed=False,
                max_steps=None,
                max_train_size=cfg.get("max_train_size"),
                save_predictions=False,
            )
            if not results.empty:
                results["n_series"] = n_series
                out_path = out_subdir / "SeasonalNaive_M4.csv"
                results.to_csv(out_path, index=False)
                all_results.append(results)
        except Exception as e:
            print(f"  ✗ SeasonalNaive @ n={n_series} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Append to M4_combined.csv so plot_data_volume.py picks it up
    if all_results:
        new_rows = pd.concat(all_results, ignore_index=True)
        m4_combined_path = SWEEP_DIR / "M4_combined.csv"
        if m4_combined_path.exists():
            existing = pd.read_csv(m4_combined_path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows
        combined.to_csv(m4_combined_path, index=False)
        print(f"\n  Appended SeasonalNaive results to {m4_combined_path}")

    elapsed = time.time() - t0
    print(f"\n{'#'*60}")
    print(f"# SeasonalNaive sweep complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'#'*60}")

    # ── Verdict ──────────────────────────────────────────────────────────
    print(f"\n  ── INTERPRETATION ──")
    if all_results:
        sn = pd.concat(all_results, ignore_index=True)
        agg = sn.groupby("n_series")["mae_mean"].mean().reset_index()
        print(f"\n  SeasonalNaive MAE by sample size:")
        for _, row in agg.iterrows():
            print(f"    n={int(row['n_series']):5d}  MAE = {row['mae_mean']:.2f}")

        n1000 = float(agg[agg["n_series"] == 1000]["mae_mean"].iloc[0]) if not agg[agg["n_series"] == 1000].empty else None
        n2000 = float(agg[agg["n_series"] == 2000]["mae_mean"].iloc[0]) if not agg[agg["n_series"] == 2000].empty else None
        if n1000 and n2000:
            jump_pct = 100 * (n2000 - n1000) / n1000
            print(f"\n  SeasonalNaive n=1000 → n=2000 change: {jump_pct:+.1f}%")
            if jump_pct > 5:
                print(f"  HYPOTHESIS A CONFIRMED: SeasonalNaive (deterministic, no training)")
                print(f"  also jumps at n=2000. The cause IS sample difficulty, not model")
                print(f"  overfitting. The n=2000 sample is intrinsically harder than n=1000")
                print(f"  due to stratified sampling reaching deeper into category tails.")
                print(f"  This validates the report's choice of n=1000 as the main-run size.")
            elif jump_pct < 1:
                print(f"  HYPOTHESIS A REJECTED: SeasonalNaive is ~flat at n=2000 even though")
                print(f"  the trained models worsen. The cause is something model-specific —")
                print(f"  maybe overfitting, maybe optimizer issues. Investigate further.")
            else:
                print(f"  AMBIGUOUS: SeasonalNaive shows some increase but smaller than the")
                print(f"  trained models. Sample difficulty is part of the story but not all of it.")


if __name__ == "__main__":
    main()
