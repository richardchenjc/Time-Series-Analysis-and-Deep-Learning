"""
Training-budget verification experiment.

Context
-------
The Night 2 data-volume sweep showed NBEATS on M4 with a U-shaped curve:
MAE dropped from n=100 → n=1000 (595 → 485) then rose back to 552 at n=2000.
The SeasonalNaive difficulty check later showed that the n=2000 sample is
NOT intrinsically harder (SN is flat at n=2000), so the U-shape is model-
related, not data-related.

The hypothesis
--------------
Our MAX_STEPS=800 training budget is expressed in gradient steps, not epochs.
When n_series grows from 1000 to 2000, the number of minibatches per epoch
also roughly doubles (since the training data contains 2x more samples).
So at n=2000, 800 gradient steps correspond to roughly half as many epochs
as they do at n=1000 — the model is legitimately undertrained at the
larger sample size.

The test
--------
Run NBEATS at n=2000 on M4 with MAX_STEPS=1600 (doubled to compensate for
the 2x data) and compare the resulting MAE against the original n=2000
run with MAX_STEPS=800. If the doubled-budget run drops back to the
n=1000 level (MAE ≈ 485), the training-budget hypothesis is confirmed.

If it doesn't improve, the U-shape has a different cause and we need to
revisit.

Expected runtime
----------------
At n=2000 with MAX_STEPS=1600, NBEATS takes roughly 2x as long as the
original (MAX_STEPS=800) run. Original was a few seconds per fit, so
this experiment should complete in ~5-10 minutes total (3 seeds x 2 windows).

Output
------
  results/training_budget/nbeats_m4_n2000_s1600.csv   (per-run results)
  Plus a verdict printed to stdout.

Usage:
    python analysis/training_budget_check.py
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
from models.nbeats import build as build_nbeats


OUT_DIR = RESULTS_DIR / "training_budget"


def _factory(_cfg):
    def _build(seed, max_steps=None):
        # Force max_steps=1600 regardless of what the caller passes
        return build_nbeats(
            horizon=_cfg["horizon"],
            input_size=_cfg["input_size"],
            freq=_cfg["freq"],
            seed=seed,
            max_steps=1600,
        )
    return _build


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#'*60}")
    print(f"# Training budget verification")
    print(f"# NBEATS on M4 at n=2000 with MAX_STEPS=1600")
    print(f"# (doubled from default 800 to compensate for 2x data)")
    print(f"{'#'*60}")
    t0 = time.time()

    cfg = M4_CONFIG
    df_train, df_test = load_m4_monthly(n_series=2000)
    df_full = pd.concat([df_train, df_test], ignore_index=True)

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
            results_dir=OUT_DIR,
            build_model_fn=_factory(cfg),
            needs_seed=True,
            max_steps=None,  # factory ignores this and hardcodes 1600
            max_train_size=cfg.get("max_train_size"),
            save_predictions=False,
        )
    except Exception as e:
        print(f"\n  ✗ Experiment FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    if results.empty:
        print("\n  ✗ No results produced")
        return

    # Rename to distinguish from main-run NBEATS results
    out_path = OUT_DIR / "nbeats_m4_n2000_s1600.csv"
    results.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\n{'#'*60}")
    print(f"# Experiment complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'#'*60}\n")

    mae_1600 = results["mae_mean"].mean()
    mae_1600_std = results["mae_mean"].std()

    # Reference numbers from the existing sweep
    ref_n1000_s800 = 484.82
    ref_n2000_s800 = 552.21

    print(f"  ── RESULTS ──")
    print(f"  NBEATS @ M4 n=1000, MAX_STEPS= 800:  MAE = {ref_n1000_s800:.2f}   (from Night 2 sweep)")
    print(f"  NBEATS @ M4 n=2000, MAX_STEPS= 800:  MAE = {ref_n2000_s800:.2f}   (from Night 2 sweep)")
    print(f"  NBEATS @ M4 n=2000, MAX_STEPS=1600:  MAE = {mae_1600:.2f}   (this experiment)")

    # Did the doubled budget recover performance?
    drop_from_800 = ref_n2000_s800 - mae_1600
    pct_recovery = 100 * drop_from_800 / (ref_n2000_s800 - ref_n1000_s800)

    print(f"\n  ── VERDICT ──")
    print(f"  MAE change: {drop_from_800:+.2f} ({'improved' if drop_from_800 > 0 else 'got worse'})")

    if mae_1600 <= ref_n1000_s800 + 10:
        print(f"  ✓ HYPOTHESIS CONFIRMED: doubled budget recovered NBEATS to the")
        print(f"    n=1000 level ({mae_1600:.1f} vs {ref_n1000_s800:.1f}). The original n=2000")
        print(f"    U-shape was a training-budget artifact: 800 gradient steps")
        print(f"    correspond to fewer epochs as n_series grows. This is a")
        print(f"    clean finding for the report.")
    elif drop_from_800 > 20 and mae_1600 < ref_n2000_s800 - 20:
        print(f"  ◐ HYPOTHESIS PARTIALLY CONFIRMED: doubling the budget recovered")
        print(f"    {pct_recovery:.0f}% of the n=1000→n=2000 degradation. Training")
        print(f"    budget is part of the story but there's residual degradation")
        print(f"    from something else (maybe stratification, optimizer dynamics,")
        print(f"    or genuine overfitting). Investigate further if time permits.")
    else:
        print(f"  ✗ HYPOTHESIS REJECTED: doubling the training budget did NOT")
        print(f"    meaningfully improve NBEATS at n=2000 ({mae_1600:.1f} vs {ref_n2000_s800:.1f}).")
        print(f"    The U-shape must have a different cause. Possible culprits:")
        print(f"    overfitting, optimizer issues, or an M4 stratification effect")
        print(f"    that affects trained models differently from SeasonalNaive.")


if __name__ == "__main__":
    main()
