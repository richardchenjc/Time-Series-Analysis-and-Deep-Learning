"""
Insurance checks for the report's defensibility.

Runs three small experiments, each addressing one specific "but did your
methodology cause this finding?" criticism that a careful reviewer would
raise. None of these are required by the assignment, but each turns a
potential weakness into a documented strength.

  1. M5 LightGBM without calendar features
       Verifies that LightGBM still loses to DL models on M5 even
       without the SNAP/event features. This rules out the criticism
       "LightGBM only loses on M5 because you handicapped it by leaving
       out lag features" — clearly false, but worth confirming.

       Estimated time: ~1 min

  2. NBEATS sampling sanity check (stratified vs random)
       The earlier sanity check confirmed LightGBM has a ~4% MAE gap
       between random and stratified M4. This re-runs the test with
       NBEATS to verify the cross-model RANKING is preserved across
       sampling methods. If NBEATS_random < LightGBM_random AND
       NBEATS_strat < LightGBM_strat, the ranking is robust and
       stratified sampling can be reported confidently.

       Estimated time: ~5 min

  3. TimesNet undertraining check
       TimesNet's training curve was dominated by its high per-step
       cost; we used MAX_STEPS=800 globally. This re-runs TimesNet on
       M5 with MAX_STEPS=2000 to check whether it was undertrained
       at 800. If MAE drops significantly, our TimesNet result is
       pessimistic and we should report the better number. If MAE is
       unchanged, 800 was sufficient and we're fine.

       Estimated time: ~10 min

All three results are saved under results/insurance_checks/ with a
summary printed at the end. Re-aggregation is NOT required after
running this — the outputs are deliberately separate from the main
results so they don't pollute the summary table.

Usage:
    python analysis/insurance_checks.py
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
import numpy as np
from mlforecast.lag_transforms import RollingMean, RollingStd

from config import M4_CONFIG, M5_CONFIG, SEEDS, RESULTS_DIR
from data_prep.m4_prep import load_m4_monthly
from data_prep.m5_prep import load_m5
from evaluation.walk_forward import run_walk_forward

from models.lightgbm import build as build_lightgbm
from models.nbeats import build as build_nbeats
from models.timesnet import build as build_timesnet


INSURANCE_DIR = RESULTS_DIR / "insurance_checks"


# ── Check 1: M5 LightGBM without features ──────────────────────────────────
def check_m5_lightgbm_no_features():
    """Run LightGBM on M5 with the bare minimum: lag features only, no
    calendar exog, no Tweedie loss. Compare to the main-run M5 LightGBM
    result which has all of those bells and whistles.
    """
    print(f"\n{'#'*60}")
    print(f"# Check 1: M5 LightGBM without calendar features / Tweedie")
    print(f"{'#'*60}")

    cfg = M5_CONFIG
    # Bare-bones features: just seasonal lags + rolling stats, no exog, no Tweedie
    bare_lags = [1, 7, 14, 28]
    bare_lag_transforms = {
        7:  [RollingMean(window_size=7),  RollingStd(window_size=7)],
        28: [RollingMean(window_size=28), RollingStd(window_size=28)],
    }
    bare_date_features = ["dayofweek", "month"]

    def _factory(_cfg):
        def _build(seed, max_steps=None):
            return build_lightgbm(
                freq=_cfg["freq"],
                season_length=_cfg["season_length"],
                seed=seed,
                lags=bare_lags,
                lag_transforms=bare_lag_transforms,
                date_features=bare_date_features,
                exog_cols=None,        # no calendar exog
                objective=None,        # no Tweedie
            )
        return _build

    # Load M5 with calendar features OFF — we don't need them and don't
    # want them attached as extra columns
    df_train, df_test = load_m5(
        n_series=cfg["n_series_sample"],
        with_calendar_features=False,
    )
    df_full = pd.concat([df_train, df_test], ignore_index=True)

    out_dir = INSURANCE_DIR / "m5_lightgbm_no_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_walk_forward(
        df_full=df_full,
        dataset_name="M5",
        horizon=cfg["horizon"],
        input_size=cfg["input_size"],
        freq=cfg["freq"],
        season_length=cfg["season_length"],
        n_windows=cfg["walk_forward_windows"],
        seeds=SEEDS,
        results_dir=out_dir,
        build_model_fn=_factory(cfg),
        needs_seed=True,
        max_steps=None,
        max_train_size=cfg.get("max_train_size"),
        save_predictions=False,
    )
    return results


# ── Check 2: NBEATS sampling sanity check ──────────────────────────────────
def check_nbeats_sampling_sanity():
    """Run NBEATS twice on M4: once stratified, once random.
    We need to verify the cross-model ranking (NBEATS vs LightGBM)
    holds across both sampling methods.
    """
    print(f"\n{'#'*60}")
    print(f"# Check 2: NBEATS sampling sanity check (stratified vs random)")
    print(f"{'#'*60}")

    cfg = M4_CONFIG
    out_base = INSURANCE_DIR / "nbeats_sampling"

    def _factory(_cfg):
        def _build(seed, max_steps=None):
            return build_nbeats(
                horizon=_cfg["horizon"],
                input_size=_cfg["input_size"],
                freq=_cfg["freq"],
                seed=seed,
                max_steps=max_steps,
            )
        return _build

    rows = []
    for label, stratified in [("stratified", True), ("random", False)]:
        print(f"\n  ── pass: {label} ──")
        df_train, df_test = load_m4_monthly(
            n_series=cfg["n_series_sample"],
            random_state=42,
            stratified=stratified,
        )
        df_full = pd.concat([df_train, df_test], ignore_index=True)

        out_dir = out_base / label
        out_dir.mkdir(parents=True, exist_ok=True)
        results = run_walk_forward(
            df_full=df_full,
            dataset_name="M4",
            horizon=cfg["horizon"],
            input_size=cfg["input_size"],
            freq=cfg["freq"],
            season_length=cfg["season_length"],
            n_windows=cfg["walk_forward_windows"],
            seeds=SEEDS,
            results_dir=out_dir,
            build_model_fn=_factory(cfg),
            needs_seed=True,
            max_steps=None,
            max_train_size=cfg.get("max_train_size"),
            save_predictions=False,
        )
        if not results.empty:
            rows.append({
                "sampling": label,
                "model": "NBEATS",
                "mae_mean": results["mae_mean"].mean(),
                "mae_std": results["mae_mean"].std(),
                "mase_mean": results["mase_mean"].mean(),
                "mase_median": results["mase_median"].mean(),
                "n_runs": int(len(results)),
            })

    comparison = pd.DataFrame(rows)
    out_path = out_base / "comparison.csv"
    comparison.to_csv(out_path, index=False)
    print(f"\n  Saved comparison → {out_path}")
    print(comparison.to_string(index=False))
    return comparison


# ── Check 3: TimesNet undertraining check ──────────────────────────────────
def check_timesnet_undertraining():
    """Run TimesNet on M5 with MAX_STEPS=2000 to see if 800 was undertrained.
    Cheaper than testing on M4 (M5 has shorter horizon) and the M5 main-run
    result was 0.9815, so we have a clean comparison point.
    """
    print(f"\n{'#'*60}")
    print(f"# Check 3: TimesNet undertraining (M5, max_steps=2000)")
    print(f"{'#'*60}")

    cfg = M5_CONFIG

    def _factory(_cfg):
        def _build(seed, max_steps=None):
            # Override max_steps directly to 2000, ignoring the 800 default
            return build_timesnet(
                horizon=_cfg["horizon"],
                input_size=_cfg["input_size"],
                freq=_cfg["freq"],
                seed=seed,
                max_steps=2000,
                e_layers=2,  # paper default for M5
            )
        return _build

    df_train, df_test = load_m5(n_series=cfg["n_series_sample"])
    df_full = pd.concat([df_train, df_test], ignore_index=True)

    out_dir = INSURANCE_DIR / "timesnet_undertraining"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_walk_forward(
        df_full=df_full,
        dataset_name="M5",
        horizon=cfg["horizon"],
        input_size=cfg["input_size"],
        freq=cfg["freq"],
        season_length=cfg["season_length"],
        n_windows=cfg["walk_forward_windows"],
        seeds=SEEDS,
        results_dir=out_dir,
        build_model_fn=_factory(cfg),
        needs_seed=True,
        max_steps=None,  # let factory set its own max_steps=2000
        max_train_size=cfg.get("max_train_size"),
        save_predictions=False,
    )
    return results


def main():
    INSURANCE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#'*60}")
    print(f"# Insurance Checks")
    print(f"# 3 quick experiments to defend the main-run findings")
    print(f"{'#'*60}")
    t0 = time.time()

    # Track main-run M5 LightGBM and TimesNet for comparison
    main_summary = pd.read_csv(RESULTS_DIR / "summary_table.csv")
    main_m5_lgbm = main_summary[
        (main_summary["dataset"] == "M5") & (main_summary["model"] == "LightGBM")
    ]
    main_m5_timesnet = main_summary[
        (main_summary["dataset"] == "M5") & (main_summary["model"] == "TimesNet")
    ]
    main_m4_nbeats = main_summary[
        (main_summary["dataset"] == "M4") & (main_summary["model"] == "NBEATS")
    ]
    main_m4_lgbm = main_summary[
        (main_summary["dataset"] == "M4") & (main_summary["model"] == "LightGBM")
    ]

    # ── Run all three checks ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Running Check 1 / 3 ...")
    print("="*60)
    bare_lgbm = check_m5_lightgbm_no_features()

    print("\n" + "="*60)
    print("  Running Check 2 / 3 ...")
    print("="*60)
    nbeats_sampling = check_nbeats_sampling_sanity()

    print("\n" + "="*60)
    print("  Running Check 3 / 3 ...")
    print("="*60)
    timesnet_2000 = check_timesnet_undertraining()

    # ── Final summary ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'#'*60}")
    print(f"# Insurance Checks complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'#'*60}\n")

    print("─" * 60)
    print(" CHECK 1: M5 LightGBM bare-bones vs full features")
    print("─" * 60)
    if not bare_lgbm.empty and not main_m5_lgbm.empty:
        bare_mae = bare_lgbm["mae_mean"].mean()
        full_mae = float(main_m5_lgbm["mae_mean"].iloc[0])
        delta_pct = 100 * (bare_mae - full_mae) / full_mae
        print(f"  Bare-bones LightGBM (no exog, no Tweedie):  MAE = {bare_mae:.4f}")
        print(f"  Main-run LightGBM (full features+Tweedie):  MAE = {full_mae:.4f}")
        print(f"  Delta: {delta_pct:+.1f}% — features {'helped' if delta_pct > 0 else 'hurt'}")
        print(f"  Verdict: LightGBM is {'still' if bare_mae > 0.955 else 'NO LONGER'} "
              f"worse than NBEATS (0.955) without features")

    print("\n" + "─" * 60)
    print(" CHECK 2: NBEATS stratified vs random sampling on M4")
    print("─" * 60)
    if not nbeats_sampling.empty:
        strat_row = nbeats_sampling[nbeats_sampling["sampling"] == "stratified"]
        rand_row = nbeats_sampling[nbeats_sampling["sampling"] == "random"]
        if not strat_row.empty and not rand_row.empty:
            strat_mae = float(strat_row["mae_mean"].iloc[0])
            rand_mae = float(rand_row["mae_mean"].iloc[0])
            print(f"  NBEATS stratified: MAE = {strat_mae:.4f}")
            print(f"  NBEATS random:     MAE = {rand_mae:.4f}")
            print(f"  Earlier sanity check (LightGBM):")
            print(f"    LightGBM stratified: 572.90 / random: 551.40")
            print(f"  Ranking check: NBEATS should beat LightGBM under both samplings")
            if not main_m4_nbeats.empty and not main_m4_lgbm.empty:
                lgbm_strat_mae = float(main_m4_lgbm["mae_mean"].iloc[0])
                if strat_mae < lgbm_strat_mae and rand_mae < 551.40:
                    print(f"    ✓ Ranking PRESERVED across both sampling methods")
                else:
                    print(f"    ✗ Ranking may have FLIPPED — investigate")

    print("\n" + "─" * 60)
    print(" CHECK 3: TimesNet at max_steps=2000 vs main-run 800")
    print("─" * 60)
    if not timesnet_2000.empty and not main_m5_timesnet.empty:
        long_mae = timesnet_2000["mae_mean"].mean()
        short_mae = float(main_m5_timesnet["mae_mean"].iloc[0])
        delta_pct = 100 * (long_mae - short_mae) / short_mae
        print(f"  TimesNet @ 2000 steps: MAE = {long_mae:.4f}")
        print(f"  TimesNet @ 800 steps:  MAE = {short_mae:.4f}")
        print(f"  Delta: {delta_pct:+.1f}%")
        if abs(delta_pct) < 2:
            print(f"  Verdict: 800 steps was SUFFICIENT (delta < 2%)")
        elif delta_pct < -2:
            print(f"  Verdict: TimesNet was UNDERTRAINED at 800 — main result is pessimistic")
        else:
            print(f"  Verdict: Longer training HURT — overfitting kicked in")

    print("\n")


if __name__ == "__main__":
    main()
