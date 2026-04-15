"""
Diebold-Mariano test on NBEATS vs AutoARIMA on M4.

The Nemenyi CD analysis already found these two models statistically
indistinguishable on M4 ({NBEATS, AutoARIMA} clique with ranks 3.61
and 3.65, gap < CD = 0.38). This script adds a SECOND statistical
test, the Diebold-Mariano test, applied per-series, to back up that
finding with a different methodology.

Why DM in addition to Nemenyi?
------------------------------
- Nemenyi works on per-series RANKS across all 9 models. It tells us
  "NBEATS and AutoARIMA have indistinguishable rank distributions
  across 991 M4 series."
- DM works on per-series RAW LOSSES for a specific pair of models.
  It uses the actual loss values rather than ranks, so it has more
  power to detect pairwise differences when they exist.

Used together, the two tests give complementary evidence.

Small-sample correction
-----------------------
M4 horizon is 18 and we have 2 windows, so each (series) has at most
36 forecast observations to run DM on. That's small enough that the
naive normal approximation is unreliable. We use the Harvey-Leybourne-
Newbold (1997) small-sample correction:

    DM* = DM × sqrt((T + 1 - 2h + h(h-1)/T) / T)

and compare DM* against a t-distribution with T-1 degrees of freedom
rather than a standard normal. This is the standard small-sample fix
for DM in the forecasting literature.

Pooling strategy
----------------
For each M4 series, we concatenate predictions from both walk-forward
windows (2 windows × 18 horizon steps = 36 observations per series)
and run a single DM test on the pooled loss differential. This gives
slightly more statistical power than running DM per-(series, window)
and aggregating.

Output
------
  results/significance/dm_nbeats_vs_autoarima_M4.csv  (per-series results)
  Plus a summary printed to stdout.

References
----------
- Diebold, F.X., Mariano, R.S. (1995). "Comparing Predictive Accuracy."
  Journal of Business & Economic Statistics, 13(3), 253-263.
- Harvey, D., Leybourne, S., Newbold, P. (1997). "Testing the equality
  of prediction mean squared errors." International Journal of
  Forecasting, 13(2), 281-291.

Usage:
    python analysis/dm_test_nbeats_vs_autoarima.py
"""

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS_DIR

PRED_DIR = RESULTS_DIR / "predictions"
SIG_DIR = RESULTS_DIR / "significance"


def diebold_mariano_hln(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    h: int = 1,
) -> tuple[float, float, int]:
    """Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample fix.

    Computes the loss differential d_t = loss_a_t - loss_b_t, estimates its
    variance using a Newey-West HAC estimator with truncation lag (h-1),
    applies the HLN correction, and compares against a t-distribution with
    T-1 degrees of freedom.

    Parameters
    ----------
    loss_a, loss_b : np.ndarray
        Per-step loss arrays for model A and B (same length T).
    h : int
        Forecast horizon. Used to determine the truncation lag for
        the HAC variance estimator.

    Returns
    -------
    dm_stat : float
        Small-sample-corrected DM test statistic.
        Negative means model A has lower mean loss (A better).
    p_value : float
        Two-sided p-value under the t-distribution with T-1 dof.
    T : int
        Number of observations used.

    Notes
    -----
    If T is too small for the given h (specifically, T < 2h), the
    truncation lag is capped at max(1, T-2) so that the autocovariance
    computation has at least 2 observations per lag.

    If the HAC variance estimate is non-positive (possible with high
    autocorrelation), we fall back to the uncorrected sample variance
    to avoid sqrt(negative).
    """
    d = loss_a - loss_b
    T = len(d)
    if T < 3:
        return np.nan, np.nan, T

    d_bar = d.mean()

    # ── HAC variance with Bartlett kernel, lag = min(h-1, T-2) ──────────
    # Bartlett kernel: w_k = 1 - k/(L+1), ensures variance is non-negative
    # for any input (unlike the naive sum of autocovariances).
    max_lag = min(h - 1, T - 2)  # Never try to compute cov on <2 points
    max_lag = max(0, max_lag)

    gamma_0 = float(np.var(d, ddof=1))
    if max_lag == 0:
        hac_var = gamma_0
    else:
        hac_var = gamma_0
        for k in range(1, max_lag + 1):
            if T - k < 2:
                break
            weight = 1.0 - k / (max_lag + 1)  # Bartlett kernel weight
            # Covariance at lag k. Note: using demeaned product instead of
            # np.cov() to avoid the ddof=1 division issue when T-k is small.
            d_shift_a = d[k:] - d_bar
            d_shift_b = d[:-k] - d_bar
            cov_k = float(np.mean(d_shift_a * d_shift_b))
            hac_var += 2 * weight * cov_k

    # Ensure variance is positive — Bartlett kernel guarantees this in
    # theory, but numerical issues can still produce tiny negatives.
    if hac_var <= 0:
        hac_var = gamma_0

    var_d_bar = hac_var / T
    if var_d_bar <= 0 or np.isnan(var_d_bar):
        return np.nan, np.nan, T

    dm_raw = d_bar / np.sqrt(var_d_bar)

    # ── Harvey-Leybourne-Newbold correction ──
    # HLN adjust factor: sqrt((T + 1 - 2h + h(h-1)/T) / T)
    hln_numerator = T + 1 - 2 * h + h * (h - 1) / T
    if hln_numerator <= 0:
        # Can happen when h is very large relative to T; fall back to raw DM
        dm_stat = dm_raw
    else:
        hln_factor = np.sqrt(hln_numerator / T)
        dm_stat = dm_raw * hln_factor

    # t-distribution with T-1 degrees of freedom
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=T - 1))

    return dm_stat, p_value, T


def load_pair(model_a: str, model_b: str, dataset: str) -> pd.DataFrame:
    """Load both models' prediction parquets and join on (series, window, h_step)."""
    path_a = PRED_DIR / f"{model_a}_{dataset}.parquet"
    path_b = PRED_DIR / f"{model_b}_{dataset}.parquet"
    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {path_a if not path_a.exists() else path_b}"
        )

    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)

    # NBEATS has 6 runs per (series, window, h_step) (3 seeds × 2 windows);
    # AutoARIMA has 2 (deterministic × 2 windows). Collapse across seeds
    # so we have exactly one prediction per (series, window, h_step) pair.
    def _collapse_seeds(df):
        return (
            df.groupby(["unique_id", "window", "horizon_step"])
              .agg(y_true=("y_true", "first"),
                   y_pred=("y_pred", "mean"))
              .reset_index()
        )

    a = _collapse_seeds(df_a).rename(columns={"y_pred": "y_pred_a"})
    b = _collapse_seeds(df_b).rename(columns={"y_pred": "y_pred_b"})
    merged = a.merge(
        b[["unique_id", "window", "horizon_step", "y_pred_b"]],
        on=["unique_id", "window", "horizon_step"],
        how="inner",
    )
    return merged


def per_series_dm(merged: pd.DataFrame, h: int = 18) -> pd.DataFrame:
    """Run DM test per series, pooling observations across all windows.

    For each unique_id, we concatenate predictions from all walk-forward
    windows into one loss differential sequence and run a single DM test
    on the combined series. This gives up to 2*h = 36 observations per
    series on M4, which is the max we can get from our setup.
    """
    rows = []
    for series_id, grp in merged.groupby("unique_id"):
        # Sort by (window, horizon_step) to respect temporal ordering
        grp = grp.sort_values(["window", "horizon_step"])
        loss_a = (grp["y_true"] - grp["y_pred_a"]).values ** 2
        loss_b = (grp["y_true"] - grp["y_pred_b"]).values ** 2
        dm_stat, p_value, T = diebold_mariano_hln(loss_a, loss_b, h=h)
        rows.append({
            "unique_id": series_id,
            "n_obs": T,
            "mse_a": float(loss_a.mean()),
            "mse_b": float(loss_b.mean()),
            "dm_stat": dm_stat,
            "p_value": p_value,
        })
    return pd.DataFrame(rows)


def main():
    SIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# Diebold-Mariano test: NBEATS vs AutoARIMA on M4")
    print(f"# (with Harvey-Leybourne-Newbold small-sample correction)")
    print(f"{'#'*60}")
    print()
    print("Loading predictions from both models...")
    merged = load_pair("NBEATS", "AutoARIMA", "M4")
    n_series = merged["unique_id"].nunique()
    print(f"  Found {len(merged)} prediction rows across {n_series} series")

    print("\nRunning per-series DM tests (pooling across windows)...")
    results = per_series_dm(merged, h=18)  # M4 horizon = 18

    # Filter out NaN p-values (series with too few observations)
    valid = results.dropna(subset=["p_value"])
    dropped = len(results) - len(valid)
    print(f"  {len(valid)} valid tests, {dropped} dropped (insufficient data)")

    if len(valid) == 0:
        print("\n  ✗ No valid DM tests could be computed. Something is wrong.")
        print(f"  Sample of results with NaN p-values:")
        print(results.head(10).to_string(index=False))
        return

    # ── Aggregate ────────────────────────────────────────────────────────
    sig = valid[valid["p_value"] < 0.05]
    nbeats_wins = sig[sig["dm_stat"] < 0]
    autoarima_wins = sig[sig["dm_stat"] > 0]
    not_sig = valid[valid["p_value"] >= 0.05]

    print(f"\n  ── Aggregate results across {len(valid)} series ──")
    print(f"  NBEATS significantly better:    {len(nbeats_wins):4d}  ({100*len(nbeats_wins)/len(valid):5.1f}%)")
    print(f"  AutoARIMA significantly better: {len(autoarima_wins):4d}  ({100*len(autoarima_wins)/len(valid):5.1f}%)")
    print(f"  No significant difference:      {len(not_sig):4d}  ({100*len(not_sig)/len(valid):5.1f}%)")

    median_p = valid["p_value"].median()
    mean_dm = valid["dm_stat"].mean()
    median_dm = valid["dm_stat"].median()
    print(f"\n  Median p-value across all tests: {median_p:.3f}")
    print(f"  Mean DM statistic:               {mean_dm:+.3f}")
    print(f"  Median DM statistic:             {median_dm:+.3f}")
    if mean_dm < 0:
        print(f"    (negative → NBEATS has lower mean loss, on average)")
    else:
        print(f"    (positive → AutoARIMA has lower mean loss, on average)")

    # ── Verdict ──────────────────────────────────────────────────────────
    print(f"\n  ── VERDICT ──")
    not_sig_pct = 100 * len(not_sig) / len(valid)
    nbeats_pct = 100 * len(nbeats_wins) / len(valid)
    autoarima_pct = 100 * len(autoarima_wins) / len(valid)

    if not_sig_pct > 50:
        print(f"  Most series ({not_sig_pct:.0f}%) show NO significant difference")
        print(f"  between NBEATS and AutoARIMA. The DM test CONFIRMS the Nemenyi")
        print(f"  finding that these two models are statistically indistinguishable")
        print(f"  on M4 mixed-domain monthly data.")
    elif nbeats_pct > 2 * autoarima_pct:
        print(f"  NBEATS wins significantly on {nbeats_pct:.0f}% of series,")
        print(f"  AutoARIMA wins on only {autoarima_pct:.0f}%. The DM test suggests")
        print(f"  NBEATS has a per-series edge that Nemenyi's rank-based test missed.")
    elif autoarima_pct > 2 * nbeats_pct:
        print(f"  AutoARIMA wins significantly on {autoarima_pct:.0f}% of series,")
        print(f"  NBEATS on only {nbeats_pct:.0f}%. Surprising — contradicts NBEATS's")
        print(f"  slightly better mean rank. Investigate.")
    else:
        print(f"  Mixed result: {nbeats_pct:.0f}% NBEATS wins, {autoarima_pct:.0f}% AutoARIMA wins,")
        print(f"  {not_sig_pct:.0f}% no significant difference. Broadly consistent with the")
        print(f"  Nemenyi finding of statistical indistinguishability, though some")
        print(f"  per-series differences exist in both directions.")

    out_path = SIG_DIR / "dm_nbeats_vs_autoarima_M4.csv"
    valid.to_csv(out_path, index=False)
    print(f"\n  Saved per-series results: {out_path}")


if __name__ == "__main__":
    main()
