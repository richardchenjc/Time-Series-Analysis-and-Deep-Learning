"""
analysis/eda.py
===============
Exploratory Data Analysis for the three benchmark datasets (M4, M5, Traffic).

Produces per-dataset statistics, distribution plots, sample series, ADF / KPSS
stationarity tests, STL-based seasonality strength, mean ACF curves, and
aggregates everything into a single markdown report suitable for the
"Datasets" and "Preprocessing" sections of the final report.

Outputs
-------
results/eda/
├── full/                              # EDA on the FULL datasets
│   ├── M4/
│   │   ├── basic_stats.csv            # one row per series
│   │   ├── summary.json               # aggregated stats for the report
│   │   ├── length_distribution.png
│   │   ├── scale_distribution.png
│   │   ├── value_distribution.png
│   │   ├── sample_series.png
│   │   ├── mean_acf.png
│   │   ├── seasonality_strength.png
│   │   └── zero_fraction.png
│   ├── M5/ …
│   └── Traffic/ …
├── sampled/                            # same structure, on sampled subsets
└── report.md                           # combined markdown report

Usage
-----
    python analysis/eda.py                  # both full and sampled
    python analysis/eda.py --full-only
    python analysis/eda.py --sampled-only
    python analysis/eda.py --quick          # smaller test-sample, fast iteration
    python analysis/eda.py --test-sample 500

Notes
-----
- Basic per-series stats run on ALL series in the dataframe.
- Expensive per-series tests (ADF, KPSS, STL) run on `--test-sample` randomly
  chosen series (default 200) — we only need a distribution of values, not
  exhaustive per-series results. Bump to 500+ for publication-quality numbers.
- Requires `statsmodels` (add to requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# statsmodels is required for stationarity tests + STL
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.seasonal import STL
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed — ADF/KPSS/STL tests will be skipped.")
    print("         Run: pip install statsmodels")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import M4_CONFIG, M5_CONFIG, TRAFFIC_CONFIG, RESULTS_DIR
from data_prep.m4_prep import load_m4_monthly
from data_prep.m5_prep import load_m5
from data_prep.traffic_prep import load_traffic

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})
sns.set_theme(style="whitegrid", palette="Set2")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

EDA_DIR = RESULTS_DIR / "eda"
DEFAULT_TEST_SAMPLE = 200
DEFAULT_QUICK_SAMPLE = 50


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DatasetEDA:
    name: str
    config: dict
    loader: Callable
    df_train: pd.DataFrame = field(default=None, repr=False)
    df_test: pd.DataFrame = field(default=None, repr=False)


DATASETS = [
    DatasetEDA("M4",      M4_CONFIG,      load_m4_monthly),
    DatasetEDA("M5",      M5_CONFIG,      load_m5),
    DatasetEDA("Traffic", TRAFFIC_CONFIG, load_traffic),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _group_to_dict(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert a long-format df into {unique_id: y-array} for O(1) series access.

    Avoids O(n_series × n_rows) boolean masking inside per-series loops.
    """
    return {uid: g["y"].to_numpy() for uid, g in df.groupby("unique_id", sort=False)}


def _sample_uids(series_dict: dict, n_sample: int, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    uids = np.array(list(series_dict.keys()))
    return list(rng.choice(uids, size=min(n_sample, len(uids)), replace=False))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic per-series statistics
# ─────────────────────────────────────────────────────────────────────────────
def compute_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per series: length, missing/zero counts, central tendency, spread."""
    tmp = df.assign(
        _is_na=df["y"].isna(),
        _is_zero=(df["y"].fillna(-1) == 0),
    )
    stats = tmp.groupby("unique_id", sort=False).agg(
        length=("y", "size"),
        n_missing=("_is_na", "sum"),
        n_zero=("_is_zero", "sum"),
        mean=("y", "mean"),
        std=("y", "std"),
        min=("y", "min"),
        median=("y", "median"),
        max=("y", "max"),
    ).reset_index()
    stats["pct_missing"] = 100 * stats["n_missing"] / stats["length"]
    stats["pct_zero"]    = 100 * stats["n_zero"]    / stats["length"]
    # Coefficient of variation — scale-free dispersion measure
    stats["cv"] = stats["std"] / stats["mean"].abs().replace(0, np.nan)
    return stats


def summarise_stats(stats: pd.DataFrame) -> dict:
    """Aggregate a per-series stats table into a JSON-serialisable summary."""
    def _pct(s, q):
        return float(np.nanpercentile(s, q))

    return {
        "n_series": int(len(stats)),
        "length": {
            "min":    int(stats["length"].min()),
            "p25":    _pct(stats["length"], 25),
            "median": _pct(stats["length"], 50),
            "p75":    _pct(stats["length"], 75),
            "max":    int(stats["length"].max()),
            "mean":   float(stats["length"].mean()),
        },
        "scale": {
            "mean_of_series_means":   float(stats["mean"].mean()),
            "median_of_series_means": float(stats["mean"].median()),
            "min_series_mean":        float(stats["mean"].min()),
            "max_series_mean":        float(stats["mean"].max()),
            # How many orders of magnitude separate the smallest and largest series?
            "scale_ratio": float(
                stats["mean"].abs().max() / max(stats["mean"].abs().min(), 1e-9)
            ),
        },
        "missing": {
            "mean_pct_per_series":         float(stats["pct_missing"].mean()),
            "series_with_any_missing_pct": float((stats["n_missing"] > 0).mean() * 100),
        },
        "zeros": {
            "mean_zero_pct_per_series":   float(stats["pct_zero"].mean()),
            "median_zero_pct_per_series": float(stats["pct_zero"].median()),
            "series_mostly_zero_pct":     float((stats["pct_zero"] > 50).mean() * 100),
        },
        "cv": {
            "median": float(stats["cv"].median()),
            "p25":    _pct(stats["cv"], 25),
            "p75":    _pct(stats["cv"], 75),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stationarity (ADF + KPSS) on a random sample of series
# ─────────────────────────────────────────────────────────────────────────────
def run_stationarity(df: pd.DataFrame, n_sample: int, seed: int = 42) -> dict:
    """Run ADF and KPSS on up to `n_sample` series; report rejection rates.

    ADF  null: unit root (non-stationary).   p < 0.05 → reject → stationary.
    KPSS null: stationary.                   p < 0.05 → reject → non-stationary.
    """
    if not HAS_STATSMODELS:
        return {"skipped": True, "reason": "statsmodels not installed"}

    series_dict = _group_to_dict(df)
    chosen = _sample_uids(series_dict, n_sample, seed)

    adf_p, kpss_p = [], []
    for uid in chosen:
        y = series_dict[uid]
        if len(y) < 20 or np.std(y) == 0:
            continue
        try:
            adf_p.append(adfuller(y, autolag="AIC")[1])
        except Exception:
            pass
        try:
            kpss_p.append(kpss(y, regression="c", nlags="auto")[1])
        except Exception:
            pass

    adf_p  = np.asarray(adf_p)
    kpss_p = np.asarray(kpss_p)

    return {
        "n_tested_adf":  int(len(adf_p)),
        "n_tested_kpss": int(len(kpss_p)),
        "adf_pct_stationary":  float((adf_p < 0.05).mean() * 100) if len(adf_p) else None,
        "adf_median_pvalue":   float(np.median(adf_p))            if len(adf_p) else None,
        "kpss_pct_stationary": float((kpss_p >= 0.05).mean() * 100) if len(kpss_p) else None,
        "kpss_median_pvalue":  float(np.median(kpss_p))              if len(kpss_p) else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. STL seasonality strength
# ─────────────────────────────────────────────────────────────────────────────
def compute_seasonality_strength(
    df: pd.DataFrame, season_length: int, n_sample: int, seed: int = 42
) -> np.ndarray:
    """STL-based seasonality strength Fs ∈ [0, 1] for each sampled series.

    Fs = max(0, 1 − Var(resid) / Var(resid + seasonal))
    following Wang, Smyl & Hyndman (2006). Larger → stronger seasonality.
    """
    if not HAS_STATSMODELS:
        return np.array([])

    series_dict = _group_to_dict(df)
    chosen = _sample_uids(series_dict, n_sample, seed)

    fs_values = []
    min_len = 2 * season_length + 1
    for uid in chosen:
        y = series_dict[uid]
        if len(y) < min_len or np.std(y) == 0:
            continue
        try:
            res = STL(y, period=season_length, robust=True).fit()
            var_r  = np.var(res.resid)
            var_rs = np.var(res.resid + res.seasonal)
            if var_rs == 0:
                continue
            fs_values.append(max(0.0, 1.0 - var_r / var_rs))
        except Exception:
            continue

    return np.asarray(fs_values)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Mean ACF curve across sampled series
# ─────────────────────────────────────────────────────────────────────────────
def compute_mean_acf(
    df: pd.DataFrame, max_lag: int, n_sample: int, seed: int = 42
) -> np.ndarray:
    """Average autocorrelation function across sampled series.

    Returns an array of length max_lag+1 (index 0 = lag 0 = 1).
    """
    series_dict = _group_to_dict(df)
    chosen = _sample_uids(series_dict, n_sample, seed)

    acfs = []
    for uid in chosen:
        y = series_dict[uid].astype(float)
        if len(y) < max_lag + 2 or np.std(y) == 0:
            continue
        y = y - y.mean()
        denom = np.dot(y, y)
        if denom == 0:
            continue
        ac = np.array([np.dot(y[:len(y) - k], y[k:]) / denom for k in range(max_lag + 1)])
        acfs.append(ac)

    if not acfs:
        return np.zeros(max_lag + 1)
    return np.mean(np.stack(acfs), axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_length_distribution(stats: pd.DataFrame, out: Path, name: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(stats["length"], bins=40, color="#3498db", edgecolor="white")
    ax.set_xlabel("Series length (time steps)")
    ax.set_ylabel("Number of series")
    ax.set_title(f"{name}: Series length distribution")
    ax.axvline(stats["length"].median(), color="red", linestyle="--",
               label=f"Median = {int(stats['length'].median())}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_scale_distribution(stats: pd.DataFrame, out: Path, name: str):
    """Log-scale histogram of per-series mean values — shows heterogeneity."""
    fig, ax = plt.subplots(figsize=(8, 4))
    means = stats["mean"].replace(0, np.nan).abs().dropna()
    if len(means) == 0:
        ax.text(0.5, 0.5, "No non-zero means", ha="center", va="center",
                transform=ax.transAxes)
    else:
        ax.hist(np.log10(means), bins=40, color="#e67e22", edgecolor="white")
    ax.set_xlabel(r"$\log_{10}$(|per-series mean|)")
    ax.set_ylabel("Number of series")
    ax.set_title(f"{name}: Cross-series scale heterogeneity")
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_value_distribution(df: pd.DataFrame, out: Path, name: str):
    """Pooled target-value distribution (sampled for speed, log-y)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    y = df["y"].sample(min(len(df), 50_000), random_state=42).values
    ax.hist(y, bins=60, color="#2ecc71", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("y (target value)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"{name}: Pooled target distribution (sampled)")
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_sample_series(df: pd.DataFrame, out: Path, name: str, n: int = 9, seed: int = 42):
    """3×3 grid of example series from the dataset."""
    rng = np.random.default_rng(seed)
    uids = df["unique_id"].unique()
    chosen = rng.choice(uids, size=min(n, len(uids)), replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=False)
    series_dict = {uid: g.sort_values("ds") for uid, g in df[df["unique_id"].isin(chosen)].groupby("unique_id")}
    for ax, uid in zip(axes.flat, chosen):
        sub = series_dict[uid]
        ax.plot(sub["ds"].values, sub["y"].values, linewidth=0.8)
        ax.set_title(str(uid)[:30], fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes.flat[len(chosen):]:
        ax.set_visible(False)
    fig.suptitle(f"{name}: Sample series", fontsize=14)
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_mean_acf(acf_vals: np.ndarray, season_length: int, out: Path, name: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    lags = np.arange(len(acf_vals))
    ax.vlines(lags, 0, acf_vals, color="#34495e", linewidth=1.5)
    ax.scatter(lags, acf_vals, color="#34495e", s=20, zorder=3)
    if season_length < len(acf_vals):
        ax.axvline(season_length, color="red", linestyle="--", alpha=0.6,
                   label=f"Seasonal lag = {season_length}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Mean ACF")
    ax.set_title(f"{name}: Average autocorrelation across sampled series")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_seasonality_strength(fs: np.ndarray, out: Path, name: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(fs) == 0:
        ax.text(0.5, 0.5, "STL unavailable", ha="center", va="center",
                transform=ax.transAxes)
    else:
        ax.hist(fs, bins=30, color="#9b59b6", edgecolor="white", range=(0, 1))
        ax.axvline(np.median(fs), color="red", linestyle="--",
                   label=f"Median Fs = {np.median(fs):.2f}")
        ax.legend()
    ax.set_xlabel("STL seasonality strength Fs")
    ax.set_ylabel("Number of series")
    ax.set_title(f"{name}: Seasonality strength distribution")
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_zero_fraction(stats: pd.DataFrame, out: Path, name: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(stats["pct_zero"], bins=30, color="#e74c3c", edgecolor="white", range=(0, 100))
    ax.set_xlabel("% of zero-valued observations per series")
    ax.set_ylabel("Number of series")
    ax.set_title(f"{name}: Zero-inflation per series")
    ax.axvline(stats["pct_zero"].median(), color="black", linestyle="--",
               label=f"Median = {stats['pct_zero'].median():.1f}%")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-dataset orchestration
# ─────────────────────────────────────────────────────────────────────────────
def run_eda_for_dataset(
    dset: DatasetEDA,
    out_dir: Path,
    test_sample: int,
) -> dict:
    """Run every EDA step for one dataset, write artefacts, return a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = dset.name
    print(f"\n{'='*60}\n  EDA — {name}\n{'='*60}")

    # Combine train+test for a full dataset-level view
    df = pd.concat([dset.df_train, dset.df_test], ignore_index=True)
    season_length = dset.config["season_length"]

    # 1. Basic stats ─────────────────────────────────────────────────
    print("  [1/6] basic stats …")
    stats = compute_basic_stats(df)
    stats.to_csv(out_dir / "basic_stats.csv", index=False)
    summary = summarise_stats(stats)

    # 2. Stationarity ─────────────────────────────────────────────────
    print(f"  [2/6] stationarity (ADF+KPSS, n={test_sample}) …")
    summary["stationarity"] = run_stationarity(df, n_sample=test_sample)

    # 3. Seasonality strength ────────────────────────────────────────
    print("  [3/6] STL seasonality strength …")
    fs = compute_seasonality_strength(df, season_length, n_sample=test_sample)
    summary["seasonality"] = {
        "n_tested": int(len(fs)),
        "median_fs": float(np.median(fs)) if len(fs) else None,
        "p25_fs":    float(np.percentile(fs, 25)) if len(fs) else None,
        "p75_fs":    float(np.percentile(fs, 75)) if len(fs) else None,
        "pct_strong_fs_gt_0p6": float((fs > 0.6).mean() * 100) if len(fs) else None,
    }

    # 4. Mean ACF ─────────────────────────────────────────────────────
    print("  [4/6] mean ACF …")
    max_lag = min(3 * season_length, 100)
    acf_vals = compute_mean_acf(df, max_lag=max_lag, n_sample=test_sample)
    summary["acf"] = {
        "lag_1":        float(acf_vals[1]) if len(acf_vals) > 1 else None,
        "lag_seasonal": float(acf_vals[season_length]) if len(acf_vals) > season_length else None,
    }

    # 5. Plots ────────────────────────────────────────────────────────
    print("  [5/6] plots …")
    plot_length_distribution(stats, out_dir / "length_distribution.png", name)
    plot_scale_distribution(stats, out_dir / "scale_distribution.png", name)
    plot_value_distribution(df, out_dir / "value_distribution.png", name)
    plot_sample_series(df, out_dir / "sample_series.png", name)
    plot_mean_acf(acf_vals, season_length, out_dir / "mean_acf.png", name)
    plot_seasonality_strength(fs, out_dir / "seasonality_strength.png", name)
    plot_zero_fraction(stats, out_dir / "zero_fraction.png", name)

    # 6. JSON summary ─────────────────────────────────────────────────
    print("  [6/6] saving summary …")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 7. Markdown report
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(v, digits=2):
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return f"{float(v):,.{digits}f}"


def write_markdown_report(results: dict[str, dict[str, dict]], out_path: Path):
    """`results[mode][dataset_name] = summary dict`."""
    lines = ["# Dataset EDA Report", ""]
    lines.append("Auto-generated by `analysis/eda.py`. Individual figures live in "
                 "`results/eda/<mode>/<dataset>/`.")
    lines.append("")

    for mode in ["full", "sampled"]:
        if mode not in results:
            continue
        lines += [f"## Mode: `{mode}`", ""]

        # Summary comparison table
        lines += [
            "### Summary table", "",
            "| Dataset | Series | Median length | % missing | Median % zero "
            "| Median Fs | ADF % stationary | ACF(lag_s) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for dname, s in results[mode].items():
            lines.append(
                f"| {dname} "
                f"| {_fmt(s['n_series'])} "
                f"| {_fmt(s['length']['median'])} "
                f"| {_fmt(s['missing']['mean_pct_per_series'])}% "
                f"| {_fmt(s['zeros']['median_zero_pct_per_series'])}% "
                f"| {_fmt(s['seasonality']['median_fs'])} "
                f"| {_fmt(s['stationarity'].get('adf_pct_stationary'))}% "
                f"| {_fmt(s['acf']['lag_seasonal'])} |"
            )
        lines.append("")

        # Per-dataset sections
        for dname, s in results[mode].items():
            lines += [f"### {dname}", ""]
            lines.append(f"- **Series**: {_fmt(s['n_series'])}")
            lines.append(f"- **Length**: min {_fmt(s['length']['min'])}, "
                         f"median {_fmt(s['length']['median'])}, "
                         f"max {_fmt(s['length']['max'])}")
            lines.append(f"- **Scale heterogeneity**: series-mean range "
                         f"{_fmt(s['scale']['min_series_mean'])} → "
                         f"{_fmt(s['scale']['max_series_mean'])} "
                         f"(ratio {_fmt(s['scale']['scale_ratio'])})")
            lines.append(f"- **Missing values**: "
                         f"{_fmt(s['missing']['mean_pct_per_series'])}% of observations; "
                         f"{_fmt(s['missing']['series_with_any_missing_pct'])}% of series affected")
            lines.append(f"- **Zero inflation**: median "
                         f"{_fmt(s['zeros']['median_zero_pct_per_series'])}% zero per series; "
                         f"{_fmt(s['zeros']['series_mostly_zero_pct'])}% of series are >50% zero")
            sstat = s["stationarity"]
            if not sstat.get("skipped"):
                lines.append(f"- **Stationarity**: ADF flags "
                             f"{_fmt(sstat.get('adf_pct_stationary'))}% stationary, "
                             f"KPSS flags {_fmt(sstat.get('kpss_pct_stationary'))}% stationary "
                             f"(n={sstat.get('n_tested_adf')})")
            sea = s["seasonality"]
            lines.append(f"- **Seasonality (STL Fs)**: median {_fmt(sea['median_fs'])}, "
                         f"{_fmt(sea['pct_strong_fs_gt_0p6'])}% of series with Fs > 0.6")
            lines.append(f"- **ACF**: lag-1 = {_fmt(s['acf']['lag_1'])}, "
                         f"lag-seasonal = {_fmt(s['acf']['lag_seasonal'])}")
            lines.append("")
            for fig in ["length_distribution", "scale_distribution", "value_distribution",
                        "sample_series", "mean_acf", "seasonality_strength", "zero_fraction"]:
                rel = f"{mode}/{dname}/{fig}.png"
                lines += [f"![{dname} {fig}]({rel})", ""]
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote markdown report → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────
def _load_dataset(dset: DatasetEDA, sampled: bool):
    if sampled:
        n = dset.config["n_series_sample"]
        print(f"\n[{dset.name}] loading sampled ({n} series)…")
        dset.df_train, dset.df_test = dset.loader(n_series=n)
    else:
        print(f"\n[{dset.name}] loading FULL dataset…")
        dset.df_train, dset.df_test = dset.loader(n_series=None)


def main():
    parser = argparse.ArgumentParser(description="Run EDA on all three datasets.")
    parser.add_argument("--full-only", action="store_true", help="Skip sampled EDA")
    parser.add_argument("--sampled-only", action="store_true", help="Skip full EDA")
    parser.add_argument("--quick", action="store_true",
                        help="Smaller test-sample for fast iteration")
    parser.add_argument("--test-sample", type=int, default=DEFAULT_TEST_SAMPLE,
                        help="Number of series for ADF/KPSS/STL (default 200)")
    args = parser.parse_args()

    test_sample = DEFAULT_QUICK_SAMPLE if args.quick else args.test_sample

    EDA_DIR.mkdir(parents=True, exist_ok=True)

    modes = []
    if not args.sampled_only:
        modes.append("full")
    if not args.full_only:
        modes.append("sampled")
    if not modes:
        print("Nothing to do (both --full-only and --sampled-only set).")
        return

    all_results: dict[str, dict[str, dict]] = {}

    for mode in modes:
        print(f"\n{'#'*60}\n# MODE: {mode.upper()}\n{'#'*60}")
        all_results[mode] = {}
        for dset in DATASETS:
            _load_dataset(dset, sampled=(mode == "sampled"))
            out_dir = EDA_DIR / mode / dset.name
            summary = run_eda_for_dataset(dset, out_dir, test_sample=test_sample)
            all_results[mode][dset.name] = summary
            # Free memory — full M5 is ~1.5 GB
            dset.df_train = None
            dset.df_test = None

    write_markdown_report(all_results, EDA_DIR / "report.md")
    print(f"\n✓ EDA complete — see {EDA_DIR}")


if __name__ == "__main__":
    main()
