"""
Central configuration for DSS5104 CA2 — Deep Learning Time-Series Forecasting.

All paths, hyperparameters, dataset configs, and random seeds are defined here
so that the entire experiment is controlled from a single place.

Sample-size reasoning (main run)
--------------------------------
We sized each dataset to balance statistical power, stratum coverage, and
overnight compute budget. Full reasoning is in the report; the short form:

  M4      : 1000 series stratified by category (6 strata, ~167/stratum)
            — comfortably above the 100-per-stratum threshold for stable
              within-stratum estimates. Leaves headroom for a 100→2000
              data-volume sweep (below the population of 48k).

  M5      : 500 series stratified by volume quintile (5 strata, 100/stratum)
            — right at the per-stratum threshold. Halving from 1000 costs
              little information because the M5 story is qualitative
              (intermittent demand) rather than precision-dependent.

  Traffic : all 862 sensors. The population is small enough to use in full,
            and at n=50 stratification added no measurable benefit
            (within-quintile noise dominated). Using everything avoids the
            "why not just use all" methodological question.

Windows (walk-forward)
----------------------
2 non-overlapping windows per dataset. On the low end of literature
practice (3–5 is more common), but our cross-series statistical power is
substantial: each window averages MAE over 500–1000 series, so per-window
means are tight estimates. The ± values we report across (seed, window)
pairs therefore capture run-to-run stability rather than intrinsic noise.
Trading a 3rd window (~1.5× compute) for the HP sensitivity and
data-volume sweeps was the better use of overnight budget given the
6-day schedule.

Seeds
-----
3 per (model, window). Spec minimum. Statistical precision comes mostly
from cross-series averaging (each run's mean MAE is already tight because
it's the average over hundreds of series), not from cross-seed variation.
More seeds would primarily tighten the std estimate, not the means —
and the means are the primary comparison quantity.
"""

import os
from pathlib import Path

# MPS fallback: DeepAR's Student-T sampling uses aten::_standard_gamma which
# is not implemented on Apple MPS. This env var makes PyTorch fall back to CPU
# for unsupported ops only. Harmless on CUDA.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent      # CA2/
DATA_DIR = ROOT / "Data"
RESULTS_DIR = ROOT / "results"

# ─── Random Seeds (3 seeds) ──────────────────────────────────────────────────
SEEDS = [42, 123, 456]

# ─── Shared DL Training Hyperparameters ──────────────────────────────────────
# Raised from 400 → 800 steps on the 5070 Ti. Closer to what the PatchTST /
# TimesNet papers use and reduces the risk of under-training.
MAX_STEPS = 800
BATCH_SIZE = 32       # 16 GB VRAM gives comfortable headroom at this batch
EARLY_STOP_PATIENCE = 5
VAL_CHECK_STEPS = 50  # validate every 50 steps

# ─── Per-model Learning Rates ────────────────────────────────────────────────
# Transformer-based and MLP-encoder models (PatchTST, TiDE, TimesNet) are
# sensitive to high LRs — 1e-4 is the standard recommendation from each
# paper. MLP residual (N-BEATS) and RNN (DeepAR) models are stable at 1e-3.
# DLinear is a single linear layer — 1e-3 is fine.
LR_TRANSFORMER = 1e-4   # PatchTST, TimesNet, TiDE
LR_MLP = 1e-3           # N-BEATS, DLinear
LR_RNN = 1e-3           # DeepAR

# ─── M4 Dataset Config (Monthly subset) ──────────────────────────────────────
M4_CONFIG = {
    "name": "M4",
    "freq": "ME",         # pandas month-end frequency
    "season_length": 12,
    "horizon": 18,        # official M4 Monthly forecast horizon
    "input_size": 36,     # 2× horizon lookback
    "train_csv": DATA_DIR / "M4" / "Monthly-train.csv",
    "test_csv":  DATA_DIR / "M4" / "Monthly-test.csv",
    "info_csv":  DATA_DIR / "M4" / "m4_info.csv",
    "n_series_sample": 1000,    # stratified across 6 M4 categories
    "walk_forward_windows": 2,
    "max_train_size": 144,      # 12 years of monthly history
}

# ─── M5 Dataset Config ───────────────────────────────────────────────────────
M5_CONFIG = {
    "name": "M5",
    "freq": "D",          # daily
    "season_length": 7,
    "horizon": 28,
    "input_size": 56,     # 2× horizon lookback
    "sales_csv":    DATA_DIR / "M5" / "sales_train_evaluation.csv",
    "calendar_csv": DATA_DIR / "M5" / "calendar.csv",
    "n_series_sample": 500,     # stratified across 5 volume quintiles
    "walk_forward_windows": 2,
    "max_train_size": 365,      # 1 year of daily history
}

# ─── Traffic Dataset Config ──────────────────────────────────────────────────
TRAFFIC_CONFIG = {
    "name": "Traffic",
    "freq": "h",          # hourly
    "season_length": 24,
    "horizon": 24,        # 24-hour-ahead prediction
    "input_size": 168,    # 7 days lookback (7 × 24)
    "data_file": DATA_DIR / "Traffic.tsf",
    "n_series_sample": None,    # None = use all 862 sensors
    "walk_forward_windows": 2,
    "max_train_size": 672,      # 4 weeks (24 × 28) of hourly history
}

# ─── PatchTST Hyperparameters ─────────────────────────────────────────────────
# Defaults follow Nie et al. (2023). patch_len and stride are overridden
# per-dataset in pipelines/run_patchtst.py to align patches with seasonal units.
PATCHTST_PARAMS = {
    "patch_len": 16,        # default; overridden per-dataset
    "stride": 8,            # default; overridden per-dataset
    "n_heads": 8,           # paper uses 16, we use 8 for memory
    "hidden_size": 128,
    "encoder_layers": 3,
}

# ─── N-BEATS Hyperparameters (paper defaults) ────────────────────────────────
NBEATS_PARAMS = {
    "stack_types": ["trend", "seasonality"],
    "n_blocks": [3, 3],
    "mlp_units": [[512, 512], [512, 512]],
}

# ─── TiDE Hyperparameters ─────────────────────────────────────────────────────
TIDE_PARAMS = {
    "hidden_size": 256,          # 2× paper default for extra capacity
    "decoder_output_dim": 32,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
}

# ─── DeepAR Hyperparameters ───────────────────────────────────────────────────
DEEPAR_PARAMS = {
    "hidden_size": 128,
    "n_layers": 2,
}

# ─── TimesNet Hyperparameters ─────────────────────────────────────────────────
TIMESNET_PARAMS = {
    "top_k": 5,
    "num_kernels": 6,
    "d_model": 64,
    "d_ff": 64,
    "e_layers": 2,
}

# ─── LightGBM Hyperparameters ────────────────────────────────────────────────
# Per-dataset feature sets are configured in pipelines/run_lightgbm.py.
# Per-dataset objective overrides (Tweedie for M5) are also there.
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
}
