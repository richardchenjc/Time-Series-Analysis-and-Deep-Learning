# DSS5104 CA2 — Deep Learning for Time-Series Forecasting

This project benchmarks **9 forecasting models** across **3 datasets** using a rigorous sliding-window walk-forward evaluation protocol with 3 random seeds. The central research question is: *when, if ever, is the added complexity of a deep learning model justified over classical or linear baselines?*

---

## Key Findings

- **N-BEATS and AutoARIMA are statistically indistinguishable on M4** (Friedman-Nemenyi clique + Diebold-Mariano 74% no significant difference). A 60-year-old classical method matches the best deep model on mixed-domain monthly data.
- **DLinear uniquely and significantly wins Traffic**, beating all deep learning models including PatchTST. Linear models suffice for strongly periodic data.
- **N-BEATS uniquely and significantly wins M5**, the only dataset where deep learning provides a clear advantage.
- **Model complexity does not predict performance.** N-BEATS (moderate complexity, explicit trend-seasonality bias) outperforms architecturally more complex models (TimesNet, PatchTST) that lack matched inductive biases.
- **AutoARIMA is the most expensive model on Traffic** (9,234s = 2.6h), 13× more than TimesNet. Classical ≠ cheap.
- **At small sample sizes (n=100), DLinear beats PatchTST on M4**, replicating Zeng et al. (2023). PatchTST overtakes DLinear only from n ≥ 300.
- **PatchTST is highly sensitive to lookback length** (48% MAE spread) but robust to patch size (1.1% spread).

---

## Models

| Model | Category | Architecture |
|-------|----------|-------------|
| PatchTST | Deep Learning | Transformer (patch-based input) |
| N-BEATS | Deep Learning | Deep residual with basis expansion (trend + seasonality stacks) |
| TiDE | Deep Learning | MLP encoder-decoder |
| DeepAR | Deep Learning | RNN (LSTM, autoregressive, probabilistic output) |
| TimesNet | Deep Learning (Additional) | CNN with FFT-based period detection and 2D convolution |
| **DLinear** | **Linear Baseline** | **Single linear layer mapping past → future** |
| Seasonal Naive | Classical Baseline | Repeat last observed seasonal cycle |
| AutoARIMA | Classical Baseline | Auto-selected ARIMA via information criteria |
| LightGBM | Classical Baseline (ML) | Gradient-boosted trees with hand-crafted lag features |

---

## Datasets

| Dataset | Type | Domain | Series Sampled | Frequency | Horizon |
|---------|------|--------|----------------|-----------|---------|
| M4 Monthly | Univariate | Mixed (6 domains) | 1,000 (stratified by category) | Monthly | 18 |
| M5 | Hierarchical | Retail sales (Walmart) | 500 (stratified by volume quintile) | Daily | 28 |
| Traffic | Multivariate | SF Bay Area road occupancy | 862 (full population) | Hourly | 24 |

### Data Download

Datasets are **not included** in this repository. Download from:

- **M4 Monthly** — [Kaggle M4 Forecasting Competition](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset). Place CSVs in `Data/M4/`.
- **M5** — [Kaggle M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data). Place CSVs in `Data/M5/`.
- **Traffic** — [Monash Time Series Repository](https://zenodo.org/records/4656132). Rename to `Traffic.tsf`, place in `Data/`.

---

## Setup

### Requirements

- **Python 3.12** (tested; 3.13 has dependency issues with `ray`)
- NVIDIA GPU with CUDA 12.8+ recommended (tested on RTX 5070 Ti)

### Installation

```bash
pip install -r requirements.txt

# For CUDA GPU support (required for deep learning models):
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Apple Silicon / MPS Note

On Apple Silicon, set `PYTORCH_ENABLE_MPS_FALLBACK=1` (done automatically in `config.py`). DeepAR's Student-t sampling partially runs on CPU.

---

## Running Experiments

### Reproduce All Results

```bash
# Step 1: Run all 9 model pipelines (main results)
python pipelines/run_all.py

# Step 2: Run data-volume and HP sensitivity sweeps
python pipelines/run_night2.py           # HP sensitivity + M4/M5 data-volume
python pipelines/run_night3.py           # AutoARIMA sweep + Traffic sweep + DM test

# Step 3: Run analysis and generate all figures
python analysis/aggregate_results.py     # Summary tables
python analysis/plot_results.py          # Main comparison plots
python analysis/plot_data_volume.py      # Data-volume curves
python analysis/plot_sensitivity.py      # HP sensitivity plots
python analysis/plot_per_horizon.py      # Per-horizon MAE decomposition
python analysis/plot_per_series_stratified.py  # Per-series subgroup analysis
python analysis/plot_cost_vs_accuracy.py # Cost frontier plots
python analysis/significance_test.py     # Friedman/Nemenyi CD diagrams
python analysis/plot_learning_curves.py  # Training/validation loss curves
python analysis/plot_val_curves.py       # Val-only clean version
python analysis/plot_example_preds.py    # Example prediction plots
```

### Smoke Test (Quick Validation)

```bash
python pipelines/run_all.py --smoke-test
```

---

## Experimental Protocol

### Preprocessing
- All series converted to long format (`unique_id`, `ds`, `y`).
- Per-series standard normalisation applied by NeuralForecast internally.
- Missing values filled with zero. No manual differencing or detrending.
- LightGBM receives dataset-specific lag features, rolling statistics, and (on M5) calendar exogenous features.

### Sampling
- **Stratified sampling** used for M4 (by domain category) and M5 (by volume quintile) to ensure all data regimes influence the comparison. Verified that model rankings are preserved across stratified and random sampling methods.

### Walk-Forward Evaluation
- **2 sliding windows** per dataset (non-overlapping test sets).
- Training history capped at `max_train_size` per dataset.
- Series with < `input_size + horizon + 1` observations filtered per window.

### Metrics
- **MAE**: within-dataset accuracy metric.
- **MASE**: scale-free metric (M4 official). Mean and median reported; mean-median gap is a robustness diagnostic.

### Seeds
- **3 random seeds** (42, 123, 456) for stochastic models.
- Deterministic baselines run once per window.

---

## Hardware and Runtime

| Item | Details |
|------|---------|
| GPU | NVIDIA RTX 5070 Ti (16 GB VRAM) |
| CUDA | 12.8 |
| Python | 3.12 |
| Total main run | ~3.2 hours |

---

## Project Structure

```
CA2/
├── config.py                   # Central configuration
├── requirements.txt            # Pinned dependencies
├── project_chronicle.md        # Complete analytical decision log
├── data_prep/                  # Dataset loading and formatting
│   ├── m4_prep.py
│   ├── m5_prep.py
│   ├── traffic_prep.py
│   └── sampling.py             # Stratified sampling helpers
├── models/                     # Model definitions (one file per model)
│   ├── __init__.py             # ModelSpec dataclass
│   ├── seasonal_naive.py
│   ├── auto_arima.py
│   ├── lightgbm.py
│   ├── patchtst.py
│   ├── nbeats.py
│   ├── tide.py
│   ├── deepar.py
│   ├── dlinear.py
│   └── timesnet.py
├── evaluation/                 # Evaluation engine
│   ├── walk_forward.py         # Sliding-window driver + prediction saving
│   ├── metrics.py              # MAE and MASE
│   └── timing.py               # Training time and GPU memory tracker
├── pipelines/                  # Per-model and orchestrator pipelines
│   ├── run_model.py            # Shared pipeline utility
│   ├── run_all.py              # Main orchestrator
│   ├── run_night2.py           # HP sensitivity + data-volume sweep
│   ├── run_night3.py           # Extensions + DM test
│   ├── run_<model>.py          # Per-model scripts (9 total)
│   ├── run_data_volume_sweep.py
│   ├── run_hp_sensitivity.py
│   └── run_sampling_sanity_check.py
├── analysis/                   # Post-experiment analysis
│   ├── aggregate_results.py
│   ├── eda.py                  # Exploratory data analysis
│   ├── plot_results.py         # Main comparison plots
│   ├── plot_data_volume.py
│   ├── plot_sensitivity.py
│   ├── plot_per_horizon.py
│   ├── plot_per_series_stratified.py
│   ├── plot_cost_vs_accuracy.py
│   ├── plot_learning_curves.py
│   ├── plot_val_curves.py
│   ├── plot_example_preds.py
│   ├── significance_test.py    # Friedman/Nemenyi + CD diagrams
│   ├── dm_test_nbeats_vs_autoarima.py
│   ├── insurance_checks.py
│   ├── training_budget_check.py
│   ├── traffic_lightgbm_bare_check.py
│   ├── sn_difficulty_check.py
│   └── export_per_horizon_csv.py
├── results/                    # Auto-created outputs
│   ├── *.csv                   # Per-model results
│   ├── predictions/            # Raw per-step predictions (parquet)
│   ├── plots/                  # All figures
│   ├── data_volume/            # Sweep results
│   ├── hp_sensitivity/         # Sensitivity results
│   ├── significance/           # Rank CSVs
│   └── learning_curves/        # Loss CSVs
└── Data/                       # Raw datasets (not in repo)
```

---

## References

- Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* (PatchTST). ICLR 2023.
- Oreshkin, B. N. et al. (2019). *N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting*.
- Das, A. et al. (2023). *Long-term Forecasting with TiDE: Time-series Dense Encoder*.
- Salinas, D. et al. (2020). *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks*. IJoF.
- Wu, H. et al. (2023). *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*. ICLR 2023.
- Zeng, A. et al. (2023). *Are Transformers Effective for Time Series Forecasting?* (DLinear). AAAI 2023.
