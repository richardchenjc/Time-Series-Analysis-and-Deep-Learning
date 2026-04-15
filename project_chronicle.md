# Complete Project Chronicle — DSS5104 CA2
# Every decision, finding, and analytical insight, in chronological order.
# This document serves as the single source of truth for report drafting.

═══════════════════════════════════════════════════════════════════════
PHASE 0: INITIAL STATE (teammate's repo)
═══════════════════════════════════════════════════════════════════════

## What existed
- 9 models implemented: PatchTST, NBEATS, TiDE, DeepAR, DLinear, TimesNet,
  SeasonalNaive, AutoARIMA, LightGBM
- 3 datasets: M4 Monthly (500 series), M5 Daily (200 series), Traffic Hourly (50 sensors)
- Walk-forward evaluation with 2 windows × 3 seeds
- Subprocess-isolated orchestrator (run_all.py)
- Hardware: 16GB MacBook Air M4 (MPS backend)
- Config: MAX_STEPS=400, BATCH_SIZE=16

## Original results (from README, pre-improvement)
- N-BEATS best on M4 and M5
- DLinear first on Traffic
- DeepAR worst everywhere
- LightGBM fastest but struggled on M5 (MASE 2.11)
- TimesNet most expensive (~3 hours alone, ~10,994s total)
- Total runtime ~4.4 hours

## Gaps identified in initial review
1. Key Question 3 (HP sensitivity, data volume, overfitting) completely unaddressed
2. Preprocessing was thin (just fillna(0) + NeuralForecast internal scaling)
3. No EDA or dataset characterization
4. Only 2 walk-forward windows (minimum defensible)
5. No significance testing
6. No per-horizon error decomposition
7. LightGBM handicapped (no calendar features on M5)
8. No stratified sampling (random only)


═══════════════════════════════════════════════════════════════════════
PHASE 1: EDA (analysis/eda.py)
═══════════════════════════════════════════════════════════════════════

## What we built
- analysis/eda.py: comprehensive EDA script reusing existing loaders
- analysis/eda_explore.ipynb: interactive companion notebook
- Added statsmodels to requirements.txt

## EDA findings (full population)
Dataset  | Median Fs | ACF lag  | ADF stationary | Scale ratio | Key finding
---------|-----------|----------|----------------|-------------|----------------------------------
M4       | 0.21      | lag12=0.66| 15% stationary| varies      | Mixed domain, mostly non-stationary
M5       | 0.14      | lag7=0.22 | 85% stationary| 16,924×     | 73% median zero, intermittent demand
Traffic  | 0.68      | lag24=0.74| 100% stationary| 13.88×     | Strongly periodic, textbook seasonal

## Key EDA insight: M5 zero-inflation
- Median 73.31% zeros per series, 78% of series >50% zero
- This is intermittent demand, not a normal forecasting problem
- MASE denominator doesn't go to zero (in-sample naive not constant), but is small
- fillna(0) is correct but report must acknowledge the sparsity

## Sampling bias discovered
- Random M4 sample had Fs=0.31 vs full population 0.21 (over-represented seasonal series)
- Random M5 sample had scale ratio 507× vs full 16,924× (missed extreme tails)
- This motivated stratified sampling (Phase 3)


═══════════════════════════════════════════════════════════════════════
PHASE 2: METRICS UPGRADE (walk_forward.py, aggregate_results.py)
═══════════════════════════════════════════════════════════════════════

## Decision: stick with MASE, not switch to sMAPE or RMSSE
- Assignment requires MAE + one of {MASE, sMAPE}
- MASE is the M4 official metric and academic standard
- sMAPE has 0/0 problem on M5 zeros AND known asymmetry bias
- RMSSE would require extra justification paragraph, not worth page budget
- Decision: MAE + MASE (mean and median), no second scale-free metric

## Changes made
- walk_forward.py now computes: mase_median, mase_n_total, mase_n_dropped per run
- aggregate_results.py now produces: mase_median_mean, mase_median_std, mase_drop_pct
- Backward-compatible: _ensure_columns() shim handles pre-patch CSVs

## Key finding: MASE drop% is 0% everywhere
- Original prediction was ~10-15% on M5. WRONG.
- Intermittent ≠ constant. Non-zero 27% of M5 observations make denominator > 0.
- mase_drop_pct is 0% on all datasets for all models.
- Mean-vs-median MASE gap is real but modest:
  M4: ratio 1.16×, M5: ratio 1.21×, Traffic: ratio 1.37×
- Traffic has biggest gap (a few extreme sensors), not M5 as expected

## Metrics paragraph for report
"We report MAE and MASE. To diagnose whether aggregate MASE could be
misleading on heterogeneous data, we additionally report median MASE.
No series produced undefined MASE values on any dataset. The mean-median
MASE ratio was 1.16× on M4, 1.21× on M5, and 1.37× on Traffic,
indicating all three datasets contain a minority of series with
disproportionately large errors."


═══════════════════════════════════════════════════════════════════════
PHASE 3: STRATIFIED SAMPLING (data_prep/sampling.py, loader patches)
═══════════════════════════════════════════════════════════════════════

## Design
- M4: stratified by category (Macro/Micro/Demographic/Industry/Finance/Other)
  using m4_info.csv, with length-quartile fallback if file missing
- M5: stratified by mean sales volume quintiles
- Traffic: stratified by mean occupancy quintiles
- Allocation: equal-per-stratum (deliberately non-representative for diversity)
- All loaders gain stratified=True/False parameter for sanity checking

## Verification
- M5 quintiles: clean 40/40/40/40/40 split confirmed
- M4 categories: 166-169 per stratum across 6 categories

## Effect on EDA metrics (sampled: old random vs new stratified)
Metric             | Full pop | Old random | New stratified | Direction
-------------------|----------|------------|----------------|----------
M4 Median Fs       | 0.21     | 0.31       | 0.25           | ✓ toward pop
M5 Median % zero   | 73.31%   | 74.27%     | 72.75%         | ✓ toward pop
M5 Scale ratio     | 16,924×  | 507×       | 475×           | barely changed

## Key insight: M5 intermittency is structural, not a sampling artifact
- Even after explicitly including high-volume items, M5 still 72.75% median zero
- High-volume Walmart items still have dead days, weekend-only sales, restocking gaps


═══════════════════════════════════════════════════════════════════════
PHASE 4: LIGHTGBM M5 CALENDAR FEATURES + TWEEDIE
═══════════════════════════════════════════════════════════════════════

## Features added to M5 only
- 5 exogenous columns: snap_CA, snap_TX, snap_WI, is_event, is_sporting_or_holiday
- Plumbed through ModelSpec.exog_cols → walk_forward._prepare_inputs → mlforecast X_df
- Non-M5 models silently strip exog columns (verified with SeasonalNaive)

## Result with L2 loss + calendar features
- LightGBM M5 MAE: 1.2547 (barely beat SeasonalNaive 1.2557 by 0.08%)
- LightGBM M5 MASE: 2.25 (much worse than SeasonalNaive 1.47)
- Calendar features helped MAE on high-volume items, hurt MASE on intermittent ones

## Tweedie loss experiment
- Switched to objective="tweedie", tweedie_variance_power=1.5
- Result: MAE 1.2734 (WORSE than L2's 1.2547)
- All three configs (SN, L2+features, Tweedie+features) within 1.5% MAE of each other
- Verdict: M5 MAE is set by intermittency noise floor, not model choice

## Key finding for report
"Even with SNAP/event features and Tweedie loss (the standard intermittent-demand
configuration), LightGBM cannot beat SeasonalNaive on M5. The MAE noise floor
on M5 is set by intermittency itself, not by the model or feature choice."


═══════════════════════════════════════════════════════════════════════
PHASE 5: SAMPLE SIZE BUMP + CONFIG CHANGES
═══════════════════════════════════════════════════════════════════════

## New sample sizes (justified)
- M4: 500 → 1000 (6 categories × ~167 per stratum, above 100-per-stratum threshold)
- M5: 200 → 500 (5 quintiles × 100 per quintile, at threshold)
- Traffic: 50 → all 862 (small population, no benefit from subsampling)

## Training config changes
- MAX_STEPS: 400 → 800 (papers use 1000+, 800 is compromise)
- BATCH_SIZE: 16 → 32 (5070 Ti has 16GB VRAM)
- All other HPs unchanged from paper defaults

## Per-model architecture overrides (from original + our additions)
- PatchTST: n_heads 4→8, patch_len/stride per-dataset aligned to seasonal units
- N-BEATS: mlp_units restored to paper default [[512,512],[512,512]]
- TiDE: hidden_size 128→256 (2× paper)
- DeepAR: hidden_size 64→128 (matches paper), 256 on Traffic
- TimesNet: e_layers=3 on Traffic (nested daily+weekly)
- LightGBM: full per-dataset feature engineering (M4: extended lags+rolling,
  M5: calendar+Tweedie, Traffic: hourly lags+rolling+date features)


═══════════════════════════════════════════════════════════════════════
PHASE 6: ENVIRONMENT FIX
═══════════════════════════════════════════════════════════════════════

## Problem chain
1. neuralforecast 0.1.0 installed (ancient, wrong API)
2. pytorch-lightning 2.6.1 (too new for old NF)
3. pip couldn't upgrade NF because ray has no Windows+Python3.13 wheel
4. Root cause: Python 3.13 on Windows

## Fix
- Created new venv with Python 3.12
- pip install neuralforecast>=1.7.0 (resolved to modern version)
- pip install torch --index-url https://download.pytorch.org/whl/cu128 (CUDA for 5070 Ti)
- Verified: torch CUDA True, Device NVIDIA GeForce RTX 5070 Ti, Capability (12,0)

## Additional fixes during testing
- UTF-8 console encoding: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  added to walk_forward.py and run_all.py (Windows cp1252 can't encode ✓/✗/→)
- DeepAR "exit code 3221226505" was NOT a CUDA issue — was the encoding bug
  cascading through subprocess error handling. Fixed by UTF-8 reconfigure.
- run_all.py success detection: Gemini fix added file-existence check instead
  of trusting Windows exit codes (0xC0000409 false alarm on process cleanup)


═══════════════════════════════════════════════════════════════════════
PHASE 7: MAIN RUN (Night 1)
═══════════════════════════════════════════════════════════════════════

## Run details
- Hardware: RTX 5070 Ti, 16GB VRAM, Python 3.12, CUDA 12.8
- Total time: 11,509s (3.20 hours) — much faster than estimated 13h
- 8/9 pipelines succeeded in orchestrator
- run_all.py crashed on summary print (UTF-8 encoding in orchestrator, NOT in models)

## M4 DL failure (all 6 neural models on M4)
- Root cause: walk_forward filter used input_size=36, but NeuralForecast
  carves out val_size=18 internally, leaving shortest series with only 18
  training timestamps < input_size=36
- Fix: changed filter to min_required = input_size + horizon (= 54 for M4)
- Re-ran 6 DL models on M4 individually, all succeeded

## DeepAR M4 special case
- DeepAR reports input_size=37 (library adds +1 for autoregressive target)
- With filter at 54, DeepAR got 3/6 runs (window 2 failed for some series)
- Fix: per-model input_size_offset parameter in walk_forward
- DeepAR pipeline passes input_size_offset=1, others pass 0
- After fix: DeepAR gets 6/6 runs, sees ~895 series vs ~900 for others

## M4 baseline re-run
- Re-ran SeasonalNaive, AutoARIMA, LightGBM on M4 with new filter
- Now all 9 models on M4 see same ~900 series (DeepAR ~895)
- mase_total_mean is uniform across models


═══════════════════════════════════════════════════════════════════════
PHASE 8: MAIN RESULTS (complete 9×3 table)
═══════════════════════════════════════════════════════════════════════

## M4 Monthly — ranked by MAE
Model          | MAE     | MASE_med | Category
---------------|---------|----------|------------------
NBEATS         | 484.8   | 0.744    | Deep Learning
AutoARIMA      | 500.2   | 0.708    | Classical
TimesNet       | 521.6   | 0.786    | Deep Learning
LightGBM       | 572.9   | 0.888    | Classical (ML)
DeepAR         | 586.9   | 0.972    | Deep Learning
PatchTST       | 598.8   | 0.915    | Deep Learning
SeasonalNaive  | 620.9   | 0.972    | Classical
DLinear        | 647.5   | 1.066    | Linear Baseline
TiDE           | 750.4   | 1.257    | Deep Learning

Key: NBEATS wins, but AutoARIMA is only 3% behind.
     PatchTST is 6th — Transformer mid-pack on mixed-domain monthly.
     TiDE is uniformly worst.

## M5 Daily — ranked by MAE
Model          | MAE    | MASE_med | Category
---------------|--------|----------|------------------
NBEATS         | 0.955  | 0.758    | Deep Learning
PatchTST       | 0.964  | 0.772    | Deep Learning
DeepAR         | 0.971  | 0.791    | Deep Learning
TimesNet       | 0.982  | 0.796    | Deep Learning
DLinear        | 0.984  | 0.811    | Linear Baseline
AutoARIMA      | 1.024  | 0.885    | Classical
TiDE           | 1.026  | 0.877    | Deep Learning
LightGBM       | 1.133  | 1.048    | Classical (ML)
SeasonalNaive  | 1.202  | 1.019    | Classical

Key: DL models dominate top 5. NBEATS uniquely best.
     LightGBM loses despite calendar features + Tweedie.
     Changed from earlier finding "nothing wins M5."

## Traffic Hourly — ranked by MAE
Model          | MAE    | MASE_med | Category
---------------|--------|----------|------------------
LightGBM       | 0.0101 | 0.491    | Classical (ML)
DLinear        | 0.0103 | 0.463    | Linear Baseline
NBEATS         | 0.0104 | 0.501    | Deep Learning
PatchTST       | 0.0107 | 0.522    | Deep Learning
TimesNet       | 0.0108 | 0.524    | Deep Learning
TiDE           | 0.0118 | 0.580    | Deep Learning
SeasonalNaive  | 0.0121 | 0.605    | Classical
DeepAR         | 0.0122 | 0.604    | Deep Learning
AutoARIMA      | 0.0124 | 0.627    | Classical

Key: LightGBM and DLinear tie for first.
     BUT: LightGBM's win depends on date features (see fairness audit).
     AutoARIMA is the SLOWEST model overall (9234s on Traffic alone).


═══════════════════════════════════════════════════════════════════════
PHASE 9: STATISTICAL SIGNIFICANCE (Friedman + Nemenyi + DM)
═══════════════════════════════════════════════════════════════════════

## Friedman test — all three datasets p ≈ 0
- M4:      χ² = 1397.3, p = 2.17e-296
- M5:      χ² = 1123.3, p = 3.63e-237
- Traffic: χ² = 1328.6, p = 1.53e-281
Overwhelming evidence that model rankings are systematic, not noise.

## Nemenyi clique structure

### M4 (CD = 0.382)
Clique 1: { NBEATS, AutoARIMA }         ranks [3.61, 3.65]  ← HEADLINE FINDING
Clique 2: { LightGBM, PatchTST, DeepAR } ranks [4.81, 5.11, 5.12]
Clique 3: { DLinear, SeasonalNaive }     ranks [5.68, 5.83]
Loners:   TimesNet (4.04), TiDE (7.16)

### M5 (CD = 0.537)
Loner:    NBEATS (2.91)                  ← uniquely best
Clique 1: { PatchTST, DeepAR }          ranks [3.58, 3.94]
Clique 2: { DeepAR, DLinear }           ranks [3.94, 4.47]  (overlaps with above)
Clique 3: { DLinear, TimesNet }         ranks [4.47, 4.51]  (overlaps)
Clique 4: { AutoARIMA, TiDE }           ranks [5.87, 5.93]
Clique 5: { LightGBM, SeasonalNaive }   ranks [6.68, 7.10]

### Traffic (CD = 0.409)
Loner:    DLinear (3.09)                 ← uniquely best
Loner:    LightGBM (3.59)               ← uniquely second
Clique 1: { NBEATS, PatchTST }          ranks [4.34, 4.58]
Clique 2: { PatchTST, TimesNet }        ranks [4.58, 4.86]  (overlaps)
Clique 3: { SeasonalNaive, AutoARIMA }  ranks [5.83, 6.13]
Clique 4: { AutoARIMA, TiDE, DeepAR }   ranks [6.13, 6.27, 6.30]

## Diebold-Mariano test (NBEATS vs AutoARIMA on M4)
- 992 valid per-series tests (pooled across windows, HLN correction)
- NBEATS significantly better:  126 series (12.7%)
- AutoARIMA significantly better: 131 series (13.2%)
- No significant difference: 735 series (74.1%)
- Median p-value: 0.312
- Mean DM statistic: -0.060 (essentially zero)
- VERDICT: DM CONFIRMS Nemenyi — the two are statistically indistinguishable.

## Headline statistical findings
1. On M4, NBEATS and AutoARIMA are statistically tied (Nemenyi + DM agree).
   A 60-year-old method matches the best DL model.
2. On M5, NBEATS is uniquely and significantly the best.
3. On Traffic, DLinear is uniquely and significantly the best.
   DLinear > LightGBM (statistically distinguishable).


═══════════════════════════════════════════════════════════════════════
PHASE 10: DATA VOLUME SWEEP (Night 2 + extensions)
═══════════════════════════════════════════════════════════════════════

## M4 sweep: NBEATS, PatchTST, DLinear, AutoARIMA, SeasonalNaive
n_series | NBEATS | PatchTST | DLinear | AutoARIMA | SeasonalNaive
---------|--------|----------|---------|-----------|-------------
100      | 595    | 729      | 657     | 545       | 672
300      | 524    | 607      | 649     | 485       | 597
1000     | 485    | 599      | 647     | 500       | 621
2000     | 552    | 643      | 729     | 514       | 624

### Findings from M4 sweep
1. AutoARIMA is UNIFORMLY competitive across all sample sizes.
   It's the flattest line (range: 485-545, ~12% spread).
   At n=100 and n=2000, AutoARIMA beats NBEATS.
   The "tie" at n=1000 is actually a brief crossover, not a stable equilibrium.

2. At n=100, DLinear (657) beats PatchTST (729).
   → Direct replication of Zeng et al. "linear beats Transformer at small n"
   At n=300+, PatchTST overtakes DLinear.

3. All trained models worsen at n=2000 (U-shape). SeasonalNaive is flat (+0.5%).
   → The degradation is model-related, not sample-difficulty.

4. Training-budget hypothesis REJECTED:
   NBEATS at n=2000 with MAX_STEPS=1600 got MAE 624 (WORSE than 552 at 800 steps).
   More training = worse. This is overfitting, not undertraining.

5. Interpretation: at n=2000, stratified sampling draws deeper into each category's
   hard tail. Models memorize diverse-but-noisy patterns → overfitting.
   n=1000 is the sweet spot for M4.

## M5 sweep: NBEATS, PatchTST, DLinear
n_series | NBEATS | PatchTST | DLinear
---------|--------|----------|--------
100      | 0.959  | 0.930    | 0.959
250      | 0.965  | 0.962    | 0.985
500      | 0.955  | 0.964    | 0.984

### Findings from M5 sweep
1. No benefit from scaling n=100 to n=500. All models ~flat.
2. PatchTST is best at n=100 (0.930), NBEATS best at n=500 (0.955).
3. DL models do NOT need large datasets on M5 — competitive at n=100.
4. Overlapping std bands → differences likely within noise.

## Traffic sweep: NBEATS, PatchTST, DLinear
n_series | NBEATS | PatchTST | DLinear
---------|--------|----------|--------
50       | 0.0128 | 0.0121   | 0.0115
150      | 0.0113 | 0.0114   | 0.0110
500      | 0.0104 | 0.0107   | 0.0102
862      | 0.0104 | 0.0107   | 0.0103

### Findings from Traffic sweep
1. DLinear wins at EVERY sample size, even n=50.
2. All models improve monotonically with more data (no U-shape).
3. At n=862, models converge within 4% of each other.
4. "More data helps everyone equally" on Traffic — the simplest story.


═══════════════════════════════════════════════════════════════════════
PHASE 11: HP SENSITIVITY (Night 2)
═══════════════════════════════════════════════════════════════════════

## Study 1: PatchTST × patch_len on M4 → 1.1% spread → ROBUST
- Replicates Nie et al. (2023) claim directly.
- patch_len ∈ {3, 6, 12} produces essentially identical MAE.

## Study 2: PatchTST × lookback on M4 → 48.3% spread → VERY SENSITIVE
- input_size ∈ {18, 36, 72, 144}
- PatchTST's M4 performance depends heavily on lookback choice.
- Our main run used input_size=36; different lookback could change ranking.
- IMPORTANT LIMITATION: cross-model comparison used shared input_size=36.

## Study 3: NBEATS × n_blocks on M4 → 1.4% spread → ROBUST
- n_blocks ∈ {[1,1], [3,3], [5,5]}
- NBEATS's win on M4 doesn't depend on depth choice.

## Study 4: DLinear × lookback on Traffic → 17.6% spread → SENSITIVE
- input_size ∈ {24, 48, 96, 168}
- DLinear's Traffic dominance depends on choosing appropriate lookback.
- At too-short lookback, it would lose to DL models.

## Overall sensitivity assessment
- Architectural HPs (patch_len, n_blocks): robust (<2% spread)
- Lookback/context length: sensitive (17-48% spread)
- Paper defaults are reasonable operating points for architectural HPs.
- Lookback should be tuned per-dataset — shared input_size is a limitation.


═══════════════════════════════════════════════════════════════════════
PHASE 12: INSURANCE CHECKS
═══════════════════════════════════════════════════════════════════════

## Check 1: M5 LightGBM bare-bones vs full features
- Bare-bones (lag features only, no exog, no Tweedie): MAE = 1.0954
- Full features (SNAP + events + Tweedie): MAE = 1.1333
- Delta: -3.4% — FEATURES HURT
- Both lose to NBEATS (0.955)
- Insight: Tweedie + calendar features interact poorly with lag space on M5.
  The "enhanced" LightGBM is actually a worse baseline than plain LightGBM.
- Report decision: keep full-features in main results (represents "practitioner
  effort"), document bare-bones in footnote.

## Check 2: NBEATS sampling sanity (stratified vs random on M4)
- NBEATS stratified: MAE = 484.8
- NBEATS random: MAE = 509.4
- LightGBM stratified: 572.9 / random: 551.4
- Ranking PRESERVED: NBEATS beats LightGBM under BOTH methods.
- Stratified sample is harder (as expected from equal-per-stratum allocation).
- Additional insight: NBEATS gains MORE from harder data than LightGBM.
  On the harder stratified sample, NBEATS improves more → DL handles hard
  tails better than tree-based ML.

## Check 3: TimesNet undertraining at 800 steps
- TimesNet M5 @ 2000 steps: MAE = 0.9784
- TimesNet M5 @ 800 steps: MAE = 0.9815
- Delta: -0.3% → 800 steps was SUFFICIENT.
- TimesNet is NOT undertrained. Its cost (~3h) is inherent, not fixable by
  reducing training.

## Check 4: Traffic LightGBM without date features
- Full LightGBM (with hour + dayofweek): MAE = 0.0101
- Bare-bones LightGBM (lag features only): MAE = 0.0105
- DLinear reference: MAE = 0.0103
- BARE-BONES LIGHTGBM LOSES TO DLINEAR.
- LightGBM's ranking #1 on Traffic DEPENDS on date features.
- This is a fairness concern: DL models don't get date features.
- Report must flag this in methodology/limitations.


═══════════════════════════════════════════════════════════════════════
PHASE 13: PER-HORIZON ANALYSIS
═══════════════════════════════════════════════════════════════════════

## Per-horizon CSV export
Best at h=1 and h=H:
  M4:      h=1 → NBEATS (MAE 257.8)     h=18 → NBEATS (MAE 680.2)
  M5:      h=1 → PatchTST (MAE 0.81)    h=28 → DeepAR (MAE 1.08)
  Traffic: h=1 → LightGBM (MAE 0.003)   h=24 → PatchTST (MAE 0.006)

## M4 per-horizon findings
- NBEATS and AutoARIMA track each other across ALL 18 horizon steps.
  Visually indistinguishable from h=1 to h=12, slight NBEATS advantage h=13-18.
  Reinforces the Nemenyi/DM "tied" finding — they're tied at every horizon too.
- TiDE is uniformly worst at every horizon.
- TimesNet is strong across horizons (2nd-3rd everywhere).

## M5 per-horizon findings
- Clear 7-day periodicity visible in all models (peaks at h=7, 14, 21, 28).
- NBEATS handles weekly peaks better than baselines.
- LightGBM and SeasonalNaive have notably higher peaks — struggle with
  weekly cycle transitions.

## Traffic per-horizon findings
- Daily cycle visible (peak h=17-18 = evening rush).
- DLinear and LightGBM track closely for h=1-10.
- At h=24, PatchTST and NBEATS actually overtake DLinear.
  → DLinear wins overall but not at every single step.


═══════════════════════════════════════════════════════════════════════
PHASE 14: FAIRNESS AUDIT
═══════════════════════════════════════════════════════════════════════

## Asymmetry 1: Per-model HP tuning effort
- PatchTST, TimesNet, DeepAR get per-dataset overrides (patch_len, e_layers, hidden_size)
- N-BEATS, TiDE, DLinear, AutoARIMA use global paper defaults
- Defense: overrides align model-specific knobs to data structure, not "tuning"
- N-BEATS won M4 without any per-dataset tuning → asymmetry didn't decide outcome

## Asymmetry 2: LightGBM gets feature engineering, DL models don't
- M5: LightGBM has 5 calendar exog columns that DL models don't see
- Traffic: LightGBM has hour + dayofweek features that DL models don't see
- Insurance check proved: LightGBM loses Traffic to DLinear WITHOUT date features
- Insurance check proved: LightGBM loses M5 to NBEATS WITH or WITHOUT features
- Impact: doesn't flip rankings, but fairness concern must be documented

## Asymmetry 3: Series filtering differences
- Baselines and DL models now see same ~900 M4 series (after Phase 7 fix)
- DeepAR sees ~895 due to +1 autoregressive offset
- M5 and Traffic unaffected (uniform series lengths)
- Documented and justified

## Asymmetry 4: No HP grid search on validation set
- All models use paper defaults or modest overrides
- DL models in published papers report post-tuning results
- Our DL numbers are plausibly 5-15% worse than tuned versions
- But classical baselines also use library defaults → fair comparison

## Asymmetry 5: Training budget fixed across model complexity
- MAX_STEPS=800 for all DL models regardless of complexity
- TimesNet undertraining check showed 800 is sufficient (delta -0.3%)
- DLinear converges in ~50-100 steps, TimesNet may use all 800

## Asymmetry 6: Cross-dataset sample size
- M4=1000, M5=500, Traffic=862
- Each justified independently by per-stratum coverage
- Cross-dataset comparisons should note different sample sizes

## Asymmetry 7: AutoARIMA stepwise search cap (nmodels=94)
- Some M4 series hit the built-in search limit
- Default behavior, not our handicap
- Could slightly underestimate AutoARIMA's potential

## Asymmetry 8: n=2000 overfitting finding
- All trained models degrade at n=2000, but SeasonalNaive doesn't
- More training steps makes it WORSE (budget hypothesis rejected)
- This is genuine overfitting on diverse stratified samples
- n=1000 is the validated sweet spot


═══════════════════════════════════════════════════════════════════════
PHASE 15: COST-VS-ACCURACY + COMPUTATIONAL FINDINGS
═══════════════════════════════════════════════════════════════════════

## Pareto-optimal models (fast + accurate)
- M4: NBEATS (fast, best MAE) and AutoARIMA (moderate speed, 2nd MAE)
- M5: NBEATS and PatchTST (both fast, both top-2 MAE)
- Traffic: DLinear and LightGBM (fastest, best MAE)

## Surprise finding: AutoARIMA is the most expensive model on Traffic
- AutoARIMA on Traffic: 9234 seconds (2.6 hours)
- TimesNet on Traffic: 691 seconds (11.5 min)
- AutoARIMA is 13× more expensive than TimesNet on Traffic
- "The most expensive model in our benchmark is the classical statistical
  baseline, not the transformer."

## GPU memory
- DeepAR has highest peak GPU: ~3.6 GB on Traffic (LSTM state caching)
- DLinear has lowest: minimal GPU usage
- NBEATS: moderate GPU, best cost/accuracy ratio


═══════════════════════════════════════════════════════════════════════
PHASE 16: REMAINING WORK
═══════════════════════════════════════════════════════════════════════

## Learning curves (in progress)
- Retraining 6 neural + LightGBM × 3 datasets = 21 fits with loss logging
- ~50-60 min runtime
- SeasonalNaive and AutoARIMA excluded (no iterative training)

## Report (not started)
- Due: April 19 (4 days from now)
- Format: Word (.docx) for team collaborative editing
- Max 10 pages
- Must address 4 key questions:
  1. Performance (do DL beat baselines?)
  2. Computational cost (training time, hardware)
  3. Robustness (HP sensitivity, data volume, overfitting)
  4. Practicality (would you recommend DL?)


═══════════════════════════════════════════════════════════════════════
MASTER FINDING SUMMARY — 20 DISCRETE FINDINGS
═══════════════════════════════════════════════════════════════════════

1.  NBEATS ≈ AutoARIMA on M4 (Nemenyi clique + DM 74% no-difference)
2.  NBEATS uniquely best on M5 (Nemenyi loner)
3.  DLinear uniquely best on Traffic (Nemenyi loner, beats LightGBM significantly)
4.  AutoARIMA uniformly competitive across M4 sample sizes (flattest curve)
5.  At small n, DLinear beats PatchTST on M4 (Zeng et al. replication)
6.  All trained models overfit at n=2000 on M4 (SN flat, doubled budget worse)
7.  No data-volume benefit on M5 in range n=100-500
8.  DLinear wins Traffic at all sample sizes
9.  PatchTST robust to patch_len (1.1% spread)
10. PatchTST very sensitive to lookback (48.3% spread) — main-run limitation
11. NBEATS robust to depth (1.4% spread)
12. DLinear sensitive to lookback on Traffic (17.6% spread)
13. LightGBM's Traffic ranking depends on date features (fairness concern)
14. LightGBM's M5 calendar features actually hurt performance (-3.4%)
15. TimesNet not undertrained at 800 steps
16. M5 intermittency is structural, not a sampling artifact (72.75% zeros even stratified)
17. MASE drop% is 0% everywhere (no series have undefined MASE)
18. AutoARIMA is the most expensive model on Traffic (13× TimesNet)
19. NBEATS and AutoARIMA track each other across ALL M4 horizon steps
20. M5 per-horizon shows clear 7-day periodicity; DL handles peaks better
