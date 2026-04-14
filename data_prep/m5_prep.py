"""
M5 dataset preparation.

Loads Walmart sales data (sales_train_evaluation.csv + calendar.csv),
converts to Nixtla long format. Samples item-level series for tractability.

Sampling strategy
-----------------
Defaults to stratified sampling by mean sales volume (quintiles) so that
the sample includes high-volume items. Pass `stratified=False` for uniform
random sampling (sanity-check mode).

Calendar features (optional)
----------------------------
When `with_calendar_features=True`, attaches SNAP days and event flags
from the retail calendar as time-varying exogenous regressors.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import M5_CONFIG
from data_prep.sampling import (
    stratified_sample_ids,
    print_strata_summary,
    value_quintile_strata,
)

# Names of calendar exog columns produced when with_calendar_features=True
M5_EXOG_COLS = ["snap_CA", "snap_TX", "snap_WI", "is_event", "is_sporting_or_holiday"]


def _build_calendar_features(calendar: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw calendar.csv to a (date → 5 binary features) lookup."""
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"])

    for c in ("snap_CA", "snap_TX", "snap_WI"):
        cal[c] = cal[c].fillna(0).astype("int8")

    cal["is_event"] = cal["event_name_1"].notna().astype("int8")
    cal["is_sporting_or_holiday"] = (
        cal["event_type_1"].isin(["Sporting", "National"]).astype("int8")
    )

    return cal[["date"] + M5_EXOG_COLS].rename(columns={"date": "ds"})


def load_m5(
    n_series: int | None = None,
    random_state: int = 42,
    with_calendar_features: bool = True,
    stratified: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load M5 data and return (df_train, df_test) in long format.

    Parameters
    ----------
    n_series : int or None
        Number of item-level series to sample. None = use all (~30k).
    random_state : int
        Seed for reproducible sampling.
    with_calendar_features : bool
        If True, attach SNAP days and event flags as exogenous columns.
    stratified : bool
        If True (default), stratified sample by volume quintile.
        If False, uniform random sampling (sanity-check mode).
    """
    cfg = M5_CONFIG
    horizon = cfg["horizon"]

    print(f"[M5] Reading calendar: {cfg['calendar_csv']}")
    calendar = pd.read_csv(cfg["calendar_csv"])
    day_to_date = dict(zip(calendar["d"], pd.to_datetime(calendar["date"])))

    cal_features = _build_calendar_features(calendar) if with_calendar_features else None

    print(f"[M5] Reading sales CSV: {cfg['sales_csv']}")
    sales = pd.read_csv(cfg["sales_csv"])

    id_col = "id"
    d_cols = [c for c in sales.columns if c.startswith("d_")]

    # --- Sample series ---
    all_ids = sales[id_col].to_numpy()
    if n_series is not None and n_series < len(all_ids):
        if stratified:
            means = sales[d_cols].mean(axis=1).to_numpy()
            strata = value_quintile_strata(means)

            selected_ids = stratified_sample_ids(
                all_ids, strata, n_series, random_state=random_state
            )
            sampled_strata_lookup = dict(zip(all_ids, strata))
            sampled_strata = np.array([sampled_strata_lookup[i] for i in selected_ids])
            print(f"[M5] Stratified sample: {len(selected_ids)} series "
                  f"out of {len(all_ids)} (strategy: volume_quintile)")
            print_strata_summary(strata, sampled_strata, "M5")
        else:
            rng = np.random.RandomState(random_state)
            selected_ids = rng.choice(all_ids, size=n_series, replace=False)
            print(f"[M5] Random sample: {len(selected_ids)} series "
                  f"out of {len(all_ids)} (stratified=False)")

        sales = sales[sales[id_col].isin(selected_ids)].reset_index(drop=True)
    else:
        print(f"[M5] Using all {len(all_ids)} series")

    # --- Melt to long format ---
    print("[M5] Melting to long format...")
    df_long = sales.melt(
        id_vars=[id_col],
        value_vars=d_cols,
        var_name="d",
        value_name="y",
    )
    df_long.rename(columns={id_col: "unique_id"}, inplace=True)
    df_long["y"] = df_long["y"].fillna(0.0)

    df_long["ds"] = df_long["d"].map(day_to_date)
    df_long.drop(columns=["d"], inplace=True)
    df_long.dropna(subset=["ds"], inplace=True)
    df_long["y"] = df_long["y"].astype(float)

    # Attach calendar exog features
    if cal_features is not None:
        print(f"[M5] Attaching {len(M5_EXOG_COLS)} calendar exog features")
        df_long = df_long.merge(cal_features, on="ds", how="left")
        for c in M5_EXOG_COLS:
            df_long[c] = df_long[c].fillna(0).astype("int8")

    df_long.sort_values(["unique_id", "ds"], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    # --- Split: last `horizon` days per series as test ---
    print(f"[M5] Splitting: last {horizon} days as test...")
    max_dates = df_long.groupby("unique_id")["ds"].max().reset_index()
    max_dates["cutoff"] = max_dates["ds"] - pd.Timedelta(days=horizon - 1)
    df_long = df_long.merge(max_dates[["unique_id", "cutoff"]], on="unique_id")

    keep_cols = ["unique_id", "ds", "y"]
    if cal_features is not None:
        keep_cols += M5_EXOG_COLS

    df_train = (
        df_long[df_long["ds"] < df_long["cutoff"]][keep_cols].reset_index(drop=True)
    )
    df_test = (
        df_long[df_long["ds"] >= df_long["cutoff"]][keep_cols].reset_index(drop=True)
    )

    print(f"[M5] Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    print(f"[M5] Unique series: {df_train['unique_id'].nunique()}")
    if cal_features is not None:
        print(f"[M5] Exog cols attached: {M5_EXOG_COLS}")

    return df_train, df_test


if __name__ == "__main__":
    df_train, df_test = load_m5(n_series=M5_CONFIG["n_series_sample"])
    print("\n--- Train sample ---")
    print(df_train.head(10))
    print(f"\n--- Test sample ---")
    print(df_test.head(10))
    print(f"\nTrain date range: {df_train['ds'].min()} — {df_train['ds'].max()}")
    print(f"Test date range:  {df_test['ds'].min()} — {df_test['ds'].max()}")
    print(f"Train columns: {list(df_train.columns)}")
