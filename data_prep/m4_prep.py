"""
M4 Monthly dataset preparation.

Loads the M4 Monthly train/test CSVs (wide format), melts to Nixtla long
format (unique_id, ds, y), samples a subset of series for tractability,
and assigns synthetic monthly timestamps.

Sampling strategy
-----------------
Defaults to stratified sampling by M4 category (Macro/Micro/Demographic/
Industry/Finance/Other) when m4_info.csv is present, falling back to
length quartiles otherwise. Pass `stratified=False` to use uniform
random sampling instead — used by the sampling sanity check to verify
that stratification doesn't bias cross-model comparisons.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import M4_CONFIG
from data_prep.sampling import (
    stratified_sample_ids,
    print_strata_summary,
    length_quartile_strata,
)


def _wide_to_long(df_wide: pd.DataFrame, freq: str, end_date: str = "2020-01-01") -> pd.DataFrame:
    """Convert M4 wide-format CSV to Nixtla long format.

    M4 wide format: first column is Series ID, remaining columns are time
    steps with values (NaN-padded at the end for shorter series).
    """
    id_col = df_wide.columns[0]
    value_cols = df_wide.columns[1:]
    end_ts = pd.Timestamp(end_date)

    # Preserve original column order: column names like "V1","V2",... sort
    # lexicographically wrong ("V10" < "V2"), so map each name to its index.
    col_order = {col: i for i, col in enumerate(value_cols)}

    df_long = df_wide.melt(
        id_vars=[id_col],
        value_vars=value_cols,
        var_name="_step",
        value_name="y",
    )
    df_long.rename(columns={id_col: "unique_id"}, inplace=True)

    # Drop NaN-padded trailing entries (shorter series)
    df_long = df_long.dropna(subset=["y"])
    df_long["y"] = df_long["y"].astype(float).fillna(0.0)

    # Sort by original column order to restore temporal sequence
    df_long["_col_idx"] = df_long["_step"].map(col_order)
    df_long.sort_values(["unique_id", "_col_idx"], inplace=True)
    df_long.drop(columns=["_step", "_col_idx"], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    # Assign dates: each series ends at end_date
    def _assign_dates(group):
        group = group.copy()
        group["ds"] = pd.date_range(end=end_ts, periods=len(group), freq=freq)
        return group

    df_long = df_long.groupby("unique_id", group_keys=False).apply(_assign_dates)
    df_long["ds"] = pd.to_datetime(df_long["ds"])
    return df_long[["unique_id", "ds", "y"]]


def _compute_m4_strata(all_ids: np.ndarray, train_wide: pd.DataFrame, cfg: dict):
    """Build stratum labels for the M4 series, with graceful fallback.

    Strategy A: read m4_info.csv and use the `category` column.
    Strategy B: fall back to length quartiles.
    """
    info_path = cfg.get("info_csv")
    if info_path is not None:
        try:
            info = pd.read_csv(info_path)
            id_col = next(
                (c for c in info.columns if c.lower() in ("m4id", "id")), None
            )
            cat_col = next((c for c in info.columns if c.lower() == "category"), None)
            if id_col and cat_col:
                info_lookup = info.set_index(id_col)[cat_col]
                strata = pd.Series(all_ids).map(info_lookup).to_numpy()
                missing_pct = pd.isna(strata).mean() * 100
                if missing_pct < 5:
                    return strata, "category"
                else:
                    print(f"[M4] m4_info.csv missing {missing_pct:.0f}% of ids — falling back")
        except Exception as e:
            print(f"[M4] Could not load m4_info.csv ({e}) — falling back to length quartiles")

    # Fallback: length quartiles
    lengths = train_wide.iloc[:, 1:].notna().sum(axis=1).to_numpy()
    return length_quartile_strata(lengths), "length_quartile"


def load_m4_monthly(
    n_series: int | None = None,
    random_state: int = 42,
    stratified: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load M4 Monthly data and return (df_train, df_test) in long format.

    Parameters
    ----------
    n_series : int or None
        Number of series to sample. None = use all.
    random_state : int
        Seed for reproducible sampling.
    stratified : bool
        If True (default), stratified sample by category / length quartile.
        If False, uniform random sampling (for sampling-method sanity check).
    """
    cfg = M4_CONFIG

    print(f"[M4] Reading train CSV: {cfg['train_csv']}")
    train_wide = pd.read_csv(cfg["train_csv"])
    print(f"[M4] Reading test CSV: {cfg['test_csv']}")
    test_wide = pd.read_csv(cfg["test_csv"])

    # --- Determine which series to use ---
    all_ids = train_wide.iloc[:, 0].to_numpy()
    if n_series is not None and n_series < len(all_ids):
        if stratified:
            strata, strategy = _compute_m4_strata(all_ids, train_wide, cfg)
            selected_ids = stratified_sample_ids(
                all_ids, strata, n_series, random_state=random_state
            )
            sampled_strata_lookup = dict(zip(all_ids, strata))
            sampled_strata = np.array([sampled_strata_lookup[i] for i in selected_ids])
            print(f"[M4] Stratified sample: {len(selected_ids)} series "
                  f"out of {len(all_ids)} (strategy: {strategy})")
            print_strata_summary(strata, sampled_strata, "M4")
        else:
            rng = np.random.RandomState(random_state)
            selected_ids = rng.choice(all_ids, size=n_series, replace=False)
            print(f"[M4] Random sample: {len(selected_ids)} series "
                  f"out of {len(all_ids)} (stratified=False)")

        train_wide = train_wide[train_wide.iloc[:, 0].isin(selected_ids)].reset_index(drop=True)
        test_wide = test_wide[test_wide.iloc[:, 0].isin(selected_ids)].reset_index(drop=True)
    else:
        print(f"[M4] Using all {len(all_ids)} series")

    # --- Convert train to long format ---
    print("[M4] Converting train to long format...")
    df_train = _wide_to_long(train_wide, freq=cfg["freq"])

    # --- Convert test to long format ---
    print("[M4] Converting test to long format...")
    id_col = test_wide.columns[0]
    value_cols = test_wide.columns[1:]
    last_dates = df_train.groupby("unique_id")["ds"].max()
    col_order = {col: i for i, col in enumerate(value_cols)}

    df_test = test_wide.melt(
        id_vars=[id_col],
        value_vars=value_cols,
        var_name="_step",
        value_name="y",
    )
    df_test.rename(columns={id_col: "unique_id"}, inplace=True)
    df_test = df_test.dropna(subset=["y"])
    df_test["y"] = df_test["y"].astype(float).fillna(0.0)
    df_test["_col_idx"] = df_test["_step"].map(col_order)
    df_test.sort_values(["unique_id", "_col_idx"], inplace=True)
    df_test.drop(columns=["_step", "_col_idx"], inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    def _assign_test_dates(group):
        uid = group["unique_id"].iloc[0]
        start = last_dates[uid] + pd.DateOffset(months=1)
        group = group.copy()
        group["ds"] = pd.date_range(start=start, periods=len(group), freq=cfg["freq"])
        return group

    df_test = df_test.groupby("unique_id", group_keys=False).apply(_assign_test_dates)
    df_test["ds"] = pd.to_datetime(df_test["ds"])
    df_test = df_test[["unique_id", "ds", "y"]]

    print(f"[M4] Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    print(f"[M4] Unique series: {df_train['unique_id'].nunique()}")

    return df_train, df_test


if __name__ == "__main__":
    df_train, df_test = load_m4_monthly(n_series=M4_CONFIG["n_series_sample"])
    print("\n--- Train sample ---")
    print(df_train.head(10))
    print(f"\n--- Test sample ---")
    print(df_test.head(10))
    print(f"\nTrain date range: {df_train['ds'].min()} — {df_train['ds'].max()}")
    print(f"Test date range:  {df_test['ds'].min()} — {df_test['ds'].max()}")
