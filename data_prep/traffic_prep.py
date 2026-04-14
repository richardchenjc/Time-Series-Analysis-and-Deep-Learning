"""
Traffic dataset preparation.

Loads the Traffic.tsf file (San Francisco road occupancy rates),
parses it into Nixtla long format, and samples a subset of sensors.

Sampling strategy
-----------------
Defaults to stratified sampling by mean occupancy (quintiles). Pass
`stratified=False` for uniform random sampling (sanity-check mode).
When n_series is None (the main-run default for Traffic), no sampling
is done and all 862 sensors are used.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import TRAFFIC_CONFIG
from data_prep.sampling import (
    stratified_sample_ids,
    print_strata_summary,
    value_quintile_strata,
)


def _parse_tsf(file_path: str) -> pd.DataFrame:
    """Parse a .tsf (Time Series Format) file into a long-format DataFrame."""
    freq_map = {
        "hourly": "h",
        "daily": "D",
        "weekly": "W",
        "monthly": "ME",
        "yearly": "YE",
        "minutely": "min",
    }
    pd_freq = "h"  # default

    series_dfs = []
    in_data = False

    print(f"[Traffic] Parsing TSF file: {file_path}")
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("@"):
                lower = line.lower()
                if lower.startswith("@frequency"):
                    frequency = line.split()[-1].strip()
                    pd_freq = freq_map.get(frequency.lower(), "h")
                    print(f"[Traffic] Frequency: {frequency}")
                if lower == "@data":
                    in_data = True
                    print(f"[Traffic] Data section starts at line {line_num}")
                continue

            if not in_data:
                continue

            parts = line.split(":", 2)
            if len(parts) < 3:
                continue

            series_name = parts[0].strip()
            value_str = parts[2].strip()

            values = np.fromstring(value_str, sep=",")
            n = len(values)
            if n == 0:
                continue

            ts_raw = parts[1].strip()
            date_parts = ts_raw.split(" ")
            if len(date_parts) == 2:
                ts_clean = f"{date_parts[0]} {date_parts[1].replace('-', ':')}"
            else:
                ts_clean = ts_raw
            try:
                start = pd.to_datetime(ts_clean)
            except Exception:
                start = pd.Timestamp("2015-01-01")

            dates = pd.date_range(start=start, periods=n, freq=pd_freq)

            series_dfs.append(pd.DataFrame({
                "unique_id": series_name,
                "ds": dates,
                "y": values,
            }))

    df = pd.concat(series_dfs, ignore_index=True)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].fillna(0.0)
    print(f"[Traffic] Parsed {df['unique_id'].nunique()} series, {len(df)} total rows")
    return df


def load_traffic(
    n_series: int | None = None,
    random_state: int = 42,
    stratified: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Traffic data and return (df_train, df_test) in long format.

    Parameters
    ----------
    n_series : int or None
        Number of sensor series to sample. None = use all 862.
    random_state : int
        Seed for reproducible sampling.
    stratified : bool
        If True (default), stratified sample by occupancy quintile.
        If False, uniform random sampling (sanity-check mode).
    """
    cfg = TRAFFIC_CONFIG
    horizon = cfg["horizon"]

    df = _parse_tsf(str(cfg["data_file"]))

    # --- Sample series ---
    all_ids = df["unique_id"].unique()
    if n_series is not None and n_series < len(all_ids):
        if stratified:
            means = df.groupby("unique_id")["y"].mean().reindex(all_ids).to_numpy()
            strata = value_quintile_strata(means)

            selected_ids = stratified_sample_ids(
                all_ids, strata, n_series, random_state=random_state
            )
            sampled_strata_lookup = dict(zip(all_ids, strata))
            sampled_strata = np.array([sampled_strata_lookup[i] for i in selected_ids])
            print(f"[Traffic] Stratified sample: {len(selected_ids)} sensors "
                  f"out of {len(all_ids)} (strategy: occupancy_quintile)")
            print_strata_summary(strata, sampled_strata, "Traffic")
        else:
            rng = np.random.RandomState(random_state)
            selected_ids = rng.choice(all_ids, size=n_series, replace=False)
            print(f"[Traffic] Random sample: {len(selected_ids)} sensors "
                  f"out of {len(all_ids)} (stratified=False)")

        df = df[df["unique_id"].isin(selected_ids)].reset_index(drop=True)
    else:
        print(f"[Traffic] Using all {len(all_ids)} sensors")

    df.sort_values(["unique_id", "ds"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Split: last `horizon` time steps per series as test ---
    print(f"[Traffic] Splitting: last {horizon} time steps as test...")
    max_dates = df.groupby("unique_id")["ds"].max().reset_index()
    max_dates["cutoff"] = max_dates["ds"] - pd.Timedelta(hours=horizon - 1)
    df = df.merge(max_dates[["unique_id", "cutoff"]], on="unique_id")
    df_train = df[df["ds"] < df["cutoff"]][["unique_id", "ds", "y"]].reset_index(drop=True)
    df_test = df[df["ds"] >= df["cutoff"]][["unique_id", "ds", "y"]].reset_index(drop=True)

    print(f"[Traffic] Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    print(f"[Traffic] Unique series: {df_train['unique_id'].nunique()}")

    return df_train, df_test


if __name__ == "__main__":
    df_train, df_test = load_traffic(n_series=TRAFFIC_CONFIG["n_series_sample"])
    print("\n--- Train sample ---")
    print(df_train.head(10))
    print(f"\n--- Test sample ---")
    print(df_test.head(10))
    print(f"\nTrain date range: {df_train['ds'].min()} — {df_train['ds'].max()}")
    print(f"Test date range:  {df_test['ds'].min()} — {df_test['ds'].max()}")
