"""
Stratified sampling utilities for dataset loaders.

Random sampling under-represents tails on long-tailed datasets:
  - M5 random samples skip the highest-volume Walmart items
  - M4 random samples are biased toward the dominant length / category
  - Traffic random samples may miss high-occupancy sensors

We address this by allocating sample slots equally across strata
(e.g. volume quintiles), trading representativeness for diversity.
This is the right call for small samples — we want every "type" of
series to be visible to the model, not a microcosm of the population.
"""

from __future__ import annotations

from collections import Counter
import numpy as np
import pandas as pd


def stratified_sample_ids(
    ids: np.ndarray,
    strata: np.ndarray,
    n_sample: int,
    random_state: int = 42,
) -> np.ndarray:
    """Sample n_sample IDs equally from each stratum.

    If a stratum is too small to fill its base quota, the deficit is
    drawn from the global pool of unsampled IDs (proportional to
    remaining stratum sizes). This guarantees that we always return
    exactly min(n_sample, len(ids)) IDs.

    Parameters
    ----------
    ids : np.ndarray
        Array of unique series IDs (any hashable type).
    strata : np.ndarray
        Stratum label per ID, parallel to `ids` (same length).
        NaN strata are placed in a special "_unknown" bucket.
    n_sample : int
        Total number of IDs to draw.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sampled IDs, length min(n_sample, len(ids)).
    """
    if len(ids) != len(strata):
        raise ValueError(
            f"ids and strata must have the same length: {len(ids)} vs {len(strata)}"
        )
    if n_sample >= len(ids):
        return np.asarray(ids)

    rng = np.random.RandomState(random_state)

    # Replace NaN strata with a placeholder so groupby keeps them
    strata_clean = pd.Series(strata).fillna("_unknown").to_numpy()

    df = pd.DataFrame({"id": ids, "stratum": strata_clean})
    # sort=True for deterministic group ordering across runs
    groups = {s: g["id"].to_numpy() for s, g in df.groupby("stratum", sort=True)}
    n_strata = len(groups)
    base_quota = n_sample // n_strata

    # Pass 1: take min(base_quota, group_size) from every stratum
    sampled_chunks = []
    for s, group_ids in groups.items():
        take = min(base_quota, len(group_ids))
        if take > 0:
            chosen = rng.choice(group_ids, size=take, replace=False)
            sampled_chunks.append(chosen)

    sampled_so_far = (
        np.concatenate(sampled_chunks) if sampled_chunks else np.array([], dtype=ids.dtype)
    )

    # Pass 2: top up the deficit from unsampled IDs
    deficit = n_sample - len(sampled_so_far)
    if deficit > 0:
        already = set(sampled_so_far.tolist())
        remaining = np.array([i for i in ids if i not in already])
        if len(remaining) > 0:
            extra = rng.choice(
                remaining, size=min(deficit, len(remaining)), replace=False
            )
            sampled_chunks.append(extra)

    return np.concatenate(sampled_chunks) if sampled_chunks else np.array([], dtype=ids.dtype)


def print_strata_summary(
    strata_all: np.ndarray, strata_sampled: np.ndarray, name: str
) -> None:
    """Log per-stratum population and sample counts for verification.

    The report should quote this breakdown to demonstrate that the
    sample isn't accidentally concentrated in one stratum.
    """
    pop = Counter(pd.Series(strata_all).fillna("_unknown"))
    samp = Counter(pd.Series(strata_sampled).fillna("_unknown"))
    print(f"[{name}] Stratification breakdown (sampled / population):")
    width = max(len(str(s)) for s in pop.keys())
    for s in sorted(pop.keys(), key=str):
        print(f"  {str(s):<{width}}  {samp.get(s, 0):>6} / {pop[s]:>6}")


def length_quartile_strata(lengths: np.ndarray) -> np.ndarray:
    """Return Q1..Q4 length-quartile labels for the given lengths.

    Used as a fallback for M4 when category metadata is unavailable.
    """
    s = pd.qcut(
        lengths, q=4, labels=["Q1_short", "Q2", "Q3", "Q4_long"], duplicates="drop"
    )
    return np.asarray(s)


def value_quintile_strata(means: np.ndarray) -> np.ndarray:
    """Return Q1..Q5 quintile labels based on per-series mean values.

    Used for M5 (sales volume) and Traffic (occupancy level).
    Assigns NaN means to the lowest quintile.
    """
    means = np.where(np.isfinite(means), means, 0.0)
    s = pd.qcut(
        means,
        q=5,
        labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
        duplicates="drop",
    )
    return np.asarray(s)
