"""
Pipeline: Seasonal Naive across all 3 datasets (M4, M5, Traffic).

Usage:
    python pipelines/run_seasonal_naive.py                # Full run
    python pipelines/run_seasonal_naive.py --smoke-test   # Quick validation
    python pipelines/run_seasonal_naive.py --m4-only      # Re-run M4 only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.seasonal_naive import build
from pipelines.run_model import run_pipeline


def _factory(cfg):
    """Return a factory closure for the given dataset config."""
    def _build(seed=None, max_steps=None):
        return build(season_length=cfg["season_length"], freq=cfg["freq"])
    return _build


def main(smoke_test: bool = False, m4_only: bool = False):
    run_pipeline(
        model_name="SeasonalNaive",
        build_fn_for_cfg=_factory,
        needs_seed=False,
        smoke_test=smoke_test,
        datasets=["M4"] if m4_only else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Seasonal Naive on all datasets.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run with minimal settings for quick validation")
    parser.add_argument("--m4-only", action="store_true",
                        help="Re-run on M4 only (used for cheap M4 fix re-runs)")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test, m4_only=args.m4_only)
