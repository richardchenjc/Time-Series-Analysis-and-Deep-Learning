"""
Pipeline: N-BEATS across all 3 datasets (M4, M5, Traffic).

Usage:
    python pipelines/run_nbeats.py                # Full run
    python pipelines/run_nbeats.py --smoke-test   # Quick validation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.nbeats import build
from pipelines.run_model import run_pipeline


def _factory(cfg):
    """Return a factory closure for the given dataset config."""
    def _build(seed, max_steps=None):
        return build(
            horizon=cfg["horizon"],
            input_size=cfg["input_size"],
            freq=cfg["freq"],
            seed=seed,
            max_steps=max_steps,
        )
    return _build


def main(smoke_test: bool = False):
    run_pipeline(
        model_name="NBEATS",
        build_fn_for_cfg=_factory,
        needs_seed=True,
        smoke_test=smoke_test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run N-BEATS on all datasets.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run with minimal settings for quick validation")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
