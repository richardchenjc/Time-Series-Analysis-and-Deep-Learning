"""
Pipeline: TimesNet across all 3 datasets (M4, M5, Traffic).

Encoder layers are scaled to the temporal richness of each dataset's input:

  Dataset          e_layers  Rationale
  ──────────────── ────────  ─────────────────────────────────────────────────
  M4 Monthly       2         36-step input, one dominant annual period → default
  M5 Daily         2         56-step input, one dominant weekly period → default
  Traffic Hourly   3         168-step (7-day) input; contains both daily AND
                             weekly periodicity → extra layer to capture both

Usage:
    python pipelines/run_timesnet.py                # Full run
    python pipelines/run_timesnet.py --smoke-test   # Quick validation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.timesnet import build
from pipelines.run_model import run_pipeline

# Dataset-specific encoder layer counts.
# TimesNet stacks 2D CNN Inception blocks; more layers let the model learn
# hierarchical periodic features. Traffic's 168-step window spans an entire
# week and contains two clear nested cycles (daily + weekly), so 3 layers
# allow the model to process them at different levels of abstraction.
_E_LAYERS = {
    "M4":      2,   # paper default; single annual cycle
    "M5":      2,   # paper default; single weekly cycle
    "Traffic": 3,   # daily (24 h) + weekly (168 h) nesting → extra layer
}


def _factory(cfg):
    """Return a factory closure for the given dataset config."""
    e_layers = _E_LAYERS[cfg["name"]]

    def _build(seed, max_steps=None):
        return build(
            horizon=cfg["horizon"],
            input_size=cfg["input_size"],
            freq=cfg["freq"],
            seed=seed,
            max_steps=max_steps,
            e_layers=e_layers,
        )
    return _build


def main(smoke_test: bool = False):
    run_pipeline(
        model_name="TimesNet",
        build_fn_for_cfg=_factory,
        needs_seed=True,
        smoke_test=smoke_test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TimesNet on all datasets.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run with minimal settings for quick validation")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
