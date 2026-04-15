"""
Pipeline: DeepAR across all 3 datasets (M4, M5, Traffic).

LSTM hidden size is scaled to each dataset's input complexity:

  Dataset          hidden_size  Rationale
  ──────────────── ───────────  ─────────────────────────────────────────────
  M4 Monthly       128          36-step input, simple univariate → default
  M5 Daily         128          56-step input, moderate → default
  Traffic Hourly   256          168-step input, complex multivariate → 2× default

input_size_offset=1
-------------------
DeepAR's autoregressive structure requires one additional timestamp
beyond the standard input_size (neuralforecast reports input_size=37
when we pass input_size=36). We pass input_size_offset=1 to walk_forward
so the series filter accounts for this; otherwise window 2 on M4 fails
on series that are ~1 timestamp too short for DeepAR but long enough
for the other DL models.

Usage:
    python pipelines/run_deepar.py                # Full run
    python pipelines/run_deepar.py --smoke-test   # Quick validation
    python pipelines/run_deepar.py --m4-only      # Re-run M4 only (cheap fix)
"""

import os
# Must be set BEFORE neuralforecast/torch is imported.
# DeepAR's Student-T distribution uses aten::_standard_gamma which is
# not implemented on Apple MPS. Setting this flag here (before torch
# initialises the MPS backend) tells PyTorch to silently fall back to
# CPU for that one op only; everything else keeps running on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.deepar import build
from pipelines.run_model import run_pipeline

# Dataset-specific LSTM hidden sizes.
# Traffic has 4.7× more input steps than M4 and richer multivariate dynamics,
# justifying a larger hidden state to capture temporal dependencies.
_HIDDEN_SIZES = {
    "M4":      128,   # default (Salinas et al. 2020 / NeuralForecast default)
    "M5":      128,   # default
    "Traffic": 256,   # 2× default — 168-step hourly input warrants larger state
}


def _factory(cfg):
    """Return a factory closure for the given dataset config."""
    hidden_size = _HIDDEN_SIZES[cfg["name"]]

    def _build(seed, max_steps=None):
        return build(
            horizon=cfg["horizon"],
            input_size=cfg["input_size"],
            freq=cfg["freq"],
            seed=seed,
            max_steps=max_steps,
            hidden_size=hidden_size,
        )
    return _build


def main(smoke_test: bool = False, m4_only: bool = False):
    run_pipeline(
        model_name="DeepAR",
        build_fn_for_cfg=_factory,
        needs_seed=True,
        smoke_test=smoke_test,
        input_size_offset=1,    # critical for DeepAR — see module docstring
        datasets=["M4"] if m4_only else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepAR on all datasets.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run with minimal settings for quick validation")
    parser.add_argument("--m4-only", action="store_true",
                        help="Re-run on M4 only (used for cheap M4 fix re-runs)")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test, m4_only=args.m4_only)
