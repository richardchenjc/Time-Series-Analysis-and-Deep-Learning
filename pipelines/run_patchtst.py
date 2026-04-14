"""
Pipeline: PatchTST across all 3 datasets (M4, M5, Traffic).

Patch sizes are chosen to align each patch with a semantically meaningful
temporal unit — ideally one seasonal period or a clean fraction of it:

  Dataset          patch_len  stride  n_patches  Rationale
  ──────────────── ─────────  ──────  ─────────  ────────────────────────────
  M4 Monthly       6          3       12         half-year per patch; 12 patches
  M5 Daily         7          7       9          one full week per patch
  Traffic Hourly   24         12      14         one full day per patch

Number of patches = floor((input_size - patch_len) / stride) + 2.

Usage:
    python pipelines/run_patchtst.py                # Full run
    python pipelines/run_patchtst.py --smoke-test   # Quick validation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.patchtst import build
from pipelines.run_model import run_pipeline

# Dataset-specific patch configurations.
# patch_len aligns to the dominant seasonal unit for each frequency;
# stride = patch_len / 2 (50 % overlap) for M4 and Traffic to increase
# contextual coverage, and stride = patch_len (no overlap) for M5 since
# weekly boundaries are clean and non-overlapping patches suffice.
_PATCH_CONFIGS = {
    "M4":      {"patch_len": 6,  "stride": 3},   # half-year patch, 50% overlap → 12 patches
    "M5":      {"patch_len": 7,  "stride": 7},   # one week patch, no overlap   → 9 patches
    "Traffic": {"patch_len": 24, "stride": 12},  # one day patch, 50% overlap   → 14 patches
}


def _factory(cfg):
    """Return a factory closure for the given dataset config."""
    patch_cfg = _PATCH_CONFIGS[cfg["name"]]

    def _build(seed, max_steps=None):
        return build(
            horizon=cfg["horizon"],
            input_size=cfg["input_size"],
            freq=cfg["freq"],
            seed=seed,
            max_steps=max_steps,
            patch_len=patch_cfg["patch_len"],
            stride=patch_cfg["stride"],
        )
    return _build


def main(smoke_test: bool = False):
    run_pipeline(
        model_name="PatchTST",
        build_fn_for_cfg=_factory,
        needs_seed=True,
        smoke_test=smoke_test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PatchTST on all datasets.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run with minimal settings for quick validation")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
