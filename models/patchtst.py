"""PatchTST — Transformer with patch-based architecture via neuralforecast.

Patch size and stride are exposed as optional overrides so that
pipelines/run_patchtst.py can pass dataset-specific values that align
patches to semantically meaningful temporal units (e.g. one seasonal period).
"""

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import (
    MAX_STEPS, BATCH_SIZE, LR_TRANSFORMER, EARLY_STOP_PATIENCE,
    VAL_CHECK_STEPS, PATCHTST_PARAMS,
)
from models import ModelSpec


def build(
    horizon: int,
    input_size: int,
    freq: str,
    seed: int,
    max_steps: int | None = None,
    patch_len: int | None = None,
    stride: int | None = None,
) -> ModelSpec:
    """Build NeuralForecast with PatchTST.

    Parameters
    ----------
    horizon : int
        Forecast horizon (h).
    input_size : int
        Lookback window length.
    freq : str
        Pandas frequency string.
    seed : int
        Random seed for reproducibility.
    max_steps : int or None
        Override default MAX_STEPS (useful for smoke tests).
    patch_len : int or None
        Override default patch length from PATCHTST_PARAMS.
        Number of patches = floor((input_size - patch_len) / stride) + 2.
    stride : int or None
        Override default stride from PATCHTST_PARAMS.

    Returns
    -------
    ModelSpec
        Wraps a NeuralForecast object (not yet fitted).
    """
    steps = max_steps if max_steps is not None else MAX_STEPS
    _patch_len = patch_len if patch_len is not None else PATCHTST_PARAMS["patch_len"]
    _stride    = stride    if stride    is not None else PATCHTST_PARAMS["stride"]

    model = PatchTST(
        h=horizon,
        input_size=input_size,
        patch_len=_patch_len,
        stride=_stride,
        n_heads=PATCHTST_PARAMS["n_heads"],
        hidden_size=PATCHTST_PARAMS["hidden_size"],
        encoder_layers=PATCHTST_PARAMS["encoder_layers"],
        max_steps=steps,
        batch_size=BATCH_SIZE,
        learning_rate=LR_TRANSFORMER,
        random_seed=seed,
        early_stop_patience_steps=EARLY_STOP_PATIENCE,
        val_check_steps=VAL_CHECK_STEPS,
        scaler_type="standard",
    )
    nf = NeuralForecast(models=[model], freq=freq)
    return ModelSpec(name="PatchTST", model_type="neural", forecaster=nf, needs_seed=True)
