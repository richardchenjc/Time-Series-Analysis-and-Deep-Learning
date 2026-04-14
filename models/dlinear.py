"""DLinear — single linear layer baseline via neuralforecast.

Critical linear baseline for comparison against Transformer-based models.
Despite its simplicity, DLinear often matches or beats complex architectures.
"""

from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import (
    MAX_STEPS, BATCH_SIZE, LR_MLP, EARLY_STOP_PATIENCE, VAL_CHECK_STEPS,
)
from models import ModelSpec


def build(
    horizon: int,
    input_size: int,
    freq: str,
    seed: int,
    max_steps: int | None = None,
) -> ModelSpec:
    """Build NeuralForecast with DLinear.

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

    Returns
    -------
    ModelSpec
        Wraps a NeuralForecast object (not yet fitted).
    """
    steps = max_steps if max_steps is not None else MAX_STEPS

    model = DLinear(
        h=horizon,
        input_size=input_size,
        max_steps=steps,
        batch_size=BATCH_SIZE,
        learning_rate=LR_MLP,
        random_seed=seed,
        early_stop_patience_steps=EARLY_STOP_PATIENCE,
        val_check_steps=VAL_CHECK_STEPS,
        scaler_type="standard",
    )
    nf = NeuralForecast(models=[model], freq=freq)
    return ModelSpec(name="DLinear", model_type="neural", forecaster=nf, needs_seed=True)
