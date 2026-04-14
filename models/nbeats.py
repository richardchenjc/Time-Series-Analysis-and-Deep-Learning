"""N-BEATS — deep residual network with basis expansion via neuralforecast."""

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import (
    MAX_STEPS, BATCH_SIZE, LR_MLP, EARLY_STOP_PATIENCE,
    VAL_CHECK_STEPS, NBEATS_PARAMS,
)
from models import ModelSpec


def build(
    horizon: int,
    input_size: int,
    freq: str,
    seed: int,
    max_steps: int | None = None,
) -> ModelSpec:
    """Build NeuralForecast with N-BEATS.

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

    model = NBEATS(
        h=horizon,
        input_size=input_size,
        stack_types=NBEATS_PARAMS["stack_types"],
        n_blocks=NBEATS_PARAMS["n_blocks"],
        mlp_units=NBEATS_PARAMS["mlp_units"],
        max_steps=steps,
        batch_size=BATCH_SIZE,
        learning_rate=LR_MLP,
        random_seed=seed,
        early_stop_patience_steps=EARLY_STOP_PATIENCE,
        val_check_steps=VAL_CHECK_STEPS,
        scaler_type="standard",
    )
    nf = NeuralForecast(models=[model], freq=freq)
    return ModelSpec(name="NBEATS", model_type="neural", forecaster=nf, needs_seed=True)
