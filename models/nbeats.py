"""N-BEATS — deep residual network with basis expansion via neuralforecast.

Hyperparameter overrides
------------------------
n_blocks and mlp_units are exposed as optional overrides so the
hyperparameter-sensitivity pipeline (pipelines/run_hp_sensitivity.py)
can sweep them on M4 without touching config.py.
"""

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
    n_blocks: list[int] | None = None,
    mlp_units: list[list[int]] | None = None,
) -> ModelSpec:
    """Build NeuralForecast with N-BEATS.

    Parameters
    ----------
    horizon, input_size, freq, seed : standard fields.
    max_steps : int or None
        Override default MAX_STEPS (useful for smoke tests).
    n_blocks : list[int] or None
        Override the number of blocks per stack from NBEATS_PARAMS.
        Length must match len(stack_types). Default [3, 3].
    mlp_units : list[list[int]] or None
        Override the MLP layer sizes. Default [[512,512],[512,512]].

    Returns
    -------
    ModelSpec wrapping an unfitted NeuralForecast object.
    """
    steps = max_steps if max_steps is not None else MAX_STEPS
    _n_blocks = n_blocks if n_blocks is not None else NBEATS_PARAMS["n_blocks"]
    _mlp_units = mlp_units if mlp_units is not None else NBEATS_PARAMS["mlp_units"]

    # mlp_units must align with n_blocks: one inner list per stack
    if len(_mlp_units) != len(NBEATS_PARAMS["stack_types"]):
        # If caller only overrode n_blocks but not mlp_units, broadcast
        # the default unit shape across all stacks
        base_units = NBEATS_PARAMS["mlp_units"][0]
        _mlp_units = [base_units for _ in NBEATS_PARAMS["stack_types"]]

    model = NBEATS(
        h=horizon,
        input_size=input_size,
        stack_types=NBEATS_PARAMS["stack_types"],
        n_blocks=_n_blocks,
        mlp_units=_mlp_units,
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
