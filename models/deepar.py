"""DeepAR — RNN-based autoregressive model with probabilistic output via neuralforecast.

Note: DeepAR's Student-T sampling uses aten::_standard_gamma which is not
implemented on Apple MPS. The env var PYTORCH_ENABLE_MPS_FALLBACK=1 (set in
config.py) makes PyTorch fall back to CPU for that op only.

hidden_size is exposed as an optional override so pipelines can scale the
LSTM state to the complexity of each dataset.
"""

from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import (
    MAX_STEPS, BATCH_SIZE, LR_RNN, EARLY_STOP_PATIENCE,
    VAL_CHECK_STEPS, DEEPAR_PARAMS,
)
from models import ModelSpec


def build(
    horizon: int,
    input_size: int,
    freq: str,
    seed: int,
    max_steps: int | None = None,
    hidden_size: int | None = None,
) -> ModelSpec:
    """Build NeuralForecast with DeepAR.

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
    hidden_size : int or None
        Override LSTM hidden state size from DEEPAR_PARAMS.
        Larger values suit longer, more complex input sequences.

    Returns
    -------
    ModelSpec
        Wraps a NeuralForecast object (not yet fitted).
    """
    steps = max_steps if max_steps is not None else MAX_STEPS
    _hidden = hidden_size if hidden_size is not None else DEEPAR_PARAMS["hidden_size"]

    model = DeepAR(
        h=horizon,
        input_size=input_size,
        lstm_hidden_size=_hidden,
        lstm_n_layers=DEEPAR_PARAMS["n_layers"],
        max_steps=steps,
        batch_size=BATCH_SIZE,
        learning_rate=LR_RNN,
        random_seed=seed,
        early_stop_patience_steps=EARLY_STOP_PATIENCE,
        val_check_steps=VAL_CHECK_STEPS,
        scaler_type="standard",
    )
    nf = NeuralForecast(models=[model], freq=freq)
    return ModelSpec(name="DeepAR", model_type="neural", forecaster=nf, needs_seed=True)
