"""Model definitions for time-series forecasting.

All model factory functions return a ModelSpec, enabling uniform handling
in the walk-forward evaluation engine regardless of underlying framework.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

ModelType = Literal["stats", "ml", "neural"]


@dataclass
class ModelSpec:
    """Uniform wrapper returned by every model factory function.

    Attributes
    ----------
    name : str
        Model name as it appears in prediction column headers and result CSVs
        (e.g. "PatchTST", "SeasonalNaive", "LightGBM").
    model_type : {"stats", "ml", "neural"}
        Framework type — determines which fit/predict API to call.
        - "stats"  : StatsForecast  — sf.fit(df) / sf.predict(h=horizon)
        - "ml"     : MLForecast     — mlf.fit(df) / mlf.predict(h=horizon, X_df=...)
        - "neural" : NeuralForecast — nf.fit(df=df, val_size=h) / nf.predict()
    forecaster : Any
        The underlying StatsForecast | MLForecast | NeuralForecast object.
    needs_seed : bool
        False for deterministic models (SeasonalNaive, AutoARIMA); True for
        ML/DL models where results vary across seeds.
    exog_cols : list[str] | None
        Names of time-varying, future-known exogenous columns the model
        expects in df_train (and as future values in X_df at predict time).
        Used by walk_forward to:
          - keep these columns visible to the model during fit
          - build the X_df parameter for ml-type predict()
        For all other model types, walk_forward strips extra columns
        from the input frames before fit, so models that don't declare
        exog_cols never accidentally see them.
    """

    name: str
    model_type: ModelType
    forecaster: Any
    needs_seed: bool
    exog_cols: list[str] | None = None
