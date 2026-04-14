"""AutoARIMA baseline via statsforecast."""

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from models import ModelSpec

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def build(season_length: int, freq: str) -> ModelSpec:
    """Build a StatsForecast object with AutoARIMA.

    Parameters
    ----------
    season_length : int
        Seasonal period (e.g. 12 for monthly, 7 for daily, 24 for hourly).
    freq : str
        Pandas frequency string (e.g. 'ME', 'D', 'h').

    Returns
    -------
    ModelSpec
        Wraps a StatsForecast object (not yet fitted).
    """
    sf = StatsForecast(
        models=[AutoARIMA(season_length=season_length)],
        freq=freq,
        n_jobs=-1,
    )
    return ModelSpec(
        name="AutoARIMA",
        model_type="stats",
        forecaster=sf,
        needs_seed=False,
    )
