"""FastAPI serving layer for the interactive degradation demo."""

from wfb.serving.inference import InferenceRegistry, RegistryConfig, ResultCache
from wfb.serving.results_store import ResultsStore, SweepRecord

__all__ = [
    "InferenceRegistry",
    "RegistryConfig",
    "ResultCache",
    "ResultsStore",
    "SweepRecord",
]


def create_app(*args: object, **kwargs: object) -> object:
    """Lazy re-export so importing :mod:`wfb.serving` does not require FastAPI."""
    from wfb.serving.app import create_app as _create_app

    return _create_app(*args, **kwargs)  # type: ignore[arg-type]
