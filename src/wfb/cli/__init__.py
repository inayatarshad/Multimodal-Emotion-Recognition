"""Command-line entrypoints (``wfb-data``, ``wfb-train``, ``wfb-eval``)."""

from wfb.cli.config import (
    axes_from_config,
    to_data_config,
    to_loader_config,
    to_model_config,
    to_train_config,
)

__all__ = [
    "axes_from_config",
    "to_data_config",
    "to_loader_config",
    "to_model_config",
    "to_train_config",
]
