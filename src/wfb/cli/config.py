"""Bridge between Hydra's ``DictConfig`` and the package's plain dataclasses.

The rest of ``src/wfb`` never imports Hydra or OmegaConf. Everything downstream takes
ordinary dataclasses, so the library is usable from a notebook, a test, or the FastAPI
service without a config framework in the way — and mypy can actually check it.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, TypeVar, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from wfb.corruption.sweeps import (
    DEFAULT_SEVERITIES,
    REMOVAL_VARIANTS,
    SweepAxis,
    graded_axis,
    misalignment_axis,
    removal_grid,
    standard_grid,
)
from wfb.data.datamodule import LoaderConfig
from wfb.data.loaders import DataConfig
from wfb.data.synthetic import SyntheticConfig
from wfb.models.base import ModelConfig
from wfb.training.trainer import TrainConfig
from wfb.types import modality_from_str

T = TypeVar("T")


def to_plain(node: Any) -> Any:
    """Resolve an OmegaConf node into plain Python containers."""
    if isinstance(node, DictConfig | ListConfig):
        return OmegaConf.to_container(node, resolve=True)
    return node


def build_dataclass(cls: type[T], node: Any, **overrides: Any) -> T:
    """Instantiate ``cls`` from a config node, ignoring keys the dataclass does not have.

    Unknown keys are dropped rather than raising: config files carry documentation and
    grouping keys (``defaults``, ``_target_``) that are not constructor arguments.
    """
    raw = to_plain(node) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(raw).__name__}")
    valid = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    kwargs = {k: v for k, v in raw.items() if k in valid}
    kwargs.update(overrides)
    return cls(**kwargs)


def to_data_config(cfg: DictConfig) -> DataConfig:
    """Build the loader config, including its nested synthetic-corpus settings."""
    node = cast(dict[str, Any], to_plain(cfg.data) or {})
    synthetic_node = node.get("synthetic") or {}
    data = build_dataclass(DataConfig, node)
    low, high = node.get("label_range", (-3.0, 3.0))
    data.label_range = (float(low), float(high))
    if synthetic_node:
        data.synthetic = build_dataclass(
            SyntheticConfig,
            synthetic_node,
            name=data.name,
            task=data.task,
            seq_len=data.seq_len,
            num_classes=data.num_classes,
        )
    return data


def to_model_config(cfg: DictConfig) -> ModelConfig:
    """Build the architecture config."""
    node = cast(dict[str, Any], to_plain(cfg.model) or {})
    model = build_dataclass(ModelConfig, node)
    model.modalities = tuple(node.get("modalities", ("text", "audio", "visual")))
    return model


def to_train_config(cfg: DictConfig) -> TrainConfig:
    """Build the training config, resolving ``device: auto``."""
    train = build_dataclass(TrainConfig, cfg.get("train", {}))
    if train.accelerator in {"auto", "", None}:
        train.accelerator = "auto"
    train.seed = int(cfg.get("seed", train.seed))
    return train


def to_loader_config(cfg: DictConfig) -> LoaderConfig:
    """Build the DataLoader config."""
    return build_dataclass(LoaderConfig, cfg.get("loader", {}))


def axes_from_config(cfg: DictConfig) -> list[SweepAxis]:
    """Assemble the corruption grid described by ``cfg.eval`` (and ``cfg.corruption``).

    With no explicit axis list, this returns the full standard grid. An explicit list
    selects individual (modality, operator) pairs — that is what the per-family configs
    in ``configs/corruption/`` are for.
    """
    eval_node = cast(dict[str, Any], to_plain(cfg.get("eval", {})) or {})
    corruption_node = cast(dict[str, Any], to_plain(cfg.get("corruption", {})) or {})
    merged: dict[str, Any] = {**eval_node, **corruption_node}

    severities = tuple(float(s) for s in merged.get("severities", DEFAULT_SEVERITIES))
    variants = tuple(merged.get("removal_variants", REMOVAL_VARIANTS))
    explicit = merged.get("axes") or []

    if not explicit:
        return standard_grid(
            severities=severities,
            removal_variants=variants,
            include_graded=bool(merged.get("include_graded", True)),
            include_misalign=bool(merged.get("include_misalign", True)),
        )

    axes: list[SweepAxis] = []
    for item in explicit:
        kind = str(item["kind"])
        params = dict(item.get("params") or {})
        modality_name = item.get("modality")
        if kind == "misalign" or modality_name in {None, "all"}:
            axes.append(misalignment_axis(severities, **params))
            continue
        axes.append(graded_axis(modality_from_str(str(modality_name)), kind, severities, **params))

    if merged.get("include_removal", False):
        axes.extend(removal_grid(variants))
    return axes


def describe_config(cfg: DictConfig) -> str:
    """Compact one-line summary printed at the start of every run."""
    return (
        f"data={cfg.data.name} model={cfg.model.name} seed={cfg.get('seed', 0)} "
        f"epochs={cfg.train.max_epochs}"
    )
