"""Architectures spanning the fusion-sophistication axis, plus the build factory."""

from __future__ import annotations

from typing import Any

from wfb.models.base import BaseFusionModel, DataSpec, ModelConfig, ModelOutput
from wfb.models.encoders import PredictionHead, TemporalEncoder
from wfb.models.fusion_simple import EarlyFusionModel, LateFusionModel
from wfb.models.lmf import LowRankFusionModel
from wfb.models.mult import MultimodalTransformerModel
from wfb.models.tfn import TensorFusionModel
from wfb.models.unimodal import UnimodalModel
from wfb.types import DatasetBundle

MODEL_REGISTRY: dict[str, type[BaseFusionModel]] = {
    "unimodal": UnimodalModel,
    "text_only": UnimodalModel,
    "audio_only": UnimodalModel,
    "visual_only": UnimodalModel,
    "early": EarlyFusionModel,
    "late": LateFusionModel,
    "tfn": TensorFusionModel,
    "lmf": LowRankFusionModel,
    "mult": MultimodalTransformerModel,
}

# Ordering along the fusion-sophistication axis — the independent variable for H1.
# Used for consistent plot ordering and for the brittleness-index correlation.
SOPHISTICATION_ORDER: tuple[str, ...] = (
    "text_only",
    "audio_only",
    "visual_only",
    "late",
    "early",
    "lmf",
    "tfn",
    "mult",
)


def available_models() -> list[str]:
    """All registered architecture names."""
    return sorted(MODEL_REGISTRY)


def build_model(cfg: ModelConfig, spec: DataSpec) -> BaseFusionModel:
    """Instantiate the architecture named by ``cfg.name``.

    The three unimodal entries are the same class with a restricted modality list; the
    restriction is applied here so ``configs/model/*.yaml`` stay one-liners.
    """
    if cfg.name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {cfg.name!r}. Available: {available_models()}")

    if cfg.name.endswith("_only"):
        modality = cfg.name.removesuffix("_only")
        cfg = ModelConfig(**{**cfg.__dict__, "modalities": (modality,)})

    return MODEL_REGISTRY[cfg.name](cfg, spec)


def build_from_bundle(cfg: ModelConfig, bundle: DatasetBundle) -> BaseFusionModel:
    """Convenience wrapper: derive the :class:`DataSpec` from a loaded dataset."""
    return build_model(cfg, DataSpec.from_bundle(bundle))


def model_summary(model: BaseFusionModel) -> dict[str, Any]:
    """Registry-facing description, used by ``GET /api/models`` and the Pareto plot."""
    return {
        "name": model.cfg.name,
        "class": type(model).__name__,
        "modalities": [m.value for m in model.active],
        "parameters": model.num_parameters,
        "encoder": model.cfg.encoder,
        "hidden": model.cfg.hidden,
        "modality_dropout": model.cfg.modality_dropout,
    }


__all__ = [
    "MODEL_REGISTRY",
    "SOPHISTICATION_ORDER",
    "BaseFusionModel",
    "DataSpec",
    "EarlyFusionModel",
    "LateFusionModel",
    "LowRankFusionModel",
    "ModelConfig",
    "ModelOutput",
    "MultimodalTransformerModel",
    "PredictionHead",
    "TemporalEncoder",
    "TensorFusionModel",
    "UnimodalModel",
    "available_models",
    "build_from_bundle",
    "build_model",
    "model_summary",
]
