"""Pydantic v2 schemas — the validated boundary of the HTTP API.

Every request and response crossing the network is described here, which is also what
generates the OpenAPI document served at ``/docs``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModalityName = Literal["text", "audio", "visual"]


class CorruptionSetting(BaseModel):
    """One modality's corruption in a request."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="none", description="Registered operator name, or 'none'")
    severity: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalised severity")
    params: dict[str, float] = Field(default_factory=dict, description="Operator overrides")


class CorruptionRequest(BaseModel):
    """Per-modality corruption block of a predict/compare request."""

    model_config = ConfigDict(extra="forbid")

    text: CorruptionSetting = Field(default_factory=CorruptionSetting)
    audio: CorruptionSetting = Field(default_factory=CorruptionSetting)
    visual: CorruptionSetting = Field(default_factory=CorruptionSetting)

    def as_pairs(self) -> list[tuple[ModalityName, CorruptionSetting]]:
        """Iterate as ``(modality, setting)`` pairs in canonical order."""
        return [("text", self.text), ("audio", self.audio), ("visual", self.visual)]


class PredictRequest(BaseModel):
    """``POST /api/predict``."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(description="Id from GET /api/samples")
    model: str = Field(default="mult", description="Architecture name from GET /api/models")
    corruption: CorruptionRequest = Field(default_factory=CorruptionRequest)
    return_attention: bool = Field(default=False, description="Include cross-modal attention")


class CompareRequest(BaseModel):
    """``POST /api/compare`` — the same corrupted input through every architecture."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    models: list[str] = Field(default_factory=list, description="Empty means all loaded models")
    corruption: CorruptionRequest = Field(default_factory=CorruptionRequest)
    return_attention: bool = False


class ModalityContribution(BaseModel):
    """Per-modality contribution estimate, by leave-one-out ablation."""

    modality: ModalityName
    contribution: float = Field(
        description="Change in prediction when this modality alone is removed"
    )
    relative: float = Field(description="Share of the total absolute contribution")


class PredictResponse(BaseModel):
    """Prediction under one corruption setting, with its delta from clean."""

    sample_id: str
    model: str
    prediction: float
    label: float | None = None
    clean_prediction: float
    delta: float = Field(description="prediction - clean_prediction")
    confidences: dict[str, float] = Field(default_factory=dict)
    sentiment: str = Field(description="Discretised label: negative / neutral / positive")
    contributions: list[ModalityContribution] = Field(default_factory=list)
    attention: dict[str, list[list[float]]] = Field(default_factory=dict)
    corruption_description: str = ""
    corruption_hash: str = ""
    latency_ms: float = 0.0
    cached: bool = False


class CompareResponse(BaseModel):
    """All architectures on the same corrupted input — the hero view's payload."""

    sample_id: str
    corruption_description: str
    results: list[PredictResponse]
    latency_ms: float = 0.0


class ModelInfo(BaseModel):
    """Registry entry from ``GET /api/models``."""

    name: str
    architecture: str
    modalities: list[ModalityName]
    parameters: int
    trained: bool = Field(description="False means randomly initialised — demo only")
    checkpoint: str | None = None
    clean_metrics: dict[str, float] = Field(default_factory=dict)
    fusion_rank: int | None = Field(
        default=None, description="Position on the fusion-sophistication axis"
    )


class SampleInfo(BaseModel):
    """A demo clip from ``GET /api/samples``."""

    id: str
    dataset: str
    split: str
    label: float
    sentiment: str
    media_url: str | None = None
    transcript: str | None = None


class CorruptionInfo(BaseModel):
    """One registered operator, from ``GET /api/corruptions``."""

    name: str
    applies_to: list[ModalityName]
    unit: str
    doc: str


class HealthResponse(BaseModel):
    """``GET /health``."""

    status: Literal["ok", "degraded"]
    version: str
    models_loaded: int
    trained_models: int
    dataset: str
    dataset_source: str
    cache: Literal["redis", "memory", "disabled"]
    uptime_seconds: float


class DegradationCurve(BaseModel):
    """One (model, axis) curve from ``GET /api/results/degradation``."""

    model: str
    axis: str
    metric: str
    severities: list[float]
    retention: list[float]
    retention_std: list[float] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)
    audc: float
    audc_std: float = 0.0
    critical: float | None = None
    seeds: int = 1


class DegradationResponse(BaseModel):
    """``GET /api/results/degradation``."""

    dataset: str
    metric: str
    provenance: str
    curves: list[DegradationCurve]
    brittleness: dict[str, float] = Field(default_factory=dict)


class RelianceEntry(BaseModel):
    """One row of the Modality Reliance Matrix."""

    model: str
    mrs: dict[str, float]
    mrs_normalized: dict[str, float]
    subset_retention: dict[str, float] = Field(default_factory=dict)


class RelianceResponse(BaseModel):
    """``GET /api/results/reliance``."""

    dataset: str
    metric: str
    provenance: str
    entries: list[RelianceEntry]


class ParetoPoint(BaseModel):
    """One point of the robustness Pareto plot."""

    label: str
    base_model: str
    modality_dropout: float
    clean_score: float
    mean_audc: float
    parameters: int
    on_frontier: bool = False


class ParetoResponse(BaseModel):
    """``GET /api/results/pareto``."""

    dataset: str
    metric: str
    points: list[ParetoPoint]


class ErrorResponse(BaseModel):
    """Structured error body."""

    detail: str
    request_id: str | None = None
