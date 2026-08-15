"""Model registry, warm cache, and single-sample inference for the demo API.

Everything expensive happens once, at startup: the dataset is loaded into memory and
every available checkpoint is instantiated and moved to eval mode. A request then costs
one small forward pass, which is what makes the live-slider interaction feel immediate.

Checkpoints are discovered rather than declared, so the API serves whatever has actually
been trained. Architectures without a checkpoint are still served — flagged
``trained: false`` — because a demo that 500s on a fresh clone is worse than one that is
honest about what it is showing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from wfb.corruption.registry import apply_plan, describe_plan, plan_generator
from wfb.data.loaders import DataConfig, load_dataset
from wfb.models import SOPHISTICATION_ORDER, ModelConfig, build_model
from wfb.models.base import BaseFusionModel, DataSpec
from wfb.types import (
    CorruptionPlan,
    CorruptionSpec,
    DatasetBundle,
    Modality,
    ModalityDict,
    SplitName,
    modality_from_str,
)

logger = logging.getLogger(__name__)

SENTIMENT_BANDS = ((-3.0, -0.35, "negative"), (-0.35, 0.35, "neutral"), (0.35, 3.0, "positive"))


def sentiment_label(value: float) -> str:
    """Discretise a continuous sentiment score for display."""
    for low, high, name in SENTIMENT_BANDS:
        if low <= value < high:
            return name
    return "positive" if value >= 0 else "negative"


@dataclass
class LoadedModel:
    """A model in the registry, with the metadata the API reports."""

    name: str
    module: BaseFusionModel
    trained: bool
    checkpoint: str | None = None
    clean_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def fusion_rank(self) -> int | None:
        """Index on the fusion-sophistication axis, or ``None`` if unranked."""
        try:
            return SOPHISTICATION_ORDER.index(self.name)
        except ValueError:
            return None


class ResultCache:
    """Response cache keyed by ``(sample_id, model, corruption_hash)``.

    Redis when reachable, an in-process dict otherwise. The demo is extremely repetitive —
    users drag a slider back and forth over the same dozen severities — so hit rates are
    high and the fallback dict is genuinely sufficient for a single-process deployment.
    """

    def __init__(self, url: str | None = None, max_entries: int = 4096) -> None:
        self.max_entries = max_entries
        self._local: dict[str, Any] = {}
        self._redis: Any = None
        self.backend = "memory"
        if url:
            try:
                import redis

                client = redis.Redis.from_url(url, socket_connect_timeout=0.5)
                client.ping()
                self._redis = client
                self.backend = "redis"
                logger.info("Result cache: redis at %s", url)
            except Exception as exc:
                logger.warning("Redis unavailable (%s); using the in-process cache", exc)

    @staticmethod
    def key(sample_id: str, model: str, corruption_hash: str) -> str:
        """Deterministic cache key."""
        return f"wfb:{sample_id}:{model}:{corruption_hash}"

    def get(self, key: str) -> dict[str, Any] | None:
        """Look up a cached response."""
        if self._redis is not None:
            raw = self._redis.get(key)
            return None if raw is None else json.loads(raw)
        return self._local.get(key)

    def set(self, key: str, value: dict[str, Any], ttl: int = 3600) -> None:
        """Store a response."""
        if self._redis is not None:
            self._redis.setex(key, ttl, json.dumps(value))
            return
        if len(self._local) >= self.max_entries:
            self._local.pop(next(iter(self._local)))
        self._local[key] = value

    def clear(self) -> None:
        """Drop everything (used by tests)."""
        self._local.clear()
        if self._redis is not None:
            self._redis.flushdb()


@dataclass
class RegistryConfig:
    """Startup configuration for the serving layer."""

    dataset: str = "mosi"
    checkpoint_dir: str = "outputs"
    results_dir: str = "experiments/results"
    split: SplitName = "test"
    redis_url: str | None = None
    seed: int = 0
    device: str = "cpu"
    models: tuple[str, ...] = ("text_only", "late", "early", "tfn", "lmf", "mult")
    force_synthetic: bool = False


class InferenceRegistry:
    """Holds the dataset and every loaded model for the lifetime of the process."""

    def __init__(self, cfg: RegistryConfig | None = None) -> None:
        self.cfg = cfg or RegistryConfig()
        self.started = time.time()
        self.cache = ResultCache(self.cfg.redis_url)
        self.bundle: DatasetBundle = load_dataset(
            DataConfig(
                name=self.cfg.dataset,
                force_synthetic=self.cfg.force_synthetic,
                cache=not self.cfg.force_synthetic,
            ),
            verify_splits=False,
        )
        self.spec = DataSpec.from_bundle(self.bundle)
        self.models: dict[str, LoadedModel] = {}
        self._index: dict[str, int] = {
            sample_id: i for i, sample_id in enumerate(self.bundle[self.cfg.split].ids)
        }
        self._load_models()

    # ------------------------------------------------------------------ startup

    def _load_models(self) -> None:
        for name in self.cfg.models:
            model_cfg = self._config_for(name)
            module = build_model(model_cfg, self.spec)
            checkpoint = self._find_checkpoint(name)
            trained = False
            clean_metrics: dict[str, float] = {}
            if checkpoint is not None:
                try:
                    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    module.load_state_dict(state.get("state_dict", state))
                    trained = True
                    clean_metrics = self._clean_metrics_for(checkpoint)
                except (RuntimeError, KeyError) as exc:
                    logger.warning("Could not load %s: %s", checkpoint, exc)
            module.eval()
            module.to(self.cfg.device)
            self.models[name] = LoadedModel(
                name=name,
                module=module,
                trained=trained,
                checkpoint=str(checkpoint) if checkpoint else None,
                clean_metrics=clean_metrics,
            )
        logger.info(
            "Registry ready: %d models (%d trained) on %s",
            len(self.models),
            sum(1 for m in self.models.values() if m.trained),
            self.bundle.describe(),
        )

    def _config_for(self, name: str) -> ModelConfig:
        """Model config, read from the training run's saved config when available."""
        run_dir = Path(self.cfg.checkpoint_dir) / f"{self.bundle.name}_{name}_s{self.cfg.seed}"
        record = run_dir / "train_result.json"
        if record.exists():
            try:
                saved = json.loads(record.read_text(encoding="utf-8"))["config"]["model"]
                saved["modalities"] = tuple(saved.get("modalities", ("text", "audio", "visual")))
                valid = ModelConfig().__dict__.keys()
                return ModelConfig(**{k: v for k, v in saved.items() if k in valid})
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Ignoring unreadable config at %s: %s", record, exc)
        if name == "mult":
            return ModelConfig(name=name, encoder="transformer", hidden=40, heads=5, layers=2)
        if name == "early":
            return ModelConfig(name=name, encoder="transformer", layers=2)
        return ModelConfig(name=name)

    def _find_checkpoint(self, name: str) -> Path | None:
        root = Path(self.cfg.checkpoint_dir)
        run_dir = root / f"{self.bundle.name}_{name}_s{self.cfg.seed}"

        # `train_result.json` is the authoritative record of which file that run actually
        # produced, and it is written next to the config the weights match. Guessing
        # "best.ckpt" instead is how a stale checkpoint from an earlier preset got loaded
        # against a newer architecture config and failed on a shape mismatch.
        recorded = self._recorded_checkpoint(run_dir)
        if recorded is not None:
            return recorded

        candidates = [
            run_dir / "best.ckpt",
            *sorted(root.glob(f"{self.bundle.name}_{name}_s*/best.ckpt")),
        ]
        return next((c for c in candidates if c.exists()), None)

    def _recorded_checkpoint(self, run_dir: Path) -> Path | None:
        """The checkpoint path a completed training run recorded for itself."""
        record = run_dir / "train_result.json"
        if not record.exists():
            return None
        try:
            recorded = json.loads(record.read_text(encoding="utf-8")).get("checkpoint")
        except (ValueError, TypeError):
            return None
        if not recorded:
            return None
        path = Path(str(recorded))
        if path.exists():
            return path
        # The run may have been moved; fall back to the same filename in this directory.
        local = run_dir / path.name
        return local if local.exists() else None

    def _clean_metrics_for(self, checkpoint: Path) -> dict[str, float]:
        record = checkpoint.parent / "train_result.json"
        if not record.exists():
            return {}
        try:
            payload = json.loads(record.read_text(encoding="utf-8"))
            return {k: float(v) for k, v in payload.get("clean_metrics", {}).items()}
        except (ValueError, TypeError):
            return {}

    # ------------------------------------------------------------------ lookups

    def sample_ids(self, limit: int = 50) -> list[str]:
        """Ids of the first ``limit`` demo samples."""
        return self.bundle[self.cfg.split].ids[:limit]

    def has_sample(self, sample_id: str) -> bool:
        """Whether ``sample_id`` exists in the served split."""
        return sample_id in self._index

    def features_for(self, sample_id: str) -> ModalityDict:
        """The clean feature tensors for one sample, batched as ``(1, T, D)``."""
        if sample_id not in self._index:
            raise KeyError(f"Unknown sample {sample_id!r}")
        index = self._index[sample_id]
        split = self.bundle[self.cfg.split]
        return {m: t[index : index + 1].clone() for m, t in split.features.items()}

    def label_for(self, sample_id: str) -> float:
        """Ground-truth label for a sample."""
        return float(self.bundle[self.cfg.split].labels[self._index[sample_id]])

    def sample_index(self, sample_id: str) -> int:
        """Row index of ``sample_id`` in the served split."""
        return self._index[sample_id]

    # ------------------------------------------------------------------ inference

    def build_plan(
        self, settings: Sequence[tuple[str, str, float, Mapping[str, float]]]
    ) -> CorruptionPlan:
        """Turn API corruption settings into a validated :class:`CorruptionPlan`."""
        specs: list[CorruptionSpec] = []
        for modality_name, kind, severity, params in settings:
            if kind in {"none", ""} or severity <= 0.0:
                continue
            specs.append(
                CorruptionSpec(
                    modality=modality_from_str(modality_name),
                    kind=kind,
                    severity=float(severity),
                    params=dict(params),
                )
            )
        return CorruptionPlan(tuple(specs))

    @torch.no_grad()
    def infer(
        self,
        sample_id: str,
        model_name: str,
        plan: CorruptionPlan,
        return_attention: bool = False,
        with_contributions: bool = True,
    ) -> dict[str, Any]:
        """Run one sample through one model under ``plan``.

        Returns a plain dict (not a Pydantic model) so it can be cached as JSON and
        reused by both the REST and WebSocket paths.
        """
        if model_name not in self.models:
            raise KeyError(f"Unknown model {model_name!r}")
        entry = self.models[model_name]
        started = time.perf_counter()

        clean = self.features_for(sample_id)
        index = self.sample_index(sample_id)
        corrupted = apply_plan(
            clean,
            plan,
            stats=self.bundle.stats,
            generator=plan_generator(plan, index, self.cfg.seed),
            mask_vectors=entry.module.mask_vectors(),
        )

        clean_out = entry.module({m: clean[m] for m in entry.module.active})
        out = entry.module({m: corrupted[m] for m in entry.module.active})
        prediction = _scalar(out.prediction)
        clean_prediction = _scalar(clean_out.prediction)

        payload: dict[str, Any] = {
            "sample_id": sample_id,
            "model": model_name,
            "prediction": prediction,
            "label": self.label_for(sample_id),
            "clean_prediction": clean_prediction,
            "delta": prediction - clean_prediction,
            "sentiment": sentiment_label(prediction),
            "confidences": _confidences(out.prediction, self.bundle),
            "contributions": (
                self._contributions(entry.module, corrupted) if with_contributions else []
            ),
            "attention": _attention_payload(out.attention) if return_attention else {},
            "corruption_description": describe_plan(plan),
            "corruption_hash": plan.hash(),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "cached": False,
        }
        return payload

    @torch.no_grad()
    def _contributions(
        self, module: BaseFusionModel, features: ModalityDict
    ) -> list[dict[str, Any]]:
        """Leave-one-out contribution estimate per modality.

        Deliberately model-agnostic: ablate one modality at a time from the *already
        corrupted* input and measure how far the prediction moves. It is cheap (one extra
        forward pass per modality), it needs no gradients, and it means the same number is
        comparable across architectures — an attention-based explanation would not be,
        since only one architecture has attention.
        """
        baseline = _scalar(module(features).prediction)
        raw: list[tuple[Modality, float]] = []
        for modality in module.active:
            ablated = dict(features)
            ablated[modality] = torch.zeros_like(features[modality])
            raw.append((modality, _scalar(module(ablated).prediction) - baseline))

        total = sum(abs(v) for _, v in raw) or 1.0
        return [
            {
                "modality": modality.value,
                "contribution": value,
                "relative": abs(value) / total,
            }
            for modality, value in raw
        ]

    # ------------------------------------------------------------------ misc

    @property
    def uptime(self) -> float:
        """Seconds since startup."""
        return time.time() - self.started

    def fingerprint(self) -> str:
        """Short hash of the served configuration, for cache invalidation across restarts."""
        payload = json.dumps(
            {
                "dataset": self.bundle.name,
                "provenance": self.bundle.provenance.to_dict(),
                "models": {n: m.checkpoint for n, m in self.models.items()},
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _scalar(prediction: torch.Tensor) -> float:
    """First element of a prediction tensor as a float (regression head)."""
    flat = prediction.reshape(-1)
    return float(flat[0]) if flat.numel() else float("nan")


def _confidences(prediction: torch.Tensor, bundle: DatasetBundle) -> dict[str, float]:
    """Class confidences for classification, or a softmax over sentiment bands."""
    if bundle.task == "classification":
        probabilities = torch.softmax(prediction.reshape(1, -1), dim=-1).reshape(-1)
        names = bundle.class_names or [f"class_{i}" for i in range(probabilities.numel())]
        return {name: float(p) for name, p in zip(names, probabilities, strict=False)}

    value = _scalar(prediction)
    # Distance-based pseudo-confidence over the three display bands. Explicitly a display
    # aid, not a calibrated probability — a regression head does not have one.
    logits = torch.tensor(
        [-(abs(value - centre)) for centre in (-1.5, 0.0, 1.5)], dtype=torch.float32
    )
    probabilities = torch.softmax(logits * 2.0, dim=0)
    return {
        name: float(p)
        for name, p in zip(("negative", "neutral", "positive"), probabilities, strict=True)
    }


def _attention_payload(attention: dict[str, torch.Tensor]) -> dict[str, list[list[float]]]:
    """Downsample attention maps to something a browser can draw."""
    out: dict[str, list[list[float]]] = {}
    for key, weights in attention.items():
        matrix = weights[0] if weights.ndim == 3 else weights
        if matrix.ndim != 2:
            continue
        out[key] = [[round(float(v), 5) for v in row] for row in matrix[:32, :32]]
    return out
