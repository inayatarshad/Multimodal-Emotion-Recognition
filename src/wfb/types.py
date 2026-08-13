"""Core datatypes shared across the whole package.

Everything downstream of :mod:`wfb.data.loaders` is written against the contract
defined here, so no consumer needs to know whether the features came from
CMU-MultimodalSDK, a local pickle, or the synthetic generator.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

TaskType = Literal["regression", "classification"]
SplitName = Literal["train", "val", "test"]

SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")


class Modality(str, Enum):
    """The three modalities. Values match the keys used in configs and the API."""

    TEXT = "text"
    AUDIO = "audio"
    VISUAL = "visual"

    @classmethod
    def all(cls) -> tuple[Modality, ...]:
        """Canonical ordering, used everywhere a modality axis is indexed."""
        return (cls.TEXT, cls.AUDIO, cls.VISUAL)

    @property
    def short(self) -> str:
        """One-letter code (``T`` / ``A`` / ``V``) for compact table headers."""
        return {"text": "T", "audio": "A", "visual": "V"}[self.value]


ModalityDict = dict[Modality, Tensor]


def modality_from_str(name: str) -> Modality:
    """Parse a modality name, accepting the common aliases used in the literature."""
    key = name.strip().lower()
    aliases = {
        "t": "text",
        "l": "text",
        "lang": "text",
        "language": "text",
        "text": "text",
        "a": "audio",
        "acoustic": "audio",
        "audio": "audio",
        "v": "visual",
        "video": "visual",
        "vision": "visual",
        "visual": "visual",
    }
    if key not in aliases:
        raise ValueError(f"Unknown modality {name!r}; expected one of {list(aliases)}")
    return Modality(aliases[key])


@dataclass(frozen=True)
class FeatureStats:
    """Per-feature training-set statistics for one modality.

    Corruption operators are calibrated against these (e.g. Gaussian noise at a target
    SNR needs the signal power), and ``mean`` is the replacement value for the ``mean``
    ablation variant. They are computed on **train only** — using val/test statistics
    would leak.
    """

    mean: Tensor  # (D,)
    std: Tensor  # (D,)
    rms: Tensor  # (D,) root-mean-square, i.e. sqrt(E[x^2]); the signal amplitude for SNR

    def to(self, device: torch.device | str) -> FeatureStats:
        """Move all statistics to ``device``."""
        return FeatureStats(
            mean=self.mean.to(device), std=self.std.to(device), rms=self.rms.to(device)
        )


@dataclass
class SplitData:
    """One split (train / val / test) of a dataset, fully materialised in memory.

    Feature tensors are ``(N, T, D)`` — sequence length ``T`` is the aligned window
    (padded/truncated at load time), so batching is trivial and corruption operators
    can assume a rectangular tensor.
    """

    ids: list[str]
    features: ModalityDict
    labels: Tensor  # (N,) float for regression, (N,) long for classification
    emotions: Tensor | None = None  # (N, 6) multi-label emotion intensities, MOSEI only
    lengths: Tensor | None = None  # (N,) true (pre-padding) sequence lengths

    def __post_init__(self) -> None:
        n = len(self.ids)
        for modality, tensor in self.features.items():
            if tensor.ndim != 3:
                raise ValueError(
                    f"{modality.value} features must be (N, T, D), got {tuple(tensor.shape)}"
                )
            if tensor.shape[0] != n:
                raise ValueError(
                    f"{modality.value} has {tensor.shape[0]} rows but there are {n} ids"
                )
        if self.labels.shape[0] != n:
            raise ValueError(f"labels has {self.labels.shape[0]} rows but there are {n} ids")

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def seq_len(self) -> int:
        """Aligned sequence length ``T`` (identical across modalities by construction)."""
        return int(next(iter(self.features.values())).shape[1])

    def dims(self) -> dict[Modality, int]:
        """Feature dimensionality per modality."""
        return {m: int(t.shape[2]) for m, t in self.features.items()}

    def subset(self, indices: Tensor | list[int]) -> SplitData:
        """Return a new :class:`SplitData` containing only ``indices`` (used by tests)."""
        idx = torch.as_tensor(indices, dtype=torch.long)
        return SplitData(
            ids=[self.ids[int(i)] for i in idx],
            features={m: t[idx].clone() for m, t in self.features.items()},
            labels=self.labels[idx].clone(),
            emotions=None if self.emotions is None else self.emotions[idx].clone(),
            lengths=None if self.lengths is None else self.lengths[idx].clone(),
        )


@dataclass
class DatasetBundle:
    """A whole dataset: three splits, plus the metadata every consumer needs."""

    name: str
    task: TaskType
    splits: dict[SplitName, SplitData]
    stats: dict[Modality, FeatureStats]
    provenance: Provenance
    num_classes: int = 1
    label_range: tuple[float, float] = (-3.0, 3.0)
    class_names: list[str] = field(default_factory=list)

    def __getitem__(self, split: SplitName) -> SplitData:
        return self.splits[split]

    def __iter__(self) -> Iterator[SplitName]:
        return iter(SPLIT_NAMES)

    @property
    def dims(self) -> dict[Modality, int]:
        """Feature dimensionality per modality, taken from the train split."""
        return self.splits["train"].dims()

    @property
    def seq_len(self) -> int:
        """Aligned sequence length."""
        return self.splits["train"].seq_len

    @property
    def is_synthetic(self) -> bool:
        """True when the features came from the synthetic generator, not a real corpus."""
        return self.provenance.source == "synthetic"

    def describe(self) -> str:
        """One-line human summary, printed by the CLI and logged into results JSON."""
        sizes = ", ".join(f"{s}={len(self.splits[s])}" for s in SPLIT_NAMES)
        dims = ", ".join(f"{m.short}={d}" for m, d in self.dims.items())
        return (
            f"{self.name} [{self.task}] {sizes} | T={self.seq_len} dims({dims}) "
            f"| source={self.provenance.source}"
        )


@dataclass(frozen=True)
class Provenance:
    """Where a dataset came from. Carried into every results file.

    This exists so a number computed on synthetic features can never be silently
    mistaken for a number computed on CMU-MOSEI.
    """

    source: Literal["cache", "local_file", "mmsdk", "synthetic"]
    detail: str = ""
    checksum: str = ""

    def to_dict(self) -> dict[str, str]:
        """JSON-serialisable form."""
        return {"source": self.source, "detail": self.detail, "checksum": self.checksum}


@dataclass(frozen=True)
class CorruptionSpec:
    """A single corruption applied to a single modality at a given severity.

    ``severity`` is always normalised to ``[0, 1]``; each operator maps it onto its own
    physical parameter (SNR in dB, word error rate, blur sigma, ...). ``severity == 0``
    is an exact identity for every operator — that invariant is unit-tested.
    """

    modality: Modality
    kind: str
    severity: float = 0.0
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {self.severity}")

    @property
    def is_identity(self) -> bool:
        """True when this spec provably does nothing."""
        return self.kind in {"none", "identity"} or self.severity == 0.0

    def key(self) -> str:
        """Stable short key used in results tables and cache keys."""
        return f"{self.modality.value}:{self.kind}@{self.severity:.4g}"


@dataclass(frozen=True)
class CorruptionPlan:
    """An ordered collection of :class:`CorruptionSpec`, applied as a pipeline."""

    specs: tuple[CorruptionSpec, ...] = ()

    @classmethod
    def clean(cls) -> CorruptionPlan:
        """The empty plan — the clean baseline."""
        return cls(())

    @classmethod
    def single(
        cls, modality: Modality, kind: str, severity: float, **params: Any
    ) -> CorruptionPlan:
        """Convenience constructor for the common one-operator case."""
        return cls((CorruptionSpec(modality, kind, severity, params),))

    @property
    def is_clean(self) -> bool:
        """True when nothing in the plan changes the input."""
        return all(spec.is_identity for spec in self.specs)

    def key(self) -> str:
        """Deterministic key: safe for filenames, dict keys and cache lookups."""
        if self.is_clean:
            return "clean"
        return "+".join(sorted(s.key() for s in self.specs if not s.is_identity))

    def hash(self) -> str:
        """Short stable hash of the plan, for cache keys on the serving side."""
        import hashlib

        payload = json.dumps(
            [
                {
                    "m": s.modality.value,
                    "k": s.kind,
                    "s": round(s.severity, 6),
                    "p": dict(sorted(s.params.items())),
                }
                for s in self.specs
            ],
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> list[dict[str, Any]]:
        """JSON-serialisable form."""
        return [
            {
                "modality": s.modality.value,
                "kind": s.kind,
                "severity": s.severity,
                "params": dict(s.params),
            }
            for s in self.specs
        ]

    @classmethod
    def from_dict(cls, raw: list[dict[str, Any]] | None) -> CorruptionPlan:
        """Inverse of :meth:`to_dict`."""
        if not raw:
            return cls.clean()
        return cls(
            tuple(
                CorruptionSpec(
                    modality=modality_from_str(str(item["modality"])),
                    kind=str(item.get("kind", "none")),
                    severity=float(item.get("severity", 0.0)),
                    params=dict(item.get("params", {})),
                )
                for item in raw
            )
        )


def resolve_path(path: str | Path) -> Path:
    """Expand ``~`` and make a path absolute without requiring it to exist."""
    return Path(path).expanduser().resolve()
