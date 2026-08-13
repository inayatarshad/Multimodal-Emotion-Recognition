"""CMU-MultimodalSDK download/alignment recipes.

This is the slow path. It runs at most once per dataset: the resulting bundle is written
to ``data/processed/`` and every later run hits the cache.

The SDK's computational-sequence key names have drifted across releases
(``CMU_MOSI_Visual_Facet_41`` vs ``CMU_MOSI_Visual_Facet_42``, ``TimestampedWordVectors``
vs ``glove_vectors``, ...), so keys are resolved by **substring matching** against
whatever the download actually produced rather than being hardcoded. That single choice
removes the most common reason this path fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from wfb.data.synthetic import compute_stats
from wfb.types import SPLIT_NAMES, DatasetBundle, Modality, Provenance, SplitData, SplitName

if TYPE_CHECKING:  # pragma: no cover
    from wfb.data.loaders import DataConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SdkRecipe:
    """How to assemble one dataset out of the SDK."""

    sdk_module: str
    """Attribute name on ``mmdatasdk`` (e.g. ``cmu_mosi``)."""
    text_hints: tuple[str, ...] = ("wordvector", "glove", "word_vectors")
    audio_hints: tuple[str, ...] = ("covarep", "opensmile", "acoustic")
    visual_hints: tuple[str, ...] = ("facet", "openface", "visual")
    label_hints: tuple[str, ...] = ("label",)
    align_to_text: bool = True
    """Align features to word boundaries first (the standard 'aligned' protocol)."""
    folds: dict[str, str] = field(
        default_factory=lambda: {"train": "train", "val": "valid", "test": "test"}
    )


RECIPES: dict[str, SdkRecipe] = {
    "mosi": SdkRecipe(sdk_module="cmu_mosi"),
    "mosei": SdkRecipe(sdk_module="cmu_mosei"),
}


def _resolve_key(keys: list[str], hints: tuple[str, ...]) -> str:
    """Find the computational-sequence key matching any of ``hints`` (case-insensitive)."""
    lowered = {k.lower(): k for k in keys}
    for hint in hints:
        for low, original in lowered.items():
            if hint in low:
                return original
    raise KeyError(f"No computational sequence matching {hints} in {keys}")


def _segment_video_id(segment_id: str) -> str:
    """``video[7]`` -> ``video``. Folds are defined at video level, not segment level."""
    return segment_id.split("[")[0]


def _stack_segment(array: Any, seq_len: int) -> torch.Tensor:
    """``(T, D)`` numpy -> padded/truncated ``(seq_len, D)`` tensor."""
    tensor = torch.as_tensor(np.asarray(array, dtype=np.float32))
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)
    t = tensor.shape[0]
    if t > seq_len:
        return tensor[-seq_len:]
    if t < seq_len:
        return torch.cat([tensor.new_zeros(seq_len - t, tensor.shape[1]), tensor], dim=0)
    return tensor


def build_bundle_from_sdk(mmdatasdk: Any, recipe: SdkRecipe, cfg: DataConfig) -> DatasetBundle:
    """Download (if needed), align, fold and package a dataset via the SDK.

    Args:
        mmdatasdk: The imported ``mmsdk.mmdatasdk`` module.
        recipe: Key-resolution hints and fold names for this dataset.
        cfg: Loader config (root directory, sequence length, task).

    Returns:
        A fully populated :class:`~wfb.types.DatasetBundle`.
    """
    dataset_root = cfg.raw_dir / cfg.name
    dataset_root.mkdir(parents=True, exist_ok=True)
    source = getattr(mmdatasdk, recipe.sdk_module)

    logger.info("Downloading %s highlevel features into %s (this is slow)", cfg.name, dataset_root)
    dataset = mmdatasdk.mmdataset(source.highlevel, str(dataset_root))

    keys = list(dataset.computational_sequences.keys())
    text_key = _resolve_key(keys, recipe.text_hints)
    audio_key = _resolve_key(keys, recipe.audio_hints)
    visual_key = _resolve_key(keys, recipe.visual_hints)

    if recipe.align_to_text:
        dataset.align(text_key, collapse_functions=[_avg])
        dataset.impute(text_key)

    dataset.add_computational_sequences(source.labels, str(dataset_root))
    label_key = _resolve_key(list(dataset.computational_sequences.keys()), recipe.label_hints)
    dataset.align(label_key)

    folds = source.standard_folds
    fold_ids: dict[SplitName, set[str]] = {
        "train": set(folds.standard_train_fold),
        "val": set(folds.standard_valid_fold),
        "test": set(folds.standard_test_fold),
    }

    buckets: dict[SplitName, dict[str, list[Any]]] = {
        name: {"ids": [], "text": [], "audio": [], "visual": [], "labels": []}
        for name in SPLIT_NAMES
    }

    label_seq = dataset.computational_sequences[label_key].data
    for segment_id in label_seq:
        video_id = _segment_video_id(segment_id)
        split = next((name for name, ids in fold_ids.items() if video_id in ids), None)
        if split is None:
            continue
        try:
            text = _stack_segment(
                dataset.computational_sequences[text_key].data[segment_id]["features"], cfg.seq_len
            )
            audio = _stack_segment(
                dataset.computational_sequences[audio_key].data[segment_id]["features"],
                cfg.seq_len,
            )
            visual = _stack_segment(
                dataset.computational_sequences[visual_key].data[segment_id]["features"],
                cfg.seq_len,
            )
        except KeyError:
            logger.debug("Segment %s missing in one modality; skipped", segment_id)
            continue
        label = float(np.asarray(label_seq[segment_id]["features"]).reshape(-1)[0])
        bucket = buckets[split]
        bucket["ids"].append(str(segment_id))
        bucket["text"].append(text)
        bucket["audio"].append(audio)
        bucket["visual"].append(visual)
        bucket["labels"].append(label)

    splits: dict[SplitName, SplitData] = {}
    for name in SPLIT_NAMES:
        bucket = buckets[name]
        if not bucket["ids"]:
            raise RuntimeError(f"SDK produced an empty {name} split for {cfg.name}")
        labels = torch.tensor(bucket["labels"], dtype=torch.float32)
        splits[name] = SplitData(
            ids=[str(i) for i in bucket["ids"]],
            features={
                Modality.TEXT: torch.stack(bucket["text"]),
                Modality.AUDIO: torch.stack(bucket["audio"]),
                Modality.VISUAL: torch.stack(bucket["visual"]),
            },
            labels=labels.long() if cfg.task == "classification" else labels,
        )

    stats = compute_stats(splits["train"])
    if cfg.normalize:
        from wfb.data.loaders import normalize_features

        normalize_features(splits, stats)
        stats = compute_stats(splits["train"])

    return DatasetBundle(
        name=cfg.name,
        task=cfg.task,
        splits=splits,
        stats=stats,
        provenance=Provenance(
            source="mmsdk",
            detail=f"{recipe.sdk_module} keys=({text_key},{audio_key},{visual_key})",
            checksum=_dir_fingerprint(dataset_root),
        ),
        num_classes=cfg.num_classes,
        label_range=cfg.label_range,
        class_names=list(cfg.class_names),
    )


def _avg(intervals: Any, features: Any) -> Any:  # noqa: ARG001 - SDK callback signature
    """Collapse function for SDK alignment: mean-pool frames within a word interval."""
    return np.average(features, axis=0)


def _dir_fingerprint(path: Path) -> str:
    """Size-based fingerprint of the downloaded CSD files."""
    import hashlib

    digest = hashlib.sha256()
    for csd in sorted(path.glob("*.csd")):
        digest.update(f"{csd.name}:{csd.stat().st_size}".encode())
    return digest.hexdigest()[:16]
