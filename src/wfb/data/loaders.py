"""Dataset loading, with a fallback chain and a disk cache of aligned feature tensors.

Resolution order (see PLAN.md, D1):

1. ``data/processed/{name}.pt`` — a previously built cache.
2. ``data/raw/{...}.pkl|.pt|.h5`` — the MMSA / MulT-format aligned archives, which are
   what practitioners actually distribute and use.
3. CMU-MultimodalSDK computational sequences, aligned to the label sequence.
4. The deterministic synthetic corpus (:mod:`wfb.data.synthetic`).

Every path produces an identical :class:`~wfb.types.DatasetBundle`, and
``bundle.provenance`` records which one was taken.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from wfb.data import splits as splits_mod
from wfb.data.synthetic import SyntheticConfig, compute_stats, make_synthetic_bundle
from wfb.types import (
    SPLIT_NAMES,
    DatasetBundle,
    FeatureStats,
    Modality,
    Provenance,
    SplitData,
    SplitName,
    TaskType,
    resolve_path,
)

logger = logging.getLogger(__name__)

# Keys used by the MMSA / MulT pickles, mapped onto our modality names.
_PICKLE_KEYS: dict[Modality, tuple[str, ...]] = {
    Modality.TEXT: ("text", "text_bert", "language", "l"),
    Modality.AUDIO: ("audio", "acoustic", "a"),
    Modality.VISUAL: ("vision", "visual", "video", "v"),
}
_SPLIT_ALIASES: dict[SplitName, tuple[str, ...]] = {
    "train": ("train",),
    "val": ("valid", "val", "dev"),
    "test": ("test",),
}

# Published feature dimensionality, used to validate that a local archive is what it
# claims to be. ``None`` means "accept whatever the file provides".
_EXPECTED_DIMS: dict[str, tuple[int, int, int]] = {
    "mosi": (300, 5, 20),
    "mosei": (300, 74, 35),
}


class DataError(RuntimeError):
    """Raised when a dataset cannot be resolved from any source."""


@dataclass
class DataConfig:
    """Everything the loader needs. Populated from ``configs/data/*.yaml`` via Hydra."""

    name: str = "mosi"
    task: TaskType = "regression"
    root: str = "data"
    seq_len: int = 50
    num_classes: int = 1
    label_range: tuple[float, float] = (-3.0, 3.0)
    class_names: list[str] = field(default_factory=list)
    normalize: bool = True
    """Z-score each feature using train statistics. Essential for COVAREP, whose feature
    scales differ by orders of magnitude; harmless for GloVe."""
    storage_dtype: str = "float32"
    """Precision for the cached feature tensors. ``float16`` halves both the cache file
    and the resident memory, which is what makes CMU-MOSEI (~4 GB of float32 features,
    dominated by 768-d BERT text) usable on a 16 GB machine. Safe here because features
    are z-scored into roughly [-5, 5] before storage, where float16 resolves to ~0.002 —
    far finer than the signal — and every batch is widened back to float32 before it
    reaches a model. Statistics are always computed in float32 regardless."""
    allow_synthetic: bool = True
    """When False, a missing corpus is a hard error instead of falling back."""
    force_synthetic: bool = False
    """Bypass the real-data search entirely (used by CI and the unit tests)."""
    raw_filename: str | None = None
    """Explicit filename inside ``{root}/raw`` — otherwise a set of conventional names
    is probed."""
    synthetic: SyntheticConfig | None = None
    cache: bool = True

    @property
    def raw_dir(self) -> Path:
        """``{root}/raw``."""
        return resolve_path(self.root) / "raw"

    @property
    def processed_dir(self) -> Path:
        """``{root}/processed``."""
        return resolve_path(self.root) / "processed"

    @property
    def cache_path(self) -> Path:
        """Where the aligned tensor cache for this dataset lives."""
        suffix = "_synthetic" if self.force_synthetic else ""
        norm = "_norm" if self.normalize else ""
        precision = "" if self.storage_dtype == "float32" else f"_{self.storage_dtype}"
        return self.processed_dir / f"{self.name}{suffix}_T{self.seq_len}{norm}{precision}.pt"


# --------------------------------------------------------------------------- helpers


def _storage_dtype(cfg: DataConfig) -> torch.dtype:
    """Resolve ``cfg.storage_dtype`` to a torch dtype."""
    try:
        dtype = getattr(torch, cfg.storage_dtype)
    except AttributeError as exc:
        raise DataError(f"Unknown storage_dtype {cfg.storage_dtype!r}") from exc
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise DataError(f"storage_dtype must name a float dtype, got {cfg.storage_dtype!r}")
    return dtype


def sanitize(tensor: Tensor, clip: float = 1e4) -> Tensor:
    """Replace NaN/Inf and clip absurd magnitudes.

    CMU-MOSEI's COVAREP features genuinely contain NaNs and occasional ``±1e30`` values
    for unvoiced frames. Left alone they poison the first backward pass and every
    statistic computed from them, which is one of the more common ways a MOSEI
    reproduction quietly fails.
    """
    out = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return out.clamp(-clip, clip)


def pad_or_truncate(tensor: Tensor, seq_len: int) -> Tensor:
    """Coerce ``(N, T, D)`` to ``(N, seq_len, D)``.

    Truncation keeps the **tail** of the sequence, matching the convention in the
    CMU-MOSI/MOSEI literature (the informative part of an utterance tends to be late),
    and padding is prepended with zeros for the same reason.
    """
    n, t, d = tensor.shape
    if t == seq_len:
        return tensor
    if t > seq_len:
        return tensor[:, -seq_len:, :].contiguous()
    pad = tensor.new_zeros(n, seq_len - t, d)
    return torch.cat([pad, tensor], dim=1)


def normalize_features(
    splits: dict[SplitName, SplitData],
    stats: dict[Modality, FeatureStats],
    chunk: int = 1024,
) -> None:
    """Z-score features in place using **train** statistics.

    Done in chunks, writing back into the original tensor. The obvious
    ``(tensor - mean) / std`` allocates a second full-size array — 4 GB for MOSEI, and
    8 GB if the source is float16 and gets promoted — which is precisely the allocation
    a 16 GB machine cannot afford. Arithmetic is done in float32 even when storage is
    float16, so precision is lost only at write-back.
    """
    for split in splits.values():
        for modality, tensor in split.features.items():
            stat = stats[modality]
            mean = stat.mean.to(torch.float32)
            std = stat.std.to(torch.float32)
            for start in range(0, tensor.shape[0], chunk):
                block = tensor[start : start + chunk].to(torch.float32)
                tensor[start : start + chunk] = ((block - mean) / std).to(tensor.dtype)
                del block


def _file_checksum(path: Path, limit: int = 1 << 20) -> str:
    """Cheap content fingerprint: size plus a hash of the first megabyte."""
    digest = hashlib.sha256()
    digest.update(str(path.stat().st_size).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- cache io


def _bundle_to_payload(bundle: DatasetBundle) -> dict[str, Any]:
    """Flatten a bundle into primitives + tensors, so ``weights_only=True`` can load it."""
    payload: dict[str, Any] = {
        "format_version": 1,
        "name": bundle.name,
        "task": bundle.task,
        "num_classes": bundle.num_classes,
        "label_range": list(bundle.label_range),
        "class_names": list(bundle.class_names),
        "provenance": bundle.provenance.to_dict(),
        "stats": {
            m.value: {"mean": s.mean, "std": s.std, "rms": s.rms} for m, s in bundle.stats.items()
        },
        "splits": {},
    }
    for name in SPLIT_NAMES:
        split = bundle[name]
        payload["splits"][name] = {
            "ids": list(split.ids),
            "labels": split.labels,
            "emotions": split.emotions,
            "lengths": split.lengths,
            "features": {m.value: t for m, t in split.features.items()},
        }
    return payload


def _payload_to_bundle(payload: dict[str, Any]) -> DatasetBundle:
    """Inverse of :func:`_bundle_to_payload`."""
    splits: dict[SplitName, SplitData] = {}
    for name in SPLIT_NAMES:
        raw = payload["splits"][name]
        splits[name] = SplitData(
            ids=[str(i) for i in raw["ids"]],
            features={Modality(k): v for k, v in raw["features"].items()},
            labels=raw["labels"],
            emotions=raw.get("emotions"),
            lengths=raw.get("lengths"),
        )
    stats = {
        Modality(k): FeatureStats(mean=v["mean"], std=v["std"], rms=v["rms"])
        for k, v in payload["stats"].items()
    }
    prov = payload["provenance"]
    lo, hi = payload["label_range"]
    return DatasetBundle(
        name=str(payload["name"]),
        task=cast(TaskType, payload["task"]),
        splits=splits,
        stats=stats,
        provenance=Provenance(
            source=prov["source"], detail=prov.get("detail", ""), checksum=prov.get("checksum", "")
        ),
        num_classes=int(payload["num_classes"]),
        label_range=(float(lo), float(hi)),
        class_names=[str(c) for c in payload["class_names"]],
    )


def save_cache(bundle: DatasetBundle, path: Path) -> Path:
    """Persist a bundle to ``path`` (atomically — a half-written cache is worse than none)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(_bundle_to_payload(bundle), tmp)
    tmp.replace(path)
    logger.info("Cached %s to %s", bundle.name, path)
    return path


def load_cache(path: Path) -> DatasetBundle:
    """Load a bundle previously written by :func:`save_cache`.

    The **original** source is preserved; the cache is recorded in ``detail``. Reporting
    ``source="cache"`` here would be a hole straight through the provenance guarantee:
    synthetic features that had passed through the cache would come back claiming to be
    something else, so ``is_synthetic`` would be False, the API would stop advertising
    synthetic data to the UI, and results files would lose the warning banner that keeps
    a pipeline-validation number from reading as a real one.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    bundle = _payload_to_bundle(payload)
    original = bundle.provenance
    detail = f"cache={path}"
    if original.detail:
        detail = f"{detail} origin={original.detail}"
    bundle.provenance = Provenance(
        source=original.source, detail=detail, checksum=original.checksum
    )
    return bundle


# --------------------------------------------------------------------- local archives


def _candidate_raw_files(cfg: DataConfig) -> list[Path]:
    """Conventional filenames for the aligned archives, most specific first."""
    if cfg.raw_filename:
        return [cfg.raw_dir / cfg.raw_filename]
    stems = [
        f"{cfg.name}_data",
        f"{cfg.name}_data_noalign",
        f"{cfg.name}",
        f"aligned_{cfg.name}",
        f"{cfg.name}_aligned_50",
        "unaligned_50",
    ]
    out: list[Path] = []
    for stem in stems:
        for ext in (".pkl", ".pt", ".pickle"):
            out.append(cfg.raw_dir / f"{stem}{ext}")
            out.append(cfg.raw_dir / cfg.name / f"{stem}{ext}")
    return out


def _first_key(container: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in container:
            return name
    return None


def _extract_split(raw: dict[str, Any], cfg: DataConfig, index: int) -> SplitData:
    """Turn one split dict from an MMSA/MulT pickle into a :class:`SplitData`.

    Source arrays are **dropped from ``raw`` as they are consumed**. CMU-MOSEI's aligned
    archive is 4.4 GB, and holding the pickle and the converted tensors at the same time
    needs roughly twice that — more than a 16 GB laptop has spare. Freeing each array the
    moment it has been converted keeps the peak near the size of the pickle alone, which
    is the difference between this loading and this thrashing the machine to a halt.
    """
    features: dict[Modality, Tensor] = {}
    for modality, keys in _PICKLE_KEYS.items():
        key = _first_key(raw, keys)
        if key is None:
            raise DataError(f"Archive split is missing a {modality.value} key (tried {keys})")
        tensor = torch.as_tensor(raw.pop(key))
        if tensor.ndim == 4:  # text_bert ships as (N, 3, T) token/mask/type triplets
            tensor = tensor.squeeze(1)
        if tensor.ndim != 3:
            raise DataError(
                f"{modality.value} array has shape {tuple(tensor.shape)}; expected (N, T, D)"
            )
        converted = pad_or_truncate(sanitize(tensor), cfg.seq_len)
        del tensor
        features[modality] = converted.to(_storage_dtype(cfg))
        del converted
        gc.collect()

    label_key = _first_key(raw, ("labels", "label", "regression_labels", "y"))
    if label_key is None:
        raise DataError("Archive split has no label key")
    labels = sanitize(torch.as_tensor(raw[label_key])).reshape(-1)

    id_key = _first_key(raw, ("id", "ids", "segment_id"))
    if id_key is not None:
        ids = [
            "_".join(str(x) for x in item) if isinstance(item, list | tuple) else str(item)
            for item in raw[id_key]
        ]
    else:
        ids = [f"{cfg.name}_{index}_{i:06d}" for i in range(labels.shape[0])]

    emotions = None
    emotion_key = _first_key(raw, ("emotion_labels", "classification_labels", "annotations"))
    if emotion_key is not None:
        candidate = torch.as_tensor(raw[emotion_key])
        if candidate.ndim == 2 and candidate.shape[0] == labels.shape[0]:
            emotions = sanitize(candidate)

    if cfg.task == "classification":
        labels = labels.long()

    return SplitData(ids=ids, features=features, labels=labels, emotions=emotions)


def load_local_archive(path: Path, cfg: DataConfig) -> DatasetBundle:
    """Read an MMSA / MulT-format aligned archive from disk."""
    if path.suffix == ".pt":
        raw_obj = torch.load(path, map_location="cpu", weights_only=False)
    else:
        with path.open("rb") as handle:
            raw_obj = pickle.load(handle)

    if not isinstance(raw_obj, dict):
        raise DataError(f"{path} does not contain a dict of splits")
    data = cast(dict[str, Any], raw_obj)

    splits: dict[SplitName, SplitData] = {}
    for index, (name, aliases) in enumerate(_SPLIT_ALIASES.items()):
        key = _first_key(data, aliases)
        if key is None:
            raise DataError(f"{path} has no {name} split (tried {aliases})")
        splits[name] = _extract_split(data[key], cfg, index)

    expected = _EXPECTED_DIMS.get(cfg.name)
    actual = tuple(splits["train"].dims()[m] for m in Modality.all())
    if expected is not None and actual != expected:
        logger.warning(
            "%s feature dims are %s but published %s uses %s — this is fine for a variant "
            "archive (e.g. BERT text features), but check it is what you meant.",
            path.name,
            actual,
            cfg.name,
            expected,
        )

    stats = compute_stats(splits["train"])
    if cfg.normalize:
        normalize_features(splits, stats)

    return DatasetBundle(
        name=cfg.name,
        task=cfg.task,
        splits=splits,
        stats=stats if not cfg.normalize else compute_stats(splits["train"]),
        provenance=Provenance(source="local_file", detail=str(path), checksum=_file_checksum(path)),
        num_classes=cfg.num_classes,
        label_range=cfg.label_range,
        class_names=list(cfg.class_names),
    )


# -------------------------------------------------------------------------- mmsdk


def load_via_mmsdk(cfg: DataConfig) -> DatasetBundle:
    """Build the bundle from CMU-MultimodalSDK computational sequences.

    Kept deliberately thin, and **expected to fail** in most environments today.

    Two things were verified on 2026-08-14: the SDK moved from ``A2Zadeh/`` to
    ``CMU-MultiComp-Lab/``, and the host that serves the feature files
    (``immortal.multicomp.cs.cmu.edu``) resolves but refuses TCP connections. So this
    path cannot currently download anything, whatever the install status of the SDK.
    That is precisely why the loader treats it as one link in a chain rather than the
    way in; see :func:`load_local_archive` for the route that actually works.
    """
    try:
        from mmsdk import mmdatasdk
    except ImportError as exc:  # pragma: no cover - exercised only with the SDK installed
        raise DataError(
            "CMU-MultimodalSDK is not installed. Note that its download host "
            "(immortal.multicomp.cs.cmu.edu) was unreachable when last checked, so "
            "installing it may not help. The reliable route is to place the aligned "
            "pickle in data/raw/ -- see docs/DATA.md. To try the SDK anyway: "
            "`uv pip install "
            "git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git`"
        ) from exc

    from wfb.data.mmsdk_recipes import RECIPES, build_bundle_from_sdk

    recipe = RECIPES.get(cfg.name)
    if recipe is None:
        raise DataError(f"No CMU-MultimodalSDK recipe for dataset {cfg.name!r}")
    return build_bundle_from_sdk(mmdatasdk, recipe, cfg)


# -------------------------------------------------------------------------- entrypoint


def load_dataset(cfg: DataConfig, verify_splits: bool = True) -> DatasetBundle:
    """Resolve ``cfg`` to a :class:`~wfb.types.DatasetBundle` via the fallback chain."""
    if cfg.cache and cfg.cache_path.exists() and not cfg.force_synthetic:
        bundle = load_cache(cfg.cache_path)
        logger.info("Loaded cache: %s", bundle.describe())
    elif cfg.force_synthetic:
        bundle = _synthesize(cfg)
    else:
        bundle = _load_real_or_synthetic(cfg)
        if cfg.cache:
            save_cache(bundle, cfg.cache_path)

    if verify_splits:
        splits_mod.verify(bundle)
    return bundle


def _load_real_or_synthetic(cfg: DataConfig) -> DatasetBundle:
    errors: list[str] = []

    for candidate in _candidate_raw_files(cfg):
        if candidate.exists():
            try:
                bundle = load_local_archive(candidate, cfg)
                logger.info("Loaded local archive: %s", bundle.describe())
                return bundle
            except (DataError, pickle.UnpicklingError, KeyError, ValueError) as exc:
                errors.append(f"{candidate.name}: {exc}")

    try:
        bundle = load_via_mmsdk(cfg)
        logger.info("Loaded via CMU-MultimodalSDK: %s", bundle.describe())
        return bundle
    except (DataError, OSError, RuntimeError) as exc:
        errors.append(f"mmsdk: {exc}")

    if not cfg.allow_synthetic:
        raise DataError(
            f"Could not load {cfg.name!r} and allow_synthetic=False. Tried:\n  "
            + "\n  ".join(errors)
        )

    logger.warning(
        "Falling back to SYNTHETIC data for %r. Results are for pipeline validation only. "
        "Reasons:\n  %s",
        cfg.name,
        "\n  ".join(errors) or "no raw archive found",
    )
    return _synthesize(cfg)


def _synthesize(cfg: DataConfig) -> DatasetBundle:
    """Build the synthetic bundle, shaped to match ``cfg``."""
    base = cfg.synthetic or SyntheticConfig()
    dims = _EXPECTED_DIMS.get(cfg.name, base.dims)
    syn = SyntheticConfig(
        name=cfg.name,
        task=cfg.task,
        n_train=base.n_train,
        n_val=base.n_val,
        n_test=base.n_test,
        seq_len=cfg.seq_len,
        dims=dims,
        num_classes=cfg.num_classes,
        noise_scale=base.noise_scale,
        seed=base.seed,
    )
    bundle = make_synthetic_bundle(syn)
    if cfg.normalize:
        normalize_features(bundle.splits, bundle.stats)
        bundle.stats = compute_stats(bundle.splits["train"])
    bundle.label_range = cfg.label_range
    bundle.class_names = list(cfg.class_names) or bundle.class_names
    return bundle


def build_cache(cfg: DataConfig, freeze_splits: bool = True) -> DatasetBundle:
    """``make data``: resolve, cache, and freeze the split manifest."""
    bundle = load_dataset(cfg, verify_splits=False)
    if cfg.cache and not cfg.cache_path.exists():
        save_cache(bundle, cfg.cache_path)
    if freeze_splits:
        splits_mod.freeze(bundle)
    else:
        splits_mod.verify(bundle)
    return bundle
