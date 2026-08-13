"""Data layer: shapes, the frozen split contract, caching, and the fallback chain."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from wfb.data import splits as splits_mod
from wfb.data.datamodule import LoaderConfig, MultimodalDataModule, collate
from wfb.data.loaders import (
    DataConfig,
    DataError,
    load_cache,
    load_dataset,
    load_local_archive,
    pad_or_truncate,
    sanitize,
    save_cache,
)
from wfb.data.synthetic import SyntheticConfig, expected_reliance_order, make_synthetic_bundle
from wfb.types import SPLIT_NAMES, CorruptionPlan, DatasetBundle, Modality


def test_bundle_has_the_expected_shapes(mosi_synthetic: DatasetBundle) -> None:
    assert mosi_synthetic.seq_len == 50
    assert mosi_synthetic.dims == {Modality.TEXT: 300, Modality.AUDIO: 5, Modality.VISUAL: 20}
    for name in SPLIT_NAMES:
        split = mosi_synthetic[name]
        n = len(split)
        assert split.labels.shape == (n,)
        for modality, tensor in split.features.items():
            assert tensor.shape == (n, 50, mosi_synthetic.dims[modality])


def test_splits_are_disjoint(mosi_synthetic: DatasetBundle) -> None:
    ids = {name: set(mosi_synthetic[name].ids) for name in SPLIT_NAMES}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]


def test_synthetic_corpus_is_deterministic(tiny_config: SyntheticConfig) -> None:
    a = make_synthetic_bundle(tiny_config)
    b = make_synthetic_bundle(tiny_config)
    assert a["test"].ids == b["test"].ids
    for modality in Modality.all():
        assert torch.equal(a["test"].features[modality], b["test"].features[modality])
    assert torch.equal(a["test"].labels, b["test"].labels)


def test_synthetic_labels_span_the_declared_range(mosi_synthetic: DatasetBundle) -> None:
    labels = mosi_synthetic["train"].labels
    lo, hi = mosi_synthetic.label_range
    assert float(labels.min()) >= lo
    assert float(labels.max()) <= hi
    assert float(labels.std()) > 0.5, "labels are nearly constant — nothing to learn"


def test_synthetic_signal_is_linearly_recoverable_and_text_dominant(
    mosi_synthetic: DatasetBundle,
) -> None:
    """The planted structure must actually be extractable, in the planted order.

    This is the guard against the failure mode where the generator produces a signal that
    cancels under pooling: every model then sits at chance, and it looks like a training
    bug rather than a data bug.
    """
    import numpy as np
    from sklearn.linear_model import Ridge

    train, test = mosi_synthetic["train"], mosi_synthetic["test"]
    correlations: dict[Modality, float] = {}
    for modality in Modality.all():
        x = train.features[modality].mean(dim=1).numpy()
        xt = test.features[modality].mean(dim=1).numpy()
        predicted = Ridge(alpha=10.0).fit(x, train.labels.numpy()).predict(xt)
        correlations[modality] = float(np.corrcoef(predicted, test.labels.numpy())[0, 1])

    assert correlations[Modality.TEXT] > 0.4, "text signal is not recoverable"
    ranked = sorted(correlations, key=lambda m: correlations[m], reverse=True)
    assert tuple(ranked) == expected_reliance_order(), (
        f"modality importance {ranked} does not match the planted order {expected_reliance_order()}"
    )


def test_train_statistics_are_computed_on_train_only(tiny_bundle: DatasetBundle) -> None:
    for modality in Modality.all():
        flat = tiny_bundle["train"].features[modality].reshape(-1, tiny_bundle.dims[modality])
        assert torch.allclose(tiny_bundle.stats[modality].mean, flat.mean(dim=0), atol=1e-5)
        assert (tiny_bundle.stats[modality].std > 0).all()


# ------------------------------------------------------------------ tensor utilities


def test_sanitize_replaces_nan_and_inf() -> None:
    """COVAREP genuinely contains NaN and huge sentinels for unvoiced frames."""
    x = torch.tensor([[float("nan"), 1.0], [float("inf"), -float("inf")], [1e30, 2.0]])
    out = sanitize(x)
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) <= 1e4


def test_pad_or_truncate_keeps_the_tail() -> None:
    x = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2)
    truncated = pad_or_truncate(x, 4)
    assert truncated.shape == (1, 4, 2)
    assert torch.equal(truncated, x[:, -4:, :])

    padded = pad_or_truncate(x, 14)
    assert padded.shape == (1, 14, 2)
    assert torch.equal(padded[:, 4:, :], x)
    assert torch.equal(padded[:, :4, :], torch.zeros(1, 4, 2))


def test_pad_or_truncate_is_a_no_op_at_the_right_length() -> None:
    x = torch.randn(3, 8, 4)
    assert pad_or_truncate(x, 8) is x


# ------------------------------------------------------------------ cache round trip


def test_cache_round_trip_preserves_everything(tiny_bundle: DatasetBundle, tmp_path: Path) -> None:
    path = tmp_path / "tiny.pt"
    save_cache(tiny_bundle, path)
    restored = load_cache(path)

    assert restored.name == tiny_bundle.name
    assert restored.task == tiny_bundle.task
    assert restored.dims == tiny_bundle.dims
    assert restored.provenance.source == "cache"
    for name in SPLIT_NAMES:
        assert restored[name].ids == tiny_bundle[name].ids
        assert torch.equal(restored[name].labels, tiny_bundle[name].labels)
        for modality in Modality.all():
            assert torch.equal(
                restored[name].features[modality], tiny_bundle[name].features[modality]
            )


def test_cache_write_is_atomic(tiny_bundle: DatasetBundle, tmp_path: Path) -> None:
    path = tmp_path / "sub" / "tiny.pt"
    save_cache(tiny_bundle, path)
    assert path.exists()
    assert not list(path.parent.glob("*.tmp")), "a temporary file was left behind"


# ------------------------------------------------------------------ frozen splits


def test_freeze_then_verify_succeeds(tiny_bundle: DatasetBundle, tmp_path: Path) -> None:
    splits_mod.freeze(tiny_bundle, splits_dir=tmp_path)
    assert splits_mod.verify(tiny_bundle, splits_dir=tmp_path)


def test_verify_rejects_a_changed_split(tiny_bundle: DatasetBundle, tmp_path: Path) -> None:
    splits_mod.freeze(tiny_bundle, splits_dir=tmp_path)
    tampered = DatasetBundle(
        name=tiny_bundle.name,
        task=tiny_bundle.task,
        splits={
            "train": tiny_bundle["train"],
            "val": tiny_bundle["val"],
            "test": tiny_bundle["test"].subset(list(range(5))),  # dropped samples
        },
        stats=tiny_bundle.stats,
        provenance=tiny_bundle.provenance,
    )
    with pytest.raises(splits_mod.SplitMismatchError, match="does not match its frozen split"):
        splits_mod.verify(tampered, splits_dir=tmp_path)


def test_refreezing_a_different_split_requires_an_explicit_overwrite(
    tiny_bundle: DatasetBundle, tmp_path: Path
) -> None:
    splits_mod.freeze(tiny_bundle, splits_dir=tmp_path)
    changed = DatasetBundle(
        name=tiny_bundle.name,
        task=tiny_bundle.task,
        splits={
            "train": tiny_bundle["train"],
            "val": tiny_bundle["val"],
            "test": tiny_bundle["test"].subset(list(range(3))),
        },
        stats=tiny_bundle.stats,
        provenance=tiny_bundle.provenance,
    )
    with pytest.raises(splits_mod.SplitMismatchError, match="overwrite=True"):
        splits_mod.freeze(changed, splits_dir=tmp_path)


def test_refreezing_an_identical_split_is_idempotent(
    tiny_bundle: DatasetBundle, tmp_path: Path
) -> None:
    first = splits_mod.freeze(tiny_bundle, splits_dir=tmp_path)
    second = splits_mod.freeze(tiny_bundle, splits_dir=tmp_path)
    assert first == second


def test_verify_passes_when_no_manifest_exists(tiny_bundle: DatasetBundle, tmp_path: Path) -> None:
    assert splits_mod.verify(tiny_bundle, splits_dir=tmp_path / "empty")


# ------------------------------------------------------------------ fallback chain


def test_local_archive_is_preferred_over_synthetic(tmp_path: Path) -> None:
    """A hand-built MMSA-format pickle must be picked up and parsed."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    payload = {
        split: {
            "text": torch.randn(6, 50, 300).numpy(),
            "audio": torch.randn(6, 50, 5).numpy(),
            "vision": torch.randn(6, 50, 20).numpy(),
            "labels": torch.randn(6, 1).numpy(),
            "id": [f"{split}_{i}" for i in range(6)],
        }
        for split in ("train", "valid", "test")
    }
    with (raw_dir / "mosi_data.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    cfg = DataConfig(name="mosi", root=str(tmp_path), cache=False)
    bundle = load_dataset(cfg, verify_splits=False)
    assert bundle.provenance.source == "local_file"
    assert len(bundle["train"]) == 6
    assert bundle.dims[Modality.AUDIO] == 5


def test_archive_missing_a_modality_is_rejected(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    payload = {
        split: {"text": torch.randn(4, 50, 300).numpy(), "labels": torch.randn(4, 1).numpy()}
        for split in ("train", "valid", "test")
    }
    path = raw_dir / "mosi_data.pkl"
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    cfg = DataConfig(name="mosi", root=str(tmp_path), cache=False)
    with pytest.raises(DataError, match="missing a audio key"):
        load_local_archive(path, cfg)


def test_synthetic_fallback_engages_when_nothing_is_available(tmp_path: Path) -> None:
    cfg = DataConfig(name="mosi", root=str(tmp_path), cache=False)
    bundle = load_dataset(cfg, verify_splits=False)
    assert bundle.provenance.source == "synthetic"
    assert bundle.is_synthetic


def test_synthetic_fallback_can_be_forbidden(tmp_path: Path) -> None:
    cfg = DataConfig(name="mosi", root=str(tmp_path), cache=False, allow_synthetic=False)
    with pytest.raises(DataError, match="allow_synthetic=False"):
        load_dataset(cfg, verify_splits=False)


def test_provenance_marks_synthetic_data_unmistakably(mosi_synthetic: DatasetBundle) -> None:
    """No number derived from synthetic features may look like a real one."""
    assert mosi_synthetic.provenance.source == "synthetic"
    assert mosi_synthetic.provenance.to_dict()["source"] == "synthetic"
    assert "source=synthetic" in mosi_synthetic.describe()


# ------------------------------------------------------------------ datamodule


def test_collate_stacks_a_batch(tiny_bundle: DatasetBundle) -> None:
    from wfb.data.datamodule import MultimodalDataset

    dataset = MultimodalDataset(tiny_bundle["test"], tiny_bundle.stats)
    batch = collate([dataset[i] for i in range(4)])
    assert batch["label"].shape == (4,)
    assert len(batch["id"]) == 4
    for modality in Modality.all():
        assert batch[modality.value].shape[0] == 4


def test_datamodule_train_loader_is_clean_even_with_an_eval_plan(
    tiny_bundle: DatasetBundle,
) -> None:
    """Corruption is an evaluation-time concern; training data must stay clean."""
    cfg = DataConfig(name="tiny", force_synthetic=True, cache=False)
    dm = MultimodalDataModule(
        cfg,
        LoaderConfig(batch_size=4),
        bundle=tiny_bundle,
        eval_plan=CorruptionPlan.single(Modality.AUDIO, "zero", 1.0),
    )
    batch = next(iter(dm.train_dataloader()))
    assert float(batch[Modality.AUDIO.value].abs().sum()) > 0

    corrupted = next(iter(dm.test_dataloader()))
    assert float(corrupted[Modality.AUDIO.value].abs().sum()) == 0


def test_same_plan_gives_identical_batches_across_datamodules(
    tiny_bundle: DatasetBundle,
) -> None:
    """The precondition for the paired significance tests."""
    cfg = DataConfig(name="tiny", force_synthetic=True, cache=False)
    plan = CorruptionPlan.single(Modality.AUDIO, "gaussian_noise", 0.6)
    batches = []
    for _ in range(2):
        dm = MultimodalDataModule(cfg, LoaderConfig(eval_batch_size=8), bundle=tiny_bundle)
        batches.append(next(iter(dm.corrupted_loader("test", plan))))
    assert torch.equal(batches[0][Modality.AUDIO.value], batches[1][Modality.AUDIO.value])
