"""Lightning datamodule and the corruption-aware dataset.

The corruption is applied inside ``__getitem__`` with an RNG seeded from
``(plan_hash, sample_index)``. Two consequences, both deliberate:

* every architecture evaluated under a given plan sees **bit-identical** inputs, which is
  the precondition for the paired significance tests;
* the corrupted tensors never need to be materialised or cached — a full 200-plan sweep
  over the test set costs one forward pass per plan and no extra memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from wfb.corruption.registry import apply_plan, plan_generator
from wfb.data.loaders import DataConfig, load_dataset
from wfb.types import (
    CorruptionPlan,
    DatasetBundle,
    FeatureStats,
    Modality,
    SplitData,
    SplitName,
)

Batch = dict[str, Any]


class MultimodalDataset(Dataset[Batch]):
    """One split, optionally viewed through a corruption plan."""

    def __init__(
        self,
        split: SplitData,
        stats: dict[Modality, FeatureStats] | None = None,
        plan: CorruptionPlan | None = None,
        seed: int = 0,
        mask_vectors: dict[Modality, Tensor] | None = None,
    ) -> None:
        self.split = split
        self.stats = stats
        self.plan = plan or CorruptionPlan.clean()
        self.seed = seed
        self.mask_vectors = mask_vectors or {}

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> Batch:
        features = {m: t[index] for m, t in self.split.features.items()}
        if not self.plan.is_clean:
            features = apply_plan(
                features,
                self.plan,
                stats=self.stats,
                generator=plan_generator(self.plan, index, self.seed),
                mask_vectors=self.mask_vectors,
            )
        item: Batch = {
            "index": index,
            "id": self.split.ids[index],
            "label": self.split.labels[index],
            **{m.value: v for m, v in features.items()},
        }
        return item

    def with_plan(self, plan: CorruptionPlan) -> MultimodalDataset:
        """A view of the same split under a different corruption plan."""
        return MultimodalDataset(
            self.split, self.stats, plan, seed=self.seed, mask_vectors=self.mask_vectors
        )


def collate(items: list[Batch]) -> Batch:
    """Stack a list of samples into a batch. Ids stay a list of strings."""
    out: Batch = {
        "index": torch.tensor([int(i["index"]) for i in items], dtype=torch.long),
        "id": [str(i["id"]) for i in items],
        "label": torch.stack([torch.as_tensor(i["label"]) for i in items]),
    }
    for modality in Modality.all():
        key = modality.value
        if key in items[0]:
            out[key] = torch.stack([i[key] for i in items])
    return out


@dataclass
class LoaderConfig:
    """DataLoader knobs, from ``configs/config.yaml``."""

    batch_size: int = 32
    eval_batch_size: int = 128
    num_workers: int = 0
    """Default 0: with features already in RAM, worker processes are pure overhead on
    Windows and would re-pickle the whole tensor cache per worker."""
    pin_memory: bool = False
    persistent_workers: bool = False
    drop_last: bool = False


class MultimodalDataModule(L.LightningDataModule):
    """Wraps a :class:`~wfb.types.DatasetBundle` for Lightning.

    Training always runs on clean data (modality dropout during training is a *model*
    concern, handled in :mod:`wfb.training.modality_dropout`); corruption belongs to
    evaluation. Set ``eval_plan`` to sweep.
    """

    def __init__(
        self,
        data_cfg: DataConfig,
        loader_cfg: LoaderConfig | None = None,
        bundle: DatasetBundle | None = None,
        eval_plan: CorruptionPlan | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = loader_cfg or LoaderConfig()
        self.eval_plan = eval_plan or CorruptionPlan.clean()
        self.seed = seed
        self._bundle = bundle

    @property
    def bundle(self) -> DatasetBundle:
        """The loaded dataset, resolved on first access."""
        if self._bundle is None:
            self._bundle = load_dataset(self.data_cfg)
        return self._bundle

    def prepare_data(self) -> None:
        """Materialise the cache once, before any worker forks."""
        _ = self.bundle

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002 - Lightning signature
        """No-op: the bundle is already fully in memory."""
        _ = self.bundle

    def dataset(self, split: SplitName, plan: CorruptionPlan | None = None) -> MultimodalDataset:
        """Build a dataset for ``split`` under ``plan`` (clean by default)."""
        return MultimodalDataset(
            self.bundle[split],
            stats=self.bundle.stats,
            plan=plan if plan is not None else CorruptionPlan.clean(),
            seed=self.seed,
        )

    def _loader(
        self, split: SplitName, shuffle: bool, plan: CorruptionPlan | None = None
    ) -> DataLoader[Batch]:
        cfg = self.loader_cfg
        return DataLoader(
            self.dataset(split, plan),
            batch_size=cfg.batch_size if shuffle else cfg.eval_batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            drop_last=cfg.drop_last and shuffle,
            collate_fn=collate,
        )

    def train_dataloader(self) -> DataLoader[Batch]:
        """Clean, shuffled training data."""
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader[Batch]:
        """Validation data under ``eval_plan``."""
        return self._loader("val", shuffle=False, plan=self.eval_plan)

    def test_dataloader(self) -> DataLoader[Batch]:
        """Test data under ``eval_plan``."""
        return self._loader("test", shuffle=False, plan=self.eval_plan)

    def corrupted_loader(self, split: SplitName, plan: CorruptionPlan) -> DataLoader[Batch]:
        """An evaluation loader for an arbitrary plan — the sweep's workhorse."""
        return self._loader(split, shuffle=False, plan=plan)
