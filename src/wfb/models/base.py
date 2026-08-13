"""The LightningModule interface every architecture implements.

A subclass supplies exactly one thing: how encoded modalities become a prediction.
Optimisation, logging, modality dropout, mask tokens and the train/val/test steps live
here, so no architecture can accidentally differ from another on anything except fusion.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import lightning as L
import torch
from torch import Tensor, nn

from wfb.types import DatasetBundle, Modality, ModalityDict, TaskType


@dataclass
class DataSpec:
    """The shape/task contract a model is built against."""

    dims: dict[Modality, int]
    seq_len: int
    task: TaskType = "regression"
    num_classes: int = 1
    label_range: tuple[float, float] = (-3.0, 3.0)

    @classmethod
    def from_bundle(cls, bundle: DatasetBundle) -> DataSpec:
        """Derive the spec from a loaded dataset."""
        return cls(
            dims=bundle.dims,
            seq_len=bundle.seq_len,
            task=bundle.task,
            num_classes=bundle.num_classes,
            label_range=bundle.label_range,
        )

    @property
    def output_dim(self) -> int:
        """Head width: 1 for regression, ``num_classes`` for classification."""
        return 1 if self.task == "regression" else self.num_classes


@dataclass
class ModelConfig:
    """Every architecture hyperparameter. Populated from ``configs/model/*.yaml``."""

    name: str = "late"
    encoder: str = "lstm"
    hidden: int = 64
    fusion_hidden: int = 128
    layers: int = 1
    heads: int = 4
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "none"
    warmup_epochs: int = 0
    modalities: tuple[str, ...] = ("text", "audio", "visual")
    """Which modalities this model consumes — unimodal baselines restrict this."""
    modality_dropout: float = 0.0
    """Probability of dropping each modality during training (the Q3 mitigation arm)."""
    modality_dropout_mode: str = "zero"
    """``zero`` or ``mask`` — with ``mask``, dropped modalities are replaced by a learned
    token, which is what makes the ``mask`` removal variant meaningful at eval time."""
    rank: int = 4
    """LMF decomposition rank."""
    tensor_dim: int = 24
    """Per-modality subspace width before TFN's outer product. The tensor is
    ``(tensor_dim + 1)^3``, so this dominates TFN's parameter count — 24 keeps it near
    LMF's scale, which is the fair comparison."""
    post_fusion_dim: int = 64
    """TFN/LMF post-fusion MLP width."""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def active_modalities(self) -> tuple[Modality, ...]:
        """Parsed :class:`~wfb.types.Modality` tuple, in canonical order."""
        wanted = {Modality(m) for m in self.modalities}
        return tuple(m for m in Modality.all() if m in wanted)


@dataclass
class ModelOutput:
    """What a forward pass returns."""

    prediction: Tensor
    """``(B,)`` regression score or ``(B, C)`` class logits."""
    fused: Tensor | None = None
    per_modality: dict[Modality, Tensor] = field(default_factory=dict)
    """Per-modality decision scores, when the architecture produces them (late fusion)."""
    attention: dict[str, Tensor] = field(default_factory=dict)
    """Cross-modal attention maps, for the demo's explanation panel."""


class BaseFusionModel(L.LightningModule):
    """Shared training/evaluation machinery for all architectures."""

    def __init__(self, cfg: ModelConfig, spec: DataSpec) -> None:
        super().__init__()
        self.cfg = cfg
        self.spec = spec
        self.save_hyperparameters({"model": cfg.__dict__, "data": _spec_dict(spec)})
        self.active = cfg.active_modalities
        if not self.active:
            raise ValueError("A model must consume at least one modality")

        if cfg.modality_dropout_mode == "mask" or cfg.modality_dropout > 0:
            self.mask_tokens = nn.ParameterDict(
                {m.value: nn.Parameter(torch.zeros(spec.dims[m])) for m in self.active}
            )
        else:
            self.mask_tokens = nn.ParameterDict()

        self._build()

    # ------------------------------------------------------------------ interface

    @abstractmethod
    def _build(self) -> None:
        """Create submodules. Called at the end of ``__init__``."""

    @abstractmethod
    def forward(self, features: ModalityDict) -> ModelOutput:
        """Run the architecture on a dict of ``(B, T, D)`` tensors."""

    # ------------------------------------------------------------------ helpers

    def mask_vectors(self) -> dict[Modality, Tensor]:
        """Learned mask tokens, for the ``mask`` removal variant at eval time."""
        return {Modality(k): v.detach() for k, v in self.mask_tokens.items()}

    def select(self, batch: dict[str, Any]) -> ModalityDict:
        """Pull this model's active modalities out of a collated batch."""
        return {m: batch[m.value] for m in self.active}

    def apply_modality_dropout(self, features: ModalityDict) -> ModalityDict:
        """Randomly drop whole modalities during training (Q3's mitigation).

        Each modality is dropped independently with probability ``modality_dropout``,
        per sample rather than per batch — per-batch dropping gives a much noisier
        gradient signal for the same expected sparsity. At least one modality always
        survives, so the model is never asked to predict from nothing.
        """
        p = self.cfg.modality_dropout
        if p <= 0.0 or not self.training or len(features) < 2:
            return features

        batch_size = next(iter(features.values())).shape[0]
        device = next(iter(features.values())).device
        keys = list(features)
        keep = torch.rand(batch_size, len(keys), device=device) >= p

        # Repair all-dropped rows by reviving one modality at random.
        empty = ~keep.any(dim=1)
        if bool(empty.any()):
            revive = torch.randint(0, len(keys), (int(empty.sum()),), device=device)
            keep[empty, revive] = True

        out: ModalityDict = {}
        for i, modality in enumerate(keys):
            x = features[modality]
            gate = keep[:, i].view(-1, *([1] * (x.ndim - 1))).to(x.dtype)
            if self.cfg.modality_dropout_mode == "mask" and modality.value in self.mask_tokens:
                token = self.mask_tokens[modality.value].view(*([1] * (x.ndim - 1)), -1)
                out[modality] = x * gate + token * (1.0 - gate)
            else:
                out[modality] = x * gate
        return out

    def loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        """L1 for regression (the MOSI/MOSEI convention), cross-entropy otherwise."""
        if self.spec.task == "regression":
            return nn.functional.l1_loss(prediction.reshape(-1), target.reshape(-1).float())
        return nn.functional.cross_entropy(prediction, target.reshape(-1).long())

    # ------------------------------------------------------------------ lightning

    def _step(self, batch: dict[str, Any], stage: str) -> Tensor:
        features = self.select(batch)
        if stage == "train":
            features = self.apply_modality_dropout(features)
        out = self(features)
        loss = self.loss(out.prediction, batch["label"])
        batch_size = batch["label"].shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=stage != "train", batch_size=batch_size)
        if self.spec.task == "regression":
            mae = (out.prediction.reshape(-1) - batch["label"].reshape(-1).float()).abs().mean()
            self.log(f"{stage}_mae", mae, prog_bar=stage == "val", batch_size=batch_size)
        else:
            acc = (out.prediction.argmax(dim=-1) == batch["label"].reshape(-1)).float().mean()
            self.log(f"{stage}_acc", acc, prog_bar=stage == "val", batch_size=batch_size)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:  # noqa: ARG002
        """One optimisation step."""
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:  # noqa: ARG002
        """One validation step."""
        return self._step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:  # noqa: ARG002
        """One test step."""
        return self._step(batch, "test")

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int = 0,  # noqa: ARG002 - Lightning's hook signature
        dataloader_idx: int = 0,  # noqa: ARG002 - Lightning's hook signature
    ) -> dict[str, Any]:
        """Return predictions plus ids, for the evaluation sweep."""
        out = self(self.select(batch))
        return {
            "id": batch["id"],
            "index": batch["index"],
            "prediction": out.prediction.detach(),
            "label": batch["label"].detach(),
        }

    def configure_optimizers(self) -> Any:
        """AdamW, optionally with cosine decay."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        if self.cfg.scheduler == "cosine":
            max_epochs = int(getattr(self.trainer, "max_epochs", 0) or 50)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": cosine}
        if self.cfg.scheduler == "plateau":
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": plateau, "monitor": "val_loss"},
            }
        return optimizer

    # ------------------------------------------------------------------ misc

    @property
    def num_parameters(self) -> int:
        """Trainable parameter count, reported in the model registry and Pareto plot."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _spec_dict(spec: DataSpec) -> dict[str, Any]:
    return {
        "dims": {m.value: d for m, d in spec.dims.items()},
        "seq_len": spec.seq_len,
        "task": spec.task,
        "num_classes": spec.num_classes,
        "label_range": list(spec.label_range),
    }


def squeeze_regression(x: Tensor, task: TaskType) -> Tensor:
    """Collapse a ``(B, 1)`` head output to ``(B,)`` for regression tasks."""
    return x.squeeze(-1) if task == "regression" else x
