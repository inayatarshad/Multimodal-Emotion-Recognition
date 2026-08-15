"""Training entrypoint: build, fit, checkpoint, and report clean metrics.

Deliberately thin. Lightning owns the loop; this module owns reproducibility (seeding,
deterministic flags, the run directory layout) and the contract that a training run
produces a checkpoint plus a small JSON record — which is all the evaluation sweep and
the experiment orchestrator need.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from wfb.data.datamodule import LoaderConfig, MultimodalDataModule
from wfb.data.loaders import DataConfig, load_dataset
from wfb.evaluation.metrics import compute_metrics, primary_metric_for
from wfb.models import ModelConfig, build_model
from wfb.models.base import BaseFusionModel, DataSpec
from wfb.types import DatasetBundle

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters. From ``configs/config.yaml``."""

    max_epochs: int = 40
    patience: int = 8
    seed: int = 0
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32-true"
    gradient_clip_val: float = 1.0
    deterministic: bool = True
    monitor: str = "val_loss"
    monitor_mode: str = "min"
    output_dir: str = "outputs"
    log_every_n_steps: int = 10
    enable_progress_bar: bool = True
    save_checkpoint: bool = True
    limit_train_batches: float = 1.0
    """Fraction of training batches per epoch — CI runs use a small value."""


@dataclass
class TrainResult:
    """What a training run produces."""

    model: str
    dataset: str
    seed: int
    checkpoint: str | None
    clean_metrics: dict[str, float]
    val_metrics: dict[str, float]
    best_val_loss: float
    epochs_run: int
    parameters: int
    seconds: float
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)


def run_name(model: str, dataset: str, seed: int, tag: str = "") -> str:
    """Canonical run directory name — stable, sortable, and unique per configuration."""
    suffix = f"_{tag}" if tag else ""
    return f"{dataset}_{model}_s{seed}{suffix}"


def train(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    loader_cfg: LoaderConfig | None = None,
    bundle: DatasetBundle | None = None,
    tag: str = "",
) -> tuple[BaseFusionModel, TrainResult]:
    """Train one architecture on one dataset with one seed.

    Returns the fitted model (best checkpoint restored) and a :class:`TrainResult`.
    """
    started = time.perf_counter()
    L.seed_everything(train_cfg.seed, workers=True)
    if train_cfg.deterministic:
        torch.use_deterministic_algorithms(False)  # cuDNN RNN kernels have no det. path
        torch.backends.cudnn.benchmark = False

    bundle = bundle or load_dataset(data_cfg)
    datamodule = MultimodalDataModule(
        data_cfg, loader_cfg or LoaderConfig(), bundle=bundle, seed=train_cfg.seed
    )
    model = build_model(model_cfg, DataSpec.from_bundle(bundle))

    name = run_name(model_cfg.name, bundle.name, train_cfg.seed, tag)
    out_dir = Path(train_cfg.output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[Any] = [
        EarlyStopping(
            monitor=train_cfg.monitor,
            mode=train_cfg.monitor_mode,
            patience=train_cfg.patience,
            check_finite=True,
        )
    ]
    checkpoint_cb: ModelCheckpoint | None = None
    if train_cfg.save_checkpoint:
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(out_dir),
            filename="best",
            monitor=train_cfg.monitor,
            mode=train_cfg.monitor_mode,
            save_top_k=1,
            save_weights_only=False,
            # Overwrite rather than writing best-v1.ckpt, best-v2.ckpt, ...
            # Lightning's version counter is the wrong default here: re-training a run
            # under a new config left the *stale* file owning the canonical name, so
            # anything loading "best.ckpt" silently got the previous architecture. That
            # is how a smoke checkpoint (hidden=24) ended up shadowing a dev one
            # (hidden=32) and failing to load into the serving registry.
            enable_version_counter=False,
        )
        callbacks.append(checkpoint_cb)

    trainer = L.Trainer(
        max_epochs=train_cfg.max_epochs,
        accelerator=train_cfg.accelerator,
        devices=train_cfg.devices,
        precision=train_cfg.precision,  # type: ignore[arg-type]
        gradient_clip_val=train_cfg.gradient_clip_val,
        callbacks=callbacks,
        default_root_dir=str(out_dir),
        log_every_n_steps=train_cfg.log_every_n_steps,
        enable_progress_bar=train_cfg.enable_progress_bar,
        enable_model_summary=False,
        limit_train_batches=train_cfg.limit_train_batches,
        logger=False,
    )
    trainer.fit(model, datamodule=datamodule)

    checkpoint_path: str | None = None
    if checkpoint_cb is not None and checkpoint_cb.best_model_path:
        checkpoint_path = checkpoint_cb.best_model_path
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])

    val_metrics = evaluate_clean(model, datamodule, split="val")
    test_metrics = evaluate_clean(model, datamodule, split="test")

    best_val = float(trainer.callback_metrics.get("val_loss", torch.tensor(float("nan"))))
    result = TrainResult(
        model=model_cfg.name,
        dataset=bundle.name,
        seed=train_cfg.seed,
        checkpoint=checkpoint_path,
        clean_metrics=test_metrics,
        val_metrics=val_metrics,
        best_val_loss=best_val,
        epochs_run=int(trainer.current_epoch),
        parameters=model.num_parameters,
        seconds=time.perf_counter() - started,
        config={
            "model": dict(model_cfg.__dict__),
            "train": dict(train_cfg.__dict__),
            "data": {"name": data_cfg.name, "provenance": bundle.provenance.to_dict()},
        },
    )
    (out_dir / "train_result.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "%s: %s=%.4f (%d epochs, %.1fs)",
        name,
        primary_metric_for(bundle.task),
        test_metrics.get(primary_metric_for(bundle.task), float("nan")),
        result.epochs_run,
        result.seconds,
    )
    return model, result


@torch.no_grad()
def evaluate_clean(
    model: BaseFusionModel, datamodule: MultimodalDataModule, split: str = "test"
) -> dict[str, float]:
    """Clean-data metrics on ``split`` — the reproduction gate's numbers."""
    from wfb.evaluation.runner import predict_under_plan
    from wfb.types import CorruptionPlan

    predictions, labels = predict_under_plan(
        model,
        datamodule,
        CorruptionPlan.clean(),
        split=split,  # type: ignore[arg-type]
    )
    bundle = datamodule.bundle
    return compute_metrics(predictions, labels, task=bundle.task, label_range=bundle.label_range)


def load_checkpoint(path: str | Path, model_cfg: ModelConfig, spec: DataSpec) -> BaseFusionModel:
    """Rebuild an architecture and restore weights from a Lightning checkpoint."""
    model = build_model(model_cfg, spec)
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("state_dict", state))
    model.eval()
    return model
