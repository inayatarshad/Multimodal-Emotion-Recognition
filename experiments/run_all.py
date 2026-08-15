"""Orchestrate the full experiment grid: train once, evaluate exhaustively.

    python experiments/run_all.py --preset smoke        # minutes, synthetic, for CI
    python experiments/run_all.py --preset main         # the reported grid
    python experiments/run_all.py --preset mitigation   # the Q3 modality-dropout arm

Resumable by construction: a configuration whose sweep JSON already exists is skipped, so
an interrupted run continues where it stopped rather than restarting. Pass ``--force`` to
recompute anyway.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from wfb.corruption.sweeps import (
    DEFAULT_SEVERITIES,
    grid_signature,
    smoke_grid,
    standard_grid,
)
from wfb.data.datamodule import LoaderConfig, MultimodalDataModule
from wfb.data.loaders import DataConfig, load_dataset
from wfb.data.synthetic import SyntheticConfig
from wfb.evaluation.runner import run_sweep
from wfb.evaluation.significance import compare_all
from wfb.models import ModelConfig
from wfb.reporting.tables import full_report, headline_table, update_readme
from wfb.serving.results_store import ResultsStore
from wfb.training.trainer import TrainConfig, run_name, train

logger = logging.getLogger("run_all")

ALL_MODELS = ("text_only", "audio_only", "visual_only", "late", "early", "lmf", "tfn", "mult")


@dataclass
class Preset:
    """One experiment configuration."""

    name: str
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    datasets: tuple[str, ...]
    dropout_probabilities: tuple[float, ...] = (0.0,)
    max_epochs: int = 40
    severities: tuple[float, ...] = DEFAULT_SEVERITIES
    full_grid: bool = True
    synthetic: bool = False
    hidden: int = 64
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_runs(self) -> int:
        """Number of training runs this preset implies."""
        return (
            len(self.models)
            * len(self.seeds)
            * len(self.datasets)
            * len(self.dropout_probabilities)
        )


PRESETS: dict[str, Preset] = {
    "smoke": Preset(
        name="smoke",
        models=("late", "mult"),
        seeds=(0,),
        datasets=("mosi",),
        max_epochs=3,
        severities=(0.0, 0.5, 1.0),
        full_grid=False,
        synthetic=True,
        hidden=24,
    ),
    "dev": Preset(
        name="dev",
        models=("text_only", "late", "early", "lmf", "tfn", "mult"),
        seeds=(0, 1),
        datasets=("mosi",),
        max_epochs=20,
        synthetic=True,
        hidden=32,
    ),
    "main": Preset(
        name="main",
        models=ALL_MODELS,
        seeds=(0, 1, 2, 3, 4),
        datasets=("mosi",),
        max_epochs=40,
    ),
    "cross": Preset(
        name="cross",
        models=ALL_MODELS,
        seeds=(0, 1, 2, 3, 4),
        datasets=("mosi", "mosei", "meld"),
        max_epochs=40,
    ),
    "mitigation": Preset(
        name="mitigation",
        models=("late", "early", "lmf", "tfn", "mult"),
        seeds=(0, 1, 2, 3, 4),
        datasets=("mosi",),
        dropout_probabilities=(0.0, 0.1, 0.3, 0.5),
        max_epochs=40,
    ),
}


def model_config(name: str, preset: Preset, dropout: float) -> ModelConfig:
    """Architecture config for one cell of the grid."""
    kwargs: dict[str, Any] = {
        "name": name,
        "hidden": preset.hidden,
        "modality_dropout": dropout,
        "modality_dropout_mode": "mask" if dropout > 0 else "zero",
    }
    if name == "mult":
        kwargs.update(
            encoder="transformer",
            hidden=max(preset.hidden - preset.hidden % 5, 20),
            heads=5,
            layers=2,
            scheduler="plateau",
        )
    elif name == "early":
        kwargs.update(encoder="transformer", layers=2, dropout=0.2)
    elif name == "tfn":
        kwargs.update(tensor_dim=16, lr=5e-4, dropout=0.2)
    elif name == "lmf":
        kwargs.update(rank=4, lr=5e-4)
    return ModelConfig(**kwargs)


def data_config(dataset: str, preset: Preset) -> DataConfig:
    """Loader config for one dataset."""
    task = "classification" if dataset == "meld" else "regression"
    num_classes = 7 if dataset == "meld" else 1
    return DataConfig(
        name=dataset,
        task=task,  # type: ignore[arg-type]
        num_classes=num_classes,
        # Must match configs/data/*.yaml, because storage_dtype is part of the cache
        # filename. Getting it wrong silently misses the cache and re-reads the 4.4 GB
        # source archive, which on a 16 GB machine means thrashing rather than training.
        storage_dtype="float16" if dataset == "mosei" else "float32",
        force_synthetic=preset.synthetic,
        cache=not preset.synthetic,
        synthetic=SyntheticConfig(name=dataset, task=task, num_classes=num_classes),  # type: ignore[arg-type]
        class_names=(
            ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
            if dataset == "meld"
            else []
        ),
    )


def run_cell(
    model_name: str,
    dataset: str,
    seed: int,
    dropout: float,
    preset: Preset,
    results_dir: Path,
    output_dir: Path,
    force: bool,
) -> dict[str, Any] | None:
    """Train (if needed) and sweep one configuration. Returns a summary row."""
    tag = f"md{dropout:g}" if dropout > 0 else ""
    name = run_name(model_name, dataset, seed, tag)
    sweep_path = results_dir / f"{name}_sweep.json"

    axes = standard_grid(severities=preset.severities) if preset.full_grid else smoke_grid()
    signature = grid_signature(axes)

    # Resuming must check *comparability*, not merely existence. A cached run produced
    # under a different corruption grid is not another seed of this measurement, and
    # averaging the two into one seed band would be silently wrong — the ladders only
    # happened to differ in length here, which turned it into a loud crash instead.
    if sweep_path.exists() and not force:
        cached = json.loads(sweep_path.read_text(encoding="utf-8"))
        cached_signature = cached.get("grid_signature")
        if cached_signature == signature:
            logger.info("skip %s (already done)", name)
            return cached
        logger.warning(
            "recomputing %s: cached grid %s does not match this preset's grid %s",
            name,
            cached_signature or "<unrecorded>",
            signature,
        )

    started = time.perf_counter()
    data_cfg = data_config(dataset, preset)
    bundle = load_dataset(data_cfg, verify_splits=False)
    loader_cfg = LoaderConfig(batch_size=32, eval_batch_size=256)
    train_cfg = TrainConfig(
        max_epochs=preset.max_epochs,
        seed=seed,
        output_dir=str(output_dir),
        enable_progress_bar=False,
        patience=max(preset.max_epochs // 5, 3),
    )

    model, train_result = train(
        model_config(model_name, preset, dropout),
        data_cfg,
        train_cfg,
        loader_cfg,
        bundle=bundle,
        tag=tag,
    )

    sweep = run_sweep(
        model,
        MultimodalDataModule(data_cfg, loader_cfg, bundle=bundle, seed=seed),
        axes,
        split="test",
        keep_predictions_for=("clean", "remove."),
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    payload = sweep.to_dict()
    payload["train"] = train_result.to_dict()
    payload["grid_signature"] = signature
    payload["preset"] = preset.name
    payload["tag"] = tag  # recorded explicitly; never re-derived from the filename
    payload["wall_seconds"] = time.perf_counter() - started
    sweep_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    logger.info(
        "%s: clean=%.4f meanAUDC=%.4f (%.1fs)",
        name,
        sweep.clean_score,
        sweep.mean_audc,
        payload["wall_seconds"],
    )
    return payload


def significance_report(results_dir: Path, dataset: str, output: Path) -> dict[str, Any] | None:
    """Paired comparisons between architectures under identical corruption.

    Only meaningful because every architecture was evaluated on bit-identical corrupted
    inputs — see :func:`wfb.corruption.registry.plan_generator`.
    """
    import numpy as np

    predictions_dir = results_dir / "predictions"
    if not predictions_dir.exists():
        return None
    errors: dict[str, np.ndarray] = {}
    for file in sorted(predictions_dir.glob(f"{dataset}_*_clean.npy")):
        errors[file.stem.split("_")[1]] = np.load(file)
    if len(errors) < 2:
        return None
    report = {"dataset": dataset, "comparisons": compare_all(errors)}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """Entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="smoke", choices=sorted(PRESETS))
    parser.add_argument("--models", nargs="*", default=None, help="Override the model list")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override the preset's epoch budget (useful for a first-pass gate check)",
    )
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--force", action="store_true", help="Recompute finished cells")
    parser.add_argument("--no-report", action="store_true", help="Skip table/figure generation")
    parser.add_argument("--threads", type=int, default=0, help="torch intra-op threads")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    preset = PRESETS[args.preset]
    if args.models:
        preset = Preset(**{**preset.__dict__, "models": tuple(args.models)})
    if args.seeds:
        preset = Preset(**{**preset.__dict__, "seeds": tuple(args.seeds)})
    if args.datasets:
        preset = Preset(**{**preset.__dict__, "datasets": tuple(args.datasets)})
    if args.max_epochs:
        preset = Preset(**{**preset.__dict__, "max_epochs": args.max_epochs})

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    started = time.perf_counter()

    print(f"preset={preset.name}  runs={preset.total_runs}  models={list(preset.models)}")
    if preset.synthetic:
        print("NOTE: synthetic features — pipeline validation only, not reportable numbers.\n")

    completed = 0
    for dataset in preset.datasets:
        for model_name in preset.models:
            for dropout in preset.dropout_probabilities:
                for seed in preset.seeds:
                    try:
                        run_cell(
                            model_name,
                            dataset,
                            seed,
                            dropout,
                            preset,
                            results_dir,
                            output_dir,
                            args.force,
                        )
                        completed += 1
                    except Exception as exc:
                        logger.exception(
                            "FAILED %s/%s/seed%d/p%.1f: %s", dataset, model_name, seed, dropout, exc
                        )

    elapsed = time.perf_counter() - started
    print(f"\n{completed}/{preset.total_runs} runs completed in {elapsed / 60:.1f} min")

    if args.no_report:
        return 0

    store = ResultsStore.load(results_dir)
    if store.is_empty:
        print("No results to report.")
        return 0

    dataset = preset.datasets[0]
    report_path = results_dir / "REPORT.md"
    report_path.write_text(full_report(store, dataset), encoding="utf-8")
    print(f"\nwrote {report_path}")

    readme = Path("README.md")
    if readme.exists():
        try:
            if update_readme(readme, headline_table(store, dataset)):
                print("updated README.md headline table")
        except ValueError as exc:
            logger.warning("README not updated: %s", exc)

    try:
        from wfb.reporting.figures import generate_all

        for figure in generate_all(results_dir, "paper/figures", dataset):
            print(f"wrote {figure}")
    except ImportError:
        print("matplotlib not installed — skipping figures (uv sync --extra viz)")

    print("\n" + headline_table(store, dataset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
