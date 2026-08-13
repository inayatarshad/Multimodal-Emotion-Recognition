"""``wfb-eval`` / ``make eval`` — run the corruption sweep against trained checkpoints.

Examples::

    uv run wfb-eval model=mult data=mosi seed=0
    uv run wfb-eval model=mult eval.severities='[0,0.5,1.0]'
    uv run wfb-eval model=late +train_if_missing=true
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from wfb.cli.config import (
    axes_from_config,
    to_data_config,
    to_loader_config,
    to_model_config,
    to_train_config,
)
from wfb.corruption.registry import describe_plan
from wfb.data.datamodule import MultimodalDataModule
from wfb.data.loaders import load_dataset
from wfb.evaluation.runner import run_sweep
from wfb.models.base import DataSpec
from wfb.training.trainer import load_checkpoint, run_name, train

logger = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main_hydra(cfg: DictConfig) -> float:
    """Sweep one checkpoint over the corruption grid. Returns its mean AUDC."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    data_cfg = to_data_config(cfg)
    model_cfg = to_model_config(cfg)
    train_cfg = to_train_config(cfg)
    loader_cfg = to_loader_config(cfg)
    tag = str(cfg.get("tag", ""))

    bundle = load_dataset(data_cfg)
    print(bundle.describe())

    name = run_name(model_cfg.name, bundle.name, train_cfg.seed, tag)
    checkpoint = Path(train_cfg.output_dir) / name / "best.ckpt"

    if checkpoint.exists():
        model = load_checkpoint(checkpoint, model_cfg, DataSpec.from_bundle(bundle))
        print(f"loaded {checkpoint}")
    elif bool(cfg.get("train_if_missing", False)):
        print(f"no checkpoint at {checkpoint} — training first")
        model, _ = train(model_cfg, data_cfg, train_cfg, loader_cfg, bundle=bundle, tag=tag)
    else:
        raise SystemExit(
            f"No checkpoint at {checkpoint}. Train it first with:\n"
            f"  uv run wfb-train model={model_cfg.name} data={bundle.name} "
            f"seed={train_cfg.seed}\n"
            "or re-run this command with +train_if_missing=true"
        )

    axes = axes_from_config(cfg)
    print(f"sweeping {len(axes)} axes")
    result = run_sweep(
        model,
        MultimodalDataModule(data_cfg, loader_cfg, bundle=bundle, seed=train_cfg.seed),
        axes,
        split=str(cfg.eval.get("split", "test")),  # type: ignore[arg-type]
        metric=cfg.eval.get("metric"),
        progress=True,
    )

    results_dir = Path(str(cfg.eval.get("results_dir", "experiments/results")))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{name}_sweep.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    print(f"\n{result.model} on {result.dataset} (seed {result.seed}) — metric {result.metric}")
    print(f"  clean {result.metric:12s} {result.clean_score:.4f}")
    print(f"  mean AUDC              {result.mean_audc:.4f}")
    for modality, score in sorted(result.mrs.items(), key=lambda kv: -kv[1]):
        print(f"  MRS[{modality:6s}]          {score:+.4f}")
    print("\n  worst axes by AUDC:")
    ranked = sorted((a for a in result.axes.values() if a.group == "graded"), key=lambda a: a.audc)
    for axis in ranked[:5]:
        critical = "never" if axis.critical is None else f"{axis.critical:.2f}"
        print(f"    {axis.axis:28s} AUDC={axis.audc:.3f}  crit@0.9={critical}")
    print(f"\nwrote {out_path}")
    print(f"example plan: {describe_plan(axes[0].plans[-1])}")
    return float(result.mean_audc)


def main() -> int:
    """Console-script wrapper."""
    main_hydra()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
