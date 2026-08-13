"""``wfb-train`` / ``make train`` — Hydra entrypoint for a single training run.

Examples::

    uv run wfb-train model=mult data=mosi seed=0
    uv run wfb-train model=lmf model.modality_dropout=0.3 tag=md0.3
    uv run wfb-train -m model=late,early,tfn,lmf,mult seed=0,1,2,3,4
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from wfb.cli.config import (
    describe_config,
    to_data_config,
    to_loader_config,
    to_model_config,
    to_train_config,
)
from wfb.data.loaders import load_dataset
from wfb.evaluation.metrics import primary_metric_for
from wfb.training.trainer import run_name, train

logger = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main_hydra(cfg: DictConfig) -> float:
    """Train one model. Returns the primary clean metric, so Hydra sweeps can optimise it."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(describe_config(cfg))

    data_cfg = to_data_config(cfg)
    model_cfg = to_model_config(cfg)
    train_cfg = to_train_config(cfg)
    loader_cfg = to_loader_config(cfg)

    bundle = load_dataset(data_cfg)
    print(bundle.describe())

    _, result = train(
        model_cfg, data_cfg, train_cfg, loader_cfg, bundle=bundle, tag=str(cfg.get("tag", ""))
    )

    metric = primary_metric_for(bundle.task)
    out_dir = Path(train_cfg.output_dir) / run_name(
        model_cfg.name, bundle.name, train_cfg.seed, str(cfg.get("tag", ""))
    )
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    (out_dir / "clean_metrics.json").write_text(
        json.dumps(result.clean_metrics, indent=2), encoding="utf-8"
    )

    print(f"\n{model_cfg.name} on {bundle.name} (seed {train_cfg.seed})")
    for key, value in sorted(result.clean_metrics.items()):
        print(f"  {key:12s} {value:.4f}")
    print(f"  checkpoint  {result.checkpoint}")
    return float(result.clean_metrics.get(metric, float("nan")))


def main() -> int:
    """Console-script wrapper."""
    main_hydra()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
