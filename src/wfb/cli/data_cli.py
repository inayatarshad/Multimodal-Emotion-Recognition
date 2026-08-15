"""``wfb-data`` / ``make data`` — resolve a dataset, cache it, freeze its split."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import NamedTuple

from wfb.data.loaders import DataConfig, DataError, build_cache
from wfb.data.splits import SPLITS_DIR, load_manifest
from wfb.data.synthetic import SyntheticConfig
from wfb.types import TaskType


class _Preset(NamedTuple):
    """Task shape for a dataset the CLI knows how to build."""

    task: TaskType
    num_classes: int


_PRESETS: dict[str, _Preset] = {
    "mosi": _Preset("regression", 1),
    "mosei": _Preset("regression", 1),
    "meld": _Preset("classification", 7),
}


def build_parser() -> argparse.ArgumentParser:
    """CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="wfb-data",
        description=(
            "Build the aligned feature cache for a dataset and freeze its evaluation "
            "split. Falls back to the deterministic synthetic corpus when the real one "
            "is unavailable, unless --no-synthetic is given."
        ),
    )
    parser.add_argument("--dataset", default="mosi", choices=sorted(_PRESETS), help="Dataset")
    parser.add_argument("--root", default="data", help="Data root (default: data)")
    parser.add_argument("--seq-len", type=int, default=50, help="Aligned sequence length")
    parser.add_argument("--raw-file", default=None, help="Explicit archive filename in data/raw")
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Fail instead of falling back to synthetic features",
    )
    parser.add_argument("--synthetic", action="store_true", help="Force the synthetic corpus")
    parser.add_argument("--no-normalize", action="store_true", help="Skip z-scoring features")
    parser.add_argument(
        "--storage-dtype",
        default=None,
        choices=["float32", "float16"],
        help=(
            "Precision of the cached features. float16 halves the cache and the resident "
            "memory, which is what makes CMU-MOSEI fit on a 16 GB machine. Defaults to "
            "float16 for mosei and float32 otherwise."
        ),
    )
    parser.add_argument(
        "--refreeze",
        action="store_true",
        help="Overwrite the committed split manifest (invalidates all prior results)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    preset = _PRESETS[args.dataset]
    cfg = DataConfig(
        name=args.dataset,
        task=preset.task,
        num_classes=preset.num_classes,
        root=args.root,
        seq_len=args.seq_len,
        normalize=not args.no_normalize,
        storage_dtype=args.storage_dtype or ("float16" if args.dataset == "mosei" else "float32"),
        allow_synthetic=not args.no_synthetic,
        force_synthetic=args.synthetic,
        raw_filename=args.raw_file,
        synthetic=SyntheticConfig(name=args.dataset, seq_len=args.seq_len),
    )

    try:
        bundle = build_cache(cfg, freeze_splits=True)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(bundle.describe())
    print(f"cache:  {cfg.cache_path}")
    manifest = load_manifest(bundle.name)
    if manifest is not None:
        print(f"splits: {Path(SPLITS_DIR) / (bundle.name + '.json')}  {manifest.sizes}")
    if bundle.is_synthetic:
        print(
            "\nNOTE: these are SYNTHETIC features. Every result derived from them records "
            "provenance=synthetic. See docs/DATA.md for how to obtain the real corpora."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
