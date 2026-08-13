"""Frozen evaluation splits.

The protocol requires fixed splits committed to the repo. This module writes a split
manifest (the ordered sample ids per split, plus a checksum) to
``src/wfb/data/splits/{dataset}.json`` and verifies any freshly loaded bundle against it.

If a manifest exists and a loaded bundle disagrees with it, that is an error, not a
warning: every number in the results tables is only comparable because the split is
identical across runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wfb.types import SPLIT_NAMES, DatasetBundle, SplitName

SPLITS_DIR = Path(__file__).parent / "splits"


class SplitMismatchError(RuntimeError):
    """Raised when a loaded dataset does not match its committed split manifest."""


def _checksum(ids: list[str]) -> str:
    joined = "\n".join(ids).encode()
    return hashlib.sha256(joined).hexdigest()[:16]


@dataclass(frozen=True)
class SplitManifest:
    """The committed record of which samples belong to which split."""

    dataset: str
    sizes: dict[str, int]
    checksums: dict[str, str]
    ids: dict[str, list[str]]

    @classmethod
    def from_bundle(cls, bundle: DatasetBundle, store_ids: bool = True) -> SplitManifest:
        """Derive a manifest from a loaded bundle."""
        ids: dict[str, list[str]] = {str(name): list(bundle[name].ids) for name in SPLIT_NAMES}
        return cls(
            dataset=bundle.name,
            sizes={name: len(v) for name, v in ids.items()},
            checksums={name: _checksum(v) for name, v in ids.items()},
            ids=ids if store_ids else {},
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "dataset": self.dataset,
            "sizes": self.sizes,
            "checksums": self.checksums,
            "ids": self.ids,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SplitManifest:
        """Inverse of :meth:`to_dict`."""
        return cls(
            dataset=str(raw["dataset"]),
            sizes={str(k): int(v) for k, v in raw["sizes"].items()},
            checksums={str(k): str(v) for k, v in raw["checksums"].items()},
            ids={str(k): [str(i) for i in v] for k, v in raw.get("ids", {}).items()},
        )


def manifest_path(dataset: str, splits_dir: Path | None = None) -> Path:
    """Path of the committed manifest for ``dataset``."""
    return (splits_dir or SPLITS_DIR) / f"{dataset}.json"


def load_manifest(dataset: str, splits_dir: Path | None = None) -> SplitManifest | None:
    """Read the committed manifest, or ``None`` if this dataset has never been frozen."""
    path = manifest_path(dataset, splits_dir)
    if not path.exists():
        return None
    return SplitManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def freeze(
    bundle: DatasetBundle,
    splits_dir: Path | None = None,
    store_ids: bool = True,
    overwrite: bool = False,
) -> Path:
    """Write the split manifest for ``bundle``.

    Args:
        bundle: The loaded dataset.
        splits_dir: Override the default committed location (tests use a tmp dir).
        store_ids: Persist the full id lists. Disable for very large corpora where the
            checksum alone is enough (MOSEI's manifest is ~2 MB with ids, which is fine).
        overwrite: Replace an existing manifest. Refuses by default — silently
            re-freezing a changed split would invalidate every previous result.

    Returns:
        The path written.
    """
    path = manifest_path(bundle.name, splits_dir)
    if path.exists() and not overwrite:
        existing = load_manifest(bundle.name, splits_dir)
        fresh = SplitManifest.from_bundle(bundle, store_ids)
        if existing is not None and existing.checksums == fresh.checksums:
            return path
        raise SplitMismatchError(
            f"A different split manifest already exists at {path}. Pass overwrite=True "
            "only if you intend to invalidate every previously recorded result."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = SplitManifest.from_bundle(bundle, store_ids)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path


def verify(bundle: DatasetBundle, splits_dir: Path | None = None, strict: bool = True) -> bool:
    """Check ``bundle`` against its committed manifest.

    Returns ``True`` when it matches (or when no manifest exists yet). Raises
    :class:`SplitMismatchError` on a mismatch when ``strict``.
    """
    manifest = load_manifest(bundle.name, splits_dir)
    if manifest is None:
        return True
    fresh = SplitManifest.from_bundle(bundle, store_ids=False)
    problems: list[str] = []
    for name in SPLIT_NAMES:
        expected = manifest.checksums.get(name)
        actual = fresh.checksums[name]
        if expected is not None and expected != actual:
            problems.append(
                f"  {name}: expected checksum {expected} (n={manifest.sizes.get(name)}), "
                f"got {actual} (n={fresh.sizes[name]})"
            )
    if not problems:
        return True
    if strict:
        raise SplitMismatchError(
            f"Dataset {bundle.name!r} does not match its frozen split:\n" + "\n".join(problems)
        )
    return False


def split_sizes(dataset: str, splits_dir: Path | None = None) -> dict[SplitName, int] | None:
    """Committed split sizes, for display in the README and CLI."""
    manifest = load_manifest(dataset, splits_dir)
    if manifest is None:
        return None
    return {name: manifest.sizes[name] for name in SPLIT_NAMES if name in manifest.sizes}
