"""Shared fixtures. Everything is synthetic and tiny — the suite must run in CI in seconds."""

from __future__ import annotations

import pytest
import torch

from wfb.data.loaders import DataConfig, load_dataset
from wfb.data.synthetic import SyntheticConfig, make_synthetic_bundle
from wfb.types import DatasetBundle, Modality


@pytest.fixture(scope="session")
def tiny_config() -> SyntheticConfig:
    """A 60-sample corpus with short sequences and small feature dims."""
    return SyntheticConfig(
        name="tiny",
        n_train=60,
        n_val=20,
        n_test=40,
        seq_len=12,
        dims=(16, 5, 8),
        seed=1234,
    )


@pytest.fixture(scope="session")
def tiny_bundle(tiny_config: SyntheticConfig) -> DatasetBundle:
    """A tiny synthetic dataset, built once per session."""
    return make_synthetic_bundle(tiny_config)


@pytest.fixture(scope="session")
def mosi_synthetic() -> DatasetBundle:
    """MOSI-shaped synthetic data — real dims, so shape assertions are meaningful."""
    return load_dataset(
        DataConfig(name="mosi", force_synthetic=True, cache=False), verify_splits=False
    )


@pytest.fixture
def sample_features(tiny_bundle: DatasetBundle) -> dict[Modality, torch.Tensor]:
    """A four-sample batch of features, ``(4, T, D)`` per modality."""
    return {m: t[:4].clone() for m, t in tiny_bundle["test"].features.items()}


@pytest.fixture
def generator() -> torch.Generator:
    """A seeded generator, so every corruption test is deterministic."""
    return torch.Generator().manual_seed(20260813)
