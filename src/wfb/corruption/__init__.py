"""Composable corruption operators — the degradation protocol's implementation.

Importing this package registers every operator, so ``registry.available()`` is complete
after ``import wfb.corruption``.
"""

from wfb.corruption import audio, common, text, visual  # noqa: F401 - registration
from wfb.corruption.base import Corruption, CorruptionContext, Identity
from wfb.corruption.registry import (
    apply_plan,
    available,
    build,
    catalogue,
    describe_plan,
    get,
    plan_generator,
    register,
)
from wfb.corruption.sweeps import (
    DEFAULT_SEVERITIES,
    REMOVAL_VARIANTS,
    SweepAxis,
    graded_axis,
    misalignment_axis,
    removal_axis,
    removal_grid,
    smoke_grid,
    standard_grid,
    unique_plans,
)

__all__ = [
    "DEFAULT_SEVERITIES",
    "REMOVAL_VARIANTS",
    "Corruption",
    "CorruptionContext",
    "Identity",
    "SweepAxis",
    "apply_plan",
    "available",
    "build",
    "catalogue",
    "describe_plan",
    "get",
    "graded_axis",
    "misalignment_axis",
    "plan_generator",
    "register",
    "removal_axis",
    "removal_grid",
    "smoke_grid",
    "standard_grid",
    "unique_plans",
]
