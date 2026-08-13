"""Generated tables and figures. Everything here reads the committed results JSON."""

from wfb.reporting.tables import (
    axis_table,
    full_report,
    headline_table,
    mitigation_table,
    reliance_matrix,
    update_readme,
)

__all__ = [
    "axis_table",
    "full_report",
    "headline_table",
    "mitigation_table",
    "reliance_matrix",
    "update_readme",
]


def generate_all_figures(*args: object, **kwargs: object) -> object:
    """Lazy re-export so importing :mod:`wfb.reporting` does not require matplotlib."""
    from wfb.reporting.figures import generate_all

    return generate_all(*args, **kwargs)  # type: ignore[arg-type]
