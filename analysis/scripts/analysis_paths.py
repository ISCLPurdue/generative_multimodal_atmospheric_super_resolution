"""Small path helpers shared by the standalone analysis scripts."""

from __future__ import annotations

import os
from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]


def required_path(variable: str) -> Path:
    """Return a path supplied through an environment variable."""
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(
            f"Set {variable} to the required local input before running this script."
        )
    return Path(value).expanduser()


def output_path(variable: str, default_name: str) -> Path:
    """Return an optional output override or a directory under analysis/generated."""
    value = os.environ.get(variable)
    if value:
        return Path(value).expanduser()
    return ANALYSIS_ROOT / "generated" / default_name
