from __future__ import annotations

from enum import Enum


class DataSetSourceLabels(str, Enum):
    """How collocation points and field values are sourced."""

    FILE = "file"
    # e.g. in future ANALYTICAL = "analytical", SAMPLED = "sampled"
