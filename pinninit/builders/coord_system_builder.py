from __future__ import annotations

import torch

from ..config.config import CoordSystemConfig
from ..geometry.coordinate_system import (
    COORD_SYSTEM_REGISTRY,
    CoordSystem,
)


def build_coord_system(
    config: CoordSystemConfig,
    device: torch.device,
) -> CoordSystem:
    """
    Build a CoordSystem object from a CoordSystemConfig.

    Parameters
    ----------
    config:
        Config describing the coordinate system label, axes, and fixed axes.
    device:
        PyTorch device to use.

    Returns
    -------
    CoordSystem
        An initialised CoordSystem object.
    """
    coord_system_cls = COORD_SYSTEM_REGISTRY[config.coord_system]

    return coord_system_cls(
        coord_axes=set(config.geometric_axis_labels),
        device=device,
        fixed_axes=config.fixed_axes,
    )


def build_coord_systems(
    configs: list[CoordSystemConfig],
    device: torch.device,
) -> dict[str, CoordSystem]:
    """
    Build a label-keyed dict of CoordSystem objects from a list of configs.
    """
    result: dict[str, CoordSystem] = {}
    for config in configs:
        if config.label in result:
            raise ValueError(f"Duplicate CoordSystem label: '{config.label}'.")
        result[config.label] = build_coord_system(config, device)
    return result
