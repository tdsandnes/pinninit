from __future__ import annotations

import torch

from ..config.config import CoordinatesConfig, CoordSystemConfig
from ..data_structure.coords import Coords, make_coords_dataclass
from ..geometry.coordinate_system import CoordSystem
from ..geometry.domain import Domain


def build_coords_class(
    coord_system_config: CoordSystemConfig,
    coord_system: CoordSystem,
    device: torch.device,
) -> type[Coords]:
    """The Coords class of a coord system's axes."""
    return make_coords_dataclass(
        coord_labels=set(coord_system_config.all_axis_labels),
        coord_system=coord_system,
        device=device,
    )


def build_domain(
    coords_config: CoordinatesConfig,
    coord_system: CoordSystem,
) -> Domain:
    """The Domain of a set of coordinates."""
    domain_config = coords_config.domain
    geometric_bounds = dict(domain_config.bounds)
    periodic_coords = (
        set(domain_config.periodic_axes) if domain_config.periodic_axes else None
    )
    return Domain(
        coord_system=coord_system,
        geometric_bounds=geometric_bounds,
        periodic_coords=periodic_coords,
    )


def build_coords_classes(
    coords_configs: list[CoordinatesConfig],
    coord_system_configs: list[CoordSystemConfig],
    coord_systems: dict[str, CoordSystem],
    device: torch.device,
) -> dict[str, tuple[type[Coords], Domain]]:
    """The Coords class and Domain of each set of coordinates, by label."""
    coord_system_configs_by_label = {cfg.label: cfg for cfg in coord_system_configs}
    result: dict[str, tuple[type[Coords], Domain]] = {}
    for coords_config in coords_configs:
        cs_label = coords_config.coord_system_label
        coord_system = coord_systems[cs_label]
        cs_config = coord_system_configs_by_label[cs_label]
        coords_class = build_coords_class(cs_config, coord_system, device)
        domain = build_domain(coords_config, coord_system)
        result[coords_config.label] = (coords_class, domain)
    return result
