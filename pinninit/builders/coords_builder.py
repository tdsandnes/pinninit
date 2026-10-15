from __future__ import annotations

import torch

from ..config.config import CoordinatesConfig, CoordSystemConfig
from ..data_structure.coords_dataclass import Coords, make_coords_dataclass
from ..geometry.coordinate_system import CoordSystem
from ..geometry.domain import Domain


def build_coords_class(
    coord_system_config: CoordSystemConfig,
    coord_system: CoordSystem,
    device: torch.device,
) -> type[Coords]:
    """
    Build the Coords *class* (not an instance) for a given CoordinatesConfig.

    The class encodes the coordinate labels, coord system, and device. An
    instance is created later once the actual coordinate tensors are available
    (e.g. loaded from HDF5 or sampled).

    Parameters
    ----------
    coord_system_config:
        The matching CoordSystemConfig.
    coord_system:
        The already-built CoordSystem runtime object.
    device:
        PyTorch device.

    Returns
    -------
    type[Coords]
        A dynamically-created Coords dataclass.
    """
    return make_coords_dataclass(
        coord_labels=set(coord_system_config.all_axis_labels),
        coord_system=coord_system,
        device=device,
    )


def build_domain(
    coords_config: CoordinatesConfig,
    coord_system: CoordSystem,
    device: torch.device,
) -> Domain:
    """
    Build a Domain object from the domain section of a CoordinatesConfig.

    Parameters
    ----------
    coords_config:
        Config containing the DomainConfig to build from.
    coord_system:
        The already-built CoordSystem this domain belongs to.
    device:
        PyTorch device.

    Returns
    -------
    Domain
    """
    domain_config = coords_config.domain
    geometric_bounds = dict(domain_config.bounds)
    periodic_coords = (
        set(domain_config.periodic_axes) if domain_config.periodic_axes else None
    )
    return Domain(
        coord_system=coord_system,
        geometric_bounds=geometric_bounds,
        periodic_coords=periodic_coords,
        device=device,
    )


def build_coords_classes(
    coords_configs: list[CoordinatesConfig],
    coord_system_configs: list[CoordSystemConfig],
    coord_systems: dict[str, CoordSystem],
    device: torch.device,
) -> dict[str, tuple[type[Coords], Domain]]:
    """
    Build a label-keyed dict of Coords *classes* and domains for all CoordinatesConfigs.
    """
    coord_system_configs_by_label = {cfg.label: cfg for cfg in coord_system_configs}
    result: dict[str, tuple[type[Coords], Domain]] = {}
    for coords_config in coords_configs:
        if coords_config.label in result:
            raise ValueError(f"Duplicate Coordinates label: '{coords_config.label}'.")
        cs_label = coords_config.coord_system_label
        if cs_label not in coord_systems:
            raise ValueError(
                f"CoordinatesConfig '{coords_config.label}' references unknown "
                f"CoordSystem '{cs_label}'."
            )
        coord_system = coord_systems[cs_label]
        cs_config = coord_system_configs_by_label[cs_label]
        coords_class = build_coords_class(cs_config, coord_system, device)
        domain = build_domain(coords_config, coord_system, device)
        result[coords_config.label] = (coords_class, domain)
    return result
