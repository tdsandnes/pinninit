from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils import check_against_data_type

if TYPE_CHECKING:
    from ..io.input_parameters import PINNParameters
    from .coordinate_system import CoordSystem


def extract_domain_bounds(
    coord_system: CoordSystem, pinn_params: PINNParameters
) -> tuple[list[float | None], list[float | None]]:
    """
    Validate and extract domain bounds for each axis in the coordinate system.

    Parameters
    ----------
    coord_system : CoordSystem
        The coordinate system object.
    pinn_params : PINNParameters
        PINN parameters.

    Returns
    -------
    tuple
        A tuple of two arrays of the lower and upper for each axis, ordered according to
         the coordinate system's axis labels. Each bound may be None if not specified.

    Raises
    ------
    ValueError
        If the number of domain entries do not match the coordinate system requirements.
    TypeError
        If domain parameters do not match the required type.
    """
    # Extract DataParams
    domain_list = pinn_params.geometry_params.domain

    # Sort input domain parameters by coord system metadata
    axes_labels = coord_system.metadata.axes_labels
    coord_dim = coord_system.coord_dim
    axes_labels = axes_labels[:coord_dim]
    axes_bounds = coord_system.metadata.axes_bounds

    # Validate the length of domain_list
    if len(domain_list) != coord_dim:
        raise ValueError(
            "The number of coordinates given to describe the domain in the parameter "
            f"file does not match the coordinate system dimensionality: {coord_dim} "
            f"(time + number of spatial dimensions)"
        )

    # Validate types of domain dict
    domain_coords = [
        check_against_data_type(item.get("coordinate"), str) for item in domain_list
    ]
    domain_lb = [
        check_against_data_type(item.get("min"), float) for item in domain_list
    ]
    domain_ub = [
        check_against_data_type(item.get("max"), float) for item in domain_list
    ]

    # Check that all axis labels are given in domain_labels
    if set(domain_coords) != set(axes_labels):
        missing = set(axes_labels) - set(domain_coords)
        extra = set(domain_coords) - set(axes_labels)
        msgs = []
        if missing:
            msgs.append(f"missing={sorted(missing)}")
        if extra:
            msgs.append(f"extra={sorted(extra)}")
        raise ValueError(
            "The coordinate labels given to describe the domain in the parameter file "
            f"do not match those required by the coordinate system: {', '.join(msgs)}"
        )

    # Contain domain within coordinate system bounds
    domain_lb_contained = []
    domain_ub_contained = []
    for i in range(coord_dim):
        coord = axes_labels[i]
        idx_domain = domain_coords.index(coord)
        lb = domain_lb[idx_domain]
        ub = domain_ub[idx_domain]

        coord_system_lb = axes_bounds[i][0]
        coord_system_ub = axes_bounds[i][1]

        if (coord_system_lb is not None) and (lb is None or lb < coord_system_lb):
            lb = coord_system_lb

        domain_lb_contained.append(lb)

        if (coord_system_ub is not None) and (ub is None or ub > coord_system_ub):
            ub = coord_system_ub

        domain_ub_contained.append(ub)

    return domain_lb_contained, domain_ub_contained
