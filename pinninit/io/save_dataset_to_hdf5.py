from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import torch

from ..data_structure.coords import make_coords_dataclass
from ..data_structure.field import (
    generate_tensor_component_labels,
    make_field_dataclass,
)

if TYPE_CHECKING:
    import numpy as np

    from ..data_structure.coords import Coords
    from ..data_structure.field import Field
    from ..geometry.coordinate_system import CoordSystem


def save_datasets_to_hdf5(
    file_path: str | Path,
    datasets: dict[str, dict[str, tuple[Coords, dict[str, Field]]]],
) -> None:
    """
    Write datasets to HDF5 in the layout expected by load_coords_from_hdf5
    and load_fields_from_hdf5:

        file[dataset_label][role]/Coords/<coord_label>
        file[dataset_label][role]/Fields/<field_label>             (rank 0)
        file[dataset_label][role]/Fields/<field_label>/<component> (rank >= 1)

    with datasets of shape (n_points, size) for an array.

    Parameters
    ----------
    file_path:
        Output path. Parent directories are created if missing.
    datasets:
        datasets[dataset_label][role] = (coords, fields), where fields maps
        field label to a Field instance bound to those coords.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        for dataset_label, roles in datasets.items():
            label_grp = f.create_group(dataset_label)
            for role, (coords, fields) in roles.items():
                role_grp = label_grp.create_group(role)

                coords_grp = role_grp.create_group("Coords")
                for label in coords.coord_labels:
                    coords_grp.create_dataset(
                        label, data=coords.get_coord(label).detach().cpu().numpy()
                    )

                fields_grp = role_grp.create_group("Fields")
                for field_label, field in fields.items():
                    if field.rank == 0:
                        fields_grp.create_dataset(
                            field_label, data=_component_values(field, "value", coords)
                        )
                    else:
                        field_grp = fields_grp.create_group(field_label)
                        labels = generate_tensor_component_labels(
                            coords.geometric_labels, field.rank
                        )
                        for comp in labels:
                            field_grp.create_dataset(
                                comp, data=_component_values(field, comp, coords)
                            )


def _component_values(field: Field, comp: str, coords: Coords) -> np.ndarray:
    """A component of a field at coords, shape (n_points,), or, for an array,
    of all its elements, shape (n_points, size)."""
    if field.size is None:
        values = field.get_component(comp, coords)
    else:
        values = torch.stack(
            [field[i].get_component(comp, coords) for i in range(len(field))], dim=1
        )
    return values.detach().cpu().numpy()


class DataSetWriter:
    """Collects datasets as Coords/Field objects and saves them to HDF5 in
    the layout the pinninit loaders expect. Construction through the Coords
    and Field dataclasses validates shapes, lengths, and devices."""

    def __init__(
        self,
        coord_system: CoordSystem,
        coord_labels: set[str],
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.coords_class = make_coords_dataclass(
            coord_labels, coord_system, self.device
        )
        self.datasets: dict[str, dict[str, tuple[Coords, dict[str, Field]]]] = {}

    def add(
        self,
        label: str,
        role: str,
        coords: dict[str, np.ndarray],
        fields: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Add one (dataset label, role) entry from numpy arrays of scalar
        (rank 0) fields, or 2D arrays of shape (n_points, size) for arrays of
        scalar fields. The arrays are stored in float64, whatever torch's
        default dtype, and converted to the training dtype when loaded."""
        if role in self.datasets.get(label, {}):
            raise ValueError(f"{label}/{role} already exists")

        coords_obj = self.coords_class(
            **{
                k: torch.as_tensor(v, dtype=torch.float64, device=self.device)
                for k, v in coords.items()
            }
        )

        field_objs = {}
        for f_label, arr in (fields or {}).items():
            values = torch.as_tensor(arr, dtype=torch.float64, device=self.device)
            size = values.shape[1] if values.ndim == 2 else None
            field = make_field_dataclass(0, type(coords_obj), size)(
                label=f_label, coords=coords_obj
            )
            field.update_from_tensor(values)
            field_objs[f_label] = field

        self.datasets.setdefault(label, {})[role] = (coords_obj, field_objs)

    def save(self, file_path: str | Path) -> None:
        save_datasets_to_hdf5(file_path, self.datasets)
