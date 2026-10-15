from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import torch

from ..data_structure.coords_dataclass import make_coords_dataclass
from ..data_structure.field_dataclass import make_field_dataclass

if TYPE_CHECKING:
    import numpy as np

    from ..data_structure.coords_dataclass import Coords
    from ..data_structure.field_dataclass import Field
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
                            field_label,
                            data=field.get_component("value", coords)
                            .detach()
                            .cpu()
                            .numpy(),
                        )
                    else:
                        field_grp = fields_grp.create_group(field_label)
                        for comp in field.component_labels:
                            field_grp.create_dataset(
                                comp,
                                data=field.get_component(comp, coords)
                                .detach()
                                .cpu()
                                .numpy(),
                            )


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
        """Add one (dataset label, role) entry from numpy arrays.
        Currently supports scalar (rank-0) fields."""
        if role in self.datasets.get(label, {}):
            raise ValueError(f"{label}/{role} already exists")

        coords_obj = self.coords_class(
            **{
                k: torch.as_tensor(
                    v, dtype=torch.get_default_dtype(), device=self.device
                )
                for k, v in coords.items()
            }
        )

        ScalarField = make_field_dataclass(0, type(coords_obj))
        field_objs = {
            f_label: ScalarField(
                label=f_label,
                coords=coords_obj,
                value=torch.as_tensor(
                    arr, dtype=torch.get_default_dtype(), device=self.device
                ),
            )
            for f_label, arr in (fields or {}).items()
        }

        self.datasets.setdefault(label, {})[role] = (coords_obj, field_objs)

    def save(self, file_path: str | Path) -> None:
        save_datasets_to_hdf5(file_path, self.datasets)
