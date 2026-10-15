from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import torch

from ..config.config import DatasetConfig
from ..data_structure.coords_dataclass import get_coords_class_default
from ..data_structure.field_dataclass import (
    generate_tensor_component_labels,
)

if TYPE_CHECKING:
    from ..data_structure.coords_dataclass import Coords


def load_coords_from_hdf5(
    dataset_config: DatasetConfig,
    role: str,
    coord_labels: tuple[str, ...],
    coords_class: type[Coords],
) -> Coords:
    """
    Load coordinate tensors from an HDF5 file and return a Coords instance.

    Expected HDF5 structure:
        file[dataset_label][role]/Coords/<coord_label>   shape: (n_points,)

    Parameters
    ----------
    dataset_config:
        Config object describing the dataset (label, file path, etc.).
    role:
        Split identifier inside the HDF5 file, e.g. "train" or "val".
    coord_labels:
        All coordinate labels expected on the returned Coords object.
    coords_class:
        The Coords dataclass produced by make_coords_dataclass.

    Returns
    -------
    Coords
        A fully-initialised Coords instance with requires_grad=True on each tensor.
    """
    dataset_label = dataset_config.label
    file_path = dataset_config.file_path

    if file_path is None:
        raise ValueError(
            f"Dataset '{dataset_label}' has no file_path. Cannot load from HDF5."
        )

    with h5py.File(file_path, "r") as f:
        # Check dataset
        if dataset_label not in f:
            raise KeyError(
                f"Dataset label '{dataset_label}' not found in '{file_path}'."
            )
        if role not in f[dataset_label]:
            raise KeyError(
                f"'{role}' not found under dataset '{dataset_label}' in '{file_path}'."
            )

        role_group = f[dataset_label][role]

        if "Coords" not in role_group:
            raise KeyError(
                f"'Coords' missing under '{dataset_label}/{role}' in '{file_path}'."
            )

        coords_group = role_group["Coords"]
        coords_dict: dict[str, torch.Tensor] = {}
        for label in coord_labels:
            if label not in coords_group:
                raise KeyError(
                    f"'{label}' not found in '{dataset_label}/{role}/Coords' "
                    f"in '{file_path}'."
                )
            data = coords_group[label][()]
            device = get_coords_class_default(coords_class, "device")
            coords_dict[label] = torch.tensor(
                data,
                dtype=torch.get_default_dtype(),
                device=device,
                requires_grad=True,
            )

    return coords_class(**coords_dict)


def load_fields_from_hdf5(
    dataset_config: DatasetConfig,
    role: str,
    coords: Coords,
    field_label_to_rank: dict[str, int],
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Load field tensors from an HDF5 file and return a dict of component dicts.

    Expected HDF5 structure (matches generate_tensor_component_labels conventions):
        Rank 0:   file[dataset_label][role]/Fields/<field_label>            shape: (n,)
        Rank >=1: file[dataset_label][role]/Fields/<field_label>/<comp_label>      (n,)
                  where comp_label is e.g. "x", "x_x", "x_y", etc.

    Parameters
    ----------
    dataset_config:
        Config object describing the dataset.
    role:
        Role identifier, e.g. "train" or "val".
    coords:
        The Coords instance these fields are bound to.
    field_label_to_rank:
        Mapping of field label to tensor rank.

    Returns
    -------
    dict[str, dict[str, torch.Tensor]]
        Outer key: field label.
        Inner key: component label (e.g. "value", "x", "x_x").
        Value: tensor of shape (n_points,).
    """
    dataset_label = dataset_config.label
    file_path = dataset_config.file_path

    if file_path is None:
        raise ValueError(
            f"Dataset '{dataset_label}' has no file_path. Cannot load from HDF5."
        )

    fields: dict[str, dict[str, torch.Tensor]] = {}

    with h5py.File(file_path, "r") as f:
        # Check dataset
        if dataset_label not in f:
            raise KeyError(
                f"Dataset label '{dataset_label}' not found in '{file_path}'."
            )
        if role not in f[dataset_label]:
            raise KeyError(
                f"'{role}' not found under dataset '{dataset_label}' in '{file_path}'."
            )

        role_group = f[dataset_label][role]

        if "Fields" not in role_group:
            return {}

        fields_group = role_group["Fields"]

        for field_label, rank in field_label_to_rank.items():
            if field_label not in fields_group:
                raise KeyError(
                    f"'{field_label}' not found in '{dataset_label}/{role}/Fields' "
                    f"in '{file_path}'."
                )

            item = fields_group[field_label]
            expected_components = generate_tensor_component_labels(
                coords.geometric_labels, rank
            )

            if rank == 0:
                if not isinstance(item, h5py.Dataset):
                    raise TypeError(
                        f"Rank-0 field '{field_label}' should be an HDF5 Dataset, "
                        f"got {type(item).__name__}."
                    )
                fields[field_label] = {
                    "value": torch.tensor(
                        item[()], dtype=torch.get_default_dtype(), device=coords.device
                    )
                }

            else:
                if not isinstance(item, h5py.Group):
                    raise TypeError(
                        f"Rank-{rank} field '{field_label}' should be an HDF5 Group "
                        f"(one dataset per component), got {type(item).__name__}."
                    )

                missing = set(expected_components) - set(item.keys())
                if missing:
                    raise KeyError(
                        f"Field '{field_label}' is missing components {missing} "
                        f"in '{dataset_label}/{role}/Fields'."
                    )

                fields[field_label] = {
                    comp: torch.tensor(
                        item[comp][()],
                        dtype=torch.get_default_dtype(),
                        device=coords.device,
                    )
                    for comp in expected_components
                }

    return fields
