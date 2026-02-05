from __future__ import annotations

import h5py
import numpy as np
import torch


def load_coords(
    file_path: str,
    dataset_label: str,
    dataset_role: str,
    expected_coord_labels: list[str],
    device: torch.device,
    sample_mask: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    coords_dict: dict[str, torch.Tensor] = {}

    with h5py.File(file_path, "r") as f:
        if dataset_label not in f:
            raise KeyError(
                f"Data set label '{dataset_label}' not found in file " f"{file_path}"
            )
        dataset_group = f[dataset_label]

        if dataset_role not in dataset_group:
            raise KeyError(
                f"Data set role '{dataset_role}' not found under dataset "
                f"'{dataset_label}'"
            )

        # Read Metadata
        metadata = dataset_group[dataset_role].get("Metadata")
        if metadata is None:
            raise KeyError(f"File {file_path} is missing 'Metadata' group")

        num_points = int(metadata["num_points"][()])
        dataset_type = metadata["dataset_type"][()]
        if isinstance(dataset_type, bytes):
            dataset_type = dataset_type.decode("utf-8")

        # Use all values if no mask is given
        if sample_mask is None:
            sample_mask = np.ones(num_points, dtype=bool)
        else:
            if len(sample_mask) != num_points:
                raise ValueError(
                    "sample_mask length does not match number of points in" " dataset"
                )

        # Load Coords
        coords_group = dataset_group[dataset_role].get("Coords")
        if coords_group is None:
            raise KeyError(f"File {file_path} is missing 'Coords' group")

        # Expected and loaded coord labels
        expected_coords = set(expected_coord_labels)
        found_coords = set(coords_group.keys())

        # Check for mismatch
        if found_coords != expected_coords:
            raise KeyError(
                f"Coordinate labels mismatch. "
                f"Expected: {sorted(expected_coords)} "
                f"Got: {sorted(found_coords)}"
            )

        for coord_label in expected_coord_labels:
            # coord array from file
            data = coords_group[coord_label][sample_mask]

            # Check shape
            if data.shape != (num_points,):
                raise ValueError(
                    f"Coord '{coord_label}' must have shape ({num_points},),"
                    f" got {data.shape}"
                )

            coords_dict[coord_label] = torch.tensor(
                data,
                device=device,
                requires_grad=True,
            )

    return coords_dict


def load_fields(
    file_path: str,
    dataset_label: str,
    dataset_role: str,
    expected_coord_labels: list[str],
    device: torch.device,
    input_fields_to_rank: dict[str, int],
    sample_mask: np.ndarray | None = None,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    fields_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}

    with h5py.File(file_path, "r") as f:
        if dataset_label not in f:
            raise KeyError(
                f"Data set label '{dataset_label}' not found in file " f"{file_path}"
            )
        dataset_group = f[dataset_label]

        if dataset_role not in dataset_group:
            raise KeyError(
                f"Data set role '{dataset_role}' not found under dataset "
                f"'{dataset_label}'"
            )

        # Read Metadata
        metadata = dataset_group[dataset_role].get("Metadata")
        if metadata is None:
            raise KeyError(f"File {file_path} is missing 'Metadata' group")

        num_points = int(metadata["num_points"][()])
        dataset_type = metadata["dataset_type"][()]
        if isinstance(dataset_type, bytes):
            dataset_type = dataset_type.decode("utf-8")

        # Use all values if no mask is given
        if sample_mask is None:
            sample_mask = np.ones(num_points, dtype=bool)
        else:
            if len(sample_mask) != num_points:
                raise ValueError(
                    "sample_mask length does not match number of points in" " dataset"
                )

        # Load Fields
        fields_group = dataset_group[dataset_role].get("Fields")
        if fields_group is None:
            raise KeyError(f"File {file_path} is missing 'Fields' group")

        for field_label, expected_rank in input_fields_to_rank.items():
            if field_label not in fields_group:
                raise KeyError(f"Field '{field_label}' not found in HDF5 dataset")

            item = fields_group[field_label]
            if expected_rank == 0:
                if not isinstance(item, h5py.Dataset):
                    raise TypeError(f"Field '{field_label}' expected scalar dataset")

                # field from file
                data = item[()]

                # Check shape
                if data.shape != (num_points,):
                    raise ValueError(
                        f"Scalar field '{field_label}' must have shape ({num_points},),"
                        f" got {data.shape}"
                    )

                fields_dict[field_label] = {
                    "value": torch.tensor(
                        data[sample_mask],
                        device=device,
                    )
                }
            else:
                # Expect group for rank >=1
                if not isinstance(item, h5py.Group):
                    raise TypeError(
                        f"Field '{field_label}' expected group for rank {expected_rank}"
                    )

                expected_components = set(expected_coord_labels)
                found_components = set(item.keys())

                # Ensure all expected components are present
                missing_components = expected_components - found_components
                if missing_components:
                    raise KeyError(
                        f"Field '{field_label}' is missing expected components: "
                        f"{missing_components}. Found components: {found_components}"
                    )
                component_dict: dict[str, np.ndarray] = {}
                for component in expected_coord_labels:
                    data = item[component][()]

                    if data.shape != (num_points,):
                        raise ValueError(
                            f"Field '{field_label}.{component}' must have shape "
                            f"({num_points},), got {data.shape}"
                        )

                    component_dict[component] = torch.tensor(
                        data[sample_mask],
                        device=device,
                    )

                fields_dict[field_label] = component_dict

    return fields_dict
