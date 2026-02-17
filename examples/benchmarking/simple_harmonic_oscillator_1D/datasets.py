from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Type aliases
Coords = dict[str, np.ndarray]
FieldComponents = dict[str, np.ndarray]
Fields = dict[str, FieldComponents] | FieldComponents


### DataSets class for managing datasets and saving to HDF5
### This is temporary, since it should be possible to do this using the data structure
### and classes that exist within the code. But for now this is a simple implementation
### to get the example working while internal and external API is still changing.


class DataSets:
    def __init__(self) -> None:
        self.datasets: dict[str, dict[str, dict[str, Any]]] = {}

    def clear_datasets(self) -> None:
        self.datasets = {}

    def make_dataset(
        self,
        dataset_type: str,
        coords: Coords,
        fields: Fields | None = None,
    ) -> dict[str, Any]:
        if not coords:
            raise ValueError("coords must not be empty")

        lengths = {k: v.shape[0] for k, v in coords.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Coordinate length mismatch: {lengths}")

        num_points = next(iter(lengths.values()))

        dataset: dict[str, Any] = {
            "dataset_type": dataset_type,
            "num_points": num_points,
            "coords": coords,
        }

        if fields:
            dataset.update(fields)

        return dataset

    def add_dataset(
        self,
        label: str,
        dataset_type: str,
        coords: Coords,
        fields: Fields | None = None,
    ) -> None:
        dataset = self.make_dataset(dataset_type, coords, fields)

        if label not in self.datasets:
            self.datasets[label] = {}

        if dataset_type in self.datasets[label]:
            raise ValueError(f"{label}/{dataset_type} already exists")

        self.datasets[label][dataset_type] = dataset

    def add_train_val_pair(
        self,
        label: str,
        coords_train: Coords,
        coords_val: Coords,
        fields_train: Fields | None = None,
        fields_val: Fields | None = None,
    ) -> None:
        self.add_dataset(label, "training", coords_train, fields_train)
        self.add_dataset(label, "validation", coords_val, fields_val)

    def save(self, filename: str) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filename, "w") as f:
            for label, dataset_types in self.datasets.items():
                grp_label = f.create_group(label)

                for dataset_type, dataset in dataset_types.items():
                    grp_dataset = grp_label.create_group(dataset_type)

                    self._write_single_dataset(grp_dataset, dataset)

    def _write_single_dataset(
        self,
        grp: h5py.Group,
        dataset: dict[str, Any],
    ) -> None:
        # Metadata
        meta = grp.create_group("Metadata")
        meta.create_dataset("dataset_type", data=dataset["dataset_type"])
        meta.create_dataset("num_points", data=dataset["num_points"])

        # Coordinates
        coords_grp = grp.create_group("Coords")
        for name, arr in dataset["coords"].items():
            coords_grp.create_dataset(name, data=arr)

        # Fields
        fields_root = grp.create_group("Fields")

        for key, value in dataset.items():
            if key in ("dataset_type", "num_points", "coords"):
                continue

            if isinstance(value, dict):
                field_grp = fields_root.create_group(key)
                for comp, arr in value.items():
                    field_grp.create_dataset(comp, data=arr)
            else:
                fields_root.create_dataset(key, data=value)
