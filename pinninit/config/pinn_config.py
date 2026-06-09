from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..io.load_parameters_from_yaml import load_parameters_from_yaml

if TYPE_CHECKING:
    from .config import (
        ConstConfig,
        CoordinatesConfig,
        CoordSystemConfig,
        DatasetConfig,
        EquationConfig,
        FieldsConfig,
        LossTermConfig,
        NNConfig,
        TrainerConfig,
    )


@dataclass(frozen=True, slots=True)
class PINNConfig:
    coord_systems: list[CoordSystemConfig]
    coordinates: list[CoordinatesConfig]
    datasets: list[DatasetConfig]
    fields: FieldsConfig
    constants: list[ConstConfig]
    equations: list[EquationConfig]
    loss: list[LossTermConfig]
    nn: NNConfig
    trainer: TrainerConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> PINNConfig:
        raw = load_parameters_from_yaml(path)

        return cls(
            coord_systems=raw["coord_systems"],
            coordinates=raw["coordinates"],
            datasets=raw["datasets"],
            fields=raw["fields"],
            constants=raw["constants"],
            equations=raw["equations"],
            loss=raw["loss"],
            nn=raw["nn"],
            trainer=raw["trainer"],
        )
