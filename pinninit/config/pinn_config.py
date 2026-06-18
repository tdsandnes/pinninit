from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..io.load_parameters_from_yaml import load_parameters_from_yaml
from ..trainer_modules.trainer_module_registry import TrainerModuleLabels

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

    def __post_init__(self) -> None:
        self._validate_staggering()
        self._validate_adaptive_point_weights()

    def _validate_staggering(self) -> None:
        """Staggered loss terms never activate without a staggering module."""
        staggered = [
            loss_term.label for loss_term in self.loss if loss_term.staggering > 0
        ]
        if not staggered:
            return

        builtin_module_names = {
            module.method_name.lower()
            for module in self.trainer.trainer_modules
            if module.method_name is not None
        }
        if TrainerModuleLabels.LOSS_TERM_STAGGERING.value not in builtin_module_names:
            raise ValueError(
                f"Loss terms {staggered} have staggering > 0 but no "
                f"'{TrainerModuleLabels.LOSS_TERM_STAGGERING.value}' trainer module "
                "is configured; they would never be activated."
            )

    def _validate_adaptive_point_weights(self) -> None:
        """Loss terms with adaptive point weights need a module that updates them."""
        adaptive = [
            loss_term.label
            for loss_term in self.loss
            if loss_term.adaptive_point_weights
        ]
        if not adaptive:
            return

        builtin_module_names = {
            module.method_name.lower()
            for module in self.trainer.trainer_modules
            if module.method_name is not None
        }
        if (
            TrainerModuleLabels.RESIDUAL_BASED_ATTENTION_A24.value
            not in builtin_module_names
        ):
            raise ValueError(
                f"Loss terms {adaptive} set adaptive_point_weights but no "
                f"'{TrainerModuleLabels.RESIDUAL_BASED_ATTENTION_A24.value}' trainer "
                "module is configured. Their point weights would never update."
            )
