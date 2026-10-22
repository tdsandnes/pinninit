from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from ..io.load_parameters_from_yaml import load_parameters_from_yaml
from ..loss.loss_term_container import WeightMethodLabels
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
    # The parameter file's contents, as read (saved in checkpoints)
    parameters: dict[str, Any]
    # Names the model, e.g. for scripts to check what they load
    model_name: str | None
    # Floating point precision, set as torch's default dtype
    dtype: torch.dtype
    random_seed: int | None
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
            parameters=raw["parameters"],
            model_name=raw["model_name"],
            dtype=raw["dtype"],
            random_seed=raw["random_seed"],
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
        self._validate_labels()
        self._validate_domain_bounds()
        self._validate_staggering()
        self._validate_adaptive_point_weights()
        self._validate_adaptive_loss_term_weights()

    @property
    def _module_names(self) -> set[str]:
        """The method names of the trainer modules."""
        return {module.method_name.lower() for module in self.trainer.trainer_modules}

    def _validate_labels(self) -> None:
        """The labels in each section are unique, and the coordinate systems,
        coordinates and datasets that other sections refer to exist."""
        for section, labels in (
            ("CoordSystems", [cs.label for cs in self.coord_systems]),
            ("Coordinates", [coords.label for coords in self.coordinates]),
            ("DataSets", [dataset.label for dataset in self.datasets]),
            ("Constants", [const.label for const in self.constants]),
            ("Equations", [equation.label for equation in self.equations]),
            ("LossTerms", [loss_term.label for loss_term in self.loss]),
        ):
            if duplicates := sorted(
                {label for label in labels if labels.count(label) > 1}
            ):
                raise ValueError(f"Duplicate labels {duplicates} in {section}.")

        coord_system_labels = [cs.label for cs in self.coord_systems]
        for coords in self.coordinates:
            if coords.coord_system_label not in coord_system_labels:
                raise ValueError(
                    f"Coordinates '{coords.label}' refer to an unknown coord system "
                    f"'{coords.coord_system_label}'. Known: {coord_system_labels}."
                )
        coordinates_labels = [coords.label for coords in self.coordinates]
        dataset_labels = [dataset.label for dataset in self.datasets]
        for loss_term in self.loss:
            if loss_term.coordinates_label not in coordinates_labels:
                raise ValueError(
                    f"Loss term '{loss_term.label}' refers to unknown coordinates "
                    f"'{loss_term.coordinates_label}'. Known: {coordinates_labels}."
                )
            if (
                loss_term.dataset_label is not None
                and loss_term.dataset_label not in dataset_labels
            ):
                raise ValueError(
                    f"Loss term '{loss_term.label}' refers to an unknown dataset "
                    f"'{loss_term.dataset_label}'. Known: {dataset_labels}."
                )

    def _validate_domain_bounds(self) -> None:
        """Domains only bound axes of their coord system, so that e.g. a
        misspelt axis isn't silently left unbounded."""
        coord_systems = {cs.label: cs for cs in self.coord_systems}
        for coords in self.coordinates:
            axes = coord_systems[coords.coord_system_label].all_axis_labels
            if unknown := sorted(set(coords.domain.bounds) - set(axes)):
                raise ValueError(
                    f"Coordinates '{coords.label}' have bounds for {unknown}, which "
                    f"aren't axes of coord system '{coords.coord_system_label}' "
                    f"({axes})."
                )

    def _validate_staggering(self) -> None:
        """Staggered loss terms never activate without a staggering module."""
        staggered = [
            loss_term.label for loss_term in self.loss if loss_term.staggering > 0
        ]
        if not staggered:
            return

        if TrainerModuleLabels.LOSS_TERM_STAGGERING.value not in self._module_names:
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

        point_weight_updaters = {
            TrainerModuleLabels.RESIDUAL_BASED_ATTENTION_A24.value,
            TrainerModuleLabels.BALANCED_RESIDUAL_DECAY_RATE_C25.value,
        }
        if not (self._module_names & point_weight_updaters):
            raise ValueError(
                f"Loss terms {adaptive} set adaptive_point_weights but no "
                f"point-weight updater module ({sorted(point_weight_updaters)}) "
                "is configured. Their point weights would never update."
            )

    def _validate_adaptive_loss_term_weights(self) -> None:
        """Loss terms with adaptive weights need the module that updates them."""
        modules = {
            WeightMethodLabels.ADAPTIVE_W21: TrainerModuleLabels.ADAPTIVE_WEIGHTS_W21,
            WeightMethodLabels.ADAPTIVE_W23: TrainerModuleLabels.ADAPTIVE_WEIGHTS_W23,
        }
        for weight_method, module in modules.items():
            adaptive = [
                loss_term.label
                for loss_term in self.loss
                if loss_term.weight_method_label == weight_method
            ]
            if adaptive and module.value not in self._module_names:
                raise ValueError(
                    f"Loss terms {adaptive} have weight_method_label "
                    f"'{weight_method.value}' but no '{module.value}' trainer module "
                    "is configured. Their weights would never update."
                )
