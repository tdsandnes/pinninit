from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import yaml

from ..config.config import (
    ConstConfig,
    CoordConfig,
    CoordinatesConfig,
    CoordSystemConfig,
    DatasetConfig,
    DomainConfig,
    EquationConfig,
    FieldConfig,
    FieldsConfig,
    LossTermConfig,
    NNConfig,
    OptimizerConfig,
    TrainerConfig,
    TrainerModuleConfig,
)
from ..geometry.coordinate_system import CoordSystemLabels
from ..io.dataset_sampler import DataSetSamplerLabels
from ..io.dataset_source import DataSetSourceLabels
from ..loss.loss_function import LossFunctionLabels
from ..loss.loss_term import LossTermLabels
from ..nn.activation_function import ActivationFuncLabels
from ..nn.initialise_biases import BiasInitLabels
from ..nn.initialise_weights import WeightInitLabels
from ..nn.input_transformation import InputTransformationLabels
from ..nn.layers import LayerTypeLabels

E = TypeVar("E", bound=Enum)

# Required top-level YAML sections
_REQUIRED_SECTIONS = {
    "CoordSystems",
    "Coordinates",
    "Fields",
    "LossTerms",
    "NN",
    "Trainer",
}

# NN aliases
_LAYER_TYPE_ALIASES: dict[str, Callable[[int], list[str]]] = {
    "siren": lambda n: ["siren_first"] + ["siren"] * (n - 1),
    "all_linear": lambda n: ["linear"] * n,
}

_ACTIVATION_ALIASES: dict[str, Callable[[int], list[str]]] = {
    "siren": lambda n: ["sin"] * (n - 1) + ["none"],
    "all_tanh": lambda n: ["tanh"] * (n - 1) + ["none"],
    "all_silu": lambda n: ["silu"] * (n - 1) + ["none"],
}

_INIT_WEIGHTS_ALIASES: dict[str, Callable[[int], list[str]]] = {
    "siren": lambda n: ["siren_first"] + ["siren"] * (n - 1),
    "all_xavier": lambda n: ["xavier"] * n,
}

_INIT_BIASES_ALIASES: dict[str, Callable[[int], list[str]]] = {
    "all_zeros": lambda n: ["zeros"] * n,
}


def load_parameters_from_yaml(path: str | Path) -> dict[str, Any]:
    """
    Parse a YAML parameter file and return a dict of section config objects.

    Parameters
    ----------
    path:
        Path to the .yml / .yaml parameter file.

    Returns
    -------
    dict
        Keys: "coord_systems", "coordinates", "datasets", "fields",
        "constants", "loss", "nn", "trainer".

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If required sections are missing or any value is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")

    with path.open("r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Parameter file must be a YAML mapping, got {type(raw)}.")

    missing = _REQUIRED_SECTIONS - set(raw)
    if missing:
        raise ValueError(
            f"Parameter file is missing required sections: {sorted(missing)}."
        )

    return {
        "coord_systems": _parse_coord_systems(raw["CoordSystems"]),
        "coordinates": _parse_coordinates(raw["Coordinates"]),
        "datasets": _parse_datasets(raw.get("DataSets", [])),
        "fields": _parse_fields(raw["Fields"]),
        "constants": _parse_constants(raw.get("Constants", [])),
        "equations": _parse_equations(raw.get("Equations", [])),
        "loss": _parse_loss(raw["LossTerms"]),
        "nn": _parse_nn(raw["NN"]),
        "trainer": _parse_trainer(raw["Trainer"]),
    }


def _parse_enum(
    value: str,
    enum_type: type[E],
) -> E:
    value = value.lower()

    if value not in enum_type._value2member_map_:
        valid = [member.value for member in enum_type]
        raise ValueError(
            f"Unknown {enum_type.__name__} value '{value}'. Valid options: {valid}."
        )

    return enum_type(value)


# Section parsers
def _parse_coord_systems(raw: list[dict[str, Any]]) -> list[CoordSystemConfig]:
    return [_parse_coord_system(coord_system) for coord_system in raw]


def _parse_coord_system(raw: dict[str, Any]) -> CoordSystemConfig:
    coord_system_str: str = raw["coord_system"]
    coord_system = _parse_enum(
        coord_system_str,
        CoordSystemLabels,
    )
    axes = [
        CoordConfig(label=ax["label"], is_geometric=bool(ax["is_geometric"]))
        for ax in raw.get("axes", [])
    ]
    fixed_axes: dict[str, float] = {
        ax["label"]: float(ax["value"]) for ax in raw.get("fixed_axes", [])
    }
    return CoordSystemConfig(
        label=raw["label"],
        coord_system=coord_system,
        axes=axes,
        fixed_axes=fixed_axes,
    )


def _parse_coordinates(raw: list[dict[str, Any]]) -> list[CoordinatesConfig]:
    return [_parse_one_coord(coord) for coord in raw]


def _parse_one_coord(raw: dict[str, Any]) -> CoordinatesConfig:
    return CoordinatesConfig(
        label=raw["label"],
        coord_system_label=raw["coord_system_label"],
        domain=_parse_domain(raw["domain"]),
    )


def _parse_domain(raw: dict[str, Any]) -> DomainConfig:
    raw_bounds = raw.get("bounds", {})
    bounds: dict[str, tuple[float | None, float | None]] = {}
    periodic_axes: set[str] = set()

    for axis_label, entry in raw_bounds.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid bounds format for axis '{axis_label}'. "
                f"Expected dict with keys 'min', 'max', 'periodic', got {type(entry)}."
            )

        lb = entry.get("min")
        ub = entry.get("max")

        bounds[axis_label] = (
            float(lb) if lb is not None else None,
            float(ub) if ub is not None else None,
        )

        if entry.get("periodic", False):
            periodic_axes.add(axis_label)

    return DomainConfig(
        label=raw.get("label", ""),
        bounds=bounds,
        periodic_axes=frozenset(periodic_axes),
    )


def _parse_datasets(raw: list[dict[str, Any]]) -> list[DatasetConfig]:
    raw = raw or []
    datasets = []
    for dataset in raw:
        source_str: str = dataset.get("source", "file").lower()
        sampling_str: str = dataset.get("sampling", "none").lower()
        source = _parse_enum(
            source_str,
            DataSetSourceLabels,
        )
        sampling = _parse_enum(
            sampling_str,
            DataSetSamplerLabels,
        )
        datasets.append(
            DatasetConfig(
                label=dataset["label"],
                source=source,
                file_path=dataset.get("file_path"),
                sampling=sampling,
            )
        )
    return datasets


def _parse_fields(raw: dict[str, Any]) -> FieldsConfig:
    input_fields = [
        FieldConfig(label=field["label"], rank=int(field["rank"]))
        for field in raw.get("input_fields", [])
    ]
    output_fields = [
        FieldConfig(label=field["label"], rank=int(field["rank"]))
        for field in raw.get("output_fields", [])
    ]
    derivable_fields = [
        FieldConfig(label=field["label"], rank=int(field["rank"]))
        for field in raw.get("derivable_fields", [])
    ]
    return FieldsConfig(
        input_fields=input_fields,
        output_fields=output_fields,
        derivable_fields=derivable_fields,
    )


def _parse_constants(raw: list[dict[str, Any]] | None) -> list[ConstConfig]:
    raw = raw or []
    return [
        ConstConfig(
            label=const["label"],
            value=const["value"],
            learnable=bool(const.get("learnable", False)),
        )
        for const in raw
    ]


def _parse_equations(raw: list[dict[str, Any]] | None) -> list[EquationConfig]:
    raw = raw or []
    equations: list[EquationConfig] = []

    for equation in raw:
        output_labels: list[str] = equation.get("output_labels", [])
        output_ranks: list[int] = equation.get(
            "output_ranks",
            [0] * len(output_labels),
        )

        equations.append(
            EquationConfig(
                label=equation["label"],
                function_name=equation["function_name"],
                file_path=equation["file_path"],
                input_labels=equation.get("input_labels", []),
                output_labels=output_labels,
                output_ranks=output_ranks,
            )
        )

    return equations


def _parse_loss(raw: list[dict[str, Any]]) -> list[LossTermConfig]:
    terms: list[LossTermConfig] = []

    for term in raw:
        loss_type_str: str = term.get("loss_type", "").lower()
        loss_fn_str: str = term.get("loss_function", "").lower()

        loss_type = _parse_enum(loss_type_str, LossTermLabels)
        loss_function = _parse_enum(loss_fn_str, LossFunctionLabels)

        terms.append(
            LossTermConfig(
                label=term["label"],
                field_label=term["field_label"],
                coordinates_label=term["coordinates_label"],
                dataset_label=term["dataset_label"],
                loss_type=loss_type,
                loss_function=loss_function,
                weight_value=float(term.get("weight_value", 1.0)),
                weight_method_label=term["weight_method_label"],
                weight_group_label=term["weight_group_label"],
            )
        )

    return terms


def _expand_nn_alias(
    value: str | list[str],
    aliases: dict[str, Callable[[int], list[str]]],
    enum_type: type[E],
    num_layers: int,
) -> list[E]:
    if isinstance(value, list):
        if len(value) != num_layers:
            raise ValueError(f"Expected {num_layers} entries, got {len(value)}.")

        return [_parse_enum(v, enum_type) for v in value]

    if value in aliases:
        expanded = aliases[value](num_layers)
        return [_parse_enum(v, enum_type) for v in expanded]

    raise ValueError(
        f"Unknown value '{value}'. Valid aliases: {sorted(aliases)} "
        f"or provide a list of {num_layers} values."
    )


def _parse_nn(raw: dict[str, Any]) -> NNConfig:
    widths: list[int] = list(raw["hidden_layer_widths"])
    num_layers = len(widths) + 1
    input_transformation_str: str | None = raw.get("input_transformation")
    input_transformation = (
        _parse_enum(input_transformation_str, InputTransformationLabels)
        if input_transformation_str
        else None
    )

    return NNConfig(
        hidden_layer_widths=widths,
        layer_types=_expand_nn_alias(
            raw["layer_types"],
            _LAYER_TYPE_ALIASES,
            LayerTypeLabels,
            num_layers,
        ),
        activation_functions=_expand_nn_alias(
            raw["activation_functions"],
            _ACTIVATION_ALIASES,
            ActivationFuncLabels,
            num_layers,
        ),
        init_weights=_expand_nn_alias(
            raw["init_weights"],
            _INIT_WEIGHTS_ALIASES,
            WeightInitLabels,
            num_layers,
        ),
        init_biases=_expand_nn_alias(
            raw["init_biases"],
            _INIT_BIASES_ALIASES,
            BiasInitLabels,
            num_layers,
        ),
        input_transformation=input_transformation,
    )


def _parse_trainer(raw: dict[str, Any]) -> TrainerConfig:
    optimizer = _parse_optimizer(raw["Optimizer"])
    modules = [
        _parse_trainer_module(module) for module in raw.get("TrainerModules", [])
    ]
    return TrainerConfig(optimizer=optimizer, trainer_modules=modules)


def _parse_optimizer(raw: dict[str, Any]) -> OptimizerConfig:
    known_keys = {"optimizer", "learning_rate"}
    hyperparameters = {
        key: value for key, value in raw.items() if key not in known_keys
    }
    return OptimizerConfig(
        optimizer=raw["optimizer"],
        learning_rate=float(raw["learning_rate"]),
        hyperparameters=hyperparameters,
    )


def _parse_trainer_module(raw: dict[str, Any]) -> TrainerModuleConfig:
    _reserved = {"label", "method_name", "file_path", "class_name"}
    hyperparameters = {key: value for key, value in raw.items() if key not in _reserved}
    return TrainerModuleConfig(
        label=raw["label"],
        method_name=raw.get("method_name"),
        file_path=raw.get("file_path"),
        class_name=raw.get("class_name"),
        hyperparameters=hyperparameters,
    )
