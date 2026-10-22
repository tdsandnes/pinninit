from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml

from ..config.config import (
    ConstConfig,
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
    PointWeightModifierConfig,
    TrainerConfig,
    TrainerModuleConfig,
)
from ..geometry.coordinate_system import CoordSystemLabels
from ..io.dataset_source import DataSetSourceLabels
from ..loss.loss_function import LossFunctionLabels
from ..loss.loss_term import LossTermLabels
from ..loss.loss_term_container import WeightMethodLabels
from ..nn.activation_function import ActivationFuncLabels
from ..nn.initialise_biases import BiasInitLabels
from ..nn.initialise_weights import WeightInitLabels
from ..nn.input_transformation import InputTransformationLabels
from ..nn.layers import LayerTypeLabels
from ..nn.neural_network import NNTypeLabels
from ..optimizer.optimizer_registry import OptimizerLabels

E = TypeVar("E", bound=Enum)


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads numbers with an exponent but no dot or no
    exponent sign, such as 1e-3 or 1.0e3, as floats (YAML 1.1, which PyYAML
    follows, reads them as strings)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)

# Floating point precision the PINN is built and trained with
_DTYPES = {"float32": torch.float32, "float64": torch.float64}

# NN aliases
_LAYER_TYPE_ALIASES: dict[str, Callable[[int], list[str]]] = {
    "siren": lambda n: ["siren_first"] + ["siren"] * (n - 1),
    "all_linear": lambda n: ["linear"] * n,
}

_ACTIVATION_ALIASES: dict[str, Callable[[int], list[str]]] = {
    "siren": lambda n: ["sin"] * (n - 1) + ["none"],
    "all_tanh": lambda n: ["tanh"] * (n - 1) + ["none"],
    "all_silu": lambda n: ["silu"] * (n - 1) + ["none"],
    "all_sigmoid": lambda n: ["sigmoid"] * (n - 1) + ["none"],
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
        Keys: "parameters" (the file's contents, as read), "model_name",
        "dtype", "random_seed", "coord_systems", "coordinates", "datasets", "fields",
        "constants", "equations", "loss", "nn", "trainer".

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If a section or option is missing or unknown, or any value is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")

    with path.open("r") as f:
        raw: dict[str, Any] = yaml.load(f, Loader=_Loader)

    _check_keys(
        raw,
        "the parameter file",
        required=(
            "CoordSystems",
            "Coordinates",
            "Fields",
            "LossTerms",
            "NN",
            "Trainer",
        ),
        optional=(
            "model_name",
            "dtype",
            "random_seed",
            "DataSets",
            "Constants",
            "Equations",
        ),
    )

    return {
        "parameters": raw,
        "model_name": raw.get("model_name"),
        "dtype": _parse_dtype(raw.get("dtype", "float64")),
        "random_seed": _parse_random_seed(raw.get("random_seed", "random")),
        "coord_systems": _parse_coord_systems(raw["CoordSystems"]),
        "coordinates": _parse_coordinates(raw["Coordinates"]),
        "datasets": _parse_datasets(raw.get("DataSets") or []),
        "fields": _parse_fields(raw["Fields"]),
        "constants": _parse_constants(raw.get("Constants") or []),
        "equations": _parse_equations(raw.get("Equations") or []),
        "loss": _parse_loss(raw["LossTerms"]),
        "nn": _parse_nn(raw["NN"]),
        "trainer": _parse_trainer(raw["Trainer"]),
    }


def _as_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"'{key}' must be true or false, got {value!r}.")
    return value


def _as_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not (
        isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    ):
        raise ValueError(f"'{key}' must be an integer, got {value!r}.")
    return int(value)


def _require_keys(raw: Any, where: str, required: Iterable[str]) -> None:
    """Check that raw is a mapping with all the required keys."""
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping for {where}, got {raw!r}.")
    if missing := set(required) - set(raw):
        raise ValueError(f"Missing keys {sorted(missing)} in {where}.")


def _check_keys(
    raw: Any,
    where: str,
    required: Iterable[str] = (),
    optional: Iterable[str] = (),
) -> None:
    """Check that raw is a mapping with all the required keys and no keys
    other than the required and optional ones (e.g. a misspelt option)."""
    required = set(required)
    _require_keys(raw, where, required)
    valid = required | set(optional)
    if unknown := set(raw) - valid:
        raise ValueError(
            f"Unknown keys {sorted(unknown)} in {where}. Valid keys: {sorted(valid)}."
        )


def _parse_enum(
    value: str,
    enum_type: type[E],
) -> E:
    valid = [member.value for member in enum_type]
    if not isinstance(value, str) or value.lower() not in valid:
        raise ValueError(
            f"Unknown {enum_type.__name__} value {value!r}. Valid options: {valid}."
        )

    return enum_type(value.lower())


def _parse_optional_enum(
    raw: dict[str, Any],
    key: str,
    enum_type: type[E],
) -> E | None:
    value = raw.get(key)
    return _parse_enum(value, enum_type) if value else None


# Section parsers
def _parse_dtype(value: str) -> torch.dtype:
    if value not in _DTYPES:
        raise ValueError(f"dtype must be one of {list(_DTYPES)}, got {value!r}.")
    return _DTYPES[value]


def _parse_random_seed(value: int | str) -> int | None:
    if isinstance(value, str) and value.lower() == "random":
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"random_seed must be an int or 'random', got {value!r}.")


def _parse_coord_systems(raw: list[dict[str, Any]]) -> list[CoordSystemConfig]:
    return [_parse_coord_system(coord_system) for coord_system in raw]


def _parse_coord_system(raw: dict[str, Any]) -> CoordSystemConfig:
    _check_keys(
        raw,
        "a CoordSystems entry",
        required=("label", "coord_system"),
        optional=("geometric_axis_labels", "non_geometric_axis_labels", "fixed_axes"),
    )
    coord_system = _parse_enum(raw["coord_system"], CoordSystemLabels)
    raw_fixed_axes = raw.get("fixed_axes") or []
    for ax in raw_fixed_axes:
        _check_keys(ax, "a fixed_axes entry", required=("label", "value"))
    fixed_axes: dict[str, float] = {
        ax["label"]: float(ax["value"]) for ax in raw_fixed_axes
    }
    return CoordSystemConfig(
        label=raw["label"],
        coord_system=coord_system,
        geometric_axis_labels=raw.get("geometric_axis_labels") or [],
        non_geometric_axis_labels=raw.get("non_geometric_axis_labels") or [],
        fixed_axes=fixed_axes,
    )


def _parse_coordinates(raw: list[dict[str, Any]]) -> list[CoordinatesConfig]:
    return [_parse_one_coord(coord) for coord in raw]


def _parse_one_coord(raw: dict[str, Any]) -> CoordinatesConfig:
    _check_keys(
        raw,
        "a Coordinates entry",
        required=("label", "coord_system_label", "domain"),
    )
    return CoordinatesConfig(
        label=raw["label"],
        coord_system_label=raw["coord_system_label"],
        domain=_parse_domain(raw["domain"]),
    )


def _parse_domain(raw: dict[str, Any]) -> DomainConfig:
    _check_keys(raw, "a domain", optional=("bounds",))
    raw_bounds = raw.get("bounds") or {}
    bounds: dict[str, tuple[float | None, float | None]] = {}
    periodic_axes: set[str] = set()

    for axis_label, entry in raw_bounds.items():
        _check_keys(
            entry,
            f"the bounds of axis '{axis_label}'",
            optional=("min", "max", "periodic"),
        )

        lb = entry.get("min")
        ub = entry.get("max")

        bounds[axis_label] = (
            float(lb) if lb is not None else None,
            float(ub) if ub is not None else None,
        )

        if _as_bool(entry.get("periodic", False), "periodic"):
            periodic_axes.add(axis_label)

    return DomainConfig(
        bounds=bounds,
        periodic_axes=frozenset(periodic_axes),
    )


def _parse_datasets(raw: list[dict[str, Any]]) -> list[DatasetConfig]:
    datasets = []
    for dataset in raw:
        _check_keys(
            dataset,
            "a DataSets entry",
            required=("label",),
            optional=("source", "file_path"),
        )
        datasets.append(
            DatasetConfig(
                label=dataset["label"],
                source=_parse_enum(dataset.get("source", "file"), DataSetSourceLabels),
                file_path=dataset.get("file_path"),
            )
        )
    return datasets


def _parse_fields(raw: dict[str, Any]) -> FieldsConfig:
    _check_keys(
        raw,
        "Fields",
        optional=("input_fields", "output_fields", "derivable_fields"),
    )
    return FieldsConfig(
        input_fields=_parse_field_list(raw.get("input_fields") or [], "input_fields"),
        output_fields=_parse_field_list(
            raw.get("output_fields") or [], "output_fields"
        ),
        derivable_fields=_parse_field_list(
            raw.get("derivable_fields") or [], "derivable_fields"
        ),
    )


def _parse_field_list(raw: list[dict[str, Any]], where: str) -> list[FieldConfig]:
    for field in raw:
        _check_keys(field, f"a {where} entry", required=("label", "rank"))
    return [
        FieldConfig(label=field["label"], rank=_as_int(field["rank"], "rank"))
        for field in raw
    ]


def _parse_constants(raw: list[dict[str, Any]]) -> list[ConstConfig]:
    for const in raw:
        _check_keys(
            const,
            "a Constants entry",
            required=("label", "value"),
            optional=("learnable",),
        )
    return [
        ConstConfig(
            label=const["label"],
            value=float(const["value"]),
            learnable=_as_bool(const.get("learnable", False), "learnable"),
        )
        for const in raw
    ]


def _parse_equations(raw: list[dict[str, Any]]) -> list[EquationConfig]:
    equations: list[EquationConfig] = []

    for equation in raw:
        _check_keys(
            equation,
            "an Equations entry",
            required=("label", "function_name", "file_path"),
            optional=("input_labels", "output_labels", "output_ranks"),
        )
        output_labels: list[str] = equation.get("output_labels") or []
        output_ranks: list[int] = equation.get("output_ranks") or [0] * len(
            output_labels
        )

        equations.append(
            EquationConfig(
                label=equation["label"],
                function_name=equation["function_name"],
                file_path=equation["file_path"],
                input_labels=equation.get("input_labels") or [],
                output_labels=output_labels,
                output_ranks=output_ranks,
            )
        )

    return equations


def _parse_loss(raw: list[dict[str, Any]]) -> list[LossTermConfig]:
    terms: list[LossTermConfig] = []

    for term in raw:
        _check_keys(
            term,
            "a LossTerms entry",
            required=("label", "field_label", "coordinates_label", "loss_type"),
            optional=(
                "dataset_label",
                "loss_function",
                "weight_value",
                "weight_method_label",
                "weight_group_label",
                "staggering",
                "adaptive_point_weights",
                "point_weight_modifier",
            ),
        )
        loss_type = _parse_enum(term["loss_type"], LossTermLabels)
        loss_function = _parse_enum(
            term.get("loss_function", "weighted_mse"), LossFunctionLabels
        )

        weight_method_str: str | None = term.get("weight_method_label")
        weight_method_label = (
            _parse_enum(weight_method_str, WeightMethodLabels)
            if weight_method_str
            else WeightMethodLabels.CONSTANT
        )

        terms.append(
            LossTermConfig(
                label=term["label"],
                field_label=term["field_label"],
                coordinates_label=term["coordinates_label"],
                dataset_label=term.get("dataset_label"),
                loss_type=loss_type,
                loss_function=loss_function,
                weight_value=float(term.get("weight_value", 1.0)),
                weight_method_label=weight_method_label,
                weight_group_label=term.get("weight_group_label"),
                staggering=_as_int(term.get("staggering", 0), "staggering"),
                adaptive_point_weights=_as_bool(
                    term.get("adaptive_point_weights", False), "adaptive_point_weights"
                ),
                point_weight_modifier=_parse_point_weight_modifier(
                    term.get("point_weight_modifier")
                ),
            )
        )

    return terms


def _parse_point_weight_modifier(
    raw: dict[str, Any] | None,
) -> PointWeightModifierConfig | None:
    if raw is None:
        return None
    _check_keys(
        raw,
        "a point_weight_modifier",
        required=("file_path", "function_name"),
        optional=("input_labels",),
    )
    return PointWeightModifierConfig(
        file_path=raw["file_path"],
        function_name=raw["function_name"],
        input_labels=list(raw.get("input_labels") or []),
    )


def _expand_nn_alias(
    raw: dict[str, Any],
    key: str,
    aliases: dict[str, Callable[[int], list[str]]],
    enum_type: type[E],
    num_layers: int,
) -> list[E]:
    value = raw[key]
    if isinstance(value, list):
        if len(value) != num_layers:
            raise ValueError(
                f"{key} needs {num_layers} entries (one per layer), got {len(value)}."
            )

        return [_parse_enum(v, enum_type) for v in value]

    if value in aliases:
        expanded = aliases[value](num_layers)
        return [_parse_enum(v, enum_type) for v in expanded]

    raise ValueError(
        f"Unknown {key} value {value!r}. Valid aliases: {sorted(aliases)} "
        f"or provide a list of {num_layers} values."
    )


def _parse_nn(raw: dict[str, Any]) -> NNConfig:
    _check_keys(
        raw,
        "NN",
        required=(
            "type",
            "hidden_layer_widths",
            "layer_types",
            "activation_functions",
            "init_weights",
            "init_biases",
        ),
        optional=(
            "input_transformation",
            "weight_norm",
            "encoder_layer_type",
            "encoder_activation_function",
            "encoder_init_weights",
            "encoder_init_biases",
            "skip_connection",
        ),
    )
    nn_type = _parse_enum(raw["type"], NNTypeLabels)
    widths: list[int] = list(raw["hidden_layer_widths"])
    num_layers = len(widths) + 1
    input_transformation_str: str | None = raw.get("input_transformation")
    input_transformation = (
        _parse_enum(input_transformation_str, InputTransformationLabels)
        if input_transformation_str
        else None
    )

    return NNConfig(
        nn_type=nn_type,
        hidden_layer_widths=widths,
        layer_types=_expand_nn_alias(
            raw,
            "layer_types",
            _LAYER_TYPE_ALIASES,
            LayerTypeLabels,
            num_layers,
        ),
        activation_functions=_expand_nn_alias(
            raw,
            "activation_functions",
            _ACTIVATION_ALIASES,
            ActivationFuncLabels,
            num_layers,
        ),
        init_weights=_expand_nn_alias(
            raw,
            "init_weights",
            _INIT_WEIGHTS_ALIASES,
            WeightInitLabels,
            num_layers,
        ),
        init_biases=_expand_nn_alias(
            raw,
            "init_biases",
            _INIT_BIASES_ALIASES,
            BiasInitLabels,
            num_layers,
        ),
        input_transformation=input_transformation,
        weight_norm=_as_bool(raw.get("weight_norm", False), "weight_norm"),
        encoder_layer_type=_parse_optional_enum(
            raw, "encoder_layer_type", LayerTypeLabels
        ),
        encoder_activation_function=_parse_optional_enum(
            raw, "encoder_activation_function", ActivationFuncLabels
        ),
        encoder_init_weights=_parse_optional_enum(
            raw, "encoder_init_weights", WeightInitLabels
        ),
        encoder_init_biases=_parse_optional_enum(
            raw, "encoder_init_biases", BiasInitLabels
        ),
        skip_connection=_as_bool(raw.get("skip_connection", False), "skip_connection"),
    )


def _parse_trainer(raw: dict[str, Any]) -> TrainerConfig:
    _check_keys(
        raw,
        "Trainer",
        required=("Optimizer",),
        optional=(
            "TrainerModules",
            "max_epoch",
            "warmup_epochs",
            "print_freq",
            "n_batches",
            "shuffle",
            "log_point_weight_info",
            "point_weight_snapshot_freq",
            "checkpoint_path",
            "checkpoint_freq",
            "stop_on_sigterm",
        ),
    )
    optimizer = _parse_optimizer(raw["Optimizer"])
    modules = [
        _parse_trainer_module(module) for module in raw.get("TrainerModules") or []
    ]

    max_epoch = raw.get("max_epoch")

    return TrainerConfig(
        optimizer=optimizer,
        trainer_modules=modules,
        max_epoch=_as_int(max_epoch, "max_epoch") if max_epoch is not None else None,
        warmup_epochs=_as_int(raw.get("warmup_epochs", 0), "warmup_epochs"),
        print_freq=_as_int(raw.get("print_freq", 1), "print_freq"),
        n_batches=_as_int(raw.get("n_batches", 1), "n_batches"),
        shuffle=_as_bool(raw.get("shuffle", True), "shuffle"),
        log_point_weight_info=_as_bool(
            raw.get("log_point_weight_info", False), "log_point_weight_info"
        ),
        point_weight_snapshot_freq=_as_int(
            raw.get("point_weight_snapshot_freq", 0), "point_weight_snapshot_freq"
        ),
        checkpoint_path=raw.get("checkpoint_path"),
        checkpoint_freq=_as_int(raw.get("checkpoint_freq", 100), "checkpoint_freq"),
        stop_on_sigterm=_as_bool(raw.get("stop_on_sigterm", False), "stop_on_sigterm"),
    )


def _parse_optimizer(raw: dict[str, Any]) -> OptimizerConfig:
    # The other keys are passed to the optimizer, which rejects unknown ones
    _require_keys(raw, "Optimizer", ("optimizer", "learning_rate"))
    known_keys = {"optimizer", "learning_rate", "weight_decay_groups"}
    hyperparameters = {k: v for k, v in raw.items() if k not in known_keys}
    weight_decay_groups = {
        name: float(wd) for name, wd in (raw.get("weight_decay_groups") or {}).items()
    }
    return OptimizerConfig(
        optimizer=_parse_enum(raw["optimizer"], OptimizerLabels),
        learning_rate=float(raw["learning_rate"]),
        weight_decay_groups=weight_decay_groups,
        hyperparameters=hyperparameters,
    )


def _parse_trainer_module(raw: dict[str, Any]) -> TrainerModuleConfig:
    # The other keys are passed to the trainer module, which rejects unknown ones
    _require_keys(raw, "a TrainerModules entry", ("label", "method_name"))
    hyperparameters = {
        key: value for key, value in raw.items() if key not in {"label", "method_name"}
    }
    return TrainerModuleConfig(
        label=raw["label"],
        method_name=raw["method_name"],
        hyperparameters=hyperparameters,
    )
