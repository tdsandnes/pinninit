from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..io.dataset_sampler import DataSetSamplerLabels
from ..io.dataset_source import DataSetSourceLabels

if TYPE_CHECKING:
    from ..geometry.coordinate_system import CoordSystemLabels
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


# Geometry Config
@dataclass(frozen=True, slots=True)
class DomainConfig:
    """Coordinate domain."""

    label: str
    bounds: dict[str, tuple[float | None, float | None]]
    periodic_axes: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for axis, (lb, ub) in self.bounds.items():
            if lb is not None and ub is not None and lb > ub:
                raise ValueError(
                    f"Axis '{axis}' lower bound {lb} must be <= upper bound {ub}."
                )
        invalid_periodic = self.periodic_axes - set(self.bounds)
        if invalid_periodic:
            raise ValueError(
                f"Periodic axes {invalid_periodic} are not in bounds {self.bounds}."
            )


@dataclass(frozen=True, slots=True)
class CoordSystemConfig:
    """Coordinate system and axes. Shared across all domains in a problem."""

    label: str
    coord_system: CoordSystemLabels
    geometric_axis_labels: list[str]
    non_geometric_axis_labels: list[str] = field(default_factory=list)
    fixed_axes: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        all_labels = self.geometric_axis_labels + self.non_geometric_axis_labels
        if len(all_labels) != len(set(all_labels)):
            raise ValueError(f"Duplicate axis labels in {all_labels}.")
        if overlap := set(all_labels) & set(self.fixed_axes):
            raise ValueError(f"Axes {overlap} appear in both axes and fixed_axes.")

    @property
    def all_axis_labels(self) -> list[str]:
        return self.geometric_axis_labels + self.non_geometric_axis_labels


@dataclass(frozen=True, slots=True)
class CoordinatesConfig:
    """One coord system and one domain for one CoordsProtocol instance."""

    label: str
    coord_system_label: str
    domain: DomainConfig


# Datasets Config
@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """A single data set."""

    label: str
    source: DataSetSourceLabels = DataSetSourceLabels.FILE
    train_label: str = "train"
    val_label: str = "val"
    file_path: str | None = None
    sampling: DataSetSamplerLabels = DataSetSamplerLabels.NONE

    def __post_init__(self) -> None:
        if self.source == DataSetSourceLabels.FILE and not self.file_path:
            raise ValueError("file_path is required when source=file.")


# Fields Config
@dataclass(frozen=True, slots=True)
class FieldConfig:
    """A single field."""

    label: str
    rank: int

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError(
                f"Rank of field {self.label} must be >= 0, got {self.rank}."
            )


@dataclass(frozen=True, slots=True)
class FieldsConfig:
    """A collection of fields."""

    input_fields: list[FieldConfig]
    output_fields: list[FieldConfig]
    derivable_fields: list[FieldConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Each list must not have internal duplicates
        for section, fields in (
            ("input_fields", self.input_fields),
            ("output_fields", self.output_fields),
            ("derivable_fields", self.derivable_fields),
        ):
            labels = [f.label for f in fields]
            if len(labels) != len(set(labels)):
                raise ValueError(f"Duplicate labels in {section}: {labels}.")

        # derivable_fields must not overlap with input or output fields.
        primitive_labels = {field.label for field in self.input_fields} | {
            field.label for field in self.output_fields
        }
        derivable_labels = {field.label for field in self.derivable_fields}
        overlap = primitive_labels & derivable_labels
        if overlap:
            raise ValueError(
                f"Labels {overlap} appear in both primitive "
                "(input/output) fields and derivable_fields."
            )

    @property
    def input_field_label_to_rank(self) -> dict[str, int]:
        return {field.label: field.rank for field in self.input_fields}

    @property
    def output_field_label_to_rank(self) -> dict[str, int]:
        return {field.label: field.rank for field in self.output_fields}

    @property
    def derivable_field_label_to_rank(self) -> dict[str, int]:
        return {field.label: field.rank for field in self.derivable_fields}

    @property
    def all_field_labels(self) -> set[str]:
        return (
            set(self.input_field_label_to_rank)
            | set(self.output_field_label_to_rank)
            | set(self.derivable_field_label_to_rank)
        )


# Constant Config
@dataclass(frozen=True, slots=True)
class ConstConfig:
    """A single constant."""

    label: str
    value: float
    learnable: bool = False


# Equations Config
@dataclass(frozen=True, slots=True)
class EquationConfig:
    """A single equation."""

    label: str
    function_name: str
    file_path: str
    input_labels: list[str]
    output_labels: list[str]
    output_ranks: list[int]

    def __post_init__(self) -> None:
        if len(self.output_labels) != len(self.output_ranks):
            raise ValueError(
                f"output_labels and output_ranks must have the same length, got "
                f"{len(self.output_labels)} and {len(self.output_ranks)}."
            )
        for label, rank in zip(self.output_labels, self.output_ranks, strict=True):
            if rank < 0:
                raise ValueError(f"output rank for '{label}' must be >= 0, got {rank}.")

    @property
    def output_label_to_rank(self) -> dict[str, int]:
        return dict(zip(self.output_labels, self.output_ranks, strict=True))


# Loss Config
@dataclass(frozen=True, slots=True)
class LossTermConfig:
    """A single loss term."""

    label: str
    field_label: str
    coordinates_label: str
    dataset_label: str | None
    loss_type: LossTermLabels
    loss_function: LossFunctionLabels
    weight_value: float
    weight_method_label: WeightMethodLabels
    weight_group_label: str | None
    staggering: int = 0

    def __post_init__(self) -> None:
        if self.staggering < 0:
            raise ValueError(
                f"LossTermConfig '{self.label}': staggering must be >= 0, got "
                f"{self.staggering}."
            )


# Neural Network Architecture Config
@dataclass(frozen=True, slots=True)
class NNConfig:
    """Neural network architecture configuration."""

    nn_type: NNTypeLabels
    hidden_layer_widths: list[int]
    layer_types: list[LayerTypeLabels]
    activation_functions: list[ActivationFuncLabels]
    init_weights: list[WeightInitLabels]
    init_biases: list[BiasInitLabels]
    input_transformation: InputTransformationLabels | None
    weight_norm: bool = False
    encoder_layer_type: LayerTypeLabels | None = None
    encoder_activation_function: ActivationFuncLabels | None = None
    encoder_init_weights: WeightInitLabels | None = None
    encoder_init_biases: BiasInitLabels | None = None
    skip_connection: bool = False

    def __post_init__(self) -> None:
        n = len(self.hidden_layer_widths) + 1

        if not self.hidden_layer_widths:
            raise ValueError("hidden_layer_widths must not be empty.")

        for i, w in enumerate(self.hidden_layer_widths):
            if w <= 0:
                raise ValueError(f"hidden_layer_widths[{i}] must be > 0, got {w}.")

        if len(self.layer_types) != n:
            raise ValueError(
                f"layer_types must have length {n}, got {len(self.layer_types)}."
            )

        if len(self.activation_functions) != n:
            raise ValueError(
                f"activation_functions must have length {n}, "
                f"got {len(self.activation_functions)}."
            )

        if len(self.init_weights) != n:
            raise ValueError(
                f"init_weights must have length {n}, got {len(self.init_weights)}."
            )

        if len(self.init_biases) != n:
            raise ValueError(
                f"init_biases must have length {n}, got {len(self.init_biases)}."
            )

        if self.nn_type == "modified_mlp":
            if len(set(self.hidden_layer_widths)) != 1:
                raise ValueError(
                    "modified_mlp requires all hidden layers to share one width, "
                    f"got {self.hidden_layer_widths}."
                )
            missing = [
                name
                for name, value in (
                    ("encoder_layer_type", self.encoder_layer_type),
                    ("encoder_activation_function", self.encoder_activation_function),
                    ("encoder_init_weights", self.encoder_init_weights),
                    ("encoder_init_biases", self.encoder_init_biases),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"modified_mlp requires {missing} to be set.")

        if self.skip_connection:
            if self.nn_type != "modified_mlp":
                raise ValueError("skip_connection is only supported for modified_mlp.")
            if len(self.hidden_layer_widths) < 2:
                raise ValueError(
                    "skip_connection requires at least 2 hidden layers "
                    "(an embedding layer plus at least one residual block)."
                )


# Trainer Config
@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Optimiser configuration."""

    optimizer: OptimizerLabels
    learning_rate: float
    weight_decay_groups: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(
                f"OptimizerConfig: learning_rate must be > 0, got "
                f"{self.learning_rate}."
            )


@dataclass(frozen=True, slots=True)
class TrainerModuleConfig:
    """A single trainer module."""

    label: str
    method_name: str | None = None
    file_path: str | None = None
    class_name: str | None = None
    hyperparameters: dict[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        is_builtin = self.method_name is not None
        is_custom = self.file_path is not None or self.class_name is not None

        if is_builtin and is_custom:
            raise ValueError(
                "method_name and file_path/class_name should be mutually exclusive."
            )
        if not is_builtin and not is_custom:
            raise ValueError("Must set either method_name or file_path + class_name.")
        if is_custom and not (self.file_path and self.class_name):
            raise ValueError(
                "Both file_path and class_name are required for custom trainer modules."
            )

    @property
    def is_builtin(self) -> bool:
        return self.method_name is not None


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Trainer config."""

    optimizer: OptimizerConfig
    trainer_modules: list[TrainerModuleConfig] = field(default_factory=list)
    max_epoch: int | None = None
    warmup_epochs: int = 0
    print_freq: int = 1
    n_batches: int = 1
    shuffle: bool = True
    num_workers: int = 0

    def __post_init__(self) -> None:
        labels = [module.label for module in self.trainer_modules]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate module labels in {labels}.")
        if self.max_epoch is not None and self.max_epoch <= 0:
            raise ValueError(f"max_epoch must be > 0, got {self.max_epoch}.")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}.")
        if self.n_batches <= 0:
            raise ValueError(f"n_batches must be > 0, got {self.n_batches}.")

    @property
    def by_label(self) -> dict[str, TrainerModuleConfig]:
        return {module.label: module for module in self.trainer_modules}
