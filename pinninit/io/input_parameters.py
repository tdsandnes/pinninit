from __future__ import annotations

import os
from typing import TYPE_CHECKING

import yaml

from ..collocation.collocation_sampler import CollocationSamplerLabels
from ..geometry.coordinate_system import (
    CoordSystemLabels,
)
from ..nn.activation_function import ActivationFuncLabels
from ..nn.initialise_biases import BiasInitLabels
from ..nn.initialise_weights import WeightInitLabels
from ..nn.layers import LayerTypeLabels
from ..optimizer.optimizer import OptimizerLabels
from .data_sampler import DataSetSamplerLabels
from .io_utils import (
    expand_if_alias,
    get_enum_label_from_value,
    load_compulsory,
    load_optional,
)

if TYPE_CHECKING:
    from typing import Any


class DataParams:
    def __init__(self, input_params: dict[str, Any]):
        # file_path
        self.file_path = load_compulsory(
            "file_path", "Data", input_params, data_type=str
        )
        # train_sampling_strategy
        train_sampling_strategy_raw = load_compulsory(
            "train_sampling_strategy", "Data", input_params, data_type=str
        )

        self.train_sampling_strategy = get_enum_label_from_value(
            train_sampling_strategy_raw, DataSetSamplerLabels
        )

        if self.train_sampling_strategy == DataSetSamplerLabels.RANDOM:
            # num_train_samples
            self.num_train_samples = load_compulsory(
                "num_train_samples", "Data", input_params, data_type=int
            )
            # train_sampling_seed
            self.train_sampling_seed = load_compulsory(
                "train_sampling_seed", "Data", input_params, data_type=int
            )


class CollocationParams:
    def __init__(self, input_params: dict[str, Any]):
        # from_file
        self.from_file = load_optional(
            "from_file", "Collocation", input_params, data_type=int, default=0
        )
        if self.from_file == 1:
            # file_path
            self.file_path = load_compulsory(
                "file_path", "Collocation", input_params, data_type=str
            )

            return
        elif self.from_file == 0:
            # collocation_strategy
            ###collocation_strategy_raw = load_compulsory(
            ###    "collocation_strategy", "Collocation", input_params, data_type=str
            ###)
            self.collocation_strategy = (
                CollocationSamplerLabels.RANDOM
            )  ###This should be:
            ###get_enum_label_from_value(collocation_strategy_raw, ...)### Add registry

            if self.collocation_strategy == CollocationSamplerLabels.RANDOM:
                # num_collocation_points
                self.num_collocation_points = load_compulsory(
                    "num_collocation_points", "Collocation", input_params, data_type=int
                )
                # collocation_sampling_seed
                self.collocation_sampling_seed = load_compulsory(
                    "collocation_sampling_seed",
                    "Collocation",
                    input_params,
                    data_type=int,
                )

        else:
            raise ValueError(
                f"Unknown option: '{self.from_file}' for input parameter 'Collocation: "
                f"from_file'. Available options are: {[0, 1]}"
            )


class EquationsParams:
    def __init__(self, input_params: dict[str, Any]):
        # equations
        self.equations = load_optional(
            "equations", "Equations", input_params, data_type=list
        )


class CoordinatesParams:
    def __init__(self, input_params: dict[str, Any]):
        # coordinate_system
        coordinate_system_raw = load_compulsory(
            "coordinate_system", "Coordinates", input_params, data_type=str
        )
        self.coordinate_system = get_enum_label_from_value(
            coordinate_system_raw, CoordSystemLabels
        )
        # coordinates
        self.coordinates = load_compulsory(
            "coordinates", "Coordinates", input_params, data_type=list
        )


class FieldsParams:
    def __init__(self, input_params: dict[str, Any]):
        # input_fields
        self.input_fields = load_compulsory(
            "input_fields", "Fields", input_params, data_type=list
        )
        # output_fields
        self.output_fields = load_compulsory(
            "output_fields", "Fields", input_params, data_type=list
        )


class ConstantsParams:
    def __init__(self, input_params: dict[str, Any]):
        # input_consts
        self.input_consts = load_optional(
            "input_consts", "Constants", input_params, data_type=list
        )
        # learnt_consts
        self.learnt_consts = load_optional(
            "learnt_consts", "Constants", input_params, data_type=list
        )


class DataLossParams:
    def __init__(self, input_params: dict[str, Any]):
        # terms
        self.terms = load_optional("terms", "DataLoss", input_params, data_type=list)


class PhysicsLossParams:
    def __init__(self, input_params: dict[str, Any]):
        # terms
        self.terms = load_optional("terms", "PhysicsLoss", input_params, data_type=list)


class OptimizerParams:
    def __init__(self, input_params: dict[str, Any]):
        # optimizer
        optimizer_raw = load_compulsory(
            "optimizer", "Optimizer", input_params, data_type=str
        )
        self.optimizer = get_enum_label_from_value(optimizer_raw, OptimizerLabels)
        # learning_rate
        self.learning_rate = load_compulsory(
            "learning_rate", "Optimizer", input_params, data_type=float
        )


class LayersParams:
    # Aliases
    alias_layer_types = {
        "all_linear": lambda num_layers: ["linear"] * num_layers,
    }
    alias_activation_functions_raw = {
        "all_tanh": lambda num_layers: ["tanh"] * (num_layers - 1) + ["none"],
    }
    alias_init_weights = {
        "all_xavier": lambda num_layers: ["xavier"] * num_layers,
    }
    alias_init_biases = {
        "all_zeros": lambda num_layers: ["zeros"] * num_layers,
    }

    def __init__(self, input_params: dict[str, Any]):
        # hidden_layer_widths
        self.hidden_layer_widths = load_compulsory(
            "hidden_layer_widths", "Layers", input_params, data_type=list
        )
        num_layers = len(self.hidden_layer_widths) + 1

        # layer_types
        layer_types_raw = load_compulsory("layer_types", "Layers", input_params)
        layer_types_raw = expand_if_alias(
            layer_types_raw,
            self.alias_layer_types,
            data_type=list,
            num_layers=num_layers,
        )
        self.layer_types = [
            get_enum_label_from_value(x, LayerTypeLabels) for x in layer_types_raw
        ]

        # activation_functions
        activation_functions_raw = load_compulsory(
            "activation_functions", "Layers", input_params
        )
        activation_functions_raw = expand_if_alias(
            activation_functions_raw,
            self.alias_activation_functions_raw,
            data_type=list,
            num_layers=num_layers,
        )
        self.activation_functions = [
            get_enum_label_from_value(x, ActivationFuncLabels)
            for x in activation_functions_raw
        ]

        # init_weights
        init_weights_raw = load_compulsory("init_weights", "Layers", input_params)
        init_weights_raw = expand_if_alias(
            init_weights_raw,
            self.alias_init_weights,
            data_type=list,
            num_layers=num_layers,
        )
        self.init_weights = [
            get_enum_label_from_value(x, WeightInitLabels) for x in init_weights_raw
        ]

        # init_biases
        init_biases_raw = load_compulsory("init_biases", "Layers", input_params)
        init_biases_raw = expand_if_alias(
            init_biases_raw,
            self.alias_init_biases,
            data_type=list,
            num_layers=num_layers,
        )
        self.init_biases = [
            get_enum_label_from_value(x, BiasInitLabels) for x in init_biases_raw
        ]


class TrainingParams:
    def __init__(self, input_params: dict[str, Any]):
        # epochs
        self.epochs = load_compulsory("epochs", "Training", input_params, data_type=int)


class PINNParameters:
    def __init__(self, yaml_path: str):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Parameter file not found: {yaml_path}")
        with open(yaml_path) as f:
            self.input_params = yaml.safe_load(f)

        # Extract data
        self._extract_parameters()

        # Set lists of labels
        self._set_label_sets()

    def _extract_parameters(self) -> None:
        """Extract parameters as objects of individual classes"""
        self.data_params = DataParams(self.input_params)
        self.collocation_params = CollocationParams(self.input_params)
        self.equations_params = EquationsParams(self.input_params)
        self.coordinates_params = CoordinatesParams(self.input_params)
        self.fields_params = FieldsParams(self.input_params)
        self.constants_params = ConstantsParams(self.input_params)
        self.dataloss_params = DataLossParams(self.input_params)
        self.physicsloss_params = PhysicsLossParams(self.input_params)
        self.optimizer_params = OptimizerParams(self.input_params)
        self.layers_params = LayersParams(self.input_params)
        self.training_params = TrainingParams(self.input_params)

    def _set_label_sets(self) -> None:
        """Sets of labels of parameters of different types"""
        self.coord_axes_labels = {
            coord["label"] for coord in (self.coordinates_params.coordinates or [])
        }
        self.input_field_labels = {
            field["label"] for field in (self.fields_params.input_fields or [])
        }
        self.output_field_labels = {
            field["label"] for field in (self.fields_params.output_fields or [])
        }
        self.input_const_labels = {
            const["label"] for const in (self.constants_params.input_consts or [])
        }
        self.learnt_const_labels = {
            const["label"] for const in (self.constants_params.learnt_consts or [])
        }

        ### Do check here to make sure that all values in these sets are unique
