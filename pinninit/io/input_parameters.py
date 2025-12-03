from __future__ import annotations

import os
from typing import TYPE_CHECKING

import yaml

from ..collocation.collocation_sampler import CollocationSamplerLabels
from ..dataset.dataset_sampler import DataSetSamplerLabels
from ..geometry.coordinate_system import (
    COORD_SYSTEM_REGISTRY,
    CoordSystemLabels,
)
from ..nn.activation_function import ActivationFuncLabels
from ..nn.initialise_biases import BiasInitLabels
from ..nn.initialise_weights import WeightInitLabels
from ..nn.layers import LayerTypeLabels
from ..optimizer.optimizer import OptimizerLabels
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
        ###train_sampling_strategy_raw = load_compulsory(
        ###    "train_sampling_strategy", "Data", input_params, data_type=str
        ###)
        self.train_sampling_strategy = DataSetSamplerLabels.RANDOM  ###This should be:
        ###get_enum_label_from_value(train_sampling_strategy_raw, ...)### Add registry

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

        # Set attributes needed for consistent ordering
        self._set_geometric_axes_labels_and_mask()

        # Set attributes needed for directing to equations
        self._set_derived_field_dicts()

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

    def _set_geometric_axes_labels_and_mask(self) -> None:
        """Set geometric axes in order required by the coordinate system"""
        # Coord system label
        coord_system_label = self.coordinates_params.coordinate_system

        # Labels of geometric axes set in parameter file
        geometric_axes_labels_unsorted = [
            axis["label"]
            for axis in self.coordinates_params.coordinates
            if axis.get("is_geometric", 0) == 1
        ]

        # Get coord system axis labels
        coord_system = COORD_SYSTEM_REGISTRY[coord_system_label]
        axes_labels = coord_system.metadata.axes_labels

        # Tuple of labels in order required for coord system
        self.geometric_axes_labels = tuple(
            [label for label in axes_labels if label in geometric_axes_labels_unsorted]
        )

        self.dim_geometry = len(self.geometric_axes_labels)

    def _set_derived_field_dicts(self) -> None:
        # Input fields

        # Add to dict if there are equations that can be used to calculate
        # new input fields from only: (1) other input fields, (2) input consts,
        # (3) coordinates

        # Labels that can be matched to satisfy any of conditions 1-3 above
        condition_labels = set(
            [field["label"] for field in (self.fields_params.input_fields or [])]
            + [const["label"] for const in (self.constants_params.input_consts or [])]
            + [coord["label"] for coord in (self.coordinates_params.coordinates or [])]
        )

        # Get dict of derived input fields and the equations needed to calculate them
        self.dervied_input_field_equations_dict = self._get_dervied_dict(
            condition_labels
        )

        # Output fields

        # Add to dict if there are equations that can be used to calculate
        # new output fields from only: (1) other output fields, (2) input consts,
        # (3) learnt consts (4) coordinates

        # Labels that can be matched to satisfy any of conditions 1-4 above
        condition_labels = set(
            [field["label"] for field in (self.fields_params.output_fields or [])]
            + [const["label"] for const in (self.constants_params.input_consts or [])]
            + [const["label"] for const in (self.constants_params.learnt_consts or [])]
            + [coord["label"] for coord in (self.coordinates_params.coordinates or [])]
        )

        # Get dict of derived output fields and the equations needed to calculate them
        self.dervied_output_field_equations_dict = self._get_dervied_dict(
            condition_labels
        )

    def _get_dervied_dict(self, param_labels: set[str]) -> dict[str, set[str]]:
        """From the param file equations gets all new parameters that can be calculated
        from a list of other parameters (based on their labels), or quantities that can
        be calculated from them, through other equations. Returns dict of {derived
        param label: set of equation labelsneeded to calculate the param}"""

        # Start from empty dict
        dervied_dict: dict[str, set[str]] = {}
        equations_needed: set[str]

        # Loop through all equations and see if any satisfy conditions.
        # If new params have been added, it's possible that other equations are
        # now available. Therefore keep looping until no new params are added.
        added_new = True
        while added_new:
            added_new = False
            # Loop through all equations and see if any satisfy conditions
            for equation in self.equations_params.equations:
                passes_condition = False

                # If all input args satisfy conditions
                if all(arg in param_labels for arg in equation["input_args"]):
                    equations_needed = set()
                    passes_condition = True

                # If it doesn't satisfy the original condition, but it does satisfy it
                # when also considering considering newly derived params
                elif all(
                    arg in param_labels or arg in dervied_dict
                    for arg in equation["input_args"]
                ):
                    # what equations are needed to get the derived inputs needed?
                    equations_needed = set()
                    for arg in equation["input_args"]:
                        if arg in dervied_dict:
                            # Each value is a set, so iterate through its elements
                            for eq in dervied_dict[arg]:
                                equations_needed.add(eq)

                    passes_condition = True

                if passes_condition:
                    # Add all output args to dervied_dict...
                    for output_arg in equation["output_args"]:
                        # ... if they're not already in param_labels or dervied_dict
                        if (
                            output_arg not in param_labels
                            and output_arg not in dervied_dict
                        ):
                            # Add entry to dict to store equation needed to derive it
                            eqs = set(equations_needed)
                            eqs.add(equation["label"])
                            dervied_dict[output_arg] = eqs

                            # Added at least one new value, so pass through outer
                            # for loop at least one more time
                            added_new = True

        return dervied_dict
