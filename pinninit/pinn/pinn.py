from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from ..collocation.collocation import load_collocation
from ..data_structure.coords_dataclass import make_coords_dataclass
from ..geometry.coordinate_system import COORD_SYSTEM_REGISTRY
from ..io.input_data import load_dataset
from ..io.input_equations import load_functions_from_file_to_registry
from ..io.input_parameters import PINNParameters
from ..io.io_utils import get_enum_label_from_value
from ..loss.loss_function import LossFunctionLabels
from ..loss.loss_term import DataLoss, PhysicsLoss
from ..loss.loss_term_group import LossTermGroup
from ..nn.layers import LAYER_TYPE_REGISTRY
from ..nn.neural_network import NeuralNetwork
from ..optimizer.optimizer import OPTIMIZER_REGISTRY


class PINN:
    def __init__(self, param_file_path: str, device: torch.device):
        # Store device
        self.device = device

        # Read PINN parameters from yaml file
        self.pinn_params = PINNParameters(param_file_path)

        # Make dataclasses
        self.Coords: type[Any] = make_coords_dataclass(
            self.pinn_params.coord_axes_labels, device=self.device
        )

        # Initialise modules
        self._init_geometry()  ### most general
        self._init_dataset()  ### depends on geometry
        self._init_collocation()  #### depends on geometry
        self._init_equations()
        self._init_loss()
        self._init_nn()
        self._init_learnt_parameters()
        self._init_optimizer()

        ### Temp
        self._init_validation_data()
        self._init_test_data()

    def _init_geometry(self) -> None:
        # Extract params
        self.coord_system_label = self.pinn_params.coordinates_params.coordinate_system
        coordinates = self.pinn_params.coordinates_params.coordinates
        self.coord_axes_labels = self.pinn_params.coord_axes_labels
        self.coord_dim = len(self.coord_axes_labels)

        # Fixed axes
        fixed_axes_list = []
        fixed_axes_values_list = []
        for axis in coordinates:
            if axis.get("is_fixed", 0) == 1:
                if "fixed_value" in axis:
                    fixed_axes_list.append(axis["label"])
                    fixed_axes_values_list.append(axis["fixed_value"])
                else:
                    raise ValueError(
                        f"Axis '{axis['label']}' is marked as fixed but no "
                        "'fixed_value' is provided."
                    )
        fixed_axes = tuple(fixed_axes_list)
        fixed_axes_values = tuple(fixed_axes_values_list)

        # Geometric axes
        geometric_axes_labels = tuple(
            [
                str(axis["label"])
                for axis in self.pinn_params.coordinates_params.coordinates
                if axis.get("is_geometric", 0) == 1
            ]
        )
        dim_geometry = len(geometric_axes_labels)

        # Set coordinate system
        self.coord_system = COORD_SYSTEM_REGISTRY[self.coord_system_label](
            dim_geometry,
            geometric_axes_labels,
            fixed_axes,
            fixed_axes_values,
            self.device,
        )

        """ removed this file until coord treatment is finalised
        ### Keep these as lists rather than tensors for now because of None
        self.domain_lb, self.domain_ub = extract_domain_bounds(
            self.coord_system, self.pinn_params
        )
        """

        ### Do boundary conditions here

    def _init_dataset(self) -> None:
        # Load datasets
        file_path = self.pinn_params.data_params.file_path
        train_sampling_strategy = self.pinn_params.data_params.train_sampling_strategy
        (
            input_coords,
            self.input_fields,
            self.input_consts,
        ) = load_dataset(file_path, train_sampling_strategy, "training", self.device)

        # Initialise coord dataclass object
        self.training_coords = self.Coords(
            coord_system=self.coord_system, **input_coords
        )

        # Lower and upper bounds of coords ### this is after filtering. Is that right?
        ### Make these attributes of coords
        self.input_coords_lb = {
            label: torch.min(getattr(self.training_coords, label))
            for label in self.training_coords.coord_labels
        }
        self.input_coords_ub = {
            label: torch.max(getattr(self.training_coords, label))
            for label in self.training_coords.coord_labels
        }
        self.input_coords_internal_lb = {
            label: torch.min(getattr(self.training_coords, label))
            for label in self.training_coords.internal_labels
        }
        self.input_coords_internal_ub = {
            label: torch.max(getattr(self.training_coords, label))
            for label in self.training_coords.internal_labels
        }

    def _init_validation_data(self) -> None:
        #### temp
        file_path = "../data/igrf_sampled_val.hdf5"
        train_sampling_strategy = self.pinn_params.data_params.train_sampling_strategy
        (
            validation_coords,
            self.validation_coords_fields,
            self.validation_coords_consts,
        ) = load_dataset(file_path, train_sampling_strategy, "validation", self.device)

        self.validation_coords = self.Coords(
            coord_system=self.coord_system, **validation_coords
        )

    def _init_test_data(self) -> None:
        ####
        file_path = "../data/igrf_sampled_test.hdf5"
        train_sampling_strategy = self.pinn_params.data_params.train_sampling_strategy
        (
            test_coords,
            self.test_fields,
            self.test_consts,
        ) = load_dataset(file_path, train_sampling_strategy, "test", self.device)

        self.test_coords = self.Coords(coord_system=self.coord_system, **test_coords)

    def _init_collocation(self) -> None:
        # Set coords of collocation points
        collocation_coords = load_collocation(
            self.input_coords_lb,
            self.input_coords_ub,
            self.pinn_params,
            self.device,
        )

        self.collocation_coords = self.Coords(
            coord_system=self.coord_system, **collocation_coords
        )

    def _init_equations(self) -> None:
        # Load functions to registry
        load_functions_from_file_to_registry(self.pinn_params.equations_params)

    def _init_loss(self) -> None:
        data_loss_terms_params = self.pinn_params.dataloss_params.terms
        physics_loss_terms_params = self.pinn_params.physicsloss_params.terms

        # Init loss term groups
        self.data_loss_terms = LossTermGroup()
        self.physics_loss_terms = LossTermGroup()

        # Data loss terms
        for term in data_loss_terms_params:
            field_label = term["field_label"]
            components = term.get("components", 1)
            for idx in range(components):
                if components > 1:
                    loss_function = get_enum_label_from_value(
                        term["loss_function"][idx], LossFunctionLabels
                    )
                    weight_method = term["weight_method"][idx]
                    weight_value = term["weight_value"][idx]
                else:
                    loss_function = get_enum_label_from_value(
                        term["loss_function"], LossFunctionLabels
                    )
                    weight_method = term["weight_method"]
                    weight_value = term["weight_value"]

                data_loss_term = DataLoss(
                    field_label,
                    idx,
                    loss_function,
                    weight_method,
                    weight_value,
                )

                # Add to group
                self.data_loss_terms.add(data_loss_term)

        # Physics loss terms
        for term in physics_loss_terms_params:
            equation_label = term["equation_label"]
            residual_label = term["residual_label"]
            components = term.get("components", 1)
            weight_method = term["weight_method"]
            weight_value = term["weight_value"]

            for idx in range(components):
                if components > 1:
                    loss_function = get_enum_label_from_value(
                        term["loss_function"][idx], LossFunctionLabels
                    )
                    weight_method = term["weight_method"][idx]
                    weight_value = term["weight_value"][idx]
                else:
                    loss_function = get_enum_label_from_value(
                        term["loss_function"], LossFunctionLabels
                    )
                    weight_method = term["weight_method"]
                    weight_value = term["weight_value"]

                physics_loss_term = PhysicsLoss(
                    equation_label,
                    residual_label,
                    idx,
                    loss_function,
                    weight_method,
                    weight_value,
                )

                # Add to group
                self.physics_loss_terms.add(physics_loss_term)

    def _init_nn(self) -> None:
        # Extract params
        layers_params = self.pinn_params.layers_params
        hidden_layer_widths = layers_params.hidden_layer_widths
        layer_types = layers_params.layer_types
        self.num_layers = len(hidden_layer_widths) + 1

        ### Make this more general
        output_fields = self.pinn_params.fields_params.output_fields

        ### Currently hard code this. Only works for feed forward NN
        in_features = [self.training_coords.internal_coord_dim] + hidden_layer_widths
        out_features = hidden_layer_widths + [len(output_fields)]

        self.layers = [
            LAYER_TYPE_REGISTRY[layer_types[i]](
                in_features=in_features[i],
                out_features=out_features[i],
            )
            for i in range(self.num_layers)
        ]

        # Get lists of lb, ub sorted alphabetically by axis label
        lb_list = []
        ub_list = []

        for coord_label in self.training_coords.internal_labels:
            lb_list.append(self.input_coords_internal_lb[coord_label])
            ub_list.append(self.input_coords_internal_ub[coord_label])

        self.model = NeuralNetwork(self.layers, layers_params, lb_list, ub_list).to(
            self.device
        )

    def _init_learnt_parameters(self) -> None:
        learnt_consts: dict[str, torch.Tensor] = {}
        for const_label in self.pinn_params.learnt_const_labels:
            learnt_consts[const_label] = nn.Parameter(
                torch.zeros(1, dtype=torch.float32, device=self.device)
            )

        self.learnt_consts = learnt_consts

    def _init_optimizer(self) -> None:
        optimizer_label = self.pinn_params.optimizer_params.optimizer
        lr = self.pinn_params.optimizer_params.learning_rate

        self.optimizer = OPTIMIZER_REGISTRY[optimizer_label](
            list(self.model.parameters()) + list(self.learnt_consts.values()),
            lr=lr,
        )

    # Function to predict values
    def predict(self, coords: dict[str, torch.Tensor]) -> Any:
        # Initialise coord dataclass object
        test_coords = self.Coords(coord_system=self.coord_system, **coords)

        output = self.model(test_coords.internal)

        return output

    def compute_loss(self) -> torch.Tensor:
        # Apply models at coords of loss term groups
        self.data_loss_terms.apply_model(
            self.model,
            self.training_coords,
            self.pinn_params.input_field_labels,
            self.pinn_params.output_field_labels,
            self.input_fields,
        )
        self.physics_loss_terms.apply_model(
            self.model,
            self.collocation_coords,
            self.pinn_params.input_field_labels,
            self.pinn_params.output_field_labels,
        )

        # Compute loss for each group
        self.loss_data = self.data_loss_terms.compute_loss()
        self.loss_phys = self.physics_loss_terms.compute_loss()

        # Total loss
        loss = self.loss_data + self.loss_phys
        return loss

    def compute_validation_loss(self) -> torch.Tensor:
        # Apply models at coords of loss term groups
        self.data_loss_terms.apply_model(
            self.model,
            self.validation_coords,
            self.pinn_params.input_field_labels,
            self.pinn_params.output_field_labels,
            self.validation_coords_fields,
        )

        # Compute loss data loss group at validation_coords
        return self.data_loss_terms.compute_loss()

    def compute_test_loss(self) -> torch.Tensor:
        # Apply models at coords of loss term groups
        self.data_loss_terms.apply_model(
            self.model,
            self.test_coords,
            self.pinn_params.input_field_labels,
            self.pinn_params.output_field_labels,
            self.test_fields,
        )

        # Compute loss data loss group at test_coords
        return self.data_loss_terms.compute_loss()

    def train(self) -> None:
        nIter = self.pinn_params.training_params.epochs

        self.loss_values = []
        self.val_loss_values = []
        self.test_loss_values = []
        # self.l1_values = []
        # self.l2_values = []
        for it in range(nIter):
            # Zero the gradients
            self.optimizer.zero_grad()
            # Compute the loss
            loss = self.compute_loss()
            val_loss = self.compute_validation_loss()
            test_loss = self.compute_test_loss()
            # Backpropagation
            loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]
            # Update the model parameters
            self.optimizer.step()
            # Print the training progress every 500 iterations
            if it % 500 == 0:
                print(
                    f"Iter: {it}, Loss: {loss.item():.3e}, Data Loss: "
                    f"{self.loss_data:.3e}, Physics Loss: {self.loss_phys:.3e}, "
                    f"Val Loss: {val_loss:.3e}, Test Loss: {test_loss:.3e}"
                )
            #  l1 = self.learnt_consts["l1"]
            #  l2 = self.learnt_consts["l2"]
            #  print(
            #      f"Iter: {it}, Loss: {loss.item():.3e}, Lambda1: "
            #      f"{l1}, Lambda2: {l2}"
            #  )
            self.loss_values.append(loss.item())
            self.val_loss_values.append(val_loss.item())
            self.test_loss_values.append(test_loss.item())
        #   self.l1_values.append(self.learnt_consts["l1"])
        #   self.l2_values.append(self.learnt_consts["l2"])

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                #   "lambda_1": self.learnt_consts["l1"],
                #   "lambda_2": self.learnt_consts["l2"],
            },
            path,
        )

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file: '{path}'")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    # self.lambda_1 = torch.tensor(
    #     checkpoint["lambda_1"], requires_grad=True, dtype=torch.float32
    # )
    # self.lambda_2 = torch.tensor(
    #     checkpoint["lambda_2"], requires_grad=True, dtype=torch.float32
    # )
