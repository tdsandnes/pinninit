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
from ..loss.loss_term import LossTermLabels
from ..loss.loss_term_group import LossTermGroup
from ..nn.layers import LAYER_TYPE_REGISTRY
from ..nn.neural_network import NeuralNetwork
from ..optimizer.optimizer import OPTIMIZER_REGISTRY, OptimizerLabels
from ..training.early_stopping import EarlyStopping
from ..training.trainer import Trainer
from ..training.update_optimizer import UpdateOptimizer


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

        ### Set torch seed. Probably want more control over this
        torch.manual_seed(12345)

        # Initialise modules
        self._init_geometry()  ### most general
        self._init_dataset()  ### depends on geometry
        self._init_collocation()  #### depends on geometry
        self._init_equations()
        self._init_loss()
        self._init_nn()
        self._init_learnt_parameters()
        self._init_optimizer()

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
        self.dim_geometry = len(geometric_axes_labels)

        # Set coordinate system
        self.coord_system = COORD_SYSTEM_REGISTRY[self.coord_system_label](
            self.dim_geometry,
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
        ### Note that eventually this should all be in pytorch Dataset class

        # Load datasets
        file_path = self.pinn_params.data_params.file_path
        train_sampling_strategy = self.pinn_params.data_params.train_sampling_strategy
        (
            training_coords,
            training_fields,
        ) = load_dataset(file_path, train_sampling_strategy, "training", self.device)

        ### temp
        file_path = "../../data/igrf_sampled_val.hdf5"
        train_sampling_strategy = self.pinn_params.data_params.train_sampling_strategy
        (
            validation_coords,
            validation_coords_fields,
        ) = load_dataset(file_path, train_sampling_strategy, "validation", self.device)

        # Initialise coord dataclass object
        self.train_coords = self.Coords(
            coord_system=self.coord_system, **training_coords
        )
        self.val_coords = self.Coords(
            coord_system=self.coord_system, **validation_coords
        )

        # Init loss term group
        fields_params = self.pinn_params.fields_params
        equations_params = self.pinn_params.equations_params
        self.data_loss_term_group = LossTermGroup(
            self.train_coords,
            self.val_coords,
            self.dim_geometry,
            fields_params,
            equations_params,
            train_input_fields=training_fields,
            val_input_fields=validation_coords_fields,
        )

        # Lower and upper bounds of coords ### this is after filtering. Is that right?
        ### Make these attributes of coords
        self.input_coords_lb = {
            label: torch.min(getattr(self.train_coords, label))
            for label in self.train_coords.coord_labels
        }
        self.input_coords_ub = {
            label: torch.max(getattr(self.train_coords, label))
            for label in self.train_coords.coord_labels
        }
        self.input_coords_internal_lb = {
            label: torch.min(getattr(self.train_coords, label))
            for label in self.train_coords.internal_labels
        }
        self.input_coords_internal_ub = {
            label: torch.max(getattr(self.train_coords, label))
            for label in self.train_coords.internal_labels
        }

    def _init_collocation(self) -> None:
        # Set coords of collocation points
        collocation_coords = load_collocation(
            self.input_coords_lb,
            self.input_coords_ub,
            self.pinn_params,
            self.device,
        )
        ### temp
        file_path = "../../data/igrf_sampled_val.hdf5"
        train_sampling_strategy = self.pinn_params.data_params.train_sampling_strategy
        (
            validation_coords,
            _,
        ) = load_dataset(file_path, train_sampling_strategy, "validation", self.device)

        # Initialise coord dataclass object
        col_coords = self.Coords(coord_system=self.coord_system, **collocation_coords)
        val_coords = self.Coords(coord_system=self.coord_system, **validation_coords)

        # Init loss term group
        fields_params = self.pinn_params.fields_params
        equations_params = self.pinn_params.equations_params
        self.phys_loss_term_group = LossTermGroup(
            col_coords,
            val_coords,
            self.dim_geometry,
            fields_params,
            equations_params,
        )

    def _init_equations(self) -> None:
        # Load functions to registry
        load_functions_from_file_to_registry(self.pinn_params.equations_params)

    def _init_loss(self) -> None:
        # Separate loss term params
        data_loss_terms_params = [
            term
            for term in self.pinn_params.loss_terms_params.terms
            if term["loss_type"] == LossTermLabels.DATA
        ]
        physics_loss_terms_params = [
            term
            for term in self.pinn_params.loss_terms_params.terms
            if term["loss_type"] == LossTermLabels.PHYSICS
        ]

        # Add loss terms to groups
        self.data_loss_term_group.init_loss_terms_and_add(
            data_loss_terms_params, self.pinn_params.fields_params
        )
        self.phys_loss_term_group.init_loss_terms_and_add(
            physics_loss_terms_params, self.pinn_params.fields_params
        )

    def _init_nn(self) -> None:
        # Extract params
        layers_params = self.pinn_params.layers_params
        hidden_layer_widths = layers_params.hidden_layer_widths
        layer_types = layers_params.layer_types
        self.num_layers = len(hidden_layer_widths) + 1

        ### Make this more general
        output_fields = self.pinn_params.fields_params.output_fields

        ### Currently hard code this. Only works for feed forward NN
        in_features = [self.train_coords.internal_coord_dim] + hidden_layer_widths
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

        for coord_label in self.train_coords.internal_labels:
            lb_list.append(self.input_coords_internal_lb[coord_label])
            ub_list.append(self.input_coords_internal_ub[coord_label])

        output_fields_params = self.pinn_params.fields_params.output_fields
        output_field_label_to_rank = {
            f["label"]: f["rank"] for f in output_fields_params
        }
        self.model = NeuralNetwork(
            output_field_label_to_rank, self.layers, layers_params, lb_list, ub_list
        ).to(self.device)

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

    def train(self) -> None:
        ### Hardcode these for now
        optimizer_label = OptimizerLabels.LBFGS
        new_optimizer = OPTIMIZER_REGISTRY[optimizer_label](
            list(self.model.parameters()) + list(self.learnt_consts.values()),
            ### Currently just use defaults here
        )
        patience = 500
        frac_tol = 1e-1
        update_optimizer = UpdateOptimizer(new_optimizer, patience, frac_tol)
        patience = 500
        frac_tol = 0.0
        early_stopping = EarlyStopping(patience, frac_tol)
        trainer_hooks = [update_optimizer, early_stopping]

        # Init trainer
        loss_groups = [self.data_loss_term_group, self.phys_loss_term_group]
        self.trainer = Trainer(self.model, self.optimizer, loss_groups, trainer_hooks)

        # Train model
        self.trainer.train()

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "train_loss_history": self.trainer.train_loss_history,
                "val_loss_history": self.trainer.val_loss_history,
                ###         "loss_values": self.loss_values,
                ###         "data_loss_values": self.data_loss_values,
                ###         "phys_loss_values": self.phys_loss_values,
                ###        "val_loss_values": self.val_loss_values,
                ###        "test_loss_values": self.test_loss_values,
                ##  "rms_residual_per_test_point": self.rms_residual_per_test_point,
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

        # Restore the additional attributes
        self.train_loss_history = checkpoint["train_loss_history"]
        self.val_loss_history = checkpoint["val_loss_history"]

    ###  self.loss_values = checkpoint.get("loss_values", [])
    ###  self.data_loss_values = checkpoint.get("data_loss_values", [])
    ###  self.phys_loss_values = checkpoint.get("phys_loss_values", [])
    ###  self.val_loss_values = checkpoint.get("val_loss_values", [])
    ###  self.test_loss_values = checkpoint.get("test_loss_values", [])
    ###  self.rms_residual_per_test_point = checkpoint.get(
    ###      "rms_residual_per_test_point", []
    ###  )

    # self.lambda_1 = torch.tensor(
    #     checkpoint["lambda_1"], requires_grad=True, dtype=torch.float32
    # )
    # self.lambda_2 = torch.tensor(
    #     checkpoint["lambda_2"], requires_grad=True, dtype=torch.float32
    # )
