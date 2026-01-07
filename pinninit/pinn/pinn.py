from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from ..data_structure.coords_dataclass import make_coords_dataclass
from ..geometry.coordinate_system import COORD_SYSTEM_REGISTRY
from ..io.input_data import load_coords, load_fields
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
        self._init_datasets()  ### depends on geometry
        self._init_equations()
        self._init_loss()  ### depends on data sets
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

    def _init_datasets(self) -> None:
        ### Note that eventually this should all be in pytorch Dataset class

        self.dataset_coords: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        self.dataset_fields: dict[
            str, dict[str, dict[str, torch.Tensor | dict[str, torch.Tensor]]]
        ] = {}

        # Lookup dataset metadata by label
        dataset_dict = {
            dataset["data_set_label"]: dataset
            for dataset in self.pinn_params.data_sets_params.data_sets
        }

        # Collect all input fields
        input_fields_to_rank = {
            field["label"]: field["rank"]
            for field in self.pinn_params.fields_params.input_fields
        }

        # Loop over all loss terms to identify required datasets
        for term in self.pinn_params.loss_terms_params.terms:
            dataset_label = term["data_set_label"]

            # Skip if already loaded
            if dataset_label in self.dataset_coords:
                continue

            dataset = dataset_dict[dataset_label]
            file_path = dataset["file_path"]

            # Load coords for both roles
            coords_dict: dict[str, dict[str, torch.Tensor]] = {}
            for role in ["training", "validation"]:
                coords_dict[role] = load_coords(
                    file_path=file_path,
                    dataset_label=dataset_label,
                    dataset_role=role,
                    expected_coord_labels=list(self.pinn_params.coord_axes_labels),
                    device=self.device,
                )
            self.dataset_coords[dataset_label] = coords_dict

            # Load fields only if at least one data loss term uses this dataset
            if any(
                term["data_set_label"] == dataset_label
                and term["loss_type"] == LossTermLabels.DATA
                for term in self.pinn_params.loss_terms_params.terms
            ):
                fields_dict: dict[
                    str, dict[str, torch.Tensor | dict[str, torch.Tensor]]
                ] = {}
                for role in ["training", "validation"]:
                    fields_dict[role] = load_fields(
                        file_path=file_path,
                        dataset_label=dataset_label,
                        dataset_role=role,
                        expected_coord_labels=list(self.pinn_params.coord_axes_labels),
                        device=self.device,
                        input_fields_to_rank=input_fields_to_rank,
                    )
                self.dataset_fields[dataset_label] = fields_dict

    def _init_equations(self) -> None:
        # Load functions to registry
        load_functions_from_file_to_registry(self.pinn_params.equations_params)

    def _init_loss(self) -> None:
        # First init loss term groups
        self.loss_term_groups: dict[str, LossTermGroup] = {}

        for dataset_label in self.dataset_coords:
            # Get training and validation coords
            coords = self.dataset_coords[dataset_label]
            train_coord = self.Coords(
                coord_system=self.coord_system, **coords["training"]
            )
            val_coord = self.Coords(
                coord_system=self.coord_system, **coords["validation"]
            )

            # Get training and validation fields
            if dataset_label in self.dataset_fields:
                fields = self.dataset_fields[dataset_label]
                train_fields = fields["training"]
                val_fields = fields["validation"]
            else:
                train_fields = None
                val_fields = None

            ### Hardcode these for now
            batch_size = 4096
            shuffle_train = True
            num_workers = 0

            # Create loss term group
            self.loss_term_groups[dataset_label] = LossTermGroup(
                train_coords=train_coord,
                val_coords=val_coord,
                dim_geometry=self.dim_geometry,
                available_fields_params=self.pinn_params.fields_params,
                available_equations_params=self.pinn_params.equations_params,
                device=self.device,
                train_input_fields=train_fields,
                val_input_fields=val_fields,
                batch_size=batch_size,
                shuffle_train=shuffle_train,
                num_workers=num_workers,
            )

        # Loop over all loss term groups
        for group_label, loss_group in self.loss_term_groups.items():
            # Find all loss terms that correspond to this dataset
            relevant_loss_terms = [
                term
                for term in self.pinn_params.loss_terms_params.terms
                if term["data_set_label"] == group_label
            ]

            # Initialise loss terms in the group
            loss_group.init_loss_terms_and_add(
                relevant_loss_terms, self.pinn_params.fields_params
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
        ### Hardcode this for now
        internal_coord_dim = 3
        in_features = [internal_coord_dim] + hidden_layer_widths
        out_features = hidden_layer_widths + [len(output_fields)]

        self.layers = [
            LAYER_TYPE_REGISTRY[layer_types[i]](
                in_features=in_features[i],
                out_features=out_features[i],
            )
            for i in range(self.num_layers)
        ]

        output_fields_params = self.pinn_params.fields_params.output_fields
        output_field_label_to_rank = {
            f["label"]: f["rank"] for f in output_fields_params
        }

        # get lower and upper bounds of all training points
        lbs = torch.stack(
            [
                group.train_coords.internal_lb.detach()
                for group in self.loss_term_groups.values()
            ]
        )
        ubs = torch.stack(
            [
                group.train_coords.internal_ub.detach()
                for group in self.loss_term_groups.values()
            ]
        )
        lb = torch.min(lbs, dim=0).values
        ub = torch.max(ubs, dim=0).values

        self.model = NeuralNetwork(
            output_field_label_to_rank, self.layers, layers_params, lb, ub
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
        frac_tol = 0.0  # 1e-1
        update_optimizer = UpdateOptimizer(new_optimizer, patience, frac_tol)
        patience = 500
        frac_tol = 0.0
        early_stopping = EarlyStopping(patience, frac_tol)
        trainer_hooks = [update_optimizer, early_stopping]

        # Init trainer
        self.trainer = Trainer(
            self.model, self.optimizer, self.loss_term_groups, trainer_hooks
        )

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
