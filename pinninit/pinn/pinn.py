from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..collocation.collocation import load_collocation
from ..dataset.dataset import load_dataset
from ..differential_equations.operators import Operators
from ..geometry.coordinate_system import COORD_SYSTEM_REGISTRY
from ..io.input_equations import EQUATION_REGISTRY, load_functions_from_file_to_registry
from ..io.input_parameters import PINNParameters
from ..loss.loss_term import DataLoss, PhysicsLoss
from ..nn.layers import LAYER_TYPE_REGISTRY
from ..nn.neural_network import NeuralNetwork
from ..optimizer.optimizer import OPTIMIZER_REGISTRY


class PINN:
    def __init__(self, param_file_path: str, device: torch.device):
        # Store device
        self.device = device

        # Read PINN parameters from yaml file
        self.pinn_params = PINNParameters(param_file_path)

        # Initialise modules
        self._init_geometry()  ### most general
        self._init_dataset()  ### depends on geometry
        self._init_collocation()  #### depends on geometry
        self._init_equations()
        self._init_loss()
        self._init_nn()
        self._init_learnable_parameters()
        self._init_optimizer()

    def _init_geometry(self) -> None:
        # Extract params
        self.coord_system_label = self.pinn_params.coordinates_params.coordinate_system
        coordinates = self.pinn_params.coordinates_params.coordinates
        self.coord_axes = [axis["label"] for axis in coordinates]
        self.coord_dim = len(self.coord_axes)

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
        geometric_axes_labels = self.pinn_params.geometric_axes_labels
        dim_geometry = self.pinn_params.dim_geometry

        # Set coordinate system
        self.coord_system = COORD_SYSTEM_REGISTRY[self.coord_system_label](
            dim_geometry,
            geometric_axes_labels,
            fixed_axes,
            fixed_axes_values,
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
        (
            self.input_coords,
            self.input_fields,
            self.input_consts,
        ) = load_dataset(self.pinn_params, self.device)

        # Lower and upper bounds of coords ### this is after filtering. Is that right?
        self.input_coords_lb = {
            label: torch.min(tensor) for label, tensor in self.input_coords.items()
        }
        self.input_coords_ub = {
            label: torch.max(tensor) for label, tensor in self.input_coords.items()
        }

        # Make dict based on order of alphabetically sorted coord labels
        self.sorted_coord_labels = sorted(self.input_coords)
        coord_to_index_dict = {}
        for i, label in enumerate(self.sorted_coord_labels):
            coord_to_index_dict[label] = i

        self.coord_to_index_dict = coord_to_index_dict
        self.index_to_coord_dict = {v: k for k, v in coord_to_index_dict.items()}

    def _init_collocation(self) -> None:
        # Set coords of collocation points
        self.collocation_coords = load_collocation(
            self.input_coords_lb,
            self.input_coords_ub,
            self.pinn_params,
            self.device,
        )

    def _init_equations(self) -> None:
        # Load functions to registry
        load_functions_from_file_to_registry(self.pinn_params.equations_params)

    def _init_loss(self) -> None:
        data_loss_terms_params = self.pinn_params.dataloss_params.terms
        physics_loss_terms_params = self.pinn_params.physicsloss_params.terms

        self.data_loss_terms = [
            DataLoss(data_loss_terms_params[i]["loss_function"])
            for i in range(len(data_loss_terms_params))
        ]
        self.physics_loss_terms = [
            PhysicsLoss(physics_loss_terms_params[i]["loss_function"])
            for i in range(len(physics_loss_terms_params))
        ]

    def _init_nn(self) -> None:
        # Extract params
        layers_params = self.pinn_params.layers_params
        hidden_layer_widths = layers_params.hidden_layer_widths
        layer_types = layers_params.layer_types
        self.num_layers = len(hidden_layer_widths) + 1

        ### Make this more general
        output_fields = self.pinn_params.fields_params.output_fields

        ### Currently hard code this. Only works for feed forward NN
        in_features = [self.coord_dim] + hidden_layer_widths
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

        for i in range(self.coord_dim):
            coord_label = self.index_to_coord_dict[i]
            lb_list.append(self.input_coords_lb[coord_label])
            ub_list.append(self.input_coords_ub[coord_label])

        self.model = NeuralNetwork(self.layers, layers_params, lb_list, ub_list).to(
            self.device
        )

    def _init_learnable_parameters(self) -> None:
        self.num_learnable_consts = len(self.pinn_params.constants_params.learnt_consts)
        self.learnable_consts = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, dtype=torch.float32, device=self.device))
                for _ in range(self.num_learnable_consts)
            ]
        )

    def _init_optimizer(self) -> None:
        optimizer_label = self.pinn_params.optimizer_params.optimizer
        lr = self.pinn_params.optimizer_params.learning_rate

        self.optimizer = OPTIMIZER_REGISTRY[optimizer_label](
            list(self.model.parameters()) + list(self.learnable_consts),
            lr=lr,
        )

    # Function to predict values
    def predict(
        self, coords: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Geometric axes
        geometric_axes_labels = self.pinn_params.geometric_axes_labels

        coords_list = []
        for i in range(self.coord_dim):
            coord_label = self.index_to_coord_dict[i]
            coords_list.append(coords[coord_label])

        # Make tensor
        coords_tensor = torch.stack(coords_list, dim=1)

        psi_and_p = self.model(coords_tensor)
        psi = psi_and_p[:, 0]
        p = psi_and_p[:, 1]

        operators = Operators(
            self.coord_system,
            coords,
            geometric_axes_labels,
            self.device,
        )

        u, v = EQUATION_REGISTRY["velocity_from_psi"](psi, operators)

        return u, v, p

    def compute_loss(self) -> torch.Tensor:
        # u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS()
        # loss = torch.mean((self.u - u_pred)**2) + \
        #       torch.mean((self.v - v_pred)**2) + \
        #       torch.mean(f_u_pred**2) + \
        #       torch.mean(f_v_pred**2)

        # Make tensors for model inputs
        data_coords_list = []
        collocation_coords_list = []
        for i in range(self.coord_dim):
            coord_label = self.index_to_coord_dict[i]
            data_coords_list.append(self.input_coords[coord_label])
            collocation_coords_list.append(self.collocation_coords[coord_label])

        # Make tensor
        data_coords_tensor = torch.stack(data_coords_list, dim=1)
        collocation_coords_tensor = torch.stack(collocation_coords_list, dim=1)

        # Geometric axes
        geometric_axes_labels = self.pinn_params.geometric_axes_labels

        # Define differential eqn operator objects for data and collocation points
        operators_data = Operators(
            self.coord_system,
            self.input_coords,
            geometric_axes_labels,
            self.device,
        )
        operators_co = Operators(
            self.coord_system,
            self.collocation_coords,
            geometric_axes_labels,
            self.device,
        )

        ### data coords
        psi_and_p_data = self.model(data_coords_tensor)
        psi_data = psi_and_p_data[:, 0]

        # velocity_from_psi
        u_data, v_data = EQUATION_REGISTRY["velocity_from_psi"](
            psi_data, operators_data
        )

        ### collocation coords
        psi_and_p_co = self.model(collocation_coords_tensor)
        psi_co = psi_and_p_co[:, 0]
        p_co = psi_and_p_co[:, 1]

        # velocity_from_psi
        u_co, v_co = EQUATION_REGISTRY["velocity_from_psi"](psi_co, operators_co)

        # navier_stokes
        l1 = self.learnable_consts[0]
        l2 = self.learnable_consts[1]
        residual_u, residual_v = EQUATION_REGISTRY["navier_stokes"](
            u_co, v_co, p_co, l1, l2, operators_co
        )

        loss_data = torch.mean((self.input_fields["u"] - u_data) ** 2) + torch.mean(
            (self.input_fields["v"] - v_data) ** 2
        )
        loss_phys = torch.mean(residual_u**2) + torch.mean(residual_v**2)

        # Total loss
        loss = loss_data + loss_phys
        return loss

    def train(self, nIter: int) -> None:
        self.loss_values = []
        self.l1_values = []
        self.l2_values = []
        for it in range(nIter):
            # Zero the gradients
            self.optimizer.zero_grad()
            # Compute the loss
            loss = self.compute_loss()
            # Backpropagation
            loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]
            # Update the model parameters
            self.optimizer.step()
            # Print the training progress every 500 iterations
            if it % 500 == 0:
                print(
                    f"Iter: {it}, Loss: {loss.item():.3e}, Lambda1: "
                    f"{self.learnable_consts[0]}, Lambda2: {self.learnable_consts[1]}"
                )
            self.loss_values.append(loss.item())
            self.l1_values.append(self.learnable_consts[0])
            self.l2_values.append(self.learnable_consts[1])

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "lambda_1": self.learnable_consts[0],
                "lambda_2": self.learnable_consts[1],
            },
            path,
        )

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file: '{path}'")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.lambda_1 = torch.tensor(
            checkpoint["lambda_1"], requires_grad=True, dtype=torch.float32
        )
        self.lambda_2 = torch.tensor(
            checkpoint["lambda_2"], requires_grad=True, dtype=torch.float32
        )
