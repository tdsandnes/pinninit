from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..collocation.collocation import load_collocation
from ..dataset.dataset import load_dataset
from ..geometry.coordinate_system import COORD_SYSTEM_REGISTRY
from ..io.input_parameters import PINNParameters
from ..loss.loss_term import DataLoss, PhysicsLoss
from ..nn.layers import LAYER_TYPE_REGISTRY
from ..nn.neural_network import NeuralNetwork
from ..optimizer.optimizer import OPTIMIZER_REGISTRY
from ..utils import execute_from_registry


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

        # Geometric axes
        self.geometric_axes = [
            axis for axis in coordinates if axis.get("is_geometric", 0) == 1
        ]
        self.geometric_axes_labels = [axis["label"] for axis in self.geometric_axes]
        self.dim_geometry = len(self.geometric_axes_labels)

        # Non-geometric axes
        self.nongeometric_axes = [
            axis for axis in coordinates if axis.get("is_geometric", 0) == 0
        ]
        self.nongeometric_axes_labels = [
            axis["label"] for axis in self.nongeometric_axes
        ]
        self.dim_nongeometry = len(self.nongeometric_axes_labels)

        # Fixed axes
        self.fixed_axes = []
        self.fixed_axes_values = []
        for axis in coordinates:
            if axis.get("is_fixed", 0) == 1:
                if "fixed_value" in axis:
                    self.fixed_axes.append(axis["label"])
                    self.fixed_axes_values.append(axis["fixed_value"])
                else:
                    raise ValueError(
                        f"Axis '{axis['label']}' is marked as fixed but no "
                        "'fixed_value' is provided."
                    )

        # Set coordinate system
        self.coord_system = execute_from_registry(
            self.coord_system_label,
            COORD_SYSTEM_REGISTRY,
            self.dim_geometry,
            self.geometric_axes_labels,
            self.fixed_axes,
            self.fixed_axes_values,
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
            self.input_coords_nongeometric,
            self.input_coords_geometric,
            self.input_fields,
            self.input_consts,
        ) = load_dataset(self.pinn_params, self.device)

        self.input_coords = torch.cat(
            [self.input_coords_nongeometric, self.input_coords_geometric], dim=1
        )

        # Lower and upper bounds of coords ### this is after filtering. Is that right?
        self.input_coords_lb = torch.min(self.input_coords, dim=0).values.detach()
        self.input_coords_ub = torch.max(self.input_coords, dim=0).values.detach()

    def _init_collocation(self) -> None:
        # Set coords of collocation points
        self.collocation_coords_nongeometric, self.collocation_coords_geometric = (
            load_collocation(
                self.dim_nongeometry,
                self.dim_geometry,
                self.input_coords_lb,
                self.input_coords_ub,
                self.pinn_params,
                self.device,
            )
        )

        self.collocation_coords = torch.cat(
            [self.collocation_coords_nongeometric, self.collocation_coords_geometric],
            dim=1,
        )

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
            execute_from_registry(
                layer_types[i],
                LAYER_TYPE_REGISTRY,
                in_features=in_features[i],
                out_features=out_features[i],
            )
            for i in range(self.num_layers)
        ]

        self.model = NeuralNetwork(
            self.layers, layers_params, self.input_coords_lb, self.input_coords_ub
        ).to(self.device)

    def _init_learnable_parameters(self) -> None:
        self.num_learnable_consts = len(self.pinn_params.constants_params.learnt_consts)
        self.learnable_consts = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, dtype=torch.float32, device=self.device))
                for _ in range(self.num_learnable_consts)
            ]
        )

    def _init_optimizer(self) -> None:
        optimizer = self.pinn_params.optimizer_params.optimizer
        lr = self.pinn_params.optimizer_params.learning_rate

        self.optimizer = execute_from_registry(
            optimizer,
            OPTIMIZER_REGISTRY,
            list(self.model.parameters()) + list(self.learnable_consts),
            lr=lr,
        )

    # Function to predict values
    def predict(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ### HARDCODED FOR NOW
        psi_and_p = self.model(coords)
        psi = psi_and_p[:, 0]
        p = psi_and_p[:, 1]

        grad_psi = self.coord_system.grad(psi, coords, self.device)

        u = grad_psi[:, 1]
        v = -grad_psi[:, 0]

        return u, v, p

    def compute_loss(self) -> torch.Tensor:
        # u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS()
        # loss = torch.mean((self.u - u_pred)**2) + \
        #       torch.mean((self.v - v_pred)**2) + \
        #       torch.mean(f_u_pred**2) + \
        #       torch.mean(f_v_pred**2)

        data_coords = self.input_coords
        data_coords_geometric = self.input_coords_geometric
        collocation_coords = self.collocation_coords
        collocation_coords_nongeometric = self.collocation_coords_nongeometric
        collocation_coords_geometric = self.collocation_coords_geometric
        ### HARDCODED FOR NOW

        ### data coords
        psi_and_p_data = self.model(data_coords)
        psi_data = psi_and_p_data[:, 0]

        grad_psi_data = self.coord_system.grad(
            psi_data, data_coords_geometric, self.device
        )

        u_data = grad_psi_data[:, 1]
        v_data = -grad_psi_data[:, 0]

        ### collocation coords
        psi_and_p_co = self.model(collocation_coords)
        psi_co = psi_and_p_co[:, 0]
        p_co = psi_and_p_co[:, 1]

        grad_psi_co = self.coord_system.grad(
            psi_co, collocation_coords_geometric, self.device
        )

        u_co = grad_psi_co[:, 1]
        v_co = -grad_psi_co[:, 0]

        grad_u_co = self.coord_system.grad(
            u_co, collocation_coords_geometric, self.device
        )
        grad_v_co = self.coord_system.grad(
            v_co, collocation_coords_geometric, self.device
        )
        ut = torch.autograd.grad(
            u_co,
            collocation_coords_nongeometric,
            torch.ones_like(u_co),
            retain_graph=True,
            create_graph=True,
        )[0][:, 0]
        ux = grad_u_co[:, 0]
        uy = grad_u_co[:, 1]
        vt = torch.autograd.grad(
            v_co,
            collocation_coords_nongeometric,
            torch.ones_like(v_co),
            retain_graph=True,
            create_graph=True,
        )[0][:, 0]
        vx = grad_v_co[:, 0]
        vy = grad_v_co[:, 1]

        # Laplacians
        laplacian_u = self.coord_system.laplacian(
            u_co, collocation_coords_geometric, self.device
        )
        laplacian_v = self.coord_system.laplacian(
            v_co, collocation_coords_geometric, self.device
        )

        grad_p_co = self.coord_system.grad(
            p_co, collocation_coords_geometric, self.device
        )
        px = grad_p_co[:, 0]
        py = grad_p_co[:, 1]

        # Navier-Stokes residuals
        f_u = (
            ut
            + self.learnable_consts[0] * (u_co * ux + v_co * uy)
            + px
            - self.learnable_consts[1] * laplacian_u
        )
        f_v = (
            vt
            + self.learnable_consts[0] * (u_co * vx + v_co * vy)
            + py
            - self.learnable_consts[1] * laplacian_v
        )

        loss_data = torch.mean((self.input_fields[:, 0] - u_data) ** 2) + torch.mean(
            (self.input_fields[:, 1] - v_data) ** 2
        )
        loss_phys = torch.mean(f_u**2) + torch.mean(f_v**2)

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

    def inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.input_coords
        x = self.input_coords
        y = self.input_coords

        u = self.input_fields
        v = self.input_fields

        return t, x, y, u, v
