from __future__ import annotations

import torch
import torch.nn as nn

from ..collocation.collocation import load_collocation
from ..dataset.dataset import load_dataset
from ..geometry.coordinate_system import COORD_SYSTEM_REGISTRY
from ..geometry.domain import extract_domain_bounds
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
        coord_system_label = self.pinn_params.geometry_params.coordinate_system
        spatial_dimensions = self.pinn_params.geometry_params.spatial_dimensions
        coord_dim = spatial_dimensions + 1  # add time dim
        coord_axes = self.pinn_params.geometry_params.coordinates
        if self.pinn_params.geometry_params.fixed_coordinates is not None:
            fixed_axes = self.pinn_params.geometry_params.fixed_coordinates[
                "coordinate"
            ]
            fixed_axes_values = self.pinn_params.geometry_params.fixed_coordinates[
                "value"
            ]
        else:
            fixed_axes = None
            fixed_axes_values = None

        # Set coordinate system
        self.coord_system = execute_from_registry(
            coord_system_label,
            COORD_SYSTEM_REGISTRY,
            coord_dim,
            coord_axes,
            fixed_axes,
            fixed_axes_values,
        )

        ### Keep these as lists rather than tensors for now because of None
        self.domain_lb, self.domain_ub = extract_domain_bounds(
            self.coord_system, self.pinn_params
        )

        ### Do boundary conditions here

    def _init_dataset(self) -> None:
        # Load datasets
        self.input_coords, self.input_fields, self.input_consts = load_dataset(
            self.pinn_params, self.device
        )

        # Lower and upper bounds of coords ### this is after filtering. Is that right?
        self.input_coords_lb = torch.min(self.input_coords, dim=0).values.detach()
        self.input_coords_ub = torch.max(self.input_coords, dim=0).values.detach()

    def _init_collocation(self) -> None:
        # Set coords of collocation points
        self.collocation_coords = load_collocation(
            self.coord_system,
            self.input_coords_lb,
            self.input_coords_ub,
            self.pinn_params,
            self.device,
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
        hidden_layer_widths = self.pinn_params.layers_params.hidden_layer_widths
        layer_types = self.pinn_params.layers_params.layer_types
        activation_functions = self.pinn_params.layers_params.activation_functions
        init_weights = self.pinn_params.layers_params.init_weights
        init_biases = self.pinn_params.layers_params.init_biases

        ### Make this more general where it's not just the input coords etc.
        input_coords = self.pinn_params.geometry_params.coordinates
        output_fields = self.pinn_params.fields_params.output_fields

        ### Currently hard code this. Only works for feed forward NN
        in_features = [len(input_coords)] + hidden_layer_widths
        out_features = hidden_layer_widths + [len(output_fields)]

        num_layers = len(hidden_layer_widths) + 1
        self.layers = [
            execute_from_registry(
                layer_types[i],
                LAYER_TYPE_REGISTRY,
                in_features=in_features[i],
                out_features=out_features[i],
                activation_func_label=activation_functions[i],
                weight_init_label=init_weights[i],
                bias_init_label=init_biases[i],
            )
            for i in range(num_layers)
        ]

        self.model = NeuralNetwork(
            self.layers, self.input_coords_lb, self.input_coords_ub
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
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]

        grad_psi = self.coord_system.grad(psi, coords)

        u = grad_psi[:, 2]
        v = -grad_psi[:, 1]

        return u, v, p

    def compute_loss(self) -> torch.Tensor:
        # u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS()
        # loss = torch.mean((self.u - u_pred)**2) + \
        #       torch.mean((self.v - v_pred)**2) + \
        #       torch.mean(f_u_pred**2) + \
        #       torch.mean(f_v_pred**2)

        coords = self.input_coords
        ### HARDCODED FOR NOW
        psi_and_p = self.model(coords)
        psi = psi_and_p[:, 0]
        p = psi_and_p[:, 1]

        grad_psi = self.coord_system.grad(psi, coords, self.device)

        u = grad_psi[:, 2]
        v = -grad_psi[:, 1]

        grad_u = self.coord_system.grad(u, coords, self.device)
        grad_v = self.coord_system.grad(v, coords, self.device)
        ut = grad_u[:, 0]
        ux = grad_u[:, 1]
        uy = grad_u[:, 2]
        vt = grad_v[:, 0]
        vx = grad_v[:, 1]
        vy = grad_v[:, 2]

        ### Maybe nicer way of doing this in gra where we don't calculate grads we don't
        ### need. Although we need all of coords for metric tensor
        uxx = self.coord_system.grad(ux, coords, self.device)[:, 1]
        uyy = self.coord_system.grad(uy, coords, self.device)[:, 2]
        vxx = self.coord_system.grad(vx, coords, self.device)[:, 1]
        vyy = self.coord_system.grad(vy, coords, self.device)[:, 2]

        grad_p = self.coord_system.grad(p, coords, self.device)
        px = grad_p[:, 1]
        py = grad_p[:, 2]

        # Navier-Stokes residuals
        f_u = (
            ut
            + self.learnable_consts[0] * (u * ux + v * uy)
            + px
            - self.learnable_consts[1] * (uxx + uyy)
        )
        f_v = (
            vt
            + self.learnable_consts[0] * (u * vx + v * vy)
            + py
            - self.learnable_consts[1] * (vxx + vyy)
        )

        loss_data = torch.mean((self.input_fields[:, 0] - u) ** 2) + torch.mean(
            (self.input_fields[:, 1] - v) ** 2
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
