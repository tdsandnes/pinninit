from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..collocation.collocation import load_collocation
from ..dataset.dataset import load_dataset
from ..differential_equations.operators import Operators
from ..geometry.coordinate_system import COORD_SYSTEM_REGISTRY
from ..io.input_equations import EQUATION_REGISTRY, load_functions_from_file_to_registry
from ..io.input_parameters import EquationsParams, PINNParameters
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
        equations_params = self.pinn_params.equations_params

        # init data loss terms
        data_loss_terms = []
        all_data_loss_input_eqns_needed = set()
        all_data_loss_output_eqns_needed = set()
        for term in data_loss_terms_params:
            data_loss_term = DataLoss(
                term,
                self.pinn_params.input_field_labels,
                self.pinn_params.output_field_labels,
                self.pinn_params.derived_input_field_eqn_dict,
                self.pinn_params.derived_output_field_eqn_dict,
            )
            data_loss_terms.append(data_loss_term)

            # Add equations to sets
            all_data_loss_input_eqns_needed.update(data_loss_term.input_eqns_needed)
            all_data_loss_output_eqns_needed.update(data_loss_term.output_eqns_needed)

        # Store
        self.data_loss_terms = data_loss_terms
        self.all_data_loss_input_eqns_needed = all_data_loss_input_eqns_needed
        self.all_data_loss_output_eqns_needed = all_data_loss_output_eqns_needed

        # init phsyics loss terms
        physics_loss_terms = []
        all_physics_loss_eqns_needed = set()
        for term in physics_loss_terms_params:
            physics_loss_term = PhysicsLoss(
                term,
                self.pinn_params.output_field_labels,
                self.pinn_params.input_const_labels,
                self.pinn_params.learnt_const_labels,
                self.pinn_params.coord_axes_labels,
                self.pinn_params.derived_output_field_eqn_dict,
                equations_params,
            )
            physics_loss_terms.append(physics_loss_term)

            # Add equations to sets
            all_physics_loss_eqns_needed.update(physics_loss_term.eqns_needed)

        # Store
        self.physics_loss_terms = physics_loss_terms
        self.all_physics_loss_eqns_needed = all_physics_loss_eqns_needed

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

    def _init_learnt_parameters(self) -> None:
        learnt_consts = {}
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

    def compute_data_loss(
        self, coords: dict[str, torch.Tensor], equations_params: EquationsParams
    ) -> torch.Tensor:
        # Define differential eqn operator objects for coords
        operators = Operators(
            self.coord_system,
            coords,
            self.pinn_params.geometric_axes_labels,
            self.device,
        )

        # Make tensors for model inputs
        coords_list = []

        for i in range(len(coords)):
            coord_label = self.index_to_coord_dict[i]
            coords_list.append(coords[coord_label])

        # Make tensor
        coords_tensor = torch.stack(coords_list, dim=1)

        # Get model output
        output_fields_tuple = self.model(coords_tensor)

        # Make dict of model output consistently assigning output value to sorted label
        sorted_output_field_labels = sorted(self.pinn_params.output_field_labels)
        output_fields = {}
        for i, label in enumerate(sorted_output_field_labels):
            output_fields[label] = output_fields_tuple[:, i]

        # Apply any equations needed to get data loss
        all_data_loss_input_eqns_needed = self.all_data_loss_input_eqns_needed
        all_data_loss_output_eqns_needed = self.all_data_loss_output_eqns_needed
        equations_by_label = {eq["label"]: eq for eq in equations_params.equations}

        # Get derived input fields
        derived_input_fields: dict[str, torch.Tensor] = {}
        eqns_to_calculate = all_data_loss_input_eqns_needed.copy()

        # Loop until all equations have been evaluated
        while len(eqns_to_calculate) > 0:
            evaluated_new = set()
            for eqn in list(eqns_to_calculate):
                eqn_input_labels = equations_by_label[eqn]["input_args"]
                eqn_output_labels = equations_by_label[eqn]["output_args"]
                eqn_inputs = []

                # Check whether we have access to all input args of eqn
                can_evaluate = True
                for label in eqn_input_labels:
                    if label in self.pinn_params.input_field_labels:
                        eqn_inputs.append(self.input_fields[label])
                    elif label in self.pinn_params.input_const_labels:
                        eqn_inputs.append(self.input_consts[label])
                    elif label in self.pinn_params.coord_axes_labels:
                        eqn_inputs.append(coords[label])
                    elif label in derived_input_fields:
                        eqn_inputs.append(derived_input_fields[label])
                    else:
                        can_evaluate = False
                        break

                if can_evaluate:
                    # Evaluate eqn and store all output values
                    eqn_outputs = EQUATION_REGISTRY[eqn](*eqn_inputs, operators)
                    for i, eqn_output in enumerate(eqn_outputs):
                        derived_input_fields[eqn_output_labels[i]] = eqn_output

                    evaluated_new.add(eqn)

            # Remove any evaluated eqns from the set of eqns left to evaluate
            eqns_to_calculate -= evaluated_new

            # Raise error if we loop through all equations and didn't evaluate a new
            # equation
            if not evaluated_new:
                raise KeyError(
                    f"Unable to evaluate equations {eqns_to_calculate} for calculation "
                    "of target component of data loss."
                )

        # Get derived output fields
        derived_output_fields: dict[str, torch.Tensor] = {}
        eqns_to_calculate = all_data_loss_output_eqns_needed.copy()

        # Loop until all equations have been evaluated
        while len(eqns_to_calculate) > 0:
            evaluated_new = set()
            for eqn in list(eqns_to_calculate):
                eqn_input_labels = equations_by_label[eqn]["input_args"]
                eqn_output_labels = equations_by_label[eqn]["output_args"]
                eqn_inputs = []

                # Check whether we have access to all input args of eqn
                can_evaluate = True
                for label in eqn_input_labels:
                    if label in self.pinn_params.output_field_labels:
                        eqn_inputs.append(output_fields[label])
                    elif label in self.pinn_params.input_const_labels:
                        eqn_inputs.append(self.input_consts[label])
                    elif label in self.pinn_params.learnt_const_labels:
                        eqn_inputs.append(self.learnt_consts[label])
                    elif label in self.pinn_params.coord_axes_labels:
                        eqn_inputs.append(coords[label])
                    elif label in derived_output_fields:
                        eqn_inputs.append(derived_output_fields[label])
                    else:
                        can_evaluate = False
                        break

                if can_evaluate:
                    # Evaluate eqn and store all output values
                    eqn_outputs = EQUATION_REGISTRY[eqn](*eqn_inputs, operators)
                    for i, eqn_output in enumerate(eqn_outputs):
                        derived_output_fields[eqn_output_labels[i]] = eqn_output

                    evaluated_new.add(eqn)

            # Remove any evaluated eqns from the set of eqns left to evaluate
            eqns_to_calculate -= evaluated_new

            # Raise error if we loop through all equations and didn't evaluate a new
            # equation
            if not evaluated_new:
                raise KeyError(
                    f"Unable to evaluate equations {eqns_to_calculate} for calculation "
                    "of predicted component of data loss."
                )

        data_loss = torch.tensor(0.0, device=self.device)
        for loss_term in self.data_loss_terms:
            data_loss += loss_term.compute_loss(
                self.input_fields,
                output_fields,
                derived_input_fields,
                derived_output_fields,
            )

        return data_loss

    def compute_physics_loss(
        self, coords: dict[str, torch.Tensor], equations_params: EquationsParams
    ) -> torch.Tensor:
        # Define differential eqn operator objects for coords
        operators = Operators(
            self.coord_system,
            coords,
            self.pinn_params.geometric_axes_labels,
            self.device,
        )

        # Make tensors for model inputs
        coords_list = []

        for i in range(len(coords)):
            coord_label = self.index_to_coord_dict[i]
            coords_list.append(coords[coord_label])

        # Make tensor
        coords_tensor = torch.stack(coords_list, dim=1)

        # Get model output
        output_fields_tuple = self.model(coords_tensor)

        # Make dict of model output consistently assigning output value to sorted label
        sorted_output_field_labels = sorted(self.pinn_params.output_field_labels)
        output_fields = {}
        for i, label in enumerate(sorted_output_field_labels):
            output_fields[label] = output_fields_tuple[:, i]

        # Apply any equations needed to get physics loss
        all_physics_loss_eqns_needed = self.all_physics_loss_eqns_needed
        equations_by_label = {eq["label"]: eq for eq in equations_params.equations}

        # Get derived output fields
        derived_output_fields: dict[str, torch.Tensor] = {}
        eqns_to_calculate = all_physics_loss_eqns_needed.copy()

        # Loop until all equations have been evaluated
        while len(eqns_to_calculate) > 0:
            evaluated_new = set()
            for eqn in list(eqns_to_calculate):
                eqn_input_labels = equations_by_label[eqn]["input_args"]
                eqn_output_labels = equations_by_label[eqn]["output_args"]
                eqn_inputs = []

                # Check whether we have access to all input args of eqn
                can_evaluate = True
                for label in eqn_input_labels:
                    if label in self.pinn_params.output_field_labels:
                        eqn_inputs.append(output_fields[label])
                    elif label in self.pinn_params.input_const_labels:
                        eqn_inputs.append(self.input_consts[label])
                    elif label in self.pinn_params.learnt_const_labels:
                        eqn_inputs.append(self.learnt_consts[label])
                    elif label in self.pinn_params.coord_axes_labels:
                        eqn_inputs.append(coords[label])
                    elif label in derived_output_fields:
                        eqn_inputs.append(derived_output_fields[label])
                    else:
                        can_evaluate = False
                        break

                if can_evaluate:
                    # Evaluate eqn and store all output values
                    eqn_outputs = EQUATION_REGISTRY[eqn](*eqn_inputs, operators)
                    for i, eqn_output in enumerate(eqn_outputs):
                        derived_output_fields[eqn_output_labels[i]] = eqn_output

                    evaluated_new.add(eqn)

            # Remove any evaluated eqns from the set of eqns left to evaluate
            eqns_to_calculate -= evaluated_new

            # Raise error if we loop through all equations and didn't evaluate a new
            # equation
            if not evaluated_new:
                raise KeyError(
                    f"Unable to evaluate equations {eqns_to_calculate} for calculation "
                    "of physics loss."
                )

        physics_loss = torch.tensor(0.0, device=self.device)
        for loss_term in self.physics_loss_terms:
            physics_loss += loss_term.compute_loss(output_fields, derived_output_fields)

        return physics_loss

    def compute_loss(self) -> torch.Tensor:
        loss_data = self.compute_data_loss(
            self.input_coords, self.pinn_params.equations_params
        )
        loss_phys = self.compute_physics_loss(
            self.collocation_coords, self.pinn_params.equations_params
        )

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
                l1 = self.learnt_consts["l1"]
                l2 = self.learnt_consts["l2"]
                print(
                    f"Iter: {it}, Loss: {loss.item():.3e}, Lambda1: "
                    f"{l1}, Lambda2: {l2}"
                )
            self.loss_values.append(loss.item())
            self.l1_values.append(self.learnt_consts["l1"])
            self.l2_values.append(self.learnt_consts["l2"])

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "lambda_1": self.learnt_consts["l1"],
                "lambda_2": self.learnt_consts["l2"],
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
