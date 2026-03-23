from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from ..data_structure.coords_dataclass import make_coords_dataclass
from ..geometry.coordinate_system import COORD_SYSTEM_REGISTRY
from ..geometry.domain import Domain
from ..io.input_data import load_coords, load_fields
from ..io.input_equations import load_functions_from_file_to_registry
from ..io.input_parameters import PINNParameters
from ..loss.loss_term import LossTermLabels
from ..loss.loss_term_group import LossTermGroup
from ..nn.layers import LAYER_TYPE_REGISTRY
from ..nn.neural_network import NeuralNetwork
from ..optimizer.optimizer_registry import OPTIMIZER_REGISTRY
from ..trainer.trainer import Trainer
from ..trainer_modules.adaptive_weights_grad_balance import AdaptiveWeightsGradBalance
from ..trainer_modules.adaptive_weights_W21 import AdaptiveWeightsW21
from ..trainer_modules.early_stopping import EarlyStopping


class PINN:
    def __init__(self, param_file_path: str, device: torch.device):
        # Store device
        self.device = device

        # Set default dtype to higher precision
        torch.set_default_dtype(torch.float64)

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

        # Initialise domains
        self.domains: dict[str, Domain] = {}

        for domain in self.pinn_params.coordinates_params.domains:
            domain_label = domain["domain_label"]

            # Convert YAML coordinate list to dict[str, (min, max)]
            bounds_dict: dict[str, tuple[float | None, float | None]] = {}

            for coord in domain.get("coordinates", []):
                label = coord["label"]
                lb = coord.get("min", None)
                ub = coord.get("max", None)
                bounds_dict[label] = (lb, ub)

            self.domains[domain_label] = Domain(
                coord_system=self.coord_system,
                geometric_bounds=bounds_dict,
                device=self.device,
            )

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
        loss_terms_params = self.pinn_params.loss_terms_params.terms

        for loss_term in loss_terms_params:
            # Get training and validation coords
            loss_term_label = loss_term["loss_term_label"]
            dataset_label = loss_term["data_set_label"]
            domain_label = loss_term["domain_label"]

            domain = self.domains[domain_label]
            coords = self.dataset_coords[dataset_label]
            train_coord = self.Coords(
                coord_system=self.coord_system, domain=domain, **coords["training"]
            )
            val_coord = self.Coords(
                coord_system=self.coord_system, domain=domain, **coords["validation"]
            )

            # Get training and validation fields
            if dataset_label in self.dataset_fields:
                fields = self.dataset_fields[dataset_label]
                train_fields = fields["training"]
                val_fields = fields["validation"]
            else:
                train_fields = None
                val_fields = None

            # Init loss term group weight
            weight_method_label = loss_term["weight_method"]
            weight_group_label = loss_term["weight_group"]
            initial_weight = torch.tensor(
                loss_term["weight_value"],
                device=self.device,
                requires_grad=False,
            )
            staggering = loss_term.get("staggering", 0)

            batch_size_train = getattr(
                self.pinn_params.training_params, "batch_size_train", None
            )
            batch_size_val = getattr(
                self.pinn_params.training_params, "batch_size_val", None
            )
            shuffle_train = getattr(
                self.pinn_params.training_params, "shuffle_train", True
            )
            shuffle_val = getattr(
                self.pinn_params.training_params, "shuffle_val", False
            )
            num_workers_train = getattr(
                self.pinn_params.training_params, "num_workers_train", 0
            )
            num_workers_val = getattr(
                self.pinn_params.training_params, "num_workers_val", 0
            )

            # Create loss term group
            self.loss_term_groups[loss_term_label] = LossTermGroup(
                train_coords=train_coord,
                dim_geometry=self.dim_geometry,
                available_fields_params=self.pinn_params.fields_params,
                available_equations_params=self.pinn_params.equations_params,
                device=self.device,
                train_input_fields=train_fields,
                val_coords=val_coord,
                val_input_fields=val_fields,
                weight_method_label=weight_method_label,
                weight_group_label=weight_group_label,
                initial_weight=initial_weight,
                batch_size_train=batch_size_train,
                batch_size_val=batch_size_val,
                shuffle_train=shuffle_train,
                shuffle_val=shuffle_val,
                num_workers_train=num_workers_train,
                num_workers_val=num_workers_val,
                staggering=staggering,
            )

            # Initialise loss terms in the group
            self.loss_term_groups[loss_term_label].init_loss_term_and_add(
                loss_term, self.pinn_params.fields_params
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
        ### Hardcode this for now. Won't work if internal coord dim is different
        ### e.g. spherical shell embedded in 3D cartesian
        internal_coord_dim = self.coord_dim
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
                torch.zeros(1, device=self.device)
            )

        self.learnt_consts = learnt_consts

    def _init_optimizer(self) -> None:
        optimizer_label = self.pinn_params.optimizer_params.optimizer
        lr = self.pinn_params.optimizer_params.learning_rate

        self.optimizer = OPTIMIZER_REGISTRY[optimizer_label](
            list(self.model.parameters()) + list(self.learnt_consts.values()),
            lr=lr,
        )

    # Function to evaluate values at coords
    def evaluate_model(self, coords: dict[str, torch.Tensor]) -> Any:
        # Set to eval mode
        self.model.eval()

        # Initialise coord dataclass object
        test_coords = self.Coords(coord_system=self.coord_system, **coords)
        with torch.no_grad():
            output = self.model(test_coords.internal).detach()

        return output

    def evaluate_losses(
        self,
        coords: dict[str, torch.Tensor],
        fields_dict: dict[str, dict[str, Any]] | None = None,
        batch_size: int | None = None,
        shuffle: bool = False,
        num_workers: int = 0,
    ) -> dict[str, float]:
        # Set to eval mode
        self.model.eval()

        test_losses = {}
        test_coords = self.Coords(coord_system=self.coord_system, **coords)
        for loss_term_label in self.loss_term_groups:
            loss_term_group = self.loss_term_groups[loss_term_label]
            test_losses[loss_term_label] = (
                loss_term_group.compute_unweighted_loss_at_coords(
                    self.model,
                    test_coords,
                    fields_dict,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                ).item()
            )
        return test_losses

    def evaluate_residuals_per_point(
        self,
        coords: dict[str, torch.Tensor],
        fields_dict: dict[str, dict[str, Any]] | None = None,
        batch_size: int | None = None,
        shuffle: bool = False,
        num_workers: int = 0,
    ) -> dict[str, dict[str, torch.Tensor]]:
        self.model.eval()
        test_coords = self.Coords(coord_system=self.coord_system, **coords)

        residuals_dict: dict[str, dict[str, torch.Tensor]] = {}

        for label, group in self.loss_term_groups.items():
            # Pass the per-group input fields if provided
            res = group.compute_residuals_at_coords(
                self.model,
                test_coords,
                fields_dict,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )

            # Detach all tensors in the returned dict
            residuals_dict[label] = {k: v.detach() for k, v in res.items()}

        return residuals_dict

    def train(self) -> None:
        ### Hardcode these for now
        trainer_modules: list[Any] = []

        """
        ### Update optimizer
        optimizer_label = OptimizerLabels.LBFGS
        new_optimizer = OPTIMIZER_REGISTRY[optimizer_label](
            list(self.model.parameters()) + list(self.learnt_consts.values()),
            line_search_fn="strong_wolfe",
            ### Currently just use default lr etc.
        )
        patience = 500
        frac_tol = 1e-2
        min_iter = 0
        coast_iters_after_update = 500
        update_optimizer = UpdateOptimizer(
            new_optimizer=new_optimizer,
            patience=patience,
            frac_tol=frac_tol,
            min_iter=min_iter,
            coast_iters_after_update=coast_iters_after_update,
        )
        trainer_modules.append(update_optimizer)
        """

        ### Early stopping
        patience = 500
        frac_tol = 0  # 1e-2
        min_iter = 0
        early_stopping = EarlyStopping(
            patience=patience, frac_tol=frac_tol, min_iter=min_iter
        )
        trainer_modules.append(early_stopping)
        """
        ### Loss term staggering
        patience = 500
        frac_tol = 1e-2
        min_iter = 0
        coast_iters_after_update = 1000
        loss_term_staggering = LossTermStaggering(
            patience=patience,
            frac_tol=frac_tol,
            min_iter=min_iter,
            coast_iters_after_update=coast_iters_after_update,
        )
        trainer_modules.append(loss_term_staggering)
        """
        """
        ### R3 resampling
        n_resample_events = 10
        n_sample_size = 60
        sampler = VolumeRejectionSampler(coord_system=self.coord_system,
        domain = self.domains["phys_domain"], device = self.device,
        seed=12345, with_sobol=True)
        patience = 200
        frac_tol = 1e-2
        coast_iters_after_update = 100

        r3_D23 = R3D23(
            n_sample_size=n_sample_size,
            n_resample_events=n_resample_events,
            sampler=sampler,
            patience=patience,
            frac_tol=frac_tol,
            coast_iters_after_update=coast_iters_after_update,
        )
        trainer_modules.append(r3_D23)
        """

        ### Hardcode this for now
        if hasattr(self.pinn_params.loss_terms_params, "weight_methods"):
            for weight_method in self.pinn_params.loss_terms_params.weight_methods:
                if weight_method["label"] == "adaptive_W21":
                    reference_weight_group_label = weight_method[
                        "reference_weight_group"
                    ]
                    alpha = weight_method["alpha"]
                    update_frequency = weight_method.get("update_frequency", 1)
                    min_iter = weight_method.get("min_iter", 0)

                    adaptive_weights_W21 = AdaptiveWeightsW21(
                        reference_weight_group_label=reference_weight_group_label,
                        alpha=alpha,
                        update_frequency=update_frequency,
                        min_iter=min_iter,
                    )
                    trainer_modules.append(adaptive_weights_W21)

                if weight_method["label"] == "adaptive_grad_balance":
                    alpha = weight_method["alpha"]
                    update_frequency = weight_method.get("update_frequency", 1)
                    min_iter = weight_method.get("min_iter", 0)

                    adaptive_weights_grad_balance = AdaptiveWeightsGradBalance(
                        alpha=alpha,
                        update_frequency=update_frequency,
                        min_iter=min_iter,
                    )
                    trainer_modules.append(adaptive_weights_grad_balance)

        # Init trainer
        max_iter = getattr(self.pinn_params.training_params, "max_epochs", None)
        warmup_iters = 1000
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_groups=self.loss_term_groups,
            trainer_modules=trainer_modules,
            max_iter=max_iter,
            warmup_iters=warmup_iters,
        )

        # Train model
        self.trainer.train()

    def save(self, path: str) -> None:
        trainer = self.trainer
        logger = self.trainer.logger
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "pre_step_train_losses_log": logger.pre_step_train_losses_log,
                "post_step_val_losses_log": logger.post_step_val_losses_log,
                "loss_term_weights_log": logger.loss_term_weights_log,
                "events": logger.events,
                "best_model_iter": trainer.best_model_iter,
                "best_weighted_avg_val_loss": trainer.best_weighted_avg_val_loss,
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
        self.pre_step_train_losses_log = checkpoint["pre_step_train_losses_log"]
        self.post_step_val_losses_log = checkpoint["post_step_val_losses_log"]
        self.loss_term_weights_log = checkpoint["loss_term_weights_log"]
        self.events = checkpoint["events"]
        self.best_model_iter = checkpoint["best_model_iter"]
        self.best_weighted_avg_val_loss = checkpoint.get("best_weighted_avg_val_loss")

    ###  self.loss_values = checkpoint.get("loss_values", [])
    ###  self.data_loss_values = checkpoint.get("data_loss_values", [])
    ###  self.phys_loss_values = checkpoint.get("phys_loss_values", [])
    ###  self.val_loss_values = checkpoint.get("val_loss_values", [])
    ###  self.test_loss_values = checkpoint.get("test_loss_values", [])
    ###  self.rms_residual_per_test_point = checkpoint.get(
    ###      "rms_residual_per_test_point", []
    ###  )

    # self.lambda_1 = torch.tensor(
    #     checkpoint["lambda_1"], requires_grad=True
    # )
    # self.lambda_2 = torch.tensor(
    #     checkpoint["lambda_2"], requires_grad=True
    # )
