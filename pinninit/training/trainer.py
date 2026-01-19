from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from typing import Any

    from ..loss.loss_term_group import LossTermGroup
    from ..nn.neural_network import NeuralNetwork
    from ..optimizer.optimizer import Optimizer


class TrainerModule(ABC):
    @abstractmethod
    def runs_during_coast(self, trainer: Trainer) -> bool:
        pass

    @abstractmethod
    def on_train_start(self, trainer: Trainer) -> None:
        pass

    @abstractmethod
    def on_train_end(self, trainer: Trainer) -> None:
        pass

    @abstractmethod
    def on_step_start(self, trainer: Trainer) -> None:
        pass

    @abstractmethod
    def on_step_end(self, trainer: Trainer) -> None:
        pass

    @abstractmethod
    def on_state_change(self, trainer: Trainer) -> None:
        """Called when another module changes the trainer state
        (e.g. optimizer updated)."""
        pass


class Trainer:
    def __init__(
        self,
        model: NeuralNetwork,
        optimizer: Optimizer,
        loss_groups: dict[str, LossTermGroup],
        trainer_modules: list[TrainerModule],
        max_iter: int | None = None,
        warmup_iters: int = 0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_groups = loss_groups
        self.trainer_modules = trainer_modules
        self.max_iter = max_iter
        self.coast_until_iter = warmup_iters

        # Pass trainer to optimizer
        self.optimizer.attach_trainer(self)

        # Initialise training state
        self.iter: int = 0
        self.pre_step_train_loss_history: list[list[float]] = []
        self.pre_step_val_loss_history: list[list[float]] = []
        self.best_weighted_average_val_loss: float | None = None
        self.best_model_iter: int | None = None
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.best_loss_group_weights: dict[str, torch.Tensor] | None = None
        self._stop_requested: bool = False
        self._stop_barriers: dict[str, bool] = {}
        self._state_updated = False
        self.state_updates: list[dict[str, Any]] = []
        self.was_prev_step_coast: bool | None = None

        # Registry to keep track of the first call of a method in a step
        self._first_call_in_step_registry = {
            "compute_train_loss": True,
            "compute_val_loss": True,
        }

        # Log loss term weights
        self.pre_step_loss_weight_history: dict[str, list[float]] = {}
        for group_label in self.loss_groups:
            self.pre_step_loss_weight_history[group_label] = []

    @property
    def is_coasting(self) -> bool:
        """Check if trainer is in coast phase (modules should not intervene)."""
        return self.iter <= self.coast_until_iter

    @property
    def is_ready_to_stop(
        self,
    ) -> bool:
        """Checks if all stop barriers have been met."""
        return all(self._stop_barriers.values()) and self._stop_requested

    def _init_stop_barrier(self, name: str) -> None:
        """Adds a stop barrier with the given name."""
        self._stop_barriers[name] = False

    def _update_stop_barrier(self, name: str, status: bool) -> None:
        """Updates the status of the stop barrier with the given name."""
        if name in self._stop_barriers:
            self._stop_barriers[name] = status
        else:
            raise KeyError(f"Stop barrier '{name}' not found.")

    def _prepare_for_step(self) -> None:
        # Store pre-step validation loss
        self.compute_val_loss()

        # Log weights
        for group_label, group in self.loss_groups.items():
            self.pre_step_loss_weight_history[group_label].append(group.weight.item())

        # Set batching of loss term groups if needed
        if self.optimizer.requires_full_batch:
            for group in self.loss_groups.values():
                group.set_batching(batch_size_train=None, batch_size_val=None)
        else:
            for group in self.loss_groups.values():
                if not group.is_default_batching:
                    group.reset_to_default_batching()

        # Prepare optimizer for step
        self.optimizer.prepare_for_step()

    def _end_step(self) -> None:
        # Validation loss at end of step
        self.post_step_val_loss = self.compute_val_loss()

        # Update best model if needed
        if not self.is_coasting:
            self.update_best_model()

        # Update coast state of previous step
        self.was_prev_step_coast = self.is_coasting

    def _check_coast_transition(self) -> None:
        if self.was_prev_step_coast is None:
            return

        # Entering coast
        if not self.was_prev_step_coast and self.is_coasting:
            print(f"Coasting started on iteration {self.iter}")
            self.state_updates.append(
                {
                    "event": "enter_coast",
                    "iter": self.iter,
                    "weighted_average_val_loss": self.weighted_average_val_loss,
                }
            )
            for module in self.trainer_modules:
                module.on_state_change(self)

        # Exiting coast
        elif self.was_prev_step_coast and not self.is_coasting:
            print(f"Coasting ended on iteration {self.iter}")
            self.state_updates.append(
                {
                    "event": "exit_coast",
                    "iter": self.iter,
                    "weighted_average_val_loss": self.weighted_average_val_loss,
                }
            )
            for module in self.trainer_modules:
                module.on_state_change(self)

    def step(self) -> None:
        # Check whether we are exiting coast
        self._check_coast_transition()

        # Execute all trainer modules at start of step
        for module in self.trainer_modules:
            if not self.is_coasting or module.runs_during_coast(self):
                module.on_step_start(self)

        # Reset registry keeping track of first calls in a step
        for key in self._first_call_in_step_registry:
            self._first_call_in_step_registry[key] = True

        # Update the model parameters
        self._prepare_for_step()
        self.optimizer.step()
        self._end_step()

        # Execute all trainer modules at end of step
        for module in self.trainer_modules:
            if not self.is_coasting or module.runs_during_coast(self):
                module.on_step_end(self)

        # Notify all modules of any state change
        if self._state_updated:
            for module in self.trainer_modules:
                if not self.is_coasting or module.runs_during_coast(self):
                    module.on_state_change(self)

            # Reset _stop_requested
            self._stop_requested = False

            # Reset _state_updated
            self._state_updated = False

    def train(self) -> None:
        # Execute all trainer modules at start of training
        for module in self.trainer_modules:
            module.on_train_start(self)

        self.iter = 1
        while not self.is_ready_to_stop:
            # Step the trainer
            self.step()

            self.print_status()

            # Update iteration count and check max iterations
            self.iter += 1
            if self.max_iter is not None and self.iter >= self.max_iter:
                self.state_updates.append(
                    {
                        "event": "stop_training_max_iter",
                        "iter": self.iter,
                        "weighted_average_val_loss": self.weighted_average_val_loss,
                    }
                )
                break
        # If while loop is broken because of early stop
        else:
            self.state_updates.append(
                {
                    "event": "stop_training_early_stop",
                    "iter": self.iter,
                    "weighted_average_val_loss": self.weighted_average_val_loss,
                }
            )

        # Restore best model
        self.load_best_model()

        # Execute all trainer modules at end of training
        for module in self.trainer_modules:
            module.on_train_end(self)

    def compute_train_loss(self) -> torch.Tensor:
        #  Compute training loss
        group_unweighted_train_losses = [
            group.compute_unweighted_loss_train(self.model)
            for group in self.loss_groups.values()
        ]
        group_weighted_train_losses = [
            group.weight * loss
            for group, loss in zip(
                self.loss_groups.values(), group_unweighted_train_losses, strict=True
            )
        ]
        train_loss = torch.stack(group_weighted_train_losses).sum()

        # Store training loss if this is the first time computating it in a step
        key = self.compute_train_loss.__name__
        if self._first_call_in_step_registry[key]:
            self.pre_step_unweighted_group_train_losses = [
                loss.detach() for loss in group_unweighted_train_losses
            ]
            self.pre_step_train_loss = train_loss.detach()
            self.pre_step_train_loss_history.append(
                [loss.item() for loss in self.pre_step_unweighted_group_train_losses]
            )
            self._first_call_in_step_registry[key] = False

        return train_loss

    def compute_val_loss(self) -> torch.Tensor:
        #  Compute validation loss
        group_unweighted_val_losses = [
            group.compute_unweighted_loss_val(self.model)
            for group in self.loss_groups.values()
        ]
        group_weighted_val_losses = [
            group.weight * loss
            for group, loss in zip(
                self.loss_groups.values(), group_unweighted_val_losses, strict=True
            )
        ]
        val_loss = torch.stack(group_weighted_val_losses).sum()

        # Store validation loss if this is the first time computating it in a step
        key = self.compute_val_loss.__name__
        if self._first_call_in_step_registry[key]:
            self.pre_step_unweighted_group_val_losses = [
                loss.detach() for loss in group_unweighted_val_losses
            ]
            self.pre_step_val_loss = val_loss.detach()
            self.pre_step_val_loss_history.append(
                [loss.item() for loss in self.pre_step_unweighted_group_val_losses]
            )
            self._first_call_in_step_registry[key] = False

        return val_loss

    @property
    def sum_of_loss_term_weights(self) -> torch.Tensor:
        return torch.stack([group.weight for group in self.loss_groups.values()]).sum()

    @property
    def weighted_average_val_loss(self) -> torch.Tensor:
        return self.post_step_val_loss / self.sum_of_loss_term_weights

    def update_best_model(
        self,
    ) -> None:
        """Updates best model state if validation loss is lower than previous best."""
        current_weighted_average_val_loss = self.weighted_average_val_loss.item()

        if (
            self.best_weighted_average_val_loss is None
            or current_weighted_average_val_loss < self.best_weighted_average_val_loss
        ):
            self.best_weighted_average_val_loss = current_weighted_average_val_loss
            self.best_model_iter = self.iter
            self.best_model_state = {
                k: v.detach().clone() for k, v in self.model.state_dict().items()
            }
            # Save loss term group weights
            self.best_loss_group_weights = {
                label: group.weight.detach().clone()
                for label, group in self.loss_groups.items()
            }

    def load_best_model(
        self,
    ) -> None:
        """Loads the best model state."""
        if (
            self.best_model_state is not None
            and self.best_loss_group_weights is not None
        ):
            print(
                f"Loading best model, corresponding to iter {self.best_model_iter}, "
                f"on iteration {self.iter}"
            )
            self.state_updates.append(
                {
                    "event": "load_best_model",
                    "iter": self.iter,
                    "weighted_average_val_loss": self.best_weighted_average_val_loss,
                }
            )
            self.model.load_state_dict(self.best_model_state)
            # Restore loss group weights
            for label, weight in self.best_loss_group_weights.items():
                group = self.loss_groups[label]
                self.loss_groups[label].weight = weight.clone().to(group.weight.device)

            # Notify modules of updated state
            self._state_updated = True
            for module in self.trainer_modules:
                module.on_state_change(self)
            self._state_updated = False
        else:
            raise ValueError("No best model state to load.")

    ###    def update_optimizer(self,):
    ### Directs to a polymorphic UpdateOptimizer class in a update_optimizer.py

    ###    def update_learning_rate(self,):
    ### Directs to a polymorphic UpdateLearningRate class in a update_learning_rate.py

    ###    def update_loss_term_weights(self,):
    ##  polymorphic UpdateLossTermWeights class in a update_loss_term_weights.py

    ##    def early_stopping(self,):
    ### Directs to a polymorphic EarlyStopping class in a early_stopping.py

    def print_status(self, verbosity: int = 1) -> None:
        if (self.iter - 1) % 500 == 0:
            ### temp

            train_losses = [
                loss.item() for loss in self.pre_step_unweighted_group_train_losses
            ]
            val_losses = [
                loss.item() for loss in self.pre_step_unweighted_group_val_losses
            ]
            loss_weights = [
                weights[-1] for weights in self.pre_step_loss_weight_history.values()
            ]
            plateau_quantity = self.weighted_average_val_loss

            ### Note (self.iter - 1) here because quantities are pre-step
            iter_str = (
                "Initial state"
                if (self.iter - 1) == 0
                else f"State after iter {(self.iter - 1)}"
            )
            print(
                "-------------------------------------------------------------------\n"
                f"{iter_str}: \n"
                "-------------------------------------------------------------------\n"
                f"Training Loss: \n"
                f"- weighted sum:               {self.pre_step_train_loss}, \n"
                f"- unweighted contributions:   {train_losses}, \n"
                f"Validation Loss: \n"
                f"- weighted sum:               {self.pre_step_val_loss}, \n"
                f"- unweighted contributions:   {val_losses}, \n"
                f"Loss group weights:           {loss_weights}, \n"
                f"Plateau detection quantity:   {plateau_quantity},"
            )
