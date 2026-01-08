from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..loss.loss_term_group import LossTermGroup
    from ..nn.neural_network import NeuralNetwork
    from ..optimizer.optimizer import Optimizer


class TrainerModule(ABC):
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
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_groups = loss_groups
        self.trainer_modules = trainer_modules
        self.max_iter = max_iter

        # Pass trainer to optimizer
        self.optimizer.attach_trainer(self)

        # Initialise training state
        self.iter: int = 0
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.best_val_loss: float | None = None
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self._stop_requested: bool = False
        self._stop_barriers: dict[str, bool] = {}
        self._state_updated = False

        # Registry to keep track of the first call of a method in a step
        self._first_call_in_step_registry = {
            name: True
            for name, attr in self.__class__.__dict__.items()
            if callable(attr)
        }

    def ready_to_stop(
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
        # Set batching of loss term groups if needed
        if self.optimizer.requires_full_batch:
            for group in self.loss_groups.values():
                group.set_batching(batch_size=None)
        else:
            for group in self.loss_groups.values():
                if not group.is_default_batching:
                    group.reset_to_default_batching()

        # Prepare optimizer for step
        self.optimizer.prepare_for_step()

    def step(self) -> None:
        # Execute all trainer modules at start of step
        for module in self.trainer_modules:
            module.on_step_start(self)

        # Reset registry keeping track of first calls in a step
        for key in self._first_call_in_step_registry:
            self._first_call_in_step_registry[key] = True

        # Store pre-step validation loss
        self.compute_val_loss()

        # Update the model parameters
        self._prepare_for_step()
        self.optimizer.step()

        # Update best model if needed
        self.update_best_model()

        # Execute all trainer modules at end of step
        for module in self.trainer_modules:
            module.on_step_end(self)

        # Notify all modules of any state change
        if self._state_updated:
            for module in self.trainer_modules:
                module.on_state_change(self)

            # Reset _stop_requested
            self._stop_requested = False

            # Reset _state_updated
            self._state_updated = False

    def train(self) -> None:
        # Execute all trainer modules at start of training
        for module in self.trainer_modules:
            module.on_train_start(self)

        self.iter = 0
        while not self.ready_to_stop():
            # Step the trainer
            self.step()

            self.print_status()

            # Update iteration count and check max iterations
            self.iter += 1
            if self.max_iter is not None and self.iter >= self.max_iter:
                break

        # Execute all trainer modules at end of training
        for module in self.trainer_modules:
            module.on_train_end(self)

    def compute_train_loss(self) -> torch.Tensor:
        #  Compute training loss
        group_train_losses = [
            group.compute_loss_train(self.model) for group in self.loss_groups.values()
        ]
        train_loss = torch.stack(group_train_losses).sum()

        # Store training loss if this is the first time computating it in a step
        key = self.compute_train_loss.__name__
        if self._first_call_in_step_registry[key]:
            self.group_train_losses = [loss.detach() for loss in group_train_losses]
            self.train_loss = train_loss.detach()
            self.train_loss_history.append(self.train_loss.item())
            self._first_call_in_step_registry[key] = False

        return train_loss

    def compute_val_loss(self) -> torch.Tensor:
        #  Compute validation loss
        group_val_losses = [
            group.compute_loss_val(self.model) for group in self.loss_groups.values()
        ]
        val_loss = torch.stack(group_val_losses).sum()

        # Store validation loss if this is the first time computating it in a step
        key = self.compute_val_loss.__name__
        if self._first_call_in_step_registry[key]:
            self.group_val_losses = [loss.detach() for loss in group_val_losses]
            self.val_loss = val_loss.detach()
            self.val_loss_history.append(self.val_loss.item())
            self._first_call_in_step_registry[key] = False

        return val_loss

    def update_best_model(
        self,
    ) -> None:
        """Updates best model state if validation loss is lower than previous best."""
        current_val_loss = self.val_loss.item()

        if self.best_val_loss is None or current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_model_state = {
                k: v.detach().clone() for k, v in self.model.state_dict().items()
            }

    def load_best_model(
        self,
    ) -> None:
        """Loads the best model state."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
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
        if self.iter % 500 == 0:
            ### temp

            train_losses = [loss.item() for loss in self.group_train_losses]
            val_losses = [loss.item() for loss in self.group_val_losses]

            print(
                f"Iter: {self.iter}, "
                f"Training Loss: {train_losses}, "
                f"Val Loss: {val_losses}, "
            )
