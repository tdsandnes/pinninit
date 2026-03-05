from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .trainer_logger import TrainerLogger
from .trainer_state import TrainerState

if TYPE_CHECKING:
    from ..loss.loss_term_group import LossTermGroup
    from ..nn.neural_network import NeuralNetwork
    from ..optimizer.optimizer import Optimizer
    from ..trainer_modules.trainer_module import TrainerModule


class Trainer:
    def __init__(
        self,
        model: NeuralNetwork,
        optimizer: Optimizer,
        loss_groups: dict[str, LossTermGroup],
        trainer_modules: list[TrainerModule],
        max_iter: int | None = None,
        warmup_iters: int = 0,
        print_freq: int = 500,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.trainer_modules = trainer_modules
        self.max_iter = max_iter

        # All loss groups
        self._loss_groups = loss_groups
        # Active loss groups
        self.loss_groups = {
            label: group
            for label, group in loss_groups.items()
            if group.staggering == 0
        }

        # Initialise iterations at 0
        self.iter = 0

        # TrainerState handles current state of training
        self.state = TrainerState()
        self.state.coast_until_iter = warmup_iters

        # Pass trainer to optimizer
        self.optimizer.attach_trainer(self)

        # Best model info
        self.best_model_iter: int | None = None
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.best_loss_group_weights: dict[str, torch.Tensor] | None = None
        self.best_weighted_avg_val_loss: torch.Tensor | None = None

        # Logger
        self.logger = TrainerLogger(self, print_freq=print_freq)

    def step(self) -> None:
        # Update trainer state for current step
        self.state.on_step_start(self, self.logger)

        # Execute all trainer modules at start of step
        for module in self.trainer_modules:
            if not self.state.is_coasting or module.runs_during_coast:
                module.on_step_start(self, self.logger)

        # Update logger for current step
        self.logger.on_step_start()

        # Update the model parameters
        self.optimizer.prepare_for_step()
        self.optimizer.step()

        # Calculate validation losses at end of step
        self.post_step_val_loss = self.compute_val_loss()

        # Update best model if needed
        if not self.state.is_coasting:
            self.update_best_model()

        # Log values
        self.logger.log_values()

        # Execute all trainer modules at end of step
        for module in self.trainer_modules:
            if not self.state.is_coasting or module.runs_during_coast:
                module.on_step_end(self, self.logger)

        # Notify all modules of any state change
        if self.state.state_updated:
            for module in self.trainer_modules:
                module.on_state_change(self, self.logger)

        # Update trainer state for next step
        self.state.on_step_end(self, self.logger)

    def train(self) -> None:
        # Execute all trainer modules at start of training
        for module in self.trainer_modules:
            module.on_train_start(self, self.logger)

        # Print initial state to terminal
        self.logger.print_losses_and_weights(self, "initial state")

        # Main training loop
        while not self.state.is_ready_to_stop:
            # Update iteration count
            self.iter += 1

            # Step the trainer
            self.step()

            # Print to terminal
            self.logger.print_after_step(self)

            # Check max iterations
            if self.max_iter is not None and self.iter >= self.max_iter:
                self.logger.log_event("max_iter_reached", self.iter)
                break

        # If while loop is broken because of early stop
        else:
            self.logger.log_event("training_stopped_early", self.iter)

        # Restore best model
        self.load_best_model()

        # Execute all trainer modules at end of training
        for module in self.trainer_modules:
            module.on_train_end(self, self.logger)

    def compute_train_loss(self) -> torch.Tensor:
        #  Compute training loss
        group_unweighted_train_losses = {
            label: group.compute_unweighted_loss_train(self.model)
            for label, group in self.loss_groups.items()
        }

        # Get loss term weights
        loss_term_weights = {
            label: group.weight for label, group in self.loss_groups.items()
        }

        # Log losses and weights
        self.logger.record_train_losses(group_unweighted_train_losses)
        self.logger.record_loss_term_weights(loss_term_weights)

        # Compute weighted training loss
        group_weighted_train_losses = [
            weight * loss
            for weight, loss in zip(
                loss_term_weights.values(),
                group_unweighted_train_losses.values(),
                strict=True,
            )
        ]
        train_loss = torch.stack(group_weighted_train_losses).sum()

        return train_loss

    def compute_val_loss(self) -> torch.Tensor:
        #  Compute validation loss
        group_unweighted_val_losses = {
            label: group.compute_unweighted_loss_val(self.model)
            for label, group in self.loss_groups.items()
        }

        # Get loss term weights
        loss_term_weights = {
            label: group.weight for label, group in self.loss_groups.items()
        }

        # Log losses and weights
        self.logger.record_val_losses(group_unweighted_val_losses)
        self.logger.record_loss_term_weights(loss_term_weights)

        # Compute weighted training loss
        group_weighted_val_losses = [
            weight * loss
            for weight, loss in zip(
                loss_term_weights.values(),
                group_unweighted_val_losses.values(),
                strict=True,
            )
        ]
        val_loss = torch.stack(group_weighted_val_losses).sum()

        return val_loss

    @property
    def weighted_avg_val_loss(self) -> torch.Tensor:
        loss_term_weight_sum = torch.stack(
            [group.weight.detach() for group in self.loss_groups.values()]
        ).sum()

        return self.post_step_val_loss.detach() / loss_term_weight_sum

    def update_best_model(
        self,
    ) -> None:
        """Updates best model state if validation loss is lower than previous best."""
        current_weighted_avg_val_loss = self.weighted_avg_val_loss

        if (
            self.best_weighted_avg_val_loss is None
            or current_weighted_avg_val_loss < self.best_weighted_avg_val_loss
        ):
            self.best_weighted_avg_val_loss = current_weighted_avg_val_loss
            self.best_model_iter = self.iter
            self.best_model_state = {
                k: v.detach().clone() for k, v in self.model.state_dict().items()
            }
            # Save loss term group weights
            self.best_loss_group_weights = {
                label: group.weight for label, group in self.loss_groups.items()
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
                f"Loading best model on iteration {self.iter}: "
                f"best model iter = {self.best_model_iter}, "
                f"weighted_avg_val_loss = {self.best_weighted_avg_val_loss})"
            )

            # Log event
            self.logger.log_event("load_best_model", self.iter)

            # Load
            self.model.load_state_dict(self.best_model_state)

            # Restore loss group weights
            for label, weight in self.best_loss_group_weights.items():
                group = self.loss_groups[label]
                self.loss_groups[label].weight = weight.clone().to(group.weight.device)

            # Notify modules of updated state
            self.state.state_updated = True
        else:
            raise ValueError("No best model state to load.")

    def reset_best_model(self) -> None:
        self.best_model_iter = None
        self.best_model_state = None
        self.best_loss_group_weights = None
        self.best_weighted_avg_val_loss = None

        # Log event
        self.logger.log_event("reset_best_model", self.iter)
