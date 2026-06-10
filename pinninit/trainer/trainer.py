from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .trainer_logger import TrainerLogger
from .trainer_state import TrainerState

if TYPE_CHECKING:
    from ..data_structure.coords_dataclass import CoordsProtocol
    from ..loss.loss_term_container import LossTermContainer
    from ..nn.neural_network import NeuralNetwork
    from ..optimizer.optimizer import Optimizer
    from ..trainer_modules.trainer_module import TrainerModule


class Trainer:
    def __init__(
        self,
        model: NeuralNetwork,
        optimizer: Optimizer,
        loss_containers: dict[str, LossTermContainer],
        trainer_modules: list[TrainerModule],
        max_iter: int | None = None,
        warmup_iters: int = 0,
        print_freq: int = 500,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.trainer_modules = trainer_modules
        self.max_iter = max_iter

        # All loss containers
        self._loss_containers = loss_containers
        # Active loss containers
        self.loss_containers = {
            label: container
            for label, container in loss_containers.items()
            if container.staggering == 0
        }

        # Initialise iterations at 0
        self.iter = 0

        # TrainerState handles current state of training
        self.state = TrainerState()
        self.state.coast_until_iter = warmup_iters

        # Pass trainer to optimizer
        self.optimizer.attach_trainer(self)

        # Batch coords used for the current step
        self._current_train_batches: dict[str, CoordsProtocol] = {}
        self._current_val_batches: dict[str, CoordsProtocol] = {}

        # Best model info
        self.best_model_iter: int | None = None
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.best_loss_term_weights: dict[str, float] | None = None
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

        # Draw training batches for this step. The same batches are reused for
        # every loss evaluation within the step (e.g. L-BFGS closures).
        self._set_current_train_batches()

        # Update the model parameters
        self.optimizer.prepare_for_step()
        self.optimizer.step()

        # Draw validation batches and calculate validation losses at end of step
        self._set_current_val_batches()
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

    def _set_current_train_batches(self) -> None:
        """Set the batch coords each active loss container uses this step."""
        if self.optimizer.requires_full_batch:
            self._current_train_batches = {
                label: container.train_eval_context.coords
                for label, container in self.loss_containers.items()
            }
        else:
            self._current_train_batches = {
                label: container.next_train_batch()
                for label, container in self.loss_containers.items()
            }

    def _set_current_val_batches(self) -> None:
        """Set the validation batch coords each active loss container uses this step."""
        if self.optimizer.requires_full_batch:
            self._current_val_batches = {
                label: container.val_eval_context.coords
                for label, container in self.loss_containers.items()
            }
        else:
            self._current_val_batches = {
                label: container.next_val_batch()
                for label, container in self.loss_containers.items()
            }

    def compute_train_loss(self) -> torch.Tensor:
        # New containers may have been activated since batches were drawn
        for label, container in self.loss_containers.items():
            if label not in self._current_train_batches:
                self._current_train_batches[label] = container.next_train_batch()

        #  Compute training loss on the current batches
        unweighted_train_losses = {
            label: container.compute_train_batch_loss(
                self.model, self._current_train_batches[label], detach=False
            )
            for label, container in self.loss_containers.items()
        }

        # Get loss term weights
        loss_term_weights = {
            label: container.loss_term_weight
            for label, container in self.loss_containers.items()
        }

        # Log losses and weights
        self.logger.record_train_losses(unweighted_train_losses)
        self.logger.record_loss_term_weights(loss_term_weights)

        # Compute weighted training loss
        weighted_train_losses = [
            weight * loss
            for weight, loss in zip(
                loss_term_weights.values(),
                unweighted_train_losses.values(),
                strict=True,
            )
        ]
        train_loss = torch.stack(weighted_train_losses).sum()

        return train_loss

    def compute_val_loss(self) -> torch.Tensor:
        # New containers may have been activated since batches were drawn
        for label, container in self.loss_containers.items():
            if label not in self._current_val_batches:
                self._current_val_batches[label] = container.next_val_batch()

        #  Compute validation loss on the current batches
        unweighted_val_losses = {
            label: container.compute_val_batch_loss(
                self.model, self._current_val_batches[label]
            )
            for label, container in self.loss_containers.items()
        }

        # Get loss term weights
        loss_term_weights = {
            label: container.loss_term_weight
            for label, container in self.loss_containers.items()
        }

        # Log losses and weights
        self.logger.record_val_losses(unweighted_val_losses)
        self.logger.record_loss_term_weights(loss_term_weights)

        # Compute weighted validation loss
        weighted_val_losses = [
            weight * loss
            for weight, loss in zip(
                loss_term_weights.values(),
                unweighted_val_losses.values(),
                strict=True,
            )
        ]
        val_loss = torch.stack(weighted_val_losses).sum()

        return val_loss

    @property
    def weighted_avg_val_loss(self) -> torch.Tensor:
        loss_term_weight_sum = sum(
            container.loss_term_weight for container in self.loss_containers.values()
        )

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
            # Save loss term weights
            self.best_loss_term_weights = {
                label: container.loss_term_weight
                for label, container in self.loss_containers.items()
            }

    def load_best_model(
        self,
    ) -> None:
        """Loads the best model state."""
        if (
            self.best_model_state is not None
            and self.best_loss_term_weights is not None
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

            # Restore loss term weights
            for label, weight in self.best_loss_term_weights.items():
                self.loss_containers[label].loss_term_weight = weight

            # Notify modules of updated state
            self.state.state_updated = True
        else:
            raise ValueError("No best model state to load.")

    def reset_best_model(self) -> None:
        self.best_model_iter = None
        self.best_model_state = None
        self.best_loss_term_weights = None
        self.best_weighted_avg_val_loss = None

        # Log event
        self.logger.log_event("reset_best_model", self.iter)
