from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ..loss.loss_term_container import LossTermContainer
from .trainer_logger import TrainerLogger
from .trainer_state import TrainerState

if TYPE_CHECKING:
    from typing import Any

    from ..data_structure.coords import Coords
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
        max_epoch: int | None = None,
        warmup_epochs: int = 0,
        print_freq: int = 1,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.trainer_modules = trainer_modules
        self.max_epoch = max_epoch

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

        # Epoch tracking
        self.epoch = 0
        self._steps_in_epoch = 0

        # TrainerState handles current state of training
        self.state = TrainerState()
        self.state.coast_until_epoch = warmup_epochs

        # Pass trainer to optimizer
        self.optimizer.attach_trainer(self)

        # Batch coords used for the current step
        self._current_train_batches: dict[str, Coords] = {}

        # Best model info
        self.best_model_epoch: int | None = None
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.best_geometric_mean_val_loss: float | None = None
        self.best_loss_term_weights: dict[str, float] | None = None
        self.best_point_weights: dict[str, torch.Tensor] | None = None

        # Restored state from checkpoint (if any)
        self._restored_state: dict[str, Any] | None = None

        # Quantities cached until the model parameters change
        self._full_val_losses_cache: dict[str, float] | None = None
        self._train_residuals_cache: dict[str, torch.Tensor] | None = None
        self._total_train_loss_cache: float | None = None

        # Logger
        self.logger = TrainerLogger(self, print_freq=print_freq)

    def step(self) -> None:
        # Model params changed last step so cached quantities are stale
        self._clear_caches()

        # Whether this step completes an epoch (full pass over the training data)
        self._steps_in_epoch += 1
        is_epoch_end = self._steps_in_epoch >= self._steps_per_epoch()

        # Update trainer state for current step
        self.state.on_step_start(self)

        # Execute all trainer modules at start of step
        for module in self.trainer_modules:
            if not self.state.is_coasting or module.runs_during_coast:
                module.on_step_start(self, self.logger)

        # Update logger for current step
        self.logger.on_step_start()

        # Draw training batches for this step. The same batches are reused for
        # every loss evaluation within the step (e.g. L-BFGS closures).
        self._set_current_train_batches()

        # Prepare optimizer for step
        self.optimizer.prepare_for_step()

        # Trainer module operations that require self.optimizer.prepare_for_step()
        for module in self.trainer_modules:
            if not self.state.is_coasting or module.runs_during_coast:
                module.on_pre_optimizer_step(self, self.logger)

        # Update the model parameters
        self.optimizer.step()

        # Log values
        self.logger.accumulate_train_losses()

        # Execute all trainer modules at end of step
        for module in self.trainer_modules:
            if not self.state.is_coasting or module.runs_during_coast:
                module.on_step_end(self, self.logger)

        # Execute epoch-end hooks when a full pass over the data completes
        if is_epoch_end:
            # Roll the counter first, so hooks see the number of completed
            # epochs and iter == epoch * steps_per_epoch at epoch boundaries
            self.epoch += 1
            self._steps_in_epoch = 0
            for container in self.loss_containers.values():
                container.reset_train_batches()

            # Track the best model on the full-set validation loss
            if not self.state.is_coasting:
                self.update_best_model()
            for module in self.trainer_modules:
                if not self.state.is_coasting or module.runs_during_coast:
                    module.on_epoch_end(self, self.logger)
            self.logger.on_epoch_end()

        # Notify all modules of any state change
        if self.state.state_updated:
            # A structural change restarts the epoch count
            self._steps_in_epoch = 0
            for container in self.loss_containers.values():
                container.reset_train_batches()
            for module in self.trainer_modules:
                module.on_state_change(self, self.logger)

        # Update trainer state for next step
        self.state.on_step_end()

    def train(self) -> None:
        # Execute all trainer modules at start of training
        for module in self.trainer_modules:
            module.on_train_start(self, self.logger)

        # Restore state from checkpoint if available
        if self._restored_state is not None:
            self._apply_restored_state(self._restored_state)
            self._restored_state = None

        # Print initial state to terminal
        self.logger.print_losses_and_weights("initial state")

        # Main training loop
        while not self.state.is_ready_to_stop:
            # Update iteration count
            self.iter += 1

            # Step the trainer
            self.step()

            # Check max epoch
            if self.max_epoch is not None and self.epoch >= self.max_epoch:
                self.logger.log_event("max_epoch_reached")
                break

        else:
            # Reached only when the while condition becomes False, i.e. all stop
            # barriers were met and stop was requested (not via the max_epoch break)
            self.logger.log_event("training_stopped_early")

        # Restore best model
        if self.best_model_state is not None:
            self.load_best_model()
        else:
            self.logger.log_event("no_best_model_to_load")

        # Execute all trainer modules at end of training
        for module in self.trainer_modules:
            module.on_train_end(self, self.logger)

    def _steps_per_epoch(self) -> int:
        """Steps for every active container to cycle its full training set once."""
        if self.optimizer.requires_full_batch:
            return 1
        return max(
            (
                len(container.train_eval_context.batches)
                for container in self.loss_containers.values()
            ),
            default=1,
        )

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

    def _clear_caches(self) -> None:
        self._full_val_losses_cache = None
        self._train_residuals_cache = None
        self._total_train_loss_cache = None

    def compute_train_loss(self) -> torch.Tensor:
        """Weighted sum of the loss terms on this step's training batches."""
        # Compute per-term losses and cache this step's residuals
        losses = {}
        self._train_residuals_cache = {}
        for label, container in self.loss_containers.items():
            loss, residuals = container.compute_train_batch_loss_and_residuals(
                self.model, self._current_train_batches[label], detach=False
            )
            losses[label] = loss
            self._train_residuals_cache[label] = residuals

        # Record for the epoch-mean train loss log
        self.logger.record_train_losses(losses)

        total = torch.stack(
            [
                container.loss_term_weight * losses[label]
                for label, container in self.loss_containers.items()
            ]
        ).sum()
        self._total_train_loss_cache = float(total.detach())
        return total

    def cached_train_residuals(self, label: str) -> torch.Tensor | None:
        """Residuals cached during this step's training-loss computation."""
        if self._train_residuals_cache is None:
            return None
        return self._train_residuals_cache.get(label)

    def compute_full_unweighted_val_losses(self) -> dict[str, float]:
        """Unweighted val loss of each active container over its full val set."""
        if self._full_val_losses_cache is None:
            self._full_val_losses_cache = {
                label: container.compute_full_val_loss(self.model).item()
                for label, container in self.loss_containers.items()
            }
        return self._full_val_losses_cache

    def cached_total_train_loss(self) -> float | None:
        return self._total_train_loss_cache

    @property
    def full_geometric_mean_val_loss(self) -> float:
        """Geometric mean of the full-set val losses of the active containers."""
        losses = self.compute_full_unweighted_val_losses().values()
        # The geometric mean is 0 if any loss is 0 (where log fails)
        if min(losses) == 0:
            return 0.0
        return math.exp(sum(math.log(loss) for loss in losses) / len(losses))

    @property
    def all_loss_containers(self) -> dict[str, LossTermContainer]:
        """All loss containers, including ones not yet activated by staggering."""
        return self._loss_containers

    @property
    def current_train_batches(self) -> dict[str, Coords]:
        """Batch coords used by each active loss container in the current step."""
        return self._current_train_batches

    def update_best_model(self) -> None:
        """Updates best model state if validation loss is lower than previous best."""
        current = self.full_geometric_mean_val_loss

        if (
            self.best_geometric_mean_val_loss is None
            or current < self.best_geometric_mean_val_loss
        ):
            self.best_geometric_mean_val_loss = current
            self.best_model_epoch = self.epoch
            self.best_model_state = {
                k: v.detach().clone() for k, v in self.model.state_dict().items()
            }
            # Save loss term weights
            self.best_loss_term_weights = {
                label: container.loss_term_weight
                for label, container in self.loss_containers.items()
            }

            # Save point weights for containers that use them
            self.best_point_weights = {
                label: container.train_eval_context.get_raw_point_weights().clone()
                for label, container in self.loss_containers.items()
                if container.adaptive_point_weights
            }

    def load_best_model(self) -> None:
        """Loads the best model state, with its loss term and point weights."""
        if (
            self.best_model_state is None
            or self.best_loss_term_weights is None
            or self.best_point_weights is None
        ):
            raise ValueError("No best model state to load.")

        print(
            f"Loading best model on epoch {self.epoch}: "
            f"best model epoch = {self.best_model_epoch}, "
            f"geometric_mean_val_loss = {self.best_geometric_mean_val_loss}"
        )
        self.logger.log_event("load_best_model")

        self.model.load_state_dict(self.best_model_state)
        self._clear_caches()
        for label, weight in self.best_loss_term_weights.items():
            self.loss_containers[label].loss_term_weight = weight
        for label, weights in self.best_point_weights.items():
            self.loss_containers[label].train_eval_context.update_point_weights(weights)

        # Notify modules of updated state
        self.state.state_updated = True

    def reset_best_model(self) -> None:
        self.best_model_epoch = None
        self.best_model_state = None
        self.best_geometric_mean_val_loss = None
        self.best_loss_term_weights = None
        self.best_point_weights = None

        # Log event
        self.logger.log_event("reset_best_model")

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "iter": self.iter,
            "loss_term_weights": {
                label: container.loss_term_weight
                for label, container in self._loss_containers.items()
            },
            "point_weights": {
                label: container.train_eval_context.get_raw_point_weights().cpu()
                for label, container in self._loss_containers.items()
                if container.adaptive_point_weights
            },
            # Checkpoints are saved at the end of an epoch, when every
            # container starts a new pass over its batches
            "batches": {
                label: container.train_eval_context.batches.state_dict()
                for label, container in self._loss_containers.items()
            },
            "best_model_epoch": self.best_model_epoch,
            "best_geometric_mean_val_loss": self.best_geometric_mean_val_loss,
            "best_model_state": self.best_model_state,
            "best_loss_term_weights": self.best_loss_term_weights,
            "best_point_weights": self.best_point_weights,
            "trainer_state": self.state.state_dict(),
            "optimizer_state_dict": self.optimizer.optimizer.state_dict(),
            "modules": {
                f"{i}:{type(module).__name__}": module.state_dict()
                for i, module in enumerate(self.trainer_modules)
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        device = next(self.model.parameters()).device
        self.epoch = state["epoch"]
        self.iter = state["iter"]
        self.best_model_epoch = state["best_model_epoch"]
        self.best_geometric_mean_val_loss = state["best_geometric_mean_val_loss"]
        if state["best_model_state"] is not None:
            self.best_model_state = {
                k: v.to(device) for k, v in state["best_model_state"].items()
            }
        self.best_loss_term_weights = state["best_loss_term_weights"]
        if state["best_point_weights"] is not None:
            self.best_point_weights = {
                label: w.to(device) for label, w in state["best_point_weights"].items()
            }
        self._restored_state = state

    def _apply_restored_state(self, state: dict[str, Any]) -> None:
        """Restore the rest of a checkpoint, once the trainer modules have
        started training (e.g. registered their stop barriers)."""
        device = next(self.model.parameters()).device

        # Module states
        module_keys = [
            f"{i}:{type(module).__name__}"
            for i, module in enumerate(self.trainer_modules)
        ]
        if list(state["modules"]) != module_keys:
            raise ValueError(
                f"Saved trainer modules {list(state['modules'])} don't match "
                f"the trainer modules {module_keys}"
            )
        for key, module in zip(module_keys, self.trainer_modules, strict=True):
            module.load_state_dict(state["modules"][key], self)

        # Loss term weights
        for label, weight in state["loss_term_weights"].items():
            self._loss_containers[label].loss_term_weight = weight

        # Point weights
        for label, weights in state["point_weights"].items():
            self._loss_containers[label].train_eval_context.update_point_weights(
                weights.to(device)
            )

        # Shuffling of the training batches
        for label, batches_state in state["batches"].items():
            self._loss_containers[label].train_eval_context.batches.load_state_dict(
                batches_state
            )

        # Trainer state
        self.state.load_state_dict(state["trainer_state"])

        # Optimizer state
        self.optimizer.optimizer.load_state_dict(state["optimizer_state_dict"])
