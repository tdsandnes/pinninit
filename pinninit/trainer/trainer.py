from __future__ import annotations

import copy
import math
import os
import signal
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..loss.loss_term_container import LossTermContainer
from .trainer_logger import TrainerLogger
from .trainer_state import TrainerState

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import FrameType
    from typing import Any

    from ..data_structure.coords import Coords
    from ..nn.neural_network import NeuralNetwork
    from ..optimizer.optimizer import Optimizer
    from ..trainer_modules.trainer_module import TrainerModule


def changed_parameters(old: Any, new: Any, path: str = "") -> list[str]:
    """The parameters that differ between old and new (nested dicts and lists,
    e.g. a parameter file's contents), as 'path: old -> new'."""
    if isinstance(old, dict) and isinstance(new, dict):
        changes = []
        for key in [*old, *(key for key in new if key not in old)]:
            key_path = f"{path}.{key}" if path else str(key)
            changes += changed_parameters(old.get(key), new.get(key), key_path)
        return changes
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [
            change
            for i, (old_item, new_item) in enumerate(zip(old, new, strict=True))
            for change in changed_parameters(old_item, new_item, f"{path}[{i}]")
        ]
    return [] if old == new else [f"{path}: {old!r} -> {new!r}"]


def geometric_mean(values: Iterable[float]) -> float:
    """Geometric mean of the values, 0 if any is 0 (where log fails)."""
    values = list(values)
    if min(values) == 0:
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


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
        log_point_weight_info: bool = False,
        point_weight_snapshot_freq: int = 0,
        checkpoint_path: str | Path | None = None,
        checkpoint_freq: int = 100,
        stop_on_sigterm: bool = False,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        if max_epoch is None and not any(
            module.stops_training for module in trainer_modules
        ):
            raise ValueError(
                "Training would never end: set max_epoch or add a trainer module "
                "that stops training (e.g. early stopping)."
            )
        self.model = model
        self.optimizer = optimizer
        self.trainer_modules = trainer_modules
        self.max_epoch = max_epoch

        # Save a checkpoint every checkpoint_freq epochs and when training ends
        self.checkpoint_path = checkpoint_path
        self.checkpoint_freq = checkpoint_freq
        # Stop at the end of the epoch on SIGTERM (e.g. sent by an HPC scheduler
        # before the time limit)
        self.stop_on_sigterm = stop_on_sigterm
        self._sigterm_received = False
        # Whether the last call to train stopped because of SIGTERM, so that
        # e.g. an HPC job can be queued again to resume from the checkpoint
        self.stopped_by_sigterm = False

        # What the trainer was built from (e.g. a parameter file's contents),
        # saved in checkpoints to warn when one is loaded into a trainer built
        # from something else
        self.parameters = parameters

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

        # Wall clock time spent training (in train), in seconds, including the
        # runs a checkpoint was resumed from
        self.train_time = 0.0
        self._train_start_time: float | None = None

        # TrainerState handles current state of training
        self.state = TrainerState()
        self.state.coast_until_epoch = warmup_epochs

        # Pass trainer to optimizer
        self.optimizer.attach_trainer(self)
        # Changes the trainer modules ask for, applied once they have all run
        self._next_optimizer: Optimizer | None = None
        self._point_weight_updates: dict[
            str, tuple[LossTermContainer, torch.Tensor, Coords | None]
        ] = {}
        self._loss_term_weight_updates: dict[str, tuple[LossTermContainer, float]] = {}

        # Batch coords used for the current step
        self._current_train_batches: dict[str, Coords] = {}

        # Full-set val losses at the end of the last epoch, from before the
        # trainer modules act at the epoch end
        self.epoch_val_losses: dict[str, float] = {}

        # Best model info
        self.best_model_epoch: int | None = None
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.best_geometric_mean_val_loss: float | None = None
        self.best_loss_term_weights: dict[str, float] | None = None
        self.best_point_weights: dict[str, torch.Tensor] | None = None
        # States of the trainer modules that adapt the loss, by module key
        self.best_module_states: dict[str, dict[str, Any]] | None = None
        # Whether the best model has been loaded in the current step
        self._best_model_loaded = False

        # Restored state from checkpoint (if any)
        self._restored_state: dict[str, Any] | None = None
        # After training, the state at the end of the last epoch, from before
        # the best model was loaded, to resume training from
        self._end_state: dict[str, Any] | None = None

        # Quantities cached until the model parameters change
        self._full_val_losses_cache: dict[str, float] | None = None
        self._train_residuals_cache: dict[str, torch.Tensor] | None = None
        self._total_train_loss_cache: float | None = None

        # Logger
        self.logger = TrainerLogger(
            self,
            print_freq=print_freq,
            log_point_weight_info=log_point_weight_info,
            point_weight_snapshot_freq=point_weight_snapshot_freq,
        )

    def step(self) -> None:
        # Model params changed last step so cached quantities are stale
        self._clear_caches()
        self._best_model_loaded = False

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

        # Trainer module operations that require self.optimizer.prepare_for_step().
        # Modules that adapt the loss wait while the optimizer needs it fixed.
        for module in self.trainer_modules:
            if module.adapts_loss and self.optimizer.requires_fixed_loss:
                continue
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

        # Apply the changes the modules asked for in this step
        self._apply_updates()

        # Execute epoch-end hooks when a full pass over the data completes
        if is_epoch_end:
            # Roll the counter first, so hooks see the number of completed
            # epochs and iter == epoch * steps_per_epoch at epoch boundaries
            self.epoch += 1
            self._steps_in_epoch = 0
            for container in self.loss_containers.values():
                container.reset_train_batches()

            # The epoch's results, which the trainer modules act on at the
            # epoch end, whatever the other modules do then
            self.epoch_val_losses = dict(self.compute_full_unweighted_val_losses())

            # Track the best model on the full-set validation loss
            if not self.state.is_coasting:
                self.update_best_model()
            self.logger.on_epoch_end()
            for module in self.trainer_modules:
                if not self.state.is_coasting or module.runs_during_coast:
                    module.on_epoch_end(self, self.logger)

            # Apply the changes the modules asked for at the epoch end
            self._apply_updates()

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
        """Train until max_epoch, until the trainer modules stop training (e.g.
        early stopping) or, with stop_on_sigterm, until the end of the epoch
        in which SIGTERM is received.

        Continues from a loaded checkpoint, or from the end of the last call
        to train."""
        self._train_start_time = time.perf_counter()
        self._sigterm_received = False
        self.stopped_by_sigterm = False
        if self.stop_on_sigterm:
            previous_sigterm_handler = signal.signal(signal.SIGTERM, self._on_sigterm)

        # Execute all trainer modules at start of training
        for module in self.trainer_modules:
            module.on_train_start(self, self.logger)

        # Restore state from checkpoint, or from the end of the last training
        if self._restored_state is None:
            self._restored_state = self._end_state
        self._end_state = None
        if self._restored_state is not None:
            self._apply_restored_state(self._restored_state)
            self._restored_state = None
            print(f"Resuming training from epoch {self.epoch} (iter {self.iter})")

        # Print initial state to terminal
        self.logger.print_losses_and_weights("initial state")

        # Main training loop
        while True:
            if self.max_epoch is not None and self.epoch >= self.max_epoch:
                self.logger.log_event("max_epoch_reached")
                break
            if self.state.is_ready_to_stop:
                self.logger.log_event("training_stopped_early")
                break
            # Stop at the end of an epoch, where the state can be saved
            if self._sigterm_received and self._steps_in_epoch == 0:
                self.logger.log_event("stopped_by_sigterm")
                self.stopped_by_sigterm = True
                break

            # Update iteration count
            self.iter += 1

            # Step the trainer
            epoch = self.epoch
            self.step()

            # Save a checkpoint at the end of every checkpoint_freq epochs
            if (
                self.checkpoint_path is not None
                and self.epoch > epoch
                and self.epoch % self.checkpoint_freq == 0
            ):
                self.save(self.checkpoint_path)

        # Keep the state to resume training from
        self._end_state = copy.deepcopy(self.state_dict())

        # Restore best model
        if self.best_model_state is not None:
            self.load_best_model()
        else:
            self.logger.log_event("no_best_model_to_load")

        # Execute all trainer modules at end of training
        for module in self.trainer_modules:
            module.on_train_end(self, self.logger)

        self.train_time = self.elapsed_train_time
        self._train_start_time = None
        self.logger.print_train_time()

        if self.checkpoint_path is not None:
            self.save(self.checkpoint_path)
        if self.stop_on_sigterm:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)

    def _on_sigterm(self, signum: int, frame: FrameType | None) -> None:
        print("SIGTERM received: stopping training at the end of the epoch.")
        self._sigterm_received = True

    @property
    def elapsed_train_time(self) -> float:
        """Wall clock time spent training so far, in seconds, including the runs
        a checkpoint was resumed from."""
        if self._train_start_time is None:
            return self.train_time
        return self.train_time + time.perf_counter() - self._train_start_time

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
        return geometric_mean(self.compute_full_unweighted_val_losses().values())

    @property
    def epoch_geometric_mean_val_loss(self) -> float:
        """Geometric mean of the full-set val losses at the end of the last
        epoch, from before the trainer modules act at the epoch end."""
        return geometric_mean(self.epoch_val_losses.values())

    @property
    def all_loss_containers(self) -> dict[str, LossTermContainer]:
        """All loss containers, including ones not yet activated by staggering."""
        return self._loss_containers

    def update_point_weights(
        self,
        container: LossTermContainer,
        weights: torch.Tensor,
        coords: Coords | None = None,
    ) -> None:
        """Set the raw point weights of container at coords (e.g. this step's
        batch), or at all points, once all trainer modules have run."""
        if container.label in self._point_weight_updates:
            raise ValueError(
                f"The point weights of loss term '{container.label}' were updated "
                "twice in one step, e.g. by two trainer modules."
            )
        self._point_weight_updates[container.label] = (container, weights, coords)

    def update_loss_term_weight(
        self, container: LossTermContainer, weight: float
    ) -> None:
        """Set the loss term weight of container, once all trainer modules have
        run."""
        if container.label in self._loss_term_weight_updates:
            raise ValueError(
                f"The loss term weight of '{container.label}' was updated twice in "
                "one step, e.g. by two trainer modules."
            )
        self._loss_term_weight_updates[container.label] = (container, weight)

    def replace_optimizer(self, optimizer: Optimizer) -> None:
        """Train with optimizer from the next step, once all trainer modules have
        run."""
        self._next_optimizer = optimizer

    def _apply_updates(self) -> None:
        """Apply the changes the trainer modules asked for."""
        for container, weights, coords in self._point_weight_updates.values():
            container.train_eval_context.update_point_weights(weights, coords=coords)
        self._point_weight_updates.clear()

        for container, weight in self._loss_term_weight_updates.values():
            container.loss_term_weight = weight
        self._loss_term_weight_updates.clear()

        if self._next_optimizer is not None:
            self.optimizer = self._next_optimizer
            self.optimizer.attach_trainer(self)
            self._next_optimizer = None

        self.state.apply_stop_barrier_updates()

    def _modules_by_key(self) -> dict[str, TrainerModule]:
        """The trainer modules by a key of their position and class, e.g.
        '0:EarlyStopping', as saved in checkpoints."""
        return {
            f"{i}:{type(module).__name__}": module
            for i, module in enumerate(self.trainer_modules)
        }

    def activate_loss_container(self, label: str) -> None:
        """Start training on a loss container held back by staggering."""
        self.loss_containers[label] = self._loss_containers[label]

    @property
    def current_train_batches(self) -> dict[str, Coords]:
        """Batch coords used by each active loss container in the current step."""
        return self._current_train_batches

    def update_best_model(self) -> None:
        """Updates best model state if validation loss is lower than previous best."""
        current = self.epoch_geometric_mean_val_loss

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

            # Save the state of the modules that adapt the loss
            self.best_module_states = {
                key: module.state_dict()
                for key, module in self._modules_by_key().items()
                if module.adapts_loss
            }

    def load_best_model(self) -> None:
        """Loads the best model state, with its loss term and point weights and
        the state of the modules that adapt the loss.

        Several trainer modules may load the best model at the same epoch end
        (and one may then reset it): it is only loaded once in a step.
        """
        if self._best_model_loaded:
            return
        if (
            self.best_model_state is None
            or self.best_loss_term_weights is None
            or self.best_point_weights is None
            or self.best_module_states is None
        ):
            raise ValueError("No best model state to load.")

        print(
            f"Loading best model on epoch {self.epoch}: "
            f"best model epoch = {self.best_model_epoch}, "
            f"geometric_mean_val_loss = {self.best_geometric_mean_val_loss}"
        )
        self.logger.log_event("load_best_model")

        self.model.load_state_dict(self.best_model_state)
        self._best_model_loaded = True
        self._clear_caches()
        for label, weight in self.best_loss_term_weights.items():
            self.loss_containers[label].loss_term_weight = weight
        for label, weights in self.best_point_weights.items():
            self.loss_containers[label].train_eval_context.update_point_weights(weights)
        modules = self._modules_by_key()
        for key, module_state in self.best_module_states.items():
            modules[key].load_state_dict(module_state, self)

        # Notify modules of updated state
        self.state.state_updated = True

    def reset_best_model(self) -> None:
        self.best_model_epoch = None
        self.best_model_state = None
        self.best_geometric_mean_val_loss = None
        self.best_loss_term_weights = None
        self.best_point_weights = None
        self.best_module_states = None

        # Log event
        self.logger.log_event("reset_best_model")

    def save(self, path: str | Path) -> None:
        """Save the model, the logs and the state to resume training from.

        After training, the model is the best model, and training resumes from
        the end of the last epoch.

        The file is written whole or not at all: if the process is killed
        while saving, the file saved before is kept."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "logs": self.logger.state_dict(),
                "trainer_state_dict": self.state_dict(),
                "parameters": self.parameters,
            },
            tmp_path,
        )
        os.replace(tmp_path, path)

    def load(self, path: str | Path) -> None:
        """Load what save saved. Calling train then resumes training, with the
        same trainer modules."""
        device = next(self.model.parameters()).device
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Checkpoints saved before parameters were stored have none
        saved_parameters = checkpoint.get("parameters")
        changes = (
            changed_parameters(saved_parameters, self.parameters)
            if saved_parameters is not None and self.parameters is not None
            else []
        )
        if changes:
            warnings.warn(
                f"The checkpoint '{path}' was saved with different parameters: "
                + "; ".join(changes),
                stacklevel=2,
            )

        self.logger.load_state_dict(checkpoint["logs"])
        self.load_state_dict(checkpoint["trainer_state_dict"])

    def state_dict(self) -> dict[str, Any]:
        """The state to resume training from: a loaded state that training
        hasn't resumed from yet, the state at the end of the last training,
        or else the current state."""
        if self._restored_state is not None:
            return self._restored_state
        if self._end_state is not None:
            return self._end_state
        return {
            "epoch": self.epoch,
            "train_time": self.elapsed_train_time,
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
            "best_module_states": self.best_module_states,
            "trainer_state": self.state.state_dict(),
            "optimizer_state_dict": self.optimizer.optimizer.state_dict(),
            "modules": {
                key: module.state_dict()
                for key, module in self._modules_by_key().items()
            },
            "model_state": self.model.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        device = next(self.model.parameters()).device
        self.epoch = state["epoch"]
        self.train_time = state["train_time"]
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
        self.best_module_states = state["best_module_states"]
        self._restored_state = state
        # Training resumes from the loaded state, not the end of the last one
        self._end_state = None

    def _apply_restored_state(self, state: dict[str, Any]) -> None:
        """Restore the rest of a checkpoint, once the trainer modules have
        started training (e.g. registered their stop barriers)."""
        device = next(self.model.parameters()).device

        # Module states
        modules = self._modules_by_key()
        if list(state["modules"]) != list(modules):
            raise ValueError(
                f"Saved trainer modules {list(state['modules'])} don't match "
                f"the trainer modules {list(modules)}"
            )
        for key, module in modules.items():
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

        # Model parameters (a saved model can be the best model instead)
        self.model.load_state_dict(state["model_state"])

        # Random number generators
        torch.set_rng_state(state["rng_state"].cpu())
        if state["cuda_rng_states"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng_states"]])
