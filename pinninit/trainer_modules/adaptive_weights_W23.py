from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..loss.loss_term_container import WeightMethodLabels
from .trainer_module import TrainerModule
from .trainer_module_utils import compute_flat_train_grads

if TYPE_CHECKING:
    from ..loss.loss_term_container import LossTermContainer
    from ..trainer.trainer import Trainer
    from ..trainer.trainer_logger import TrainerLogger


class AdaptiveWeightsW23(TrainerModule):
    """
    Adaptive loss weights that balance the gradient norms across loss terms.
    See Wang et al. 2023
    """

    # Class attributes
    runs_during_coast = True

    def __init__(
        self,
        alpha: float = 0.1,  # NOTE: Wang et al. 2021 alpha definition
        update_frequency: int = 1,
        min_iter: int = 0,
    ):
        self.alpha = alpha
        self.update_frequency = update_frequency
        self.min_iter = min_iter

        # Maps weight_group_label to LossTermContainers
        self.adaptive_groups: dict[str, list[LossTermContainer]] = {}

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        self.adaptive_groups.clear()

        for container in trainer.loss_containers.values():
            # Initialise adaptive groups
            if container.weight_method_label == WeightMethodLabels.ADAPTIVE_W23:
                self._add_container(container)

    def on_epoch_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if trainer.iter < self.min_iter:
            return
        if trainer.epoch % self.update_frequency == 0:
            self._update_weights(trainer, logger)

    def on_state_change(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if trainer.iter < self.min_iter:
            return

        # Add any newly activated containers
        for container in trainer.loss_containers.values():
            if container.weight_method_label != WeightMethodLabels.ADAPTIVE_W23:
                continue

            label = container.weight_group_label
            if label is None:
                raise ValueError("Parameter 'weight_group_label' not given.")

            if label in self.adaptive_groups:
                # Existing adaptive group: initialise new member to group weight
                existing = self.adaptive_groups[label]
                if container not in existing:
                    container.loss_term_weight = existing[0].loss_term_weight
                    existing.append(container)
            else:
                self._add_container(container)

        # Update weights
        self._update_weights(trainer, logger)

    def _add_container(self, container: LossTermContainer) -> None:
        if container.weight_group_label is None:
            raise ValueError("Parameter 'weight_group_label' not given.")

        self.adaptive_groups.setdefault(container.weight_group_label, []).append(
            container
        )

        # Set container weight to its initial value
        container.loss_term_weight = container.initial_weight

    def _update_weights(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Compute grad norm of each adaptive group
        group_grad_norms: dict[str, torch.Tensor] = {}
        for label, container_list in self.adaptive_groups.items():
            all_grads_flat = compute_flat_train_grads(trainer.model, container_list)
            group_grad_norms[label] = torch.norm(all_grads_flat, p=2)

        # Set loss term weights
        sum_of_grad_norms = torch.stack(list(group_grad_norms.values())).sum()
        for label, container_list in self.adaptive_groups.items():
            # Skip if we get 0 grads
            if group_grad_norms[label] == 0:
                continue

            # Wang et al. (2023) Eqns. 2.12--2.14
            lambda_hat = float(sum_of_grad_norms / group_grad_norms[label])

            for container in container_list:
                # Wang et al. (2023) Eqn. 2.15
                container.loss_term_weight = (
                    1 - self.alpha
                ) * container.loss_term_weight + self.alpha * lambda_hat

        # Log weight update
        if logger is not None:
            logger.log_event("weight_update_W23", trainer.iter)
