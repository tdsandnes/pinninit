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


class AdaptiveWeightsW21(TrainerModule):
    """
    Adaptive loss term weights following Wang et al. (2021).
    """

    # Class attributes
    runs_during_coast = True

    def __init__(
        self,
        reference_weight_group_label: str,
        alpha: float = 0.1,  # NOTE: outdated arxiv version of W+21 has incorrect value
        update_iter_frequency: int = 1,
        min_iter: int = 0,
    ):
        self.reference_weight_group_label = reference_weight_group_label
        self.alpha = alpha
        self.update_iter_frequency = update_iter_frequency
        self.min_iter = min_iter

        # Maps weight_group_label to LossTermContainers
        self.adaptive_groups: dict[str, list[LossTermContainer]] = {}
        self.reference_weight_group: list[LossTermContainer] = []

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        self.adaptive_groups.clear()
        self.reference_weight_group.clear()

        for container in trainer.loss_containers.values():
            self._register_container(container)

        if self.reference_weight_group == []:
            raise ValueError(
                f"Reference weight group '{self.reference_weight_group_label}' not "
                "found."
            )

    def on_pre_optimizer_step(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if trainer.iter < self.min_iter:
            return
        if trainer.iter % self.update_iter_frequency == 0:
            self._update_weights(trainer, logger)

    def on_state_change(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if trainer.iter < self.min_iter:
            return

        # Register any newly activated containers
        for container in trainer.loss_containers.values():
            self._register_container(container)

        # Update weights
        self._update_weights(trainer, logger)

    def _register_container(self, container: LossTermContainer) -> None:
        # Adaptive containers
        if container.weight_method_label == WeightMethodLabels.ADAPTIVE_W21:
            if container.weight_group_label is None:
                raise ValueError("Parameter 'weight_group_label' not given.")

            label = container.weight_group_label
            if (
                label not in self.adaptive_groups
                or container not in self.adaptive_groups[label]
            ):
                self.adaptive_groups.setdefault(label, []).append(container)

                # Set container weight to its initial value
                container.loss_term_weight = container.initial_weight

        # Reference containers
        if container.weight_group_label == self.reference_weight_group_label:
            if container.weight_method_label != WeightMethodLabels.CONSTANT:
                raise ValueError(
                    f"Reference group '{self.reference_weight_group_label}' must "
                    f"use constant weighting, not "
                    f"'{container.weight_method_label.value}'."
                )

            if container not in self.reference_weight_group:
                self.reference_weight_group.append(container)

    def _update_weights(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if self.reference_weight_group == []:
            raise RuntimeError(
                f"Reference group '{self.reference_weight_group_label}' not set."
            )

        # Compute gradients for the reference group
        batches = trainer.current_train_batches
        all_ref_grads_flat = compute_flat_train_grads(
            trainer.model, self.reference_weight_group, batches
        )
        ref_max_abs = torch.max(all_ref_grads_flat.abs())

        # Compute and update adaptive groups
        for container_list in self.adaptive_groups.values():
            all_grads_flat = compute_flat_train_grads(
                trainer.model, container_list, batches
            )
            group_mean_abs = torch.mean(all_grads_flat.abs())

            # Wang et al. (2021) Eqn. 40
            lambda_hat = float(ref_max_abs / group_mean_abs)

            # Wang et al. (2021) Eqn. 41
            for container in container_list:
                container.loss_term_weight = (
                    1 - self.alpha
                ) * container.loss_term_weight + self.alpha * lambda_hat
