from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..loss.loss_function import LossFunctionLabels
from ..loss.loss_term_container import WeightMethodLabels
from .trainer_module import TrainerModule
from .trainer_module_utils import (
    active_weight_groups,
    adaptive_weight_groups,
    check_adaptive_weight_params,
    flat_train_grad,
)

if TYPE_CHECKING:
    from ..loss.loss_term_container import LossTermContainer
    from ..trainer.trainer import Trainer
    from ..trainer.trainer_logger import TrainerLogger


class AdaptiveWeightsW21(TrainerModule):
    """
    Adaptive loss term weights following Wang et al. (2021).

    Each weight group is one loss term of the paper: its gradient is the
    gradient of the summed loss of its loss terms, and they share one update.
    The reference weight group is the paper's residual loss L_r, whose loss
    terms must have constant weights. Unlike in the paper, these needn't be 1,
    so the reference gradient is of the weighted sum of its losses. A loss
    term activated by staggering starts from its initial weight.
    """

    # Class attributes
    runs_during_coast = True
    adapts_loss = True

    def __init__(
        self,
        reference_weight_group_label: str,
        alpha: float = 0.1,  # NOTE: outdated arxiv version of W+21 has incorrect value
        update_iter_frequency: int = 1,
        min_iter: int = 0,
        loss_function: str = "mse",
    ):
        check_adaptive_weight_params(alpha, update_iter_frequency, min_iter)
        self.reference_weight_group_label = reference_weight_group_label
        self.alpha = alpha
        self.update_iter_frequency = update_iter_frequency
        self.min_iter = min_iter
        self.loss_function = LossFunctionLabels(loss_function.lower())

        # Maps weight_group_label to LossTermContainers
        self.adaptive_groups: dict[str, list[LossTermContainer]] = {}
        self.reference_weight_group: list[LossTermContainer] = []

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        self.adaptive_groups = adaptive_weight_groups(
            trainer, WeightMethodLabels.ADAPTIVE_W21
        )
        for container_list in self.adaptive_groups.values():
            for container in container_list:
                container.loss_term_weight = container.initial_weight

        self.reference_weight_group = [
            container
            for container in trainer.all_loss_containers.values()
            if container.weight_group_label == self.reference_weight_group_label
        ]
        if not self.reference_weight_group:
            raise ValueError(
                f"Reference weight group '{self.reference_weight_group_label}' not "
                "found."
            )
        for container in self.reference_weight_group:
            if container.weight_method_label != WeightMethodLabels.CONSTANT:
                raise ValueError(
                    f"Reference group '{self.reference_weight_group_label}' must "
                    f"use constant weighting, not "
                    f"'{container.weight_method_label.value}'."
                )

    def on_pre_optimizer_step(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if trainer.iter < self.min_iter:
            return
        if trainer.iter % self.update_iter_frequency == 0:
            self._update_weights(trainer)

    def _update_weights(self, trainer: Trainer) -> None:
        reference_group = [
            container
            for container in self.reference_weight_group
            if container.label in trainer.loss_containers
        ]
        active_groups = active_weight_groups(trainer, self.adaptive_groups)
        if not reference_group or not active_groups:
            return

        # Gradient of the weighted reference loss
        ref_grad = flat_train_grad(
            trainer, reference_group, self.loss_function, weighted=True
        )
        ref_max_abs = torch.max(ref_grad.abs())

        # Compute and update adaptive groups
        for container_list in active_groups.values():
            group_grad = flat_train_grad(trainer, container_list, self.loss_function)
            group_mean_abs = torch.mean(group_grad.abs())

            # Skip if we get 0 grads
            if group_mean_abs == 0:
                continue

            # Wang et al. (2021) Eqn. 40
            lambda_hat = float(ref_max_abs / group_mean_abs)

            for container in container_list:
                # Wang et al. (2021) Eqn. 41
                trainer.update_loss_term_weight(
                    container,
                    (1 - self.alpha) * container.loss_term_weight
                    + self.alpha * lambda_hat,
                )
