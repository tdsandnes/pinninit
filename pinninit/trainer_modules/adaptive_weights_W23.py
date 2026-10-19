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


class AdaptiveWeightsW23(TrainerModule):
    """
    Adaptive loss weights that balance the gradient norms across loss terms.
    See Wang et al. 2023

    Each weight group is one loss term of the paper: its gradient is the
    gradient of the summed loss of its loss terms, and they share one update.
    A loss term activated by staggering starts from its initial weight.
    """

    # Class attributes
    runs_during_coast = True
    adapts_loss = True

    def __init__(
        self,
        alpha: float = 0.9,
        update_iter_frequency: int = 1,
        min_iter: int = 0,
        loss_function: str = "mse",
    ):
        check_adaptive_weight_params(alpha, update_iter_frequency, min_iter)
        self.alpha = alpha
        self.update_iter_frequency = update_iter_frequency
        self.min_iter = min_iter
        self.loss_function = LossFunctionLabels(loss_function.lower())

        # Maps weight_group_label to LossTermContainers
        self.adaptive_groups: dict[str, list[LossTermContainer]] = {}

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        self.adaptive_groups = adaptive_weight_groups(
            trainer, WeightMethodLabels.ADAPTIVE_W23
        )
        for container_list in self.adaptive_groups.values():
            for container in container_list:
                container.loss_term_weight = container.initial_weight

    def on_pre_optimizer_step(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if trainer.iter < self.min_iter:
            return
        if trainer.iter % self.update_iter_frequency == 0:
            self._update_weights(trainer)

    def _update_weights(self, trainer: Trainer) -> None:
        active_groups = active_weight_groups(trainer, self.adaptive_groups)
        if not active_groups:
            return

        # Compute grad norm of each adaptive group
        group_grad_norms: dict[str, torch.Tensor] = {}
        for label, container_list in active_groups.items():
            group_grad = flat_train_grad(trainer, container_list, self.loss_function)
            group_grad_norms[label] = torch.norm(group_grad, p=2)

        # Set loss term weights
        sum_of_grad_norms = torch.stack(list(group_grad_norms.values())).sum()
        for label, container_list in active_groups.items():
            # Skip if we get 0 grads
            if group_grad_norms[label] == 0:
                continue

            # Wang et al. (2023) Eqns. 2.12--2.14
            lambda_hat = float(sum_of_grad_norms / group_grad_norms[label])

            for container in container_list:
                # Wang et al. (2023) Eqn. 2.15
                trainer.update_loss_term_weight(
                    container,
                    self.alpha * container.loss_term_weight
                    + (1 - self.alpha) * lambda_hat,
                )
