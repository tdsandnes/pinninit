from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..loss.loss_term_group import LossTermGroup
from ..trainer.trainer import Trainer
from .trainer_module import TrainerModule

if TYPE_CHECKING:
    from ..trainer.trainer_logger import TrainerLogger


class AdaptiveWeightsGradBalance(TrainerModule):
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

        # Maps weight_group_label to LossTermGroup
        self.adaptive_groups: dict[str, list[LossTermGroup]] = {}

        # Keep track of the previous weight update
        self._prev_update_iter: int | None = None

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        self._prev_update_iter = None
        self.adaptive_groups.clear()

        for group in trainer.loss_groups.values():
            # Initialise adaptive groups
            if group.weight_method_label == "adaptive_grad_balance":  ### Make this enum
                if group.weight_group_label is None:
                    raise ValueError("Parameter 'weight_group_label' not given.")

                self.adaptive_groups.setdefault(group.weight_group_label, []).append(
                    group
                )
                if group.initial_weight is None:
                    raise ValueError(
                        f"Initial weight for group "
                        f"'{group.weight_group_label}' is not given"
                    )

                # Set group weight
                group.weight = group.initial_weight

    def on_step_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if trainer.iter < self.min_iter:
            return

        if (
            self._prev_update_iter is None
            or (trainer.iter - self._prev_update_iter) >= self.update_frequency
        ):
            self._update_weights(trainer, logger)
            self._prev_update_iter = trainer.iter

    def on_step_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        pass

    def on_train_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        pass

    def on_state_change(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if trainer.iter < self.min_iter:
            return

        for group in trainer.loss_groups.values():
            if group.weight_method_label != "adaptive_grad_balance":
                continue

            if group.weight_group_label is None:
                raise ValueError("Parameter 'weight_group_label' not given.")

            label = group.weight_group_label

            # Case 1: new adaptive weight group
            if label not in self.adaptive_groups:
                if group.initial_weight is None:
                    raise ValueError(
                        f"Initial weight for group '{label}' is not given."
                    )

                group.weight = group.initial_weight
                self.adaptive_groups[label] = [group]
                continue

            # Case 2: existing adaptive group, but new member
            if group not in self.adaptive_groups[label]:
                existing_groups = self.adaptive_groups[label]

                # Use the first group's weight as canonical
                ref_weight = existing_groups[0].weight

                # Enforce consistency
                for g in existing_groups:
                    if not torch.allclose(g.weight, ref_weight):
                        raise ValueError(
                            f"Inconsistent weights in weight group '{label}'."
                        )

                # Initialise new group to same value
                group.weight = ref_weight
                self.adaptive_groups[label].append(group)

        # Update weights
        self._update_weights(trainer, logger)
        self._prev_update_iter = trainer.iter

    def _update_weights(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Compute norm of each adaptive group
        group_grad_norms: dict[str, torch.Tensor] = {}
        for label, group_list in self.adaptive_groups.items():
            all_grads_flat_list = []
            for group in group_list:
                group_loss = group.compute_unweighted_loss_train(trainer.model)
                grads = torch.autograd.grad(
                    group_loss,
                    list(trainer.model.parameters()),
                    retain_graph=True,
                    allow_unused=True,
                )
                grads_flat = torch.cat(
                    [g.detach().reshape(-1) for g in grads if g is not None]
                )
                all_grads_flat_list.append(grads_flat)

            all_grads_flat = torch.cat(all_grads_flat_list)
            group_grad_norms[label] = torch.norm(all_grads_flat, p=2)

        # Set loss term weights
        sum_of_grad_norms = torch.stack(list(group_grad_norms.values())).sum()
        for label, group_list in self.adaptive_groups.items():
            # Wang et al. (2023) Eqns. 2.12--2.14
            lambda_hat = sum_of_grad_norms / group_grad_norms[label]

            for group in group_list:
                # Wang et al. (2023) Eqn. 2.15
                group.weight = (1 - self.alpha) * group.weight + self.alpha * lambda_hat

        # Log weight update
        if logger is not None:
            logger.log_event("weight_update_grad_balance", trainer.iter)
