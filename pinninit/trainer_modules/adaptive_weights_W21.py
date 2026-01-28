from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..loss.loss_term_group import LossTermGroup
from ..trainer.trainer import Trainer
from .trainer_module import TrainerModule

if TYPE_CHECKING:
    from ..trainer.trainer_logger import TrainerLogger


class AdaptiveWeightsW21(TrainerModule):
    """
    Adaptive loss term weights following Wang et al. (2021).
    """

    # Class attributes
    runs_during_coast = True

    def __init__(
        self,
        reference_group_label: str,
        alpha: float = 0.1,  # NOTE: outdated arxiv version of W+21 has incorrect value
        update_frequency: int = 1,
        min_iter: int = 0,
    ):
        self.reference_group_label = reference_group_label
        self.alpha = alpha
        self.update_frequency = update_frequency
        self.min_iter = min_iter

        # Maps weight_group_label to LossTermGroup
        self.adaptive_groups: dict[str, list[LossTermGroup]] = {}
        self.reference_group: LossTermGroup | None = None

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        self.adaptive_groups.clear()

        for group in trainer.loss_groups.values():
            # Initialise adaptive groups
            if group.weight_method_label == "adaptive_W21":  ### Make this enum
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

            # Identify reference group
            if group.weight_group_label == self.reference_group_label:
                self.reference_group = group
                if group.weight_method_label != "constant":  ### Make this enum
                    raise ValueError(
                        f"Reference group '{self.reference_group_label}' must use "
                        f"constant weighting, not '{group.weight_method_label}'."
                    )

        if self.reference_group is None:
            raise ValueError(
                f"Reference weight group '{self.reference_group_label}' not found."
            )

    def on_step_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if trainer.iter >= self.min_iter and trainer.iter % self.update_frequency == 0:
            self._update_weights(trainer)

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
        pass

    def _update_weights(self, trainer: Trainer) -> None:
        if self.reference_group is None:
            raise RuntimeError(
                f"Reference group '{self.reference_group_label}' not set."
            )

        # Compute gradients for the reference group
        ref_loss = self.reference_group.compute_unweighted_loss_train(trainer.model)
        ref_grads = torch.autograd.grad(
            ref_loss,
            list(trainer.model.parameters()),
            retain_graph=True,
            allow_unused=True,
        )
        ref_grads_flat = torch.cat(
            [g.detach().reshape(-1) for g in ref_grads if g is not None]
        )
        ref_max_abs = torch.max(ref_grads_flat.abs())

        # Compute and update adaptive groups
        for group_list in self.adaptive_groups.values():
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
            group_mean_abs = torch.mean(all_grads_flat.abs())

            # Wang et al. (2021) Eqn. 40
            lambda_hat = ref_max_abs / group_mean_abs

            # Wang et al. (2021) Eqn. 41
            for group in group_list:
                group.weight = (1 - self.alpha) * group.weight + self.alpha * lambda_hat
