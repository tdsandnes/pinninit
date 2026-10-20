from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .trainer_module import TrainerModule
from .trainer_module_utils import (
    adaptive_point_weight_containers,
    train_residual_magnitudes,
)

if TYPE_CHECKING:
    from ..loss.loss_term_container import LossTermContainer
    from ..trainer.trainer import Trainer
    from ..trainer.trainer_logger import TrainerLogger


class ResidualBasedAttentionA24(TrainerModule):
    """
    Residual-based attention point weights following Anagnostopoulos et al. (2024).

    Per-point multipliers lambda are updated every step on the current
    training batch:

        lambda_i <- gamma * lambda_i + eta * |r_i| / max_j |r_j|

    where the max runs over the batch.

    The loss functions compute mean(w * r^2), while the paper defines the
    RBA loss as mean((lambda * r)^2), so the stored point weights are
    w = lambda^2.

    Differences from the paper: lambda starts at lambda_init (the paper uses
    0), so that the loss is not zero (and a weight-normalised loss is defined)
    in the first step, and lambda is updated after this step's loss and
    gradients have been computed, so the new weights are used from the next
    time each point is trained on.
    """

    # Class attributes
    runs_during_coast = True
    adapts_loss = True

    def __init__(
        self,
        gamma: float = 0.999,
        eta: float = 0.01,
        lambda_init: float = 1.0,
        min_iter: int = 0,
    ) -> None:
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        if eta <= 0.0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if lambda_init <= 0.0:
            raise ValueError(f"lambda_init must be > 0, got {lambda_init}")
        if min_iter < 0:
            raise ValueError(f"min_iter must be >= 0, got {min_iter}")

        self.gamma = gamma
        self.eta = eta
        self.lambda_init = lambda_init
        self.min_iter = min_iter

        # Containers whose training loss uses adaptive point weights
        self.containers: dict[str, LossTermContainer] = {}

    def on_train_start(self, trainer: Trainer, logger: TrainerLogger) -> None:
        self.containers = adaptive_point_weight_containers(trainer, type(self).__name__)

        # Initialise lambda (stored as w = lambda^2)
        for container in self.containers.values():
            coords = container.train_eval_context.coords
            container.train_eval_context.update_point_weights(
                torch.full(
                    (coords.n_points,), self.lambda_init**2, device=coords.device
                )
            )

    def on_pre_optimizer_step(self, trainer: Trainer, logger: TrainerLogger) -> None:
        if trainer.iter < self.min_iter:
            return

        for label, container in self.containers.items():
            # Skip containers not yet activated by staggering
            if label in trainer.loss_containers:
                self._update_point_weights(trainer, container)

    def _update_point_weights(
        self, trainer: Trainer, container: LossTermContainer
    ) -> None:
        magnitudes = train_residual_magnitudes(trainer, container)
        max_magnitude = magnitudes.max()
        if max_magnitude == 0:
            return

        # Recover lambda from the stored raw weights w = lambda^2
        eval_context = container.train_eval_context
        batch_coords = trainer.current_train_batches[container.label]
        lambdas = eval_context.get_raw_point_weights(batch_coords).sqrt()

        # Anagnostopoulos et al. (2024) Eqn. 12 (max over the current batch)
        lambdas = self.gamma * lambdas + self.eta * magnitudes / max_magnitude

        trainer.update_point_weights(container, lambdas**2, batch_coords)
