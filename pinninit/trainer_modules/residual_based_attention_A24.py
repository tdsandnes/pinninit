from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .trainer_module import TrainerModule

if TYPE_CHECKING:
    from ..data_structure.coords_dataclass import CoordsProtocol
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
    """

    # Class attributes
    runs_during_coast = True

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

        self.gamma = gamma
        self.eta = eta
        self.lambda_init = lambda_init
        self.min_iter = min_iter

        # Containers whose training loss uses point weights
        self.rba_containers: dict[str, LossTermContainer] = {}

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        self.rba_containers.clear()

        # Register all containers (including staggered ones) with point weights
        for label, container in trainer.all_loss_containers.items():
            eval_context = container.train_eval_context
            if eval_context.point_weights is None:
                continue

            self.rba_containers[label] = container

            # Initialise lambda (stored as w = lambda^2)
            coords = eval_context.coords
            eval_context.update_point_weights(
                torch.full(
                    (coords.n_points,), self.lambda_init**2, device=coords.device
                )
            )

        if not self.rba_containers:
            raise ValueError(
                "ResidualBasedAttention found no loss term containers with "
                "point weights. Use a weighted loss function (e.g. "
                "'weighted_point_avg_mse') on at least one loss term."
            )

    def on_step_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if trainer.iter < self.min_iter:
            return

        for label, container in self.rba_containers.items():
            # Skip containers not yet activated by staggering
            if label not in trainer.loss_containers:
                continue

            batch_coords = trainer.current_train_batches.get(label)
            if batch_coords is None:
                continue

            self._update_point_weights(trainer, container, batch_coords)

    def _update_point_weights(
        self,
        trainer: Trainer,
        container: LossTermContainer,
        batch_coords: CoordsProtocol,
    ) -> None:
        eval_context = container.train_eval_context
        assert eval_context.point_weights is not None

        # Per-point residual magnitudes on the current batch
        residuals = trainer.cached_train_residuals(container.label)
        if residuals is None:
            residuals = container.compute_train_residuals(
                trainer.model, coords=batch_coords, detach=True
            )
        magnitudes = residuals.abs() if residuals.ndim == 1 else residuals.norm(dim=1)

        max_magnitude = magnitudes.max()
        if max_magnitude == 0:
            return

        # Recover lambda from the stored weights w = lambda^2
        weights = eval_context.point_weights.get_component("value", batch_coords)
        lambdas = weights.clamp_min(0.0).sqrt()

        # Anagnostopoulos et al. (2024) Eqn. 3 (max over the current batch)
        lambdas = self.gamma * lambdas + self.eta * magnitudes / max_magnitude

        # Scatter the batch's updated weights back into the full field
        eval_context.update_point_weights(lambdas**2, coords=batch_coords)
