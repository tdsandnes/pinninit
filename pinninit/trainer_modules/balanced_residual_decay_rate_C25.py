from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..data_structure.field import make_field_dataclass
from .trainer_module import TrainerModule
from .trainer_module_utils import (
    adaptive_point_weight_containers,
    train_residual_magnitudes,
)

if TYPE_CHECKING:
    from typing import Any

    from ..data_structure.field import Field
    from ..loss.loss_term_container import LossTermContainer
    from ..trainer.trainer import Trainer
    from ..trainer.trainer_logger import TrainerLogger


class BalancedResidualDecayRateC25(TrainerModule):
    """
    Point weights that balance the residual decay rates, following Chen et al.
    (2025), with the adaptive scaling factor s and the effective smoothing
    factors for mini-batches.

    Difference from the paper: the weights are updated after this step's loss
    and gradients have been computed, so the new weights are used from the
    next time each point is trained on.
    """

    # Class attributes
    runs_during_coast = True
    adapts_loss = True

    def __init__(
        self,
        beta_c: float = 0.999,
        beta_w: float = 0.999,
        use_scaling_factor_s: bool = True,
        use_minibatch_effective_betas: bool = True,
        eps: float = 1e-14,
        min_iter: int = 0,
    ) -> None:
        if not 0.0 < beta_c < 1.0:
            raise ValueError(f"beta_c must be in (0, 1), got {beta_c}")
        if not 0.0 < beta_w < 1.0:
            raise ValueError(f"beta_w must be in (0, 1), got {beta_w}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if min_iter < 0:
            raise ValueError(f"min_iter must be >= 0, got {min_iter}")
        self.beta_c = beta_c
        self.beta_w = beta_w
        self.use_scaling_factor_s = use_scaling_factor_s
        self.use_minibatch_effective_betas = use_minibatch_effective_betas
        self.eps = eps
        self.min_iter = min_iter

        # Containers whose training loss uses adaptive point weights
        self.containers: dict[str, LossTermContainer] = {}
        # Per-point moving average of R^4, and the iteration each point was
        # last trained on, by container label
        self.r4_bar: dict[str, Field] = {}
        self.n_last: dict[str, Field] = {}
        # Adaptive scaling factor s, and the iteration of the last update
        self.scaling_factor_s = 1.0
        self.last_update_iter = 0

    def summary(self) -> str:
        return f"BRDR scaling factor s: {self.scaling_factor_s:.6g} \n"

    def on_train_start(self, trainer: Trainer, logger: TrainerLogger) -> None:
        self.containers = adaptive_point_weight_containers(trainer, type(self).__name__)
        self.scaling_factor_s = 1.0
        self.last_update_iter = 0

        self.r4_bar = {}
        self.n_last = {}
        for label, container in self.containers.items():
            coords = container.train_eval_context.coords

            # Weights start at 1
            container.train_eval_context.update_point_weights(
                torch.ones(coords.n_points, device=coords.device)
            )

            # Per-point R4 moving average and last-seen iteration
            ScalarField = make_field_dataclass(0, type(coords))
            self.r4_bar[label] = ScalarField(
                label="brdr_r4_bar",
                coords=coords,
                value=torch.zeros(coords.n_points, device=coords.device),
            )
            if self.use_minibatch_effective_betas:
                self.n_last[label] = ScalarField(
                    label="brdr_n_last",
                    coords=coords,
                    value=torch.zeros(
                        coords.n_points, dtype=torch.long, device=coords.device
                    ),
                )

    def on_pre_optimizer_step(self, trainer: Trainer, logger: TrainerLogger) -> None:
        if trainer.iter < self.min_iter:
            return

        self._update_point_weights(trainer)
        if self.use_scaling_factor_s:
            self._scale_gradients(trainer)

    def _update_point_weights(self, trainer: Trainer) -> None:
        n = trainer.iter
        if self.use_minibatch_effective_betas:
            # Loading the best model restores the state from an earlier
            # iteration. Shift n_last by the steps since then, so they don't
            # count as steps in which the points weren't trained on.
            gap = n - 1 - self.last_update_iter if self.last_update_iter > 0 else 0
            if gap > 0:
                for field in self.n_last.values():
                    field.update_from_tensor(field.tensor() + gap)
            self.last_update_iter = n

        # Containers trained in this step (not held back by staggering)
        containers = {
            label: container
            for label, container in self.containers.items()
            if label in trainer.loss_containers
        }
        c_by_label: dict[str, torch.Tensor] = {}
        bw_eff_by_label: dict[str, torch.Tensor | float] = {}
        for label, container in containers.items():
            c_by_label[label], bw_eff_by_label[label] = self._irdr(
                trainer, container, n
            )
        if not c_by_label:
            return

        # Mean across all weighted containers
        c_bar = torch.cat(list(c_by_label.values())).mean()

        for label, c in c_by_label.items():
            container = containers[label]
            batch_coords = trainer.current_train_batches[label]
            prev_w = container.train_eval_context.get_raw_point_weights(batch_coords)
            bw_eff = bw_eff_by_label[label]

            # Eq. 19
            w_ref = c / c_bar

            # Eq. 20
            new_w = bw_eff * prev_w + (1.0 - bw_eff) * w_ref
            trainer.update_point_weights(container, new_w, batch_coords)

    def _irdr(
        self, trainer: Trainer, container: LossTermContainer, n: int
    ) -> tuple[torch.Tensor, torch.Tensor | float]:
        """The inverse residual decay rate at each point of the container's
        batch, and the smoothing factor of its weights."""
        batch_coords = trainer.current_train_batches[container.label]
        r2 = train_residual_magnitudes(trainer, container) ** 2
        r4 = r2**2

        bc_eff: torch.Tensor | float
        bw_eff: torch.Tensor | float

        r4_field = self.r4_bar[container.label]
        prev_r4 = r4_field.tensor(batch_coords)

        if self.use_minibatch_effective_betas:
            # Effective smoothing factors beta^(n - n_last)  (Eqs. 27-28).
            nlast_field = self.n_last[container.label]
            prev_nlast = nlast_field.tensor(batch_coords)
            dn = n - prev_nlast
            bc_eff = self.beta_c**dn
            bw_eff = self.beta_w**dn
            nlast_field.update_from_tensor(
                torch.full_like(prev_nlast, n), coords=batch_coords
            )
        else:
            # Plain betas.
            bc_eff = self.beta_c
            bw_eff = self.beta_w

        # Eq. 28 (Eq. 10 with effective smoothing when enabled).
        new_r4 = bc_eff * prev_r4 + (1.0 - bc_eff) * r4
        r4_field.update_from_tensor(new_r4.detach(), coords=batch_coords)

        # Eq. 9. Algorithm 1 in the paper also divides the R^4 average by
        # (1 - beta_c^n) to correct its bias from starting at 0. That factor is
        # the same for every point of the current batch, so it cancels in the
        # normalised weights (Eq. 19), apart from where eps enters.
        c = r2 / (torch.sqrt(new_r4 + self.eps))
        return c, bw_eff

    def _scale_gradients(self, trainer: Trainer) -> None:
        """Update the scaling factor s and multiply the gradients by it."""
        s0 = self.scaling_factor_s

        # Unscaled weighted total loss L = sum_term loss_term_weight * loss_term.
        l_un = trainer.cached_total_train_loss()
        if l_un is None:
            raise RuntimeError(
                "No cached total training loss while updating the BRDR scaling factor."
            )

        # sum_theta ||grad_theta L||^2 of the unscaled loss. Parameters the
        # loss doesn't depend on (e.g. a learnt constant only used by a loss
        # term not yet activated by staggering) have no gradient.
        grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        sum_sq_grad_L = float(sum(g.pow(2).sum() for g in grads))

        lr = trainer.optimizer.optimizer.param_groups[0]["lr"]

        # Eq. 26 with Eq. 25 substituted in and taking beta_s = 1 - lr.
        # Note, here s0 factors that are included in L and grad L^2 in paper cancel s0
        # in second term i.e. 2 * s0 * (s0 * L) / (s0^2 * grad L^2) = 2 * L / grad L^2.
        s_new = (1.0 - lr) * s0 + 2.0 * l_un / sum_sq_grad_L

        # Gradient correction.
        for g in grads:
            g.mul_(s_new)

        self.scaling_factor_s = s_new

    def state_dict(self) -> dict[str, Any]:
        return {
            "r4_bar": {
                label: field.tensor().detach().cpu().clone()
                for label, field in self.r4_bar.items()
            },
            "n_last": {
                label: field.tensor().detach().cpu().clone()
                for label, field in self.n_last.items()
            },
            "scaling_factor_s": self.scaling_factor_s,
            "last_update_iter": self.last_update_iter,
        }

    def load_state_dict(self, state: dict[str, Any], trainer: Trainer) -> None:
        for label, field in self.r4_bar.items():
            field.update_from_tensor(state["r4_bar"][label].to(field.device))
        for label, field in self.n_last.items():
            field.update_from_tensor(state["n_last"][label].to(field.device))
        self.scaling_factor_s = state["scaling_factor_s"]
        self.last_update_iter = state["last_update_iter"]
