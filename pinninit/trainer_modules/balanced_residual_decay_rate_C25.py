from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

from ..data_structure.field_dataclass import make_field_dataclass
from .trainer_module import TrainerModule

if TYPE_CHECKING:
    from typing import Any

    from ..data_structure.coords_dataclass import Coords
    from ..data_structure.field_dataclass import Field
    from ..loss.loss_term_container import LossTermContainer
    from ..trainer.trainer import Trainer
    from ..trainer.trainer_logger import TrainerLogger


class BalancedResidualDecayRateC25(TrainerModule):
    # Class attributes
    runs_during_coast = True

    def __init__(
        self,
        beta_c: float = 0.999,
        beta_w: float = 0.999,
        use_scaling_factor_s: bool = True,
        use_minibatch_effective_betas: bool = True,
        eps: float = 1e-14,
        min_iter: int = 0,
        log_point_weight_info: bool = False,
        snapshot_epoch_freq: int = 0,
    ) -> None:
        if not 0.0 < beta_c < 1.0:
            raise ValueError(f"beta_c must be in (0, 1), got {beta_c}")
        if not 0.0 < beta_w < 1.0:
            raise ValueError(f"beta_w must be in (0, 1), got {beta_w}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if snapshot_epoch_freq < 0:
            raise ValueError(
                f"snapshot_epoch_freq must be >= 0, got {snapshot_epoch_freq}"
            )

        self.beta_c = beta_c
        self.beta_w = beta_w
        self.use_scaling_factor_s = use_scaling_factor_s
        self.use_minibatch_effective_betas = use_minibatch_effective_betas
        self.eps = eps
        self.min_iter = min_iter
        self.log_point_weight_info = log_point_weight_info
        self.snapshot_epoch_freq = snapshot_epoch_freq

        # Containers whose training loss uses adaptive point weights
        self.brdr_containers: dict[str, LossTermContainer] = {}
        self.r4_bar: dict[str, Field] = {}
        self.n_last: dict[str, Field] = {}
        # Global adaptive scaling factor s and last-update iteration
        self._s_buffer: torch.Tensor | None = None
        self._last_update_iter: torch.Tensor | None = None

    @property
    def scaling_factor_s(self) -> float:
        return 1.0 if self._s_buffer is None else float(self._s_buffer)

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        self.brdr_containers.clear()
        self.r4_bar.clear()
        self.n_last.clear()
        self._s_buffer = None
        self._last_update_iter = None

        for label, container in trainer.all_loss_containers.items():
            if not container.adaptive_point_weights:
                continue

            eval_context = container.train_eval_context
            coords = eval_context.coords
            self.brdr_containers[label] = container

            # Weights start at 1
            eval_context.update_point_weights(
                torch.ones(coords.n_points, device=coords.device)
            )

            # Per-point R4 moving average and last-seen iteration
            ScalarField = make_field_dataclass(0, type(coords))
            self.r4_bar[label] = ScalarField(
                label="brdr_r4_bar",
                coords=coords,
                value=torch.zeros(coords.n_points, device=coords.device),
            )
            trainer.model.register_buffer(
                f"brdr_r4_bar_{label}", self.r4_bar[label].get_component("value")
            )
            if self.use_minibatch_effective_betas:
                self.n_last[label] = ScalarField(
                    label="brdr_n_last",
                    coords=coords,
                    value=torch.zeros(
                        coords.n_points, dtype=torch.long, device=coords.device
                    ),
                )
                trainer.model.register_buffer(
                    f"brdr_n_last_{label}", self.n_last[label].get_component("value")
                )

        if not self.brdr_containers:
            raise ValueError(
                "BalancedResidualDecayRate found no loss term containers with "
                "adaptive point weights."
            )

        device = next(trainer.model.parameters()).device
        self._s_buffer = torch.ones((), device=device)
        trainer.model.register_buffer("brdr_scaling_factor_s", self._s_buffer)
        if self.use_minibatch_effective_betas:
            self._last_update_iter = torch.zeros((), dtype=torch.long, device=device)
            trainer.model.register_buffer(
                "brdr_last_update_iter", self._last_update_iter
            )

    def on_pre_optimizer_step(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if trainer.iter < self.min_iter:
            return

        if self._last_update_iter is not None:
            last = int(self._last_update_iter)
            gap = trainer.iter - 1 - last if last > 0 else 0
            if gap > 0:
                for field in self.n_last.values():
                    field.get_component("value").add_(gap)
            self._last_update_iter.fill_(trainer.iter)

        self._update_point_weights(trainer)
        if self.use_scaling_factor_s:
            self._update_scaling_factor_s(trainer)

    def on_epoch_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if logger is None:
            return

        active = {
            label: container
            for label, container in self.brdr_containers.items()
            if label in trainer.loss_containers  # skip staggered-off terms
        }

        if self.log_point_weight_info:
            logger.epoch_point_weight_info_log.append(
                {
                    label: {
                        stat: float(getattr(w, stat)())
                        for stat in ("mean", "median", "min", "max")
                    }
                    for label, container in active.items()
                    for w in (container.train_eval_context.get_raw_point_weights(),)
                }
            )

        if (
            self.snapshot_epoch_freq > 0
            and trainer.epoch % self.snapshot_epoch_freq == 0
        ):
            logger.point_weight_snapshots.append(
                {
                    "epoch": trainer.epoch,
                    "iter": trainer.iter,
                    "weights": {
                        label: container.train_eval_context.get_raw_point_weights()
                        .clone()
                        .cpu()
                        for label, container in active.items()
                    },
                }
            )

    def _update_point_weights(self, trainer: Trainer) -> None:
        n = trainer.iter
        c_by_label: dict[str, torch.Tensor] = {}
        bw_eff_by_label: dict[str, torch.Tensor | float] = {}
        batch_by_label: dict[str, Coords] = {}
        for label, container in self.brdr_containers.items():
            # Skip staggered-off terms
            if label not in trainer.loss_containers:
                continue
            batch_coords = trainer.current_train_batches.get(label)
            if batch_coords is None:
                continue
            c, bw_eff = self._compute_irdr(trainer, container, batch_coords, n)
            c_by_label[label] = c
            bw_eff_by_label[label] = bw_eff
            batch_by_label[label] = batch_coords

        if not c_by_label:
            return

        # Mean across all weighted containers
        c_bar = torch.cat(list(c_by_label.values())).mean()

        # Update weights
        for label, c in c_by_label.items():
            eval_context = self.brdr_containers[label].train_eval_context
            batch_coords = batch_by_label[label]
            bw_eff = bw_eff_by_label[label]
            prev_w = eval_context.get_raw_point_weights(batch_coords)

            # Eq. 19
            w_ref = c / c_bar

            # Eq. 20
            new_w = bw_eff * prev_w + (1.0 - bw_eff) * w_ref
            eval_context.update_point_weights(new_w, coords=batch_coords)

    def _compute_irdr(
        self,
        trainer: Trainer,
        container: LossTermContainer,
        batch_coords: Coords,
        n: int,
    ) -> tuple[torch.Tensor, torch.Tensor | float]:
        residuals = trainer.cached_train_residuals(container.label)
        if residuals is None:
            if not trainer.optimizer.requires_closure:
                warnings.warn(
                    "BRDR is recomputing residuals: the train-residual cache was "
                    "empty for a non-closure optimizer (expected populated).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            _, residuals = container.compute_train_batch_loss_and_residuals(
                trainer.model, batch_coords=batch_coords, detach=True
            )
        residuals = residuals.detach()
        mag = residuals.abs() if residuals.ndim == 1 else residuals.norm(dim=1)
        r2 = mag**2
        r4 = r2**2

        bc_eff: torch.Tensor | float
        bw_eff: torch.Tensor | float

        # Get this batch's per-point state via the Fields.
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
            trainer.model.register_buffer(
                f"brdr_n_last_{container.label}", nlast_field.get_component("value")
            )
        else:
            # Plain betas.
            bc_eff = self.beta_c
            bw_eff = self.beta_w

        # Eq. 28 (Eq. 10 with effective smoothing when enabled).
        new_r4 = bc_eff * prev_r4 + (1.0 - bc_eff) * r4
        r4_field.update_from_tensor(new_r4.detach(), coords=batch_coords)
        trainer.model.register_buffer(
            f"brdr_r4_bar_{container.label}", r4_field.get_component("value")
        )

        # Eq. 9 (Note: Algorithm 1 in paper does it slightly differently)
        c = r2 / (torch.sqrt(new_r4 + self.eps))
        return c, bw_eff

    def _update_scaling_factor_s(self, trainer: Trainer) -> None:
        if trainer.optimizer.requires_closure:
            raise ValueError(
                "BalancedResidualDecayRate's adaptive scaling factor requires a "
                "non-closure optimizer."
            )

        s0 = self.scaling_factor_s

        # Unscaled weighted total loss L = sum_term loss_term_weight * loss_term.
        l_un = trainer.cached_total_train_loss()
        if l_un is None:
            raise RuntimeError(
                "No cached total training loss while updating the BRDR scaling factor."
            )

        # sum_theta ||grad_theta L||^2 of the unscaled loss.
        sum_sq_grad_L = torch.zeros((), device=next(trainer.model.parameters()).device)
        for p in trainer.model.parameters():
            if p.grad is None:
                raise RuntimeError(
                    "Parameter with no gradient while updating the BRDR scaling factor."
                )
            sum_sq_grad_L = sum_sq_grad_L + p.grad.pow(2).sum()

        lr = trainer.optimizer.optimizer.param_groups[0]["lr"]

        # Eq. 26 with Eq. 25 substituted in and taking beta_s = 1 - lr.
        # Note, here s0 factors that are included in L and grad L^2 in paper cancel s0
        # in second term i.e. 2 * s0 * (s0 * L) / (s0^2 * grad L^2) = 2 * L / grad L^2.
        s_new = (1.0 - lr) * s0 + 2.0 * l_un / float(sum_sq_grad_L)

        # Gradient correction.
        for p in trainer.model.parameters():
            assert p.grad is not None
            p.grad.mul_(s_new)

        # Store new scale factor.
        assert self._s_buffer is not None
        self._s_buffer.fill_(s_new)

    def state_dict(self) -> dict[str, Any]:
        return {
            "r4_bar": {
                label: field.get_component("value").detach().cpu().clone()
                for label, field in self.r4_bar.items()
            },
            "n_last": {
                label: field.get_component("value").detach().cpu().clone()
                for label, field in self.n_last.items()
            },
            "scaling_factor_s": self.scaling_factor_s,
            "last_update_iter": (
                None if self._last_update_iter is None else int(self._last_update_iter)
            ),
        }

    def load_state_dict(self, state: dict[str, Any], trainer: Trainer) -> None:
        for label, saved in state["r4_bar"].items():
            if label in self.r4_bar:
                self.r4_bar[label].get_component("value").copy_(saved)
        for label, saved in state["n_last"].items():
            if label in self.n_last:
                self.n_last[label].get_component("value").copy_(saved)
        if self._s_buffer is not None:
            self._s_buffer.fill_(state["scaling_factor_s"])
        if self._last_update_iter is not None and state["last_update_iter"] is not None:
            self._last_update_iter.fill_(state["last_update_iter"])
