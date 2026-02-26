from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..loss.loss_term_group import LossTermGroup
from ..trainer.trainer import Trainer
from .trainer_module import TrainerModule
from .trainer_module_utils import LossPlateauDetector

if TYPE_CHECKING:
    from ..trainer.trainer_logger import TrainerLogger
    from .trainer_module_utils import VolumeRejectionSampler


class R3D23(TrainerModule):
    """
    Retain-Resample-Release (R3) sampling of collocation points (Daw et al. 2023).
    """

    runs_during_coast = False

    def __init__(
        self,
        n_resample_events: int,
        n_sample_size: int,
        sampler: VolumeRejectionSampler,
        patience: int,
        frac_tol: float = 0.0,
        coast_iters_after_update: int = 0,
    ):
        self.n_resample_events = n_resample_events
        self.n_sample_size = n_sample_size
        self.sampler = sampler
        self.patience = patience
        self.frac_tol = frac_tol
        self.coast_iters_after_update = coast_iters_after_update

        self.resample_groups: set[LossTermGroup] = set()
        self._prev_resample_iter: int | None = None
        self._resamples_completed = 0

        self.stop_barrier_label = f"{self.__class__.__name__}"

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Initialise plateau detectors
        self._init_plateau_detectors(trainer)

        # Training can only stop after optimizer has been updated
        trainer.state.init_stop_barrier(self.stop_barrier_label)

        # Initialise coords with an initial sample from the sampler

        self._resamples_completed = 0
        self._prev_resample_iter = None
        ### HARDCODE THIS TO WORK ON ALL PHYSICS LOSS GROUPS FOR NOW
        self.resample_groups = {
            group
            for label, group in trainer.loss_groups.items()
            if label == "pde_residual"
        }

        for group in self.resample_groups:
            group.train_coords = self.sampler.sample(self.n_sample_size)

            # Rebuild data containers and loaders
            group.train_input_data, group.train_output_data = (
                group._init_data_containers(group.train_coords)
            )
            group._set_train_val_data_loaders()

    def on_train_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        pass

    def on_step_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        pass

    def on_step_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Barrier must still be active
        if trainer.state._stop_barriers[self.stop_barrier_label] is True:
            return

        ### HARDCODED: Start resampling when this stop barrier is the only one remaining
        if any(
            not status
            for label, status in trainer.state._stop_barriers.items()
            if label != self.stop_barrier_label
        ):
            return

        # Get validation losses
        if trainer.logger.unweighted_val_losses.has_value:
            unweighted_val_losses = trainer.logger.unweighted_val_losses.value
        else:
            raise RuntimeError(
                f"Validation losses have not been recorded for iter {trainer.iter}."
            )

        # Update loss plateau detectors
        for label, plateau_detector in self.plateau_detectors.items():
            if label in unweighted_val_losses:
                plateau_detector.step(unweighted_val_losses[label])
            elif label == self.weighted_avg_label:
                plateau_detector.step(float(trainer.weighted_avg_val_loss))
            else:
                raise KeyError(f"No loss value found for label {label}")

        # Resample if validation loss has plateaued
        if all(
            detector.plateau_detected for detector in self.plateau_detectors.values()
        ):
            # Reload the best model before switching optimizer
            ### Does loading best model cause more issues?
            ###  trainer.load_best_model()

            # Reset the best model tracker
            trainer.reset_best_model()

            # Resample points
            self._resample(trainer)
            self._prev_resample_iter = trainer.iter
            self._resamples_completed += 1
            print(f"R3 resample on iteration {trainer.iter}.")

            # Clear optimizer state
            trainer.optimizer.clear()

            # Log resampling update
            if logger is not None:
                logger.log_event("R3_resample", trainer.iter)

            # Notify trainer that its state has changed
            trainer.state.state_updated = True

            # Set coast phase for a number of iterations
            if self.coast_iters_after_update > 0:
                trainer.state.coast_until_iter = max(
                    trainer.state.coast_until_iter,
                    trainer.iter + self.coast_iters_after_update,
                )

            # Remove stop barrier
            if self._resamples_completed >= self.n_resample_events:
                trainer.state.update_stop_barrier(self.stop_barrier_label, True)

            # Reset plateau detectors
            for label, plateau_detector in self.plateau_detectors.items():
                if label in unweighted_val_losses or label == self.weighted_avg_label:
                    plateau_detector.reset()
                else:
                    raise KeyError(f"No loss value found for label {label}")

    def on_state_change(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if trainer.state._stop_barriers[self.stop_barrier_label] is True:
            return

        # Fully re-initialise plateau detectors in case new loss groups have been added
        self._init_plateau_detectors(trainer)

    def _resample(self, trainer: Trainer) -> None:
        model = trainer.model

        for group in self.resample_groups:
            coords = group.train_coords

            # Compute residuals at current collocation points
            residual_dict = group.compute_residuals_at_coords(
                model, coords, shuffle=False
            )

            # Aggregate residual magnitude per point
            residual_components = torch.stack(
                list(residual_dict.values()),
                dim=1,
            )

            F = torch.sqrt((residual_components**2).sum(dim=1))

            # Compute threshold
            tau = F.mean()

            # Select retained population
            retain_mask = tau < F
            retained_tensor = coords.full_tensor[retain_mask]
            n_retain = retained_tensor.shape[0]

            # Resample remaining points
            n_new = self.n_sample_size - n_retain

            if n_new > 0:
                new_coords = self.sampler.sample(n_new)
                new_tensor = new_coords.full_tensor
                merged_tensor = (
                    torch.cat([retained_tensor, new_tensor], dim=0)
                    .detach()
                    .requires_grad_(True)
                )
            else:
                merged_tensor = retained_tensor.detach().requires_grad_(True)

            # Build new coords object
            coords_dict = {
                label: merged_tensor[:, i]
                for i, label in enumerate(coords.coord_system.axes_labels_selected)
            }

            updated_coords = self.sampler.Coords(
                coord_system=coords.coord_system,
                domain=coords.domain,
                **coords_dict,
            )

            # Replace group training coords
            group.train_coords = updated_coords

            # Rebuild data containers and loaders
            group.train_input_data, group.train_output_data = (
                group._init_data_containers(updated_coords)
            )
            group._set_train_val_data_loaders()

    def _init_plateau_detectors(self, trainer: Trainer) -> None:
        # Initialise plateau detector for each loss term group
        self.plateau_detectors = {}
        #    group_label: LossPlateauDetector(self.patience, self.frac_tol)
        #    for group_label in trainer.loss_groups
        # }
        # Add plateau detector for weighted average val loss
        self.weighted_avg_label = "weighted_avg"
        self.plateau_detectors[self.weighted_avg_label] = LossPlateauDetector(
            self.patience, self.frac_tol
        )
