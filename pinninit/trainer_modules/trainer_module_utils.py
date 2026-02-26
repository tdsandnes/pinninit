from __future__ import annotations

from typing import Any

import torch

from ..data_structure.coords_dataclass import make_coords_dataclass
from ..geometry.coordinate_system import CoordSystem
from ..geometry.domain import Domain


class LossPlateauDetector:
    """Detects plateau in loss based on significant relative improvement.

    A plateau is detected when we've gone `patience` steps without achieving
    a relative improvement of at least `frac_tol` over the previous significant best.
    """

    def __init__(self, patience: int, frac_tol: float = 0.0) -> None:
        if not 0.0 <= frac_tol <= 1.0:
            raise ValueError(f"frac_tol must be between 0 and 1, got {frac_tol}")

        self.patience = patience
        self.frac_tol = frac_tol

        self.prev_significant_best: float | None = None
        self.steps_since_significant_improvement = 0
        self.plateau_detected = False

    def step(self, current_value: float) -> None:
        if current_value <= 0.0:
            raise ValueError(
                "Value must be strictly positive for relative comparison, "
                f"got {current_value}."
            )

        # If best value hasn't been set, set it as current value and return
        if self.prev_significant_best is None:
            self.prev_significant_best = current_value
            return

        # Is the current value an improvement on prev_significant_best
        is_improved = current_value < self.prev_significant_best * (1.0 - self.frac_tol)

        if is_improved:
            # If improvement, plateau counting resets
            self.prev_significant_best = current_value
            self.steps_since_significant_improvement = 0
            self.plateau_detected = False
        else:
            self.steps_since_significant_improvement += 1
            if self.steps_since_significant_improvement >= self.patience:
                self.plateau_detected = True

    def reset(self) -> None:
        self.prev_significant_best = None
        self.steps_since_significant_improvement = 0
        self.plateau_detected = False


class VolumeRejectionSampler:
    """
    Draws points uniformly in physical volume using rejection sampling.
    """

    def __init__(
        self,
        coord_system: CoordSystem,
        domain: Domain,
        device: torch.device,
        safety_factor: float = 2.0,
        n_propose_factor: float = 10.0,
        seed: int | None = None,
    ) -> None:
        self.coord_system = coord_system
        self.domain = domain
        self.device = device
        self.safety_factor = safety_factor
        self.n_propose_factor = n_propose_factor

        # Create dedicated RNG
        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(seed)
        else:
            self.rng.seed()

        self.Coords: type[Any] = make_coords_dataclass(
            set(coord_system.axes_labels_selected),
            device=device,
        )

    def sample(self, n_points: int) -> Any:
        accepted_points: list[torch.Tensor] = []
        n_accepted = 0

        sqrt_det_approx_upper_bound = None
        lb, ub = self.domain.geometric_lb, self.domain.geometric_ub

        while n_accepted < n_points:
            remaining = n_points - n_accepted
            n_propose = int(self.n_propose_factor * remaining)

            proposed = lb + (ub - lb) * torch.rand(
                n_propose,
                self.coord_system.dim_geometry,
                device=self.device,
                generator=self.rng,
            )

            sqrt_det = self.coord_system.metric_tensor(n_propose, proposed).sqrt_det
            in_domain = self.domain.contains_geometric(proposed)
            sqrt_det = sqrt_det[in_domain]
            proposed = proposed[in_domain]

            max_sqrt_det_iter = sqrt_det.max()

            if (
                sqrt_det_approx_upper_bound is None
                or max_sqrt_det_iter > sqrt_det_approx_upper_bound
            ):
                sqrt_det_approx_upper_bound = max_sqrt_det_iter * self.safety_factor
                accepted_points.clear()
                n_accepted = 0

            accepted = proposed[
                torch.rand(proposed.shape[0], device=self.device, generator=self.rng)
                < sqrt_det / sqrt_det_approx_upper_bound
            ]
            accepted = accepted[: n_points - n_accepted]

            accepted_points.append(accepted)
            n_accepted += accepted.shape[0]

        raw = torch.cat(accepted_points, dim=0)

        coords_dict = {
            label: raw[:, i]
            for i, label in enumerate(self.coord_system.axes_labels_selected)
        }

        return self.Coords(
            coord_system=self.coord_system,
            domain=self.domain,
            **coords_dict,
        )
