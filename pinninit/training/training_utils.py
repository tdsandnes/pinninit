from __future__ import annotations

import torch


class LossPlateauDetector:
    """Detects plateau in loss."""

    def __init__(self, patience: int, frac_tol: float = 0.0) -> None:
        if not 0.0 <= frac_tol <= 1.0:
            raise ValueError(f"frac_tol must be between 0 and 1, got {frac_tol}")

        self.patience = patience
        self.frac_tol = frac_tol

        self.best_value: torch.Tensor | None = None
        self.counter = 0
        self.plateau_detected = False

    def step(self, current_value: torch.Tensor) -> None:
        # If best value hasn't been set, set it as current value and return
        if self.best_value is None:
            self.best_value = current_value
            return

        # Is the current value an improvement on best_value
        is_improved = current_value < self.best_value * (1.0 - self.frac_tol)

        if is_improved:
            # If improvement, plateau counting resets
            self.best_value = current_value
            self.counter = 0
            self.plateau_detected = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.plateau_detected = True

    def reset_counter(self) -> None:
        self.counter = 0
        self.plateau_detected = False
