from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..loss.loss_term_container import LossTermContainer
    from ..nn.neural_network import NeuralNetwork
    from ..trainer.trainer import Trainer


def compute_flat_train_grads(
    model: NeuralNetwork,
    containers: list[LossTermContainer],
    trainer: Trainer,
) -> torch.Tensor:
    """Concatenated flat gradient of the containers' unweighted training losses
    with respect to the model parameters."""
    params = list(model.parameters())
    all_grads = []
    for container in containers:
        loss = trainer.cached_train_loss(container.label)
        if loss is None:
            # Closure optimizers (e.g. LBFGS) populate the cache only inside the
            # closure, which runs after this hook, so an empty cache is expected.
            if not trainer.optimizer.requires_closure:
                warnings.warn(
                    "Recomputing loss: the train-loss cache was empty "
                    "(expected populated).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            batch = (
                trainer.current_train_batches.get(container.label)
                or container.next_train_batch()
            )
            loss, _ = container.compute_train_batch_loss_and_residuals(
                model, batch, detach=False
            )
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        all_grads.append(
            torch.cat([g.detach().reshape(-1) for g in grads if g is not None])
        )
    return torch.cat(all_grads)


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
