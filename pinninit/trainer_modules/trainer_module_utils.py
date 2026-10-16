from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..loss.loss_function import LOSS_FUNCTION_REGISTRY, LossFunctionLabels

if TYPE_CHECKING:
    from ..loss.loss_term_container import LossTermContainer
    from ..nn.neural_network import NeuralNetwork
    from ..trainer.trainer import Trainer


def compute_flat_train_grads(
    model: NeuralNetwork,
    containers: list[LossTermContainer],
    trainer: Trainer,
    loss_function_label: LossFunctionLabels,
) -> torch.Tensor:
    """Flat gradient of each container's loss (using loss_function_label)."""
    loss_fn = LOSS_FUNCTION_REGISTRY[loss_function_label]
    params = list(model.parameters())
    all_grads = []
    for container in containers:
        residuals = trainer.cached_train_residuals(container.label)
        if residuals is None or not residuals.requires_grad:
            raise RuntimeError(
                f"No graph-carrying cached residuals for loss term '{container.label}'."
            )
        # The batch the cached residuals were computed on
        batch = trainer.current_train_batches[container.label]
        loss = loss_fn(residuals, container.train_eval_context, batch).sum()
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        all_grads.append(
            torch.cat([g.detach().reshape(-1) for g in grads if g is not None])
        )
    return torch.cat(all_grads)


class LossPlateauDetector:
    """Detects a plateau when the best loss stops improving.

    A plateau is flagged when the best loss so far has improved by less than a
    fraction `frac_tol` over the last `patience` steps:

        best_now >= (1 - frac_tol) * best_patience_steps_ago

    where the "best" loss is the lowest value seen up to that point.
    """

    def __init__(self, patience: int, frac_tol: float = 0.0) -> None:
        if not 0.0 <= frac_tol <= 1.0:
            raise ValueError(f"frac_tol must be between 0 and 1, got {frac_tol}")

        self.patience = patience
        self.frac_tol = frac_tol

        self.best: float | None = None
        self.best_history: list[float] = []
        self.plateau_detected = False

    def step(self, current_value: float) -> None:
        # Update the running best
        if self.best is None or current_value < self.best:
            self.best = current_value

        # Record it, keeping only the most recent patience + 1 values
        self.best_history.append(self.best)
        if len(self.best_history) > self.patience + 1:
            self.best_history.pop(0)

        # We can only judge a plateau once we have a full window to compare over
        if len(self.best_history) < self.patience + 1:
            self.plateau_detected = False
            return

        # Get the best value now and the best value as it was patience steps ago
        best_now = self.best_history[-1]
        best_patience_steps_ago = self.best_history[0]
        self.plateau_detected = (
            best_now >= (1.0 - self.frac_tol) * best_patience_steps_ago
        )

    def reset(self) -> None:
        self.best = None
        self.best_history = []
        self.plateau_detected = False
