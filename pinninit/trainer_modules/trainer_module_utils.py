from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..data_structure.coords_dataclass import CoordsProtocol
    from ..loss.loss_term_container import LossTermContainer
    from ..nn.neural_network import NeuralNetwork


def compute_flat_train_grads(
    model: NeuralNetwork,
    containers: list[LossTermContainer],
    batches: dict[str, CoordsProtocol] | None = None,
) -> torch.Tensor:
    """Concatenated flat gradient of the containers' unweighted training losses
    with respect to the model parameters.

    If `batches` is given, each container's loss is evaluated on its current
    batch (falling back to drawing the next batch for containers without one);
    otherwise on its full training set in a single graph."""
    all_grads_flat_list = []
    for container in containers:
        if batches is not None:
            batch = batches.get(container.label)
            if batch is None:
                batch = container.next_train_batch()
            loss = container.compute_train_batch_loss(model, batch, detach=False)
        else:
            loss = container.compute_unweighted_train_loss(model, detach=False)
        grads = torch.autograd.grad(
            loss,
            list(model.parameters()),
            retain_graph=True,
            allow_unused=True,
        )
        grads_flat = torch.cat([g.detach().reshape(-1) for g in grads if g is not None])
        all_grads_flat_list.append(grads_flat)

    return torch.cat(all_grads_flat_list)


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
