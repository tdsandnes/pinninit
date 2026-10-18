from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..loss.loss_function import LOSS_FUNCTION_REGISTRY, LossFunctionLabels

if TYPE_CHECKING:
    from ..loss.loss_term_container import LossTermContainer, WeightMethodLabels
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


def adaptive_weight_groups(
    trainer: Trainer, method: WeightMethodLabels
) -> dict[str, list[LossTermContainer]]:
    """The containers weighted by method, including ones not yet activated by
    staggering, by their weight group label."""
    groups: dict[str, list[LossTermContainer]] = {}
    for label, container in trainer.all_loss_containers.items():
        if container.weight_method_label != method:
            continue
        if container.weight_group_label is None:
            raise ValueError(
                f"Loss term '{label}' uses '{method.value}' weighting but has no "
                "weight_group_label."
            )
        groups.setdefault(container.weight_group_label, []).append(container)
    return groups


def active_weight_groups(
    trainer: Trainer, groups: dict[str, list[LossTermContainer]]
) -> dict[str, list[LossTermContainer]]:
    """The containers of each group that are trained in the current step,
    leaving out groups with none."""
    active = {}
    for label, containers in groups.items():
        trained = [c for c in containers if c.label in trainer.loss_containers]
        if trained:
            active[label] = trained
    return active


def adaptive_point_weight_containers(
    trainer: Trainer, module_name: str
) -> dict[str, LossTermContainer]:
    """The containers with adaptive point weights, including ones not yet
    activated by staggering."""
    containers = {
        label: container
        for label, container in trainer.all_loss_containers.items()
        if container.adaptive_point_weights
    }
    if not containers:
        raise ValueError(
            f"{module_name} found no loss terms with adaptive point weights. Set "
            "'adaptive_point_weights: true' on at least one loss term."
        )
    return containers


def check_adaptive_weight_params(
    alpha: float, update_iter_frequency: int, min_iter: int
) -> None:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if update_iter_frequency < 1:
        raise ValueError(
            f"update_iter_frequency must be >= 1, got {update_iter_frequency}"
        )
    if min_iter < 0:
        raise ValueError(f"min_iter must be >= 0, got {min_iter}")


def check_coast_epochs(coast_epochs_after_update: int) -> None:
    if coast_epochs_after_update < 0:
        raise ValueError(
            "coast_epochs_after_update must be >= 0, got "
            f"{coast_epochs_after_update}"
        )


def train_residual_magnitudes(
    trainer: Trainer, container: LossTermContainer
) -> torch.Tensor:
    """Magnitude of the residual at each point of this step's training batch,
    shape (n_points,), with the norm over the components of a vector residual."""
    residuals = trainer.cached_train_residuals(container.label)
    if residuals is None:
        raise RuntimeError(
            f"No cached training residuals for loss term '{container.label}'."
        )
    residuals = residuals.detach()
    if residuals.ndim == 1:
        return residuals.abs()
    magnitudes: torch.Tensor = residuals.norm(dim=1)
    return magnitudes


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
