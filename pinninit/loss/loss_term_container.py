from __future__ import annotations

from collections.abc import Generator
from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..data_structure.coords_dataclass import CoordsProtocol
    from ..geometry.domain import Domain
    from ..nn.neural_network import NeuralNetwork
    from .evaluation_context import EvaluationContext
    from .loss_term import LossTerm


class WeightMethodLabels(str, Enum):
    """Enumeration of loss term weight method labels."""

    CONSTANT = "constant"
    ADAPTIVE_W21 = "adaptive_w21"
    ADAPTIVE_W23 = "adaptive_w23"


def _iter_batches(
    dataloader: DataLoader[CoordsProtocol],
) -> Iterator[CoordsProtocol]:
    """Typed boundary around DataLoader iteration: torch's stubs type the
    iterator's __next__ as Any, hiding the collate_fn's return type."""
    yield from dataloader


class LossTermContainer:
    """Container of a loss term and relevant parameters needed for it's evaluation
    and its contribution to the full loss function."""

    def __init__(
        self,
        label: str,
        train_eval_context: EvaluationContext,
        val_eval_context: EvaluationContext,
        domain: Domain,
        loss_term: LossTerm,
        loss_term_weight: float = 1.0,
        weight_method_label: WeightMethodLabels = WeightMethodLabels.CONSTANT,
        weight_group_label: str | None = None,
        staggering: int = 0,
    ) -> None:
        self.label = label
        self.train_eval_context = train_eval_context
        self.val_eval_context = val_eval_context
        self.domain = domain
        self.loss_term = loss_term
        self.loss_term_weight = loss_term_weight

        # Weighting and staggering parameters used by the trainer and its modules
        self.initial_weight = loss_term_weight
        self.weight_method_label = weight_method_label
        self.weight_group_label = weight_group_label
        self.staggering = staggering

        # Iterators that cycle through training and validation batches across steps
        self._train_batch_iter: Iterator[CoordsProtocol] | None = None
        self._val_batch_iter: Iterator[CoordsProtocol] | None = None

        # Check that training and validation points are in the specified domain
        train_eval_context.coords.validate_in_domain(domain)
        val_eval_context.coords.validate_in_domain(domain)

    def _apply_model(
        self,
        model: NeuralNetwork,
        coords: CoordsProtocol,
        eval_context: EvaluationContext,
    ) -> None:
        """Run model at coords and write results into eval_context.output_data."""
        output_tensor = model.evaluate(coords)
        idx = 0
        for field_label in model.output_field_labels:
            # Get field from output_data to set n_components
            field = eval_context.output_data.get_field(field_label)
            n_components = len(field.component_labels)

            # Set field based on NN output
            value = output_tensor[:, idx : idx + n_components]
            eval_context.output_data.set_field_from_component_tensor(
                field_label, value, coords=coords
            )
            idx += n_components

        # Evaluate equations if the loss term needs a derived field
        if self.loss_term.field_label in eval_context.output_data.derived_field_labels:
            eval_context.output_data.evaluate_derived_fields(coords)

    def compute_loss(
        self,
        model: NeuralNetwork,
        eval_context: EvaluationContext,
        eval_coords: CoordsProtocol,
        detach: bool,
    ) -> torch.Tensor:
        self._apply_model(model, eval_coords, eval_context)
        loss = self.loss_term.compute_loss(eval_context, eval_coords).sum()
        return loss.detach() if detach else loss

    def _loss_generator(
        self,
        model: NeuralNetwork,
        eval_context: EvaluationContext,
        detach: bool,
    ) -> Generator[torch.Tensor, None, None]:
        """Yield one batch loss at a time."""
        for batch_coords in eval_context.dataloader:
            yield self.compute_loss(model, eval_context, batch_coords, detach)

    def next_train_batch(self) -> CoordsProtocol:
        """Return the next batch of training coords, cycling the dataloader."""
        if self._train_batch_iter is None:
            self._train_batch_iter = _iter_batches(self.train_eval_context.dataloader)
        try:
            return next(self._train_batch_iter)
        except StopIteration:
            self._train_batch_iter = _iter_batches(self.train_eval_context.dataloader)
            return next(self._train_batch_iter)

    def reset_train_batches(self) -> None:
        """Restart batch cycling from the beginning of the dataloader."""
        self._train_batch_iter = None

    def next_val_batch(self) -> CoordsProtocol:
        """Return the next batch of validation coords, cycling the dataloader."""
        if self._val_batch_iter is None:
            self._val_batch_iter = _iter_batches(self.val_eval_context.dataloader)
        try:
            return next(self._val_batch_iter)
        except StopIteration:
            self._val_batch_iter = _iter_batches(self.val_eval_context.dataloader)
            return next(self._val_batch_iter)

    def reset_val_batches(self) -> None:
        """Restart batch cycling from the beginning of the dataloader."""
        self._val_batch_iter = None

    def compute_train_batch_loss(
        self, model: NeuralNetwork, batch_coords: CoordsProtocol, detach: bool = False
    ) -> torch.Tensor:
        return self.compute_loss(
            model, self.train_eval_context, batch_coords, detach=detach
        )

    def compute_val_batch_loss(
        self, model: NeuralNetwork, batch_coords: CoordsProtocol, detach: bool = True
    ) -> torch.Tensor:
        return self.compute_loss(
            model, self.val_eval_context, batch_coords, detach=detach
        )

    def compute_unweighted_train_loss(
        self, model: NeuralNetwork, detach: bool = True
    ) -> torch.Tensor:
        return self.compute_loss(
            model,
            self.train_eval_context,
            self.train_eval_context.coords,
            detach=detach,
        )

    def compute_weighted_train_loss(
        self, model: NeuralNetwork, detach: bool = True
    ) -> torch.Tensor:
        loss = self.compute_unweighted_train_loss(model, detach=detach)
        return self.loss_term_weight * loss

    def compute_unweighted_val_loss(
        self, model: NeuralNetwork, detach: bool = True
    ) -> torch.Tensor:
        return self.compute_loss(
            model, self.val_eval_context, self.val_eval_context.coords, detach=detach
        )

    def compute_weighted_val_loss(
        self, model: NeuralNetwork, detach: bool = True
    ) -> torch.Tensor:
        loss = self.compute_unweighted_val_loss(model, detach=detach)
        return self.loss_term_weight * loss

    def compute_train_residuals(
        self, model: NeuralNetwork, detach: bool = True
    ) -> torch.Tensor:
        """Compute per-point residuals on the full training set."""
        coords = self.train_eval_context.coords
        self._apply_model(model, coords, self.train_eval_context)
        residuals = self.loss_term.compute_residual(self.train_eval_context, coords)
        return residuals.detach() if detach else residuals

    def iter_train_loss(
        self, model: NeuralNetwork, detach: bool = False
    ) -> Generator[torch.Tensor, None, None]:
        yield from self._loss_generator(model, self.train_eval_context, detach)

    def iter_weighted_train_loss(
        self, model: NeuralNetwork, detach: bool = False
    ) -> Generator[torch.Tensor, None, None]:
        for loss in self._loss_generator(model, self.train_eval_context, detach):
            yield self.loss_term_weight * loss

    def iter_val_loss(
        self, model: NeuralNetwork, detach: bool = True
    ) -> Generator[torch.Tensor, None, None]:
        yield from self._loss_generator(model, self.val_eval_context, detach)

    def iter_weighted_val_loss(
        self, model: NeuralNetwork, detach: bool = True
    ) -> Generator[torch.Tensor, None, None]:
        for loss in self._loss_generator(model, self.val_eval_context, detach):
            yield self.loss_term_weight * loss

    def compute_full_val_loss(self, model: NeuralNetwork) -> torch.Tensor:
        """Exact unweighted loss over the full validation set, evaluated in
        dataloader-sized chunks."""
        total: torch.Tensor | None = None
        n_total = 0
        for batch_coords in self.val_eval_context.dataloader:
            batch_loss = self.compute_loss(
                model, self.val_eval_context, batch_coords, detach=True
            )
            n = batch_coords.n_points
            total = batch_loss * n if total is None else total + batch_loss * n
            n_total += n

        if total is None:
            raise RuntimeError(f"Validation dataloader for '{self.label}' is empty.")

        return total / n_total
