from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

from ..data_structure.batching import BatchCycler, iter_batches

if TYPE_CHECKING:
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


class LossTermContainer:
    """Container of a loss term and relevant parameters needed for its evaluation
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
        adaptive_point_weights: bool = False,
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
        self.adaptive_point_weights = adaptive_point_weights

        # Cycle through training and validation batches across steps
        self._train_batches = BatchCycler(train_eval_context.dataloader)

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

    def compute_loss_and_residuals(
        self,
        model: NeuralNetwork,
        eval_context: EvaluationContext,
        eval_coords: CoordsProtocol,
        detach: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._apply_model(model, eval_coords, eval_context)
        loss, residuals = self.loss_term.compute_loss_and_residuals(
            eval_context, eval_coords
        )
        loss = loss.sum().detach() if detach else loss.sum()
        return loss, residuals.detach()

    def _compute_full_loss(
        self, model: NeuralNetwork, eval_context: EvaluationContext
    ) -> torch.Tensor:
        """Loss over an eval context's full point set, evaluated
        in dataloader-sized chunks so only one chunk's autograd graph is alive
        at a time.
        """
        total: torch.Tensor | None = None
        n_total = 0
        for batch_coords in iter_batches(eval_context.dataloader):
            batch_loss, _ = self.compute_loss_and_residuals(
                model, eval_context, batch_coords, detach=True
            )
            n = batch_coords.n_points
            total = batch_loss * n if total is None else total + batch_loss * n
            n_total += n

        if total is None:
            raise RuntimeError(f"Dataloader for '{self.label}' is empty.")

        return total / n_total

    def next_train_batch(self) -> CoordsProtocol:
        """Return the next batch of training coords, cycling the dataloader."""
        return self._train_batches.next()

    def reset_train_batches(self) -> None:
        """Restart batch cycling from the beginning of the dataloader."""
        self._train_batches.reset()

    def compute_train_batch_loss_and_residuals(
        self, model: NeuralNetwork, batch_coords: CoordsProtocol, detach: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.compute_loss_and_residuals(
            model, self.train_eval_context, batch_coords, detach=detach
        )

    def compute_val_batch_loss_and_residuals(
        self, model: NeuralNetwork, batch_coords: CoordsProtocol, detach: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.compute_loss_and_residuals(
            model, self.val_eval_context, batch_coords, detach=detach
        )

    def compute_full_train_loss(self, model: NeuralNetwork) -> torch.Tensor:
        """Full unweighted training loss over the full set, chunked and detached."""
        return self._compute_full_loss(model, self.train_eval_context)

    def compute_full_val_loss(self, model: NeuralNetwork) -> torch.Tensor:
        """Full unweighted validation loss over the full set, chunked and detached."""
        return self._compute_full_loss(model, self.val_eval_context)
