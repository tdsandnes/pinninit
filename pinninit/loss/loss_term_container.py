from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

from .loss_function import mse_loss

if TYPE_CHECKING:
    from ..data_structure.coords import Coords
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

        # Point weights only have an effect with a weighted loss function
        uses_point_weights = (
            adaptive_point_weights
            or train_eval_context.point_weight_modifier is not None
        )
        if uses_point_weights and train_eval_context.loss_function is mse_loss:
            raise ValueError(
                f"Loss term '{label}' has point weights, but its 'mse' loss "
                "function ignores them. Use a weighted loss function."
            )

        # Check that the loss term can be computed, and that training and
        # validation points are in the specified domain
        for eval_context in (train_eval_context, val_eval_context):
            loss_term.check(eval_context)
            eval_context.coords.validate_in_domain(domain)

    def _apply_model(
        self,
        model: NeuralNetwork,
        coords: Coords,
        eval_context: EvaluationContext,
    ) -> None:
        """Run model at coords and write results into eval_context.output_data."""
        output_tensor = model.evaluate(coords)
        idx = 0
        for field_label in model.output_field_labels:
            field_class = eval_context.output_data.field_classes[field_label]
            n_components = len(field_class.component_labels)

            # Set field based on NN output
            value = output_tensor[:, idx : idx + n_components]
            eval_context.output_data.set_field_from_component_tensor(
                field_label, value, coords=coords
            )
            idx += n_components

        # Evaluate the equations the loss term's field needs, if it is derived
        field_label = self.loss_term.field_label
        if field_label in eval_context.output_data.derived_field_labels:
            eval_context.output_data.evaluate_derived_fields(coords, [field_label])

    def compute_loss_and_residuals(
        self,
        model: NeuralNetwork,
        eval_context: EvaluationContext,
        eval_coords: Coords,
        detach: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._apply_model(model, eval_coords, eval_context)
        loss, residuals = self.loss_term.compute_loss_and_residuals(
            eval_context, eval_coords
        )
        loss = loss.sum().detach() if detach else loss.sum()
        residuals = residuals.detach() if detach else residuals
        return loss, residuals

    def _compute_full_loss(
        self, model: NeuralNetwork, eval_context: EvaluationContext
    ) -> torch.Tensor:
        """Loss over all points of an eval context, detached.

        The residuals are computed one batch at a time, so only one batch's
        autograd graph is alive at a time, and the loss is then computed from
        the residuals at all points, so it is exact for any loss function.
        """
        residuals = []
        for batch_coords in eval_context.batches.in_order():
            self._apply_model(model, batch_coords, eval_context)
            residuals.append(
                self.loss_term.compute_residual(eval_context, batch_coords).detach()
            )
            # Free the batch's autograd graph, which its fields hold
            eval_context.output_data.clear_fields()

        loss = eval_context.loss_function(
            residuals=torch.cat(residuals),
            eval_context=eval_context,
            eval_coords=eval_context.coords,
        )
        return loss.sum()

    def next_train_batch(self) -> Coords:
        """Return the next batch of training coords."""
        return self.train_eval_context.batches.next()

    def reset_train_batches(self) -> None:
        """Start a new pass over the training batches at the next batch."""
        self.train_eval_context.batches.reset()

    def compute_train_batch_loss_and_residuals(
        self, model: NeuralNetwork, batch_coords: Coords, detach: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.compute_loss_and_residuals(
            model, self.train_eval_context, batch_coords, detach=detach
        )

    def compute_full_train_loss(self, model: NeuralNetwork) -> torch.Tensor:
        """Training loss over all training points, detached. Uses the training
        loss function (with point weights), but not the loss term weight."""
        return self._compute_full_loss(model, self.train_eval_context)

    def compute_full_val_loss(self, model: NeuralNetwork) -> torch.Tensor:
        """Validation loss over all validation points, detached. Not multiplied
        by the loss term weight."""
        return self._compute_full_loss(model, self.val_eval_context)
