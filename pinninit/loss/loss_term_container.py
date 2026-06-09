from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..data_structure.coords_dataclass import CoordsProtocol
    from ..geometry.domain import Domain
    from ..nn.neural_network import NeuralNetwork
    from .evaluation_context import EvaluationContext
    from .loss_term import LossTerm


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
    ) -> None:
        self.label = label
        self.train_eval_context = train_eval_context
        self.val_eval_context = val_eval_context
        self.domain = domain
        self.loss_term = loss_term
        self.loss_term_weight = loss_term_weight

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
