from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.config import LossTermConfig
from ..loss.loss_term import LOSS_TERM_REGISTRY, LossTerm
from ..loss.loss_term_container import LossTermContainer

if TYPE_CHECKING:
    from ..geometry.domain import Domain
    from ..loss.evaluation_context import EvaluationContext


def build_loss_term(config: LossTermConfig) -> LossTerm:
    """
    Build a LossTerm (DataLoss or PhysicsLoss) from a LossTermConfig.

    Parameters
    ----------
    config:
        Config for this loss term.

    Returns
    -------
    LossTerm
    """
    loss_term_cls = LOSS_TERM_REGISTRY[config.loss_type]
    return loss_term_cls(field_label=config.field_label)


def build_loss_term_container(
    loss_term_config: LossTermConfig,
    train_eval_context: EvaluationContext,
    val_eval_context: EvaluationContext,
    domain: Domain,
) -> LossTermContainer:
    """
    Build a LossTermContainer from a single LossTermConfig, combining
    training and validation data sets

    Parameters
    ----------
    loss_term_config:
        Config for the loss term in this container, including weighting and
        staggering parameters.
    train_eval_context:
        Pre-built EvaluationContext for training.
    val_eval_context:
        Pre-built EvaluationContext for validation.
    domain:
        Domain object for this coordinate set.

    Returns
    -------
    LossTermContainer
    """
    loss_term = build_loss_term(loss_term_config)
    return LossTermContainer(
        label=loss_term_config.label,
        train_eval_context=train_eval_context,
        val_eval_context=val_eval_context,
        domain=domain,
        loss_term=loss_term,
        loss_term_weight=loss_term_config.weight_value,
        weight_method_label=loss_term_config.weight_method_label,
        weight_group_label=loss_term_config.weight_group_label,
        staggering=loss_term_config.staggering,
        adaptive_point_weights=loss_term_config.adaptive_point_weights,
    )
