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
    label: str,
    loss_term_config: LossTermConfig,
    train_eval_context: EvaluationContext,
    val_eval_context: EvaluationContext,
    domain: Domain,
    loss_term_weight: float = 1.0,
) -> LossTermContainer:
    """
    Build a LossTermContainer from a single LossTermConfigs, combining
    training and validation data sets

    Parameters
    ----------
    label:
        Label for this container.
    loss_term_config:
        Configs for loss term in this container.
    train_eval_context:
        Pre-built EvaluationContext for training.
    val_eval_context:
        Pre-built EvaluationContext for validation.
    domain:
        Domain object for this coordinate set.
    loss_term_weight:
        Scalar weight applied to the container's total loss.

    Returns
    -------
    LossTermContainer
    """
    loss_term = build_loss_term(loss_term_config)
    return LossTermContainer(
        label=label,
        train_eval_context=train_eval_context,
        val_eval_context=val_eval_context,
        domain=domain,
        loss_term=loss_term,
        loss_term_weight=loss_term_weight,
    )
