from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config.config import TrainerConfig, TrainerModuleConfig
from ..trainer.trainer import Trainer
from ..trainer_modules.trainer_module import TrainerModule
from ..trainer_modules.trainer_module_registry import (
    TRAINER_MODULE_REGISTRY,
    TrainerModuleLabels,
)

if TYPE_CHECKING:
    from ..loss.loss_term_container import LossTermContainer
    from ..nn.neural_network import NeuralNetwork
    from ..optimizer.optimizer import Optimizer


def build_trainer_module(config: TrainerModuleConfig) -> TrainerModule:
    """
    Build a TrainerModule from a TrainerModuleConfig.

    Parameters
    ----------
    config:
        Config with the module's method_name, plus hyperparameters passed as
        keyword arguments to its constructor.

    Returns
    -------
    TrainerModule
    """
    method_name = config.method_name.lower()
    valid = [member.value for member in TrainerModuleLabels]
    if method_name not in valid:
        raise ValueError(
            f"Unknown trainer module '{config.method_name}'. Valid options: {valid}."
        )

    module_class = TRAINER_MODULE_REGISTRY[TrainerModuleLabels(method_name)]
    return module_class(**config.hyperparameters)


def build_trainer(
    config: TrainerConfig,
    parameters: dict[str, Any] | None,
    model: NeuralNetwork,
    optimizer: Optimizer,
    loss_containers: dict[str, LossTermContainer],
    base_dir: Path | None = None,
) -> Trainer:
    """
    Build a Trainer from a TrainerConfig and already-built components.

    Parameters
    ----------
    config:
        Trainer config with loop parameters and trainer module configs.
    parameters:
        The parameter file's contents, saved in checkpoints.
    model:
        The neural network to train.
    optimizer:
        The already-built optimizer.
    loss_containers:
        Label-keyed loss term containers.
    base_dir:
        Optional base directory for resolving the checkpoint path.

    Returns
    -------
    Trainer
    """
    trainer_modules = [
        build_trainer_module(module) for module in config.trainer_modules
    ]

    # A relative checkpoint path resolves against base_dir
    checkpoint_path = None
    if config.checkpoint_path is not None:
        checkpoint_path = Path(base_dir or ".") / config.checkpoint_path

    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_containers=loss_containers,
        trainer_modules=trainer_modules,
        max_epoch=config.max_epoch,
        warmup_epochs=config.warmup_epochs,
        print_freq=config.print_freq,
        log_point_weight_info=config.log_point_weight_info,
        point_weight_snapshot_freq=config.point_weight_snapshot_freq,
        checkpoint_path=checkpoint_path,
        checkpoint_freq=config.checkpoint_freq,
        stop_on_sigterm=config.stop_on_sigterm,
        parameters=parameters,
    )
