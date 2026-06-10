from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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


def build_trainer_module(
    config: TrainerModuleConfig,
    base_dir: Path | None = None,
) -> TrainerModule:
    """
    Build a TrainerModule from a TrainerModuleConfig.

    Parameters
    ----------
    config:
        Config with either method_name or file_path + class_name, plus
        hyperparameters passed as keyword arguments to the constructor.
    base_dir:
        Optional base directory for resolving relative file paths.

    Returns
    -------
    TrainerModule
    """
    if config.is_builtin:
        assert config.method_name is not None
        method_name = config.method_name.lower()
        if method_name not in TrainerModuleLabels._value2member_map_:
            valid = [member.value for member in TrainerModuleLabels]
            raise ValueError(
                f"Unknown trainer module '{config.method_name}'. "
                f"Valid options: {valid}."
            )
        module_cls = TRAINER_MODULE_REGISTRY[TrainerModuleLabels(method_name)]
    else:
        raise NotImplementedError(
            "Loading custom trainer modules is not yet implemented."
        )

    return module_cls(**config.hyperparameters)


def build_trainer_modules(
    configs: list[TrainerModuleConfig],
    base_dir: Path | None = None,
) -> list[TrainerModule]:
    """
    Build all TrainerModules from a list of TrainerModuleConfigs.
    """
    return [build_trainer_module(config, base_dir=base_dir) for config in configs]


def build_trainer(
    config: TrainerConfig,
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
    model:
        The neural network to train.
    optimizer:
        The already-built optimizer.
    loss_containers:
        Label-keyed loss term containers.
    base_dir:
        Optional base directory for resolving custom trainer module paths.

    Returns
    -------
    Trainer
    """
    trainer_modules = build_trainer_modules(config.trainer_modules, base_dir=base_dir)

    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_containers=loss_containers,
        trainer_modules=trainer_modules,
        max_iter=config.max_iter,
        warmup_iters=config.warmup_iters,
        print_freq=config.print_freq,
    )
