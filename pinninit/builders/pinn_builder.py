from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..pinn.pinn import PINN
from .coord_system_builder import build_coord_systems
from .coords_builder import build_coords_classes
from .equation_builder import build_equations, build_point_weight_modifier_equation
from .evaluation_context_builder import build_evaluation_contexts
from .loss_builder import build_loss_term_container
from .nn_builder import build_nn, collect_periodic_coords, get_internal_labels
from .optimizer_builder import build_optimizer
from .trainer_builder import build_trainer

if TYPE_CHECKING:
    from pathlib import Path

    from ..config.pinn_config import PINNConfig
    from ..loss.loss_term_container import LossTermContainer
    from ..nn.neural_network import NeuralNetwork


def build_pinn(
    config: PINNConfig,
    device: torch.device,
    base_dir: Path | None = None,
) -> PINN:
    """
    Build a complete PINN from a PINNConfig.

    Builds, in order: coordinate systems and domains, equations, evaluation
    contexts and loss term containers, learnable constants, the neural
    network, the optimizer, and the trainer.

    Parameters
    ----------
    config:
        Full PINN configuration, e.g. loaded via PINNConfig.from_yaml().
    device:
        PyTorch device.
    base_dir:
        Optional base directory for resolving relative file paths (equations,
        custom trainer modules).

    Returns
    -------
    PINN
    """
    # Geometry
    coord_systems = build_coord_systems(config.coord_systems, device)
    coords_classes = build_coords_classes(
        config.coordinates, config.coord_systems, coord_systems, device
    )

    # Equations
    equations = build_equations(config.equations, base_dir=base_dir)

    # Loss term containers
    datasets_by_label = {dataset.label: dataset for dataset in config.datasets}
    loss_containers: dict[str, LossTermContainer] = {}

    for loss_term_config in config.loss:
        if loss_term_config.label in loss_containers:
            raise ValueError(f"Duplicate loss term label: '{loss_term_config.label}'.")

        if loss_term_config.coordinates_label not in coords_classes:
            raise ValueError(
                f"LossTermConfig '{loss_term_config.label}' references unknown "
                f"Coordinates '{loss_term_config.coordinates_label}'."
            )
        coords_class, domain = coords_classes[loss_term_config.coordinates_label]

        dataset_config = (
            datasets_by_label.get(loss_term_config.dataset_label)
            if loss_term_config.dataset_label is not None
            else None
        )

        point_weight_equation = (
            build_point_weight_modifier_equation(
                loss_term_config.label,
                loss_term_config.point_weight_modifier,
                base_dir=base_dir,
            )
            if loss_term_config.point_weight_modifier is not None
            else None
        )

        train_eval_context, val_eval_context = build_evaluation_contexts(
            loss_term_config=loss_term_config,
            dataset_config=dataset_config,
            coords_class=coords_class,
            fields_config=config.fields,
            const_configs=config.constants,
            equations=equations,
            point_weight_equation=point_weight_equation,
            n_batches=config.trainer.n_batches,
            shuffle=config.trainer.shuffle,
            num_workers=config.trainer.num_workers,
        )

        loss_containers[loss_term_config.label] = build_loss_term_container(
            loss_term_config=loss_term_config,
            train_eval_context=train_eval_context,
            val_eval_context=val_eval_context,
            domain=domain,
        )

    # Learnable constants
    learnt_consts: dict[str, nn.Parameter] = {
        const_config.label: nn.Parameter(
            torch.tensor(const_config.value, device=device)
        )
        for const_config in config.constants
        if const_config.learnable
    }
    for container in loss_containers.values():
        for eval_context in (
            container.train_eval_context,
            container.val_eval_context,
        ):
            for label, param in learnt_consts.items():
                eval_context.output_data.set_const(label, param)

    # Neural network
    domains = [container.domain for container in loss_containers.values()]
    internal_labels = get_internal_labels(loss_containers)
    periodic_coords = collect_periodic_coords(domains)
    dim_geometric = next(iter(coord_systems.values())).dim_geometric

    model = build_nn(
        nn_config=config.nn,
        fields_config=config.fields,
        domains=domains,
        internal_labels=internal_labels,
        periodic_coords=periodic_coords,
        dim_geometric=dim_geometric,
        device=device,
    )

    # Register learnable constants on the model so they live in its
    # state_dict: best-model tracking and checkpoints then include them
    for label, param in learnt_consts.items():
        model.register_parameter(f"learnt_const_{label}", param)

    # Optimizer over model parameters
    params: list[nn.Parameter] | list[dict[str, object]]
    wd_groups = config.trainer.optimizer.weight_decay_groups
    if wd_groups:
        params = _build_param_groups(model, wd_groups)
    else:
        params = list(model.parameters())
    optimizer = build_optimizer(config.trainer.optimizer, params)

    # Trainer
    trainer = build_trainer(
        config=config.trainer,
        model=model,
        optimizer=optimizer,
        loss_containers=loss_containers,
        base_dir=base_dir,
    )

    return PINN(
        config=config,
        device=device,
        model=model,
        loss_containers=loss_containers,
        learnt_consts=learnt_consts,
        optimizer=optimizer,
        trainer=trainer,
    )


def _build_param_groups(
    model: NeuralNetwork, weight_decay_groups: dict[str, float]
) -> list[dict[str, object]]:
    """Map a {group_name: weight_decay} onto the model's named parameter groups."""
    named = model.named_parameter_groups()
    unknown = set(weight_decay_groups) - set(named)
    if unknown:
        raise ValueError(
            f"weight_decay_groups references {sorted(unknown)}, but the model "
            f"only declares {sorted(named)}."
        )

    decayed_groups: list[dict[str, object]] = []
    claimed: set[int] = set()
    for name, decay in weight_decay_groups.items():
        params = named[name]
        decayed_groups.append({"params": params, "weight_decay": decay})
        claimed.update(id(p) for p in params)

    default = [p for p in model.parameters() if id(p) not in claimed]
    return [{"params": default}, *decayed_groups]
