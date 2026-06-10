from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..config.config import FieldsConfig, NNConfig
from ..nn.input_transformation import (
    InputTransformation,
    InputTransformationLabels,
    PeriodicW23,
    ScaleLinear,
)
from ..nn.layers import LAYER_TYPE_REGISTRY
from ..nn.neural_network import NN_TYPE_REGISTRY, NeuralNetwork

if TYPE_CHECKING:
    from ..geometry.domain import Domain
    from ..loss.loss_term_container import LossTermContainer


def build_nn(
    nn_config: NNConfig,
    fields_config: FieldsConfig,
    domains: list[Domain],
    internal_labels: tuple[str, ...],
    periodic_coords: dict[str, float],
    dim_geometric: int,
    device: torch.device,
) -> NeuralNetwork:
    """
    Build a NeuralNetwork from an NNConfig.

    Parameters
    ----------
    nn_config:
        Network architecture config.
    fields_config:
        Fields config needed for output fields determine the network output width.
    domains:
        list of domains for getting upper and lower bounds for coord scaling
    internal_labels:
        Ordered internal coordinate labels of the network input.
    periodic_coords:
        Mapping of periodic coordinate label to period. Empty if no periodicity.
    dim_geometric:
        Geometric dimension, used to compute the number of components per field.
    device:
        PyTorch device.

    Returns
    -------
    NeuralNetwork
    """
    # Get upper and lower bounds
    lb, ub = _collect_domain_bounds(domains, internal_labels)

    # Get input transformation
    input_transformation = _build_input_transformation(
        nn_config, internal_labels, lb, ub, periodic_coords
    )

    # Size of input and output features
    n_input_features = (
        input_transformation.out_features
        if input_transformation is not None
        else len(internal_labels)
    )
    n_output_features = sum(
        dim_geometric**rank
        for rank in fields_config.output_field_label_to_rank.values()
    )

    # Set up layers
    hidden_layer_widths = nn_config.hidden_layer_widths
    num_layers = len(hidden_layer_widths) + 1
    in_features = [n_input_features] + hidden_layer_widths
    out_features = hidden_layer_widths + [n_output_features]

    layers = [
        LAYER_TYPE_REGISTRY[nn_config.layer_types[i]](
            in_features=in_features[i],
            out_features=out_features[i],
        )
        for i in range(num_layers)
    ]

    # Get Neural Network
    output_field_labels = tuple(field.label for field in fields_config.output_fields)
    nn_cls = NN_TYPE_REGISTRY[nn_config.nn_type]
    model = nn_cls(
        output_field_labels,
        layers,
        nn_config,
        input_transformation=input_transformation,
    )
    return model.to(device)


def _build_input_transformation(
    nn_config: NNConfig,
    internal_labels: tuple[str, ...],
    lb: torch.Tensor,
    ub: torch.Tensor,
    periodic_coords: dict[str, float],
) -> InputTransformation | None:
    """Build the input transformation from config and domain periodicity."""
    has_scaling = (
        nn_config.input_transformation == InputTransformationLabels.SCALE_LINEAR
    )
    has_periodic = bool(periodic_coords)

    if has_periodic:
        return PeriodicW23(
            internal_labels=list(internal_labels),
            periodic_coords=periodic_coords,
            lb=lb,
            ub=ub,
            scale_non_periodic=has_scaling,
        )
    if has_scaling:
        return ScaleLinear(lb=lb, ub=ub)
    return None


def _collect_domain_bounds(
    domains: list[Domain],
    internal_labels: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get internal bounds from list of domains."""
    lbs = torch.stack(
        [
            torch.stack([domain.internal_lb[label] for label in internal_labels])
            for domain in domains
        ]
    )
    ubs = torch.stack(
        [
            torch.stack([domain.internal_ub[label] for label in internal_labels])
            for domain in domains
        ]
    )
    return torch.min(lbs, dim=0).values, torch.max(ubs, dim=0).values


def get_internal_labels(
    loss_containers: dict[str, LossTermContainer],
) -> tuple[str, ...]:
    """Get labels of internal coords from loss term containers."""
    if not loss_containers:
        raise ValueError("No loss term containers available to initialise network.")

    all_coords = [
        container.train_eval_context.coords for container in loss_containers.values()
    ]
    internal_labels = all_coords[0].internal_labels
    for coords in all_coords[1:]:
        if coords.internal_labels != internal_labels:
            raise ValueError(
                "Inconsistent internal coordinate labels across loss term "
                f"containers: {[c.internal_labels for c in all_coords]}"
            )
    return internal_labels


def collect_periodic_coords(domains: list[Domain]) -> dict[str, float]:
    """Collect periodic coordinate labels and periods across domains."""
    periodic_coords: dict[str, float] = {}
    for domain in domains:
        for label, period in domain.periodic_coords.items():
            if label in periodic_coords and periodic_coords[label] != period:
                raise ValueError(
                    f"Periodic coordinate '{label}' defined with different periods "
                    "in multiple domains."
                )
            periodic_coords[label] = period
    return periodic_coords
