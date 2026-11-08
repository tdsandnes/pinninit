from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from .activation_function import ACTIVATION_FUNC_REGISTRY
from .initialise_biases import BIAS_INIT_REGISTRY
from .initialise_weights import WEIGHT_INIT_REGISTRY
from .input_transformation import InputTransformation
from .layers import LAYER_TYPE_REGISTRY

if TYPE_CHECKING:
    from ..config.config import NNConfig
    from ..data_structure.coords import Coords


class NNTypeLabels(str, Enum):
    """Enumeration of neural network type labels."""

    FEEDFORWARD = "feedforward"
    MODIFIED_MLP = "modified_mlp"


class NeuralNetwork(nn.Module, ABC):
    """Base class for all network architectures."""

    output_field_labels: tuple[str, ...]

    @abstractmethod
    def evaluate(self, coords: Coords) -> torch.Tensor:
        pass

    def named_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Subsets of parameters that the builder can assign
        custom optimizer settings (e.g. weight decay) to."""
        return {}


class FeedforwardNN(NeuralNetwork):
    def __init__(
        self,
        output_field_labels: tuple[str, ...],
        layers: list[nn.Module],
        nn_config: NNConfig,
        input_transformation: InputTransformation | None = None,
    ):
        super().__init__()
        self.output_field_labels = output_field_labels
        self.layers = nn.ModuleList(layers)
        activation_function_labels = nn_config.activation_functions
        init_weight_labels = nn_config.init_weights
        init_bias_labels = nn_config.init_biases

        for i, layer in enumerate(self.layers):
            assert isinstance(layer.weight, torch.Tensor)
            assert isinstance(layer.bias, torch.Tensor)
            WEIGHT_INIT_REGISTRY[init_weight_labels[i]](layer.weight)
            BIAS_INIT_REGISTRY[init_bias_labels[i]](layer.bias)

        self.activation_funcs = nn.ModuleList(
            [
                ACTIVATION_FUNC_REGISTRY[activation_function_labels[i]]()
                for i in range(len(self.layers))
            ]
        )
        self.input_transformation = input_transformation

        # Weight normalisation
        if nn_config.weight_norm:
            self.layers = nn.ModuleList([weight_norm(layer) for layer in self.layers])

    def evaluate(self, coords: Coords) -> torch.Tensor:
        if self.input_transformation is not None:
            tensor = self.input_transformation.transform(coords)
        else:
            tensor = coords.internal
        return self.forward(tensor)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            tensor = layer(tensor)
            tensor = self.activation_funcs[i](tensor)
        return tensor


class ModifiedMLP(NeuralNetwork):
    """Modified MLP (Wang et al. 2021), as used in Anagnostopoulos et al. (2024).

    Two encoders U and V embed the input into the hidden width, and the
    output Z of each hidden layer is mixed as H = (1 - Z) * U + Z * V.
    Requires all hidden layers to share a single width."""

    def __init__(
        self,
        output_field_labels: tuple[str, ...],
        layers: list[nn.Module],
        nn_config: NNConfig,
        input_transformation: InputTransformation | None = None,
    ):
        super().__init__()
        self.output_field_labels = output_field_labels
        self.layers = nn.ModuleList(layers)
        activation_function_labels = nn_config.activation_functions
        init_weight_labels = nn_config.init_weights
        init_bias_labels = nn_config.init_biases

        for i, layer in enumerate(self.layers):
            assert isinstance(layer.weight, torch.Tensor)
            assert isinstance(layer.bias, torch.Tensor)
            WEIGHT_INIT_REGISTRY[init_weight_labels[i]](layer.weight)
            BIAS_INIT_REGISTRY[init_bias_labels[i]](layer.bias)

        self.activation_funcs = nn.ModuleList(
            [
                ACTIVATION_FUNC_REGISTRY[activation_function_labels[i]]()
                for i in range(len(self.layers))
            ]
        )
        self.input_transformation = input_transformation

        # Encoders U and V map the input to the hidden width.
        # Their layer type, activation, and init are set independently in the config.
        assert nn_config.encoder_layer_type is not None
        assert nn_config.encoder_activation_function is not None
        assert nn_config.encoder_init_weights is not None
        assert nn_config.encoder_init_biases is not None

        in_features = layers[0].in_features
        hidden_width = layers[0].out_features
        self.encoder_u = LAYER_TYPE_REGISTRY[nn_config.encoder_layer_type](
            in_features=in_features, out_features=hidden_width
        )
        self.encoder_v = LAYER_TYPE_REGISTRY[nn_config.encoder_layer_type](
            in_features=in_features, out_features=hidden_width
        )

        for encoder in (self.encoder_u, self.encoder_v):
            assert isinstance(encoder.weight, torch.Tensor)
            assert isinstance(encoder.bias, torch.Tensor)
            WEIGHT_INIT_REGISTRY[nn_config.encoder_init_weights](encoder.weight)
            BIAS_INIT_REGISTRY[nn_config.encoder_init_biases](encoder.bias)

        self.encoder_activation = ACTIVATION_FUNC_REGISTRY[
            nn_config.encoder_activation_function
        ]()

        # Skip connections
        self.skip_connection = nn_config.skip_connection
        self.skip_alphas = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(len(self.layers) - 2)]
            if self.skip_connection
            else []
        )

        # Weight normalisation
        if nn_config.weight_norm:
            self.layers = nn.ModuleList([weight_norm(layer) for layer in self.layers])
            self.encoder_u = weight_norm(self.encoder_u)
            self.encoder_v = weight_norm(self.encoder_v)

    def named_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "encoder": list(self.encoder_u.parameters())
            + list(self.encoder_v.parameters())
        }

    def evaluate(self, coords: Coords) -> torch.Tensor:
        if self.input_transformation is not None:
            tensor = self.input_transformation.transform(coords)
        else:
            tensor = coords.internal
        return self.forward(tensor)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Encoders
        u = self.encoder_activation(self.encoder_u(tensor))
        v = self.encoder_activation(self.encoder_v(tensor))

        # Apply encoders and then skip back to the block's input if skip_connection
        state = tensor
        for i in range(0, len(self.layers) - 1):
            layer_input = state
            after_act_func = self.activation_funcs[i](self.layers[i](layer_input))
            after_encoders = (1.0 - after_act_func) * u + after_act_func * v
            if i == 0 or not self.skip_connection:
                # No skip connection to first layer since this is needed to map to the
                # correct matrix shape
                state = after_encoders
            else:
                skip_alpha = self.skip_alphas[i - 1]
                state = skip_alpha * after_encoders + (1.0 - skip_alpha) * layer_input

        # Output layer: no encoders and no skip
        output: torch.Tensor = self.activation_funcs[-1](self.layers[-1](state))
        return output


# Mapping from NN type labels to classes
NN_TYPE_REGISTRY: dict[NNTypeLabels, type[NeuralNetwork]] = {
    NNTypeLabels.FEEDFORWARD: FeedforwardNN,
    NNTypeLabels.MODIFIED_MLP: ModifiedMLP,
}
