"""How this (and other bits) should work:

We need to make sure that:
    (1) every Dataloss label corresponds to
        (a) an input_field or a derived_fields(input_fields, input_consts,
            derived_consts(input_consts)); and
        (b) an output_field or a derived_fields(output_field, input_consts,
            derived_consts(input_consts), learnt_consts, derived_consts(learnt_consts))

    (2) All inputs for physcis loss residual functions can also be calculated similarly
        to 1b

We need to keep track of all of these in a way that makes sense, so that everything is
calculated and we know exactly what order things are in in tensors

Have a function here that loads equations from the file_path given

Make a registry here based on the labels of equations in yaml file that allows us to
execute from registry

For calculations using inputed equations:
    (1) work out what parameters need to be calculated
    (2) work out what quantities are needed for calculation.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from .input_parameters import EquationsParams


def load_functions_from_file_to_registry(equations_params: EquationsParams) -> None:
    # Loop through each equation given in the parameter file
    for equation in equations_params.equations:
        # Extract function params
        label = equation["label"]
        function_name = equation["function_name"]
        file_path = os.path.abspath(equation["file_path"])

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Equation filepath {file_path} does not exist")

        # Namespace for exec
        namespace: dict[str, Any] = {}

        # Read and execute the file in the isolated namespace
        with open(file_path) as f:
            code = f.read()
            exec(code, namespace)

        # Check that function exists
        if function_name not in namespace:
            raise AttributeError(f"Function '{function_name}' not found in {file_path}")

        # Store callable to registry dict
        EQUATION_REGISTRY[label] = namespace[function_name]


EQUATION_REGISTRY: dict[str, Callable[..., Any]] = {}
