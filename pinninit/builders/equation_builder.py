from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

from ..config.config import EquationConfig, PointWeightModifierConfig
from ..data_structure.equation import Equation

POINT_WEIGHT_MODIFIER_LABEL = "point_weight_modifier"


def build_equation(config: EquationConfig, base_dir: Path | None = None) -> Equation:
    """
    Build an Equation object from an EquationConfig.

    Parameters
    ----------
    config:
        Config with file_path, function_name, output labels/ranks.
    base_dir:
        Optional base directory for resolving relative file paths.

    Returns
    -------
    Equation
    """
    function = _load_function(config.file_path, config.function_name, base_dir)

    # Validate config input_labels against function signature (excluding 'operators')
    sig_labels = tuple(
        name for name in inspect.signature(function).parameters if name != "operators"
    )
    config_labels = tuple(config.input_labels)
    if config_labels != sig_labels:
        raise ValueError(
            f"Equation '{config.label}': input_labels in config {list(config_labels)} "
            f"do not match function signature {list(sig_labels)}."
        )

    return Equation(
        label=config.label,
        function=function,
        output_label_to_rank=config.output_label_to_rank,
    )


def build_equations(
    configs: list[EquationConfig],
    base_dir: Path | None = None,
) -> list[Equation]:
    """
    Build all Equation objects from a list of EquationConfigs.

    Parameters
    ----------
    configs:
        List of equation configs to build.
    base_dir:
        Optional base directory for resolving relative file paths.

    Returns
    -------
    list[Equation]
    """
    labels_seen: set[str] = set()
    equations: list[Equation] = []
    for config in configs:
        if config.label in labels_seen:
            raise ValueError(f"Duplicate equation label: '{config.label}'.")
        labels_seen.add(config.label)
        equations.append(build_equation(config, base_dir=base_dir))
    return equations


def build_point_weight_modifier_equation(
    loss_term_label: str,
    config: PointWeightModifierConfig,
    base_dir: Path | None = None,
) -> Equation:
    """Build an Equation for a loss term's inline point-weight modifier.

    The modifier reuses the Equation machinery but is defined inline on the loss
    term rather than in the Equations section, so its (single, rank-0) output
    label is supplied internally and never needs to be declared by the user.
    """
    equation_config = EquationConfig(
        label=f"{loss_term_label}_{POINT_WEIGHT_MODIFIER_LABEL}",
        function_name=config.function_name,
        file_path=config.file_path,
        input_labels=config.input_labels,
        output_labels=[POINT_WEIGHT_MODIFIER_LABEL],
        output_ranks=[0],
    )
    return build_equation(equation_config, base_dir=base_dir)


def _load_function(
    file_path: str,
    function_name: str,
    base_dir: Path | None,
) -> Any:
    """Dynamically load a named function from a Python file."""
    path = Path(file_path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Equation file not found: '{path}'. "
            f"(Resolved from file_path='{file_path}', base_dir='{base_dir}')"
        )

    # Use a unique module name to avoid collisions if multiple files are loaded
    module_name = f"_pinninit_eq_{path.stem}_{id(path)}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from '{path}'.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]

    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in '{path}'.")

    return getattr(module, function_name)
