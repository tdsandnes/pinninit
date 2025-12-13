from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from ..utils import check_against_data_type

if TYPE_CHECKING:
    from typing import Any


def load_compulsory(
    field: str,
    section: str,
    input_params: dict[str, Any],
    data_type: type | None = None,
) -> Any:
    """
    Load a compulsory parameter from a nested dictionary and validate its type.

    Parameters
    ----------
    field : str
        The parameter name to load.
    section : str
        The section name in the input dictionary.
    input_params : dict[str, Any]
        The dictionary containing all parameters.
    data_type : type | None
        The expected data type for the parameter.

    Returns
    -------
    Any
        The validated parameter value.

    Raises
    ------
    ValueError
        If the parameter is missing or cannot be converted to the specified type.
    """
    if field in input_params[section]:
        # Read value from input params
        read_value = input_params[section][field]
        # Check data type
        return check_against_data_type(read_value, data_type)

    else:
        raise ValueError(
            f"Missing compulsory parameter '{section}: {field}' in " "parameter file."
        )


def load_optional(
    field: str,
    section: str,
    input_params: dict[str, Any],
    data_type: type | None = None,
    default: Any = None,
) -> Any:
    """
    Load an optional parameter from a nested dictionary and validate its type.

    Parameters
    ----------
    field : str
        The parameter name to load.
    section : str
        The section name in the input dictionary.
    input_params : dict[str, Any]
        The dictionary containing all parameters.
    data_type : type | None
        The expected data type for the parameter.
    default : Any
        The default value to return if the parameter is missing.

    Returns
    -------
    Any
        The validated parameter value or the default value if provided, otherwise None.

    Raises
    ------
    ValueError
        If conversion to the specified data type fails.
    """
    # If Group doesn't exit in file
    if section not in input_params and data_type is not None:
        return data_type()

    if input_params[section] is not None and field in input_params[section]:
        # Read value from input params
        read_value = input_params[section][field]
        # Check data type
        return check_against_data_type(read_value, data_type)

    elif default is not None:
        # Return given default value
        return check_against_data_type(default, data_type)
    elif data_type is not None:
        # Return empty if no value in parameter file and no default value is given
        return data_type()
    else:
        return None


def get_enum_label_from_value(enum_value: Any, lookup_enum: type[Enum]) -> Any:
    """
    Get an enum label from its value.

    Parameters
    ----------
    enum_value : Any
        The value to match against enum labels.
    lookup_enum : Enum
        The enum class to search.

    Returns
    -------
    Enum
        The matching enum label.

    Raises
    ------
    ValueError
        If no enum label matches the given value.
    """
    # Look for value in registry
    for enum_label in lookup_enum:
        if enum_value == enum_label.value:
            # Return the found label
            return enum_label

    # Raise error if enum_value not found in registry values
    raise ValueError(
        f"Unknown option: '{enum_value}' for input parameter '{enum_value}'. "
        f"Available options are: {[e.value for e in lookup_enum]}"
    )


def expand_if_alias(
    read_value: Any,
    alias_dict: dict[Any, Any],
    data_type: type | None = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Check input against dict of aliases.

    If input is not a key in an alias dict, return the input value after validation of
    type. If it is an alias, either return the alias value, or call it (with the given
    args) if the value is callable.

    Parameters
    ----------
    read_value : Any
        The raw value read from input parameters.
    alias_dict : dict
        A dictionary mapping aliases to values or callables.
    data_type : type | None
        The expected data type for the expanded value.
    *args : Any
        Positional arguments passed to callable aliases.
    **kwargs : Any
        Keyword arguments passed to callable aliases.

    Returns
    -------
    Any
        The expanded and validated value.

    Raises
    ------
    ValueError
        If conversion to the specified data type fails.
    """
    # If the param value is not in the alias dict, we want to use the read value
    param_value = read_value

    # Check if alias exists
    if param_value in alias_dict:
        alias_value = alias_dict[param_value]
        if callable(alias_value):
            param_value = alias_value(*args, **kwargs)
        else:
            param_value = alias_value

    # Check data type
    return check_against_data_type(param_value, data_type)
