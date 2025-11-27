from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

# Generic type for Enum
EnumType = TypeVar("EnumType", bound=Enum)


def lookup_from_registry(
    label: EnumType, registry: dict[EnumType, Callable[..., Any]]
) -> Any:
    """
    Generic function to look up a class or function from a registry.

    Parameters
    ----------
    label : Enum
        The key to look up in the registry.
    registry : dict
        Mapping from labels to callables or classes.

    Returns
    -------
    Any
        The class or function associated with the given label and registry.

    Raises
    ------
    ValueError
        If the label is not found in the registry.
    """
    if label in registry:
        # Create an object or call a function
        return registry[label]
    else:
        enum_class_name = type(label).__name__
        raise ValueError(
            f"Unknown label: '{label.value}' in registry of type '{enum_class_name}'. "
            f"Available options are: {[e.value for e in registry]}"
        )


def execute_from_registry(
    label: EnumType,
    registry: dict[EnumType, Callable[..., Any]],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Generic function to create an object or call a function from a registry.

    Parameters
    ----------
    label : Enum
        The key to look up in the registry.
    registry : dict
        Mapping from labels to callables or classes.
    *args, **kwargs
        Arguments to pass to the callable/class.

    Returns
    -------
    Any
        The result of calling the registry entry with provided args/kwargs.

    Raises
    ------
    ValueError
        If the label is not found in the registry.
    """
    if label in registry:
        # Create an object or call a function
        return registry[label](*args, **kwargs)
    else:
        enum_class_name = type(label).__name__
        raise ValueError(
            f"Unknown label: '{label.value}' in registry of type '{enum_class_name}'. "
            f"Available options are: {[e.value for e in registry]}"
        )


def check_against_data_type(data: Any, data_type: type | None = None) -> Any:
    """
    Validate that input is a specified data type. Try to convert to it otherwise.

    Parameters
    ----------
    data : Any
        The input value to validate or convert.
    data_type : type | None
        The target data type. If None, no conversion is performed.

    Returns:
    ----------
    Any
        The original or converted value.

    Raises:
    ----------
        ValueError: If conversion to the specified data type fails.
    """
    # if a data type is given, convert to that type
    if data is not None and data_type is not None:
        if isinstance(data, data_type):
            return data
        else:
            try:
                # Return data if it can be converted to desired data type
                ### Proabably need something to handle e.g. list[int]
                return data_type(data)
            except ValueError as e:
                raise ValueError(
                    f"Parameter '{data}' cannot be converted to {data_type.__name__}."
                ) from e
    else:
        # If no data type is not specified, just return the loaded value
        return data


def interval_str(
    interval: tuple[float | None, float | None],
    inclusive: tuple[bool | None, bool | None],
) -> str:
    """
    Convert an interval and its inclusivity to a string representation.

    Parameters
    ----------
    interval : tuple
        A tuple (lower_bound, upper_bound) specifying the interval.
    inclusive : tuple
        A tuple (is_lower_inclusive, is_upper_inclusive) specifying if the bounds are
        inclusive.

    Returns
    -------
    str
        String representation of the interval.
    """

    lower_bound, upper_bound = interval
    is_lower_inclusive, is_upper_inclusive = inclusive

    # Left bracket
    if lower_bound is None:
        left_bracket = "("
        lower_str = "-∞"
    else:
        left_bracket = "[" if is_lower_inclusive else "("
        lower_str = str(lower_bound)

    # Right bracket
    if upper_bound is None:
        right_bracket = ")"
        upper_str = "∞"
    else:
        right_bracket = "]" if is_upper_inclusive else ")"
        upper_str = str(upper_bound)

    return f"{left_bracket}{lower_str}, {upper_str}{right_bracket}"
