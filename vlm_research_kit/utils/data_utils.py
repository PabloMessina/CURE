from typing import Any
import collections.abc # Use this for robust sequence/mapping checks
import numpy as np

def convert_to_serializable(data: Any) -> Any:
    """
    Recursively converts items in nested data structures (dicts, lists, tuples)
    that have an `.item()` method (like single-element tensors or numpy arrays)
    into standard Python types (int, float).

    Args:
        data: The data structure to convert. Can be a dictionary, list, tuple,
              or a single value.

    Returns:
        A new data structure with compatible items converted to standard types.
        Non-compatible items and structures are returned as is.
    """
    if isinstance(data, collections.abc.Mapping): # Handles dict and dict-like objects
        # Create a new dict to avoid modifying the original
        new_dict = {}
        for key, value in data.items():
            new_dict[key] = convert_to_serializable(value)
        return new_dict
    elif isinstance(data, (list, tuple)): # Handles lists and tuples
        # Create a new list/tuple to avoid modifying the original
        new_sequence = []
        for item in data:
            new_sequence.append(convert_to_serializable(item))
        # Return the same type (list or tuple) as the input
        return type(data)(new_sequence)
    elif hasattr(data, 'item') and callable(data.item):
        # Check if it has .item() and it's callable (handles tensors, numpy scalars)
        try:
            return data.item()
        except Exception:
            # If .item() fails for some reason, return original? Or raise?
            # Returning original is safer if unsure about all types with .item()
            return data
    # Add specific checks if needed, e.g., for full numpy arrays:
    elif isinstance(data, np.ndarray):
        return data.tolist() # Convert full numpy arrays to lists
    else:
        # Return the item as is if it's already serializable or not convertible
        return data

