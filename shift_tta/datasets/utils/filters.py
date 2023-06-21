"""Dataset filtering utils."""
from typing import Any, Dict, List, Optional, Union


def _check_attributes(
    attributes: Union[bool, float, str],
    allowed_attributes: Union[bool, float, str, List[float], List[str]],
) -> bool:
    """Check if attributes are allowed.
    Args:
        attributes: Attributes of current frame.
        allowed_attributes: Attributes allowed.
    Return:
        boolean, whether frame attributes are allowed.
    """
    if isinstance(allowed_attributes, list):
        # assert frame_attributes not in allowed_attributes
        return attributes in allowed_attributes
    return attributes == allowed_attributes


def check_attributes(attributes, allowed_attributes=None):
    """Check if a dictionary of attributes is allowed.
    Args:
        attributes (Dict[str, str]): attributes to check
        allowed_attributes (Dict[str, List[str]]): allowed attributes
    Return:
        boolean, whether frame attributes are allowed.
    """
    check = True
    if allowed_attributes:
        for key in allowed_attributes:
            allowed_attribute = allowed_attributes[key]
            check = check and _check_attributes(
                attributes[key], allowed_attribute
            )
            if not check:
                return check            
    return check