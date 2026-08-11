"""Helpers for renaming public API symbols without breaking callers.

Several routines were renamed in 0.2.0 because their previous names implied
established methods that they did not implement (see ``CHANGELOG.md``). The
old names keep working for one release cycle and emit a
:class:`DeprecationWarning` pointing at the replacement.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

#: Release in which the deprecated aliases will be removed.
REMOVAL_VERSION = "0.3.0"


def deprecated_alias(new: Callable[..., Any], old_name: str, *, reason: str = "") -> Any:
    """Return a wrapper around ``new`` that warns when called as ``old_name``.

    Parameters
    ----------
    new:
        The replacement callable.
    old_name:
        The deprecated name being kept as an alias.
    reason:
        Optional sentence explaining why the symbol was renamed.

    Returns
    -------
    Callable
        A wrapper with the same signature as ``new``.
    """

    message = (
        f"{old_name}() is deprecated and will be removed in {REMOVAL_VERSION}; "
        f"use {new.__name__}() instead."
    )
    if reason:
        message = f"{message} {reason}"

    @functools.wraps(new)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return new(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__qualname__ = old_name
    doc = f"Deprecated alias for :func:`{new.__name__}`."
    if reason:
        doc = f"{doc}\n\n{reason}"
    wrapper.__doc__ = doc
    return wrapper


def warn_renamed(old_name: str, new_name: str, reason: str = "") -> None:
    """Emit the standard rename :class:`DeprecationWarning`."""

    message = (
        f"{old_name} is deprecated and will be removed in {REMOVAL_VERSION}; "
        f"use {new_name} instead."
    )
    if reason:
        message = f"{message} {reason}"
    warnings.warn(message, DeprecationWarning, stacklevel=3)


__all__ = ["deprecated_alias", "warn_renamed", "REMOVAL_VERSION"]
