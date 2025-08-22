"""Lightweight plugin registry with hot-swappable components."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class PluginRegistry(Generic[T]):
    """Registry mapping names to callables.

    Parameters
    ----------
    plugins:
        Initial mapping of plugin names.
    """

    plugins: dict[str, T]

    def register(self, name: str, plugin: T) -> None:
        """Register a plugin under ``name``.

        Raises
        ------
        ValueError
            If ``name`` is already registered.
        """

        if name in self.plugins:
            raise ValueError(f"Plugin {name!r} already registered")
        self.plugins[name] = plugin

    def get(self, name: str) -> T:
        """Retrieve a plugin by name."""

        try:
            return self.plugins[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown plugin {name!r}") from exc

    def swap(self, name: str, plugin: T) -> None:
        """Replace an existing plugin with a new implementation."""

        if name not in self.plugins:
            raise KeyError(f"Cannot swap unregistered plugin {name!r}")
        self.plugins[name] = plugin


registry: PluginRegistry[Callable[..., object]] = PluginRegistry({})
