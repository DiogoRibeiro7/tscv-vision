"""Base class for representations with parameters learned from data.

This module is deliberately abstract. It fixes the contract that trainable
representations must satisfy — explicit seeding, a fit/transform split that
cannot silently see evaluation data, and round-trippable state — so that the
concrete models added later all behave the same way.

No trainable model lives here yet; see ``ROADMAP.md``.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import FittedRepresentation, FloatArray

__all__ = ["LearnedRepresentation", "require_torch"]


def require_torch() -> Any:
    """Return the :mod:`torch` module, or raise a directed :class:`ImportError`.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """

    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PyTorch is required for learned representations; install it with "
            "`pip install 'tscv-vision[torch]'`"
        ) from exc
    return torch


class LearnedRepresentation(FittedRepresentation):
    """A representation whose parameters are estimated by training.

    Subclasses implement :meth:`_fit`, :meth:`transform`, :attr:`info`,
    :meth:`state_dict` and :meth:`load_state_dict`.

    Parameters
    ----------
    random_state:
        Seed for every stochastic step (initialisation, shuffling, dropout,
        augmentation). Required rather than optional: an unseeded trainable
        representation cannot be reproduced, and results obtained from one are
        not verifiable.

    Notes
    -----
    Two rules that subclasses must not break:

    * ``fit`` sees training data only. Anything selected using validation or
      test data — the number of epochs, an early-stopping point, a layer
      choice, a fusion weight — leaks, and cross-validation applied afterwards
      cannot detect it.
    * ``transform`` is a pure function of the fitted state. It must not update
      parameters, running statistics or caches keyed on the input.
    """

    def __init__(self, *, random_state: int = 0) -> None:
        self.random_state = int(random_state)
        self._rng = np.random.default_rng(self.random_state)

    @property
    def rng(self) -> np.random.Generator:
        """Seeded generator for every stochastic step."""

        return self._rng

    def reset_rng(self) -> None:
        """Restore the generator to its initial state, making a re-fit exact."""

        self._rng = np.random.default_rng(self.random_state)

    @abstractmethod
    def _fit(self, X: Sequence[FloatArray], y: Any | None = None) -> None:
        """Train on ``X`` (and optionally ``y``). Training data only."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return the learned state, sufficient to reconstruct ``transform``."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the learned state produced by :meth:`state_dict`."""

    def fit(
        self, X: Sequence[FloatArray], y: Any | None = None
    ) -> LearnedRepresentation:
        """Reset the generator, train, and return ``self``.

        The reset makes repeated ``fit`` calls on the same data produce the
        same parameters, which is what a seed is for.
        """

        self.reset_rng()
        super().fit(X, y)
        return self
