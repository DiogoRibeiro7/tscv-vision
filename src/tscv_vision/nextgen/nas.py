"""Simple neural architecture search stub."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def neural_architecture_search(candidates: Sequence[str]) -> str:
    """Return the best candidate according to a random score.

    This placeholder demonstrates how a NAS interface may look.
    """

    scores = np.random.default_rng(0).random(len(candidates))
    return candidates[int(np.argmax(scores))]
