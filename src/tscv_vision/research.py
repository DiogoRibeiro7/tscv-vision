"""Research utilities for reproducibility, fairness and privacy."""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# simple plugin registry -----------------------------------------------------
Plugin = Callable[..., Any]
PLUGIN_REGISTRY: dict[str, Plugin] = {}


def register_plugin(name: str, func: Plugin) -> None:
    """Register a plugin under ``name``.

    Parameters
    ----------
    name:
        Identifier for the plugin.
    func:
        Callable implementing the plugin.
    """

    PLUGIN_REGISTRY[name] = func


def load_plugins(modules: Sequence[str]) -> None:
    """Import plugin modules, triggering their registrations."""

    for mod in modules:
        importlib.import_module(mod)


# reproducibility ------------------------------------------------------------

def track_experiment(
    config: Mapping[str, Any], dataset: str | Path, out_dir: str | Path
) -> Path:
    """Record configuration and dataset hash for reproducibility.

    Parameters
    ----------
    config:
        Experiment configuration parameters.
    dataset:
        Path to the dataset file.
    out_dir:
        Directory where the log will be written.

    Returns
    -------
    Path
        Path to the created JSON log file.
    """

    path = Path(dataset)
    data_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    timestamp = _dt.datetime.now(_dt.timezone.utc).isoformat()
    record = {
        "timestamp": timestamp,
        "dataset": str(path),
        "hash": data_hash,
        "config": dict(config),
    }
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file = out_path / f"exp_{timestamp.replace(':', '-')}.json"
    file.write_text(json.dumps(record, indent=2))
    return file


# fairness & privacy ---------------------------------------------------------

def bias_report(features: Array, groups: Array) -> dict[str, float]:
    """Compute mean feature value per group and max disparity.

    Parameters
    ----------
    features:
        1D feature array.
    groups:
        Array of group labels of the same shape as ``features``.

    Returns
    -------
    dict
        Mapping with per-group means and overall ``max_diff``.
    """

    if features.shape != groups.shape:
        raise ValueError("features and groups must have the same shape")
    uniq = np.unique(groups)
    means = {str(g): float(features[groups == g].mean()) for g in uniq}
    if len(uniq) < 2:
        max_diff = 0.0
    else:
        diffs = [abs(means[str(a)] - means[str(b)]) for a in uniq for b in uniq]
        max_diff = float(max(diffs))
    res: dict[str, float] = {**means, "max_diff": max_diff}
    return res


def add_dp_noise(
    features: Array, epsilon: float, *, rng: np.random.Generator | None = None
) -> Array:
    """Apply Laplace noise for simple differential privacy.

    Parameters
    ----------
    features:
        Feature array.
    epsilon:
        Privacy budget; must be > 0.
    rng:
        Optional random generator for reproducibility.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if rng is None:
        rng = np.random.default_rng()
    scale = 1.0 / epsilon
    noise: Array = rng.laplace(0.0, scale, size=features.shape)
    return features + noise


# reporting ------------------------------------------------------------------

def generate_paper(config: Mapping[str, Any], out_file: str | Path) -> Path:
    """Create a minimal markdown report describing a pipeline."""

    lines = ["# Experiment Report", "", "## Configuration", ""]
    for k, v in config.items():
        lines.append(f"- **{k}**: {v}")
    path = Path(out_file)
    path.write_text("\n".join(lines) + "\n")
    return path


__all__ = [
    "register_plugin",
    "load_plugins",
    "track_experiment",
    "bias_report",
    "add_dp_noise",
    "generate_paper",
]
