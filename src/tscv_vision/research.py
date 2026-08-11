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

from ._deprecation import deprecated_alias

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

def group_mean_disparity(features: Array, groups: Array) -> dict[str, float]:
    """Report the mean feature value per group and the largest gap between them.

    .. note::
       This is a single descriptive statistic, not a fairness audit. It says
       nothing about calibration, equalised odds, demographic parity of a
       *model*, intersectional subgroups, or statistical significance of the
       observed gap. Treat it as a screening signal only.

    Parameters
    ----------
    features:
        1D feature array.
    groups:
        Array of group labels of the same shape as ``features``.

    Returns
    -------
    dict
        Mapping with per-group means and the overall ``max_diff``.

    Raises
    ------
    ValueError
        If shapes disagree.
    """

    if features.shape != groups.shape:
        raise ValueError("features and groups must have the same shape")
    uniq = np.unique(groups)
    means = {str(g): float(features[groups == g].mean()) for g in uniq}
    if len(uniq) < 2:
        max_diff = 0.0
    else:
        values = list(means.values())
        max_diff = float(max(values) - min(values))
    res: dict[str, float] = {**means, "max_diff": max_diff}
    return res


bias_report = deprecated_alias(
    group_mean_disparity,
    "bias_report",
    reason="The function reports group-mean disparity only, not a fairness analysis.",
)


def add_laplace_noise(
    features: Array, scale: float, *, rng: np.random.Generator | None = None
) -> Array:
    """Add zero-mean Laplace noise of the given ``scale`` to ``features``.

    This is the raw mechanism primitive with no privacy semantics attached;
    see :func:`add_dp_noise` for the calibrated Laplace mechanism.

    Parameters
    ----------
    features:
        Array to perturb.
    scale:
        Laplace scale parameter ``b > 0``.
    rng:
        Optional random generator for reproducibility.

    Raises
    ------
    ValueError
        If ``scale`` is not positive.
    """

    if scale <= 0:
        raise ValueError("scale must be positive")
    if rng is None:
        rng = np.random.default_rng()
    noise: Array = rng.laplace(0.0, float(scale), size=np.shape(features))
    return np.asarray(features, dtype=float) + noise


def add_dp_noise(
    features: Array,
    epsilon: float,
    *,
    sensitivity: float,
    rng: np.random.Generator | None = None,
) -> Array:
    """Apply the Laplace mechanism calibrated to ``sensitivity``.

    Noise is drawn from ``Laplace(0, sensitivity / epsilon)``, which gives
    ``epsilon``-differential privacy **only if** ``sensitivity`` is a correct
    upper bound on the L1 sensitivity of the query that produced ``features``:

    .. math::

        \\Delta_1 = \\max_{D \\sim D'} \\lVert f(D) - f(D') \\rVert_1

    over all pairs of neighbouring datasets under your chosen neighbouring
    relation. The bound cannot be inferred from the data — it follows from the
    query and from the clipping/normalisation applied beforehand. Passing an
    under-estimate silently voids the guarantee.

    Composition is also the caller's responsibility: releasing ``m`` such
    vectors costs ``m * epsilon`` under basic (sequential) composition.

    Parameters
    ----------
    features:
        Query output to privatise.
    epsilon:
        Privacy budget; must be ``> 0``.
    sensitivity:
        L1 sensitivity of the query; must be ``> 0``. Required keyword —
        there is no meaningful default.
    rng:
        Optional random generator for reproducibility.

    Returns
    -------
    Array
        Perturbed copy of ``features``.

    Raises
    ------
    ValueError
        If ``epsilon`` or ``sensitivity`` is not positive.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = np.array([0.2, 0.9])          # a mean over values clipped to [0, 1]
    >>> add_dp_noise(x, epsilon=1.0, sensitivity=1.0 / 100, rng=rng).shape
    (2,)
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if sensitivity <= 0:
        raise ValueError(
            "sensitivity must be positive; it is the L1 sensitivity of the query "
            "that produced 'features' and must be derived from the query, not the data"
        )
    return add_laplace_noise(features, float(sensitivity) / float(epsilon), rng=rng)


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
    "group_mean_disparity",
    "bias_report",  # deprecated alias
    "add_laplace_noise",
    "add_dp_noise",
    "generate_paper",
]
