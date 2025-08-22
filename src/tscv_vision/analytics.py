"""Analytics and interpretability utilities."""

from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def shap_values(model: Callable[[Array], Array], data: Array) -> Array:
    """Compute SHAP values for ``model`` on ``data``.

    Parameters
    ----------
    model
        Callable mapping feature vectors to outputs.
    data
        2D array ``(n_samples, n_features)``.

    Returns
    -------
    Array
        SHAP values with shape ``(n_samples, n_features)``.

    Raises
    ------
    ImportError
        If :mod:`shap` is not installed.
    ValueError
        If ``data`` is not 2D.
    """
    feats = np.asarray(data, dtype=float)
    if feats.ndim != 2:
        raise ValueError("data must be 2D")
    try:
        import shap
    except Exception as exc:  # pragma: no cover
        raise ImportError("shap is required for shap_values") from exc
    explainer = shap.KernelExplainer(model, feats)
    vals = explainer.shap_values(feats)
    return np.asarray(vals, dtype=float)


def lime_explain(model: Callable[[Array], Array], data: Array, sample: Array) -> Array:
    """Explain ``model`` prediction for ``sample`` using LIME.

    Requires optional :mod:`lime` dependency. ``data`` provides feature statistics.
    """
    feats = np.asarray(data, dtype=float)
    point = np.asarray(sample, dtype=float)
    if feats.ndim != 2:
        raise ValueError("data must be 2D")
    if point.ndim != 1 or point.shape[0] != feats.shape[1]:
        raise ValueError("sample shape mismatch")
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:  # pragma: no cover
        raise ImportError("lime is required for lime_explain") from exc
    explainer = LimeTabularExplainer(feats)
    exp = explainer.explain_instance(point, lambda x: model(np.asarray(x, dtype=float)))
    weights = np.array([w for _, w in exp.as_list()], dtype=float)
    return weights


def saliency_map(model: Callable[[Array], float], series: Array, eps: float = 1e-4) -> Array:
    """Estimate gradient of ``model`` output with respect to ``series``."""
    sig = np.asarray(series, dtype=float)
    if sig.ndim != 1:
        raise ValueError("series must be 1D")
    base = model(sig)
    grad = np.empty_like(sig)
    for i in range(sig.size):
        pert = sig.copy()
        pert[i] += eps
        grad[i] = (model(pert) - base) / eps
    return grad


def counterfactual_replace(
    series: Array,
    start: int,
    stop: int,
    value: float,
    model: Callable[[Array], float],
) -> tuple[Array, float]:
    """Replace slice ``[start, stop)`` with ``value`` and return model change."""
    sig = np.asarray(series, dtype=float)
    if start < 0 or stop > sig.size or start >= stop:
        raise ValueError("invalid slice")
    base = model(sig)
    pert = sig.copy()
    pert[start:stop] = value
    diff = model(pert) - base
    return pert, float(diff)


def project_features(features: Array, method: str = "tsne", **kwargs: Any) -> Array:
    """Project ``features`` to 2D using t-SNE/UMAP with PCA fallback."""
    feats = np.asarray(features, dtype=float)
    if feats.ndim != 2:
        raise ValueError("features must be 2D")
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE

            proj = TSNE(n_components=2, **kwargs).fit_transform(feats)
            return np.asarray(proj, dtype=float)
        except Exception:  # pragma: no cover
            pass
    if method == "umap":
        try:
            import umap

            proj = umap.UMAP(n_components=2, **kwargs).fit_transform(feats)
            return np.asarray(proj, dtype=float)
        except Exception:  # pragma: no cover
            pass
    feats_center = feats - feats.mean(axis=0)
    u, s, _ = np.linalg.svd(feats_center, full_matrices=False)
    proj = (u[:, :2] * s[:2]).astype(float)
    return cast(Array, proj)


def group_significance(a: Array, b: Array) -> tuple[float, float]:
    """Welch t-test between 1D samples ``a`` and ``b``."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("inputs must be 1D")
    nx, ny = x.size, y.size
    mean_diff = x.mean() - y.mean()
    var = x.var(ddof=1) / nx + y.var(ddof=1) / ny
    t = mean_diff / np.sqrt(var)
    from math import erf, sqrt

    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return float(t), float(p)


def cross_causal_lag(x: Array, y: Array, max_lag: int = 10) -> int:
    """Lag of maximum cross-correlation of ``x`` leading ``y``."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.ndim != 1 or ys.ndim != 1:
        raise ValueError("inputs must be 1D")
    if xs.size != ys.size:
        raise ValueError("inputs must have equal length")
    lags = range(-max_lag, max_lag + 1)
    corrs: list[float] = []
    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(xs[:lag], ys[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(xs[lag:], ys[:-lag])[0, 1]
        else:
            corr = np.corrcoef(xs, ys)[0, 1]
        corrs.append(corr)
    best = int(list(lags)[int(np.nanargmax(corrs))])
    return best


def generate_report(results: dict[str, Any]) -> str:
    """Generate a simple Markdown report from ``results`` mapping."""
    lines = ["# Analysis Report", ""]
    for key, value in results.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


__all__ = [
    "shap_values",
    "lime_explain",
    "saliency_map",
    "counterfactual_replace",
    "project_features",
    "group_significance",
    "cross_causal_lag",
    "generate_report",
]
