"""Analytics and interpretability utilities."""

from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray

from .encoders import _validate_series

Array = NDArray[np.float64]


def _scale_unit(x: Array) -> Array:
    """Return ``x`` scaled to ``[0, 1]``."""
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    denom = x_max - x_min if x_max != x_min else 1.0
    return (x - x_min) / denom


class TSHAPExplainer:
    """Time-series SHAP explainer using window and frequency occlusion.

    Parameters
    ----------
    model:
        Callable mapping a 1D series to a scalar prediction.
    baseline:
        Value used to occlude windows. Defaults to ``0.0``.

    Examples
    --------
    >>> model = lambda x: float(x.mean())
    >>> explainer = TSHAPExplainer(model)
    >>> t_imp, f_imp = explainer.explain(np.arange(8.0), window=4)
    >>> t_imp.shape, f_imp.shape
    ((8,), (5,))
    """

    def __init__(self, model: Callable[[Array], float], baseline: float | None = None):
        self.model = model
        self.baseline = 0.0 if baseline is None else float(baseline)

    def explain(self, series: Array, window: int = 10) -> tuple[Array, Array]:
        """Return time and frequency attributions for ``series``.

        Parameters
        ----------
        series:
            1D input series.
        window:
            Window length for time occlusion. Must be ``>=1``.

        Returns
        -------
        time_importance, freq_importance:
            Arrays of shape ``(N,)`` and ``(N//2 + 1,)`` scaled to ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``window`` is invalid or ``series`` is not 1D.
        """

        sig = _validate_series(series)
        n = sig.size
        if window < 1 or window > n:
            raise ValueError("window must be between 1 and len(series)")

        base_pred = float(self.model(sig))

        time_vals = np.zeros(n, dtype=float)
        for start in range(0, n, window):
            stop = min(start + window, n)
            pert = sig.copy()
            pert[start:stop] = self.baseline
            diff = base_pred - float(self.model(pert))
            time_vals[start:stop] = diff
        time_vals = _scale_unit(time_vals)

        fft = np.fft.rfft(sig)
        freq_vals = np.zeros(fft.size, dtype=float)
        for i in range(fft.size):
            pert_fft = fft.copy()
            pert_fft[i] = 0
            pert = np.fft.irfft(pert_fft, n)
            diff = base_pred - float(self.model(pert))
            freq_vals[i] = diff
        freq_vals = _scale_unit(freq_vals)

        return time_vals, freq_vals


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


def gaf_attribution(importance: Array) -> Array:
    """Collapse GAF importance matrix back to time domain.

    Parameters
    ----------
    importance:
        Square matrix ``(N, N)`` with attribution weights.

    Returns
    -------
    Array
        1D importance over time scaled to ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``importance`` is not a square matrix.

    Examples
    --------
    >>> gaf_attribution(np.ones((3, 3)))
    array([1., 1., 1.])
    """

    mat = np.asarray(importance, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("importance must be a square matrix")
    return _scale_unit(mat.sum(axis=0))


def rp_attribution(importance: Array) -> Array:
    """Collapse recurrence plot importance matrix to time domain."""

    return gaf_attribution(importance)


def spectrogram_attribution(importance: Array) -> Array:
    """Collapse spectrogram importance to time windows.

    Parameters
    ----------
    importance:
        2D array ``(freq, time)`` with attribution weights.

    Returns
    -------
    Array
        1D array ``(time,)`` scaled to ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``importance`` is not 2D.
    """

    mat = np.asarray(importance, dtype=float)
    if mat.ndim != 2:
        raise ValueError("importance must be 2D")
    return _scale_unit(mat.sum(axis=0))


def plot_importance(series: Array, importance: Array, title: str | None = None) -> Any:
    """Interactive plot of ``series`` with ``importance`` overlay.

    Clicking highlights the nearest time index.

    Examples
    --------
    >>> fig = plot_importance(np.arange(5.0), np.linspace(0, 1, 5))
    >>> isinstance(fig, object)
    True
    """

    sig = _validate_series(series)
    imp = _validate_series(importance)
    if sig.size != imp.size:
        raise ValueError("series and importance must match in length")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(sig, label="series")
    ax.bar(np.arange(sig.size), imp, alpha=0.3, label="importance")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    if title is not None:
        ax.set_title(title)
    ax.legend()

    def on_click(event: Any) -> None:  # pragma: no cover - interactive callback
        if event.xdata is None:
            return
        idx = int(round(event.xdata))
        if 0 <= idx < sig.size:
            ax.axvline(idx, color="red", alpha=0.5)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    return fig


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


def generate_counterfactual(
    series: Array,
    model: Callable[[Array], float],
    target: float,
    step: float = 0.1,
    max_iter: int = 100,
) -> Array:
    """Generate a counterfactual series moving prediction toward ``target``.

    Parameters
    ----------
    series:
        1D input series.
    model:
        Callable returning a scalar prediction.
    target:
        Desired prediction value.
    step:
        Gradient descent step size.
    max_iter:
        Maximum optimisation iterations.

    Returns
    -------
    Array
        Counterfactual series with shape ``(N,)``.

    Examples
    --------
    >>> model = lambda x: float(x.mean())
    >>> cf = generate_counterfactual(np.zeros(4), model, target=1.0, step=0.5)
    >>> round(model(cf), 1)
    1.0
    """

    sig = _validate_series(series)
    cf = sig.copy()
    for _ in range(max_iter):
        pred = float(model(cf))
        diff = pred - target
        if abs(diff) < 1e-3:
            break
        grad = saliency_map(model, cf)
        grad_norm = float(np.linalg.norm(grad)) or 1.0
        cf -= step * diff * grad / grad_norm
    return cf


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
    "TSHAPExplainer",
    "gaf_attribution",
    "rp_attribution",
    "spectrogram_attribution",
    "plot_importance",
    "generate_counterfactual",
    "shap_values",
    "lime_explain",
    "saliency_map",
    "counterfactual_replace",
    "project_features",
    "group_significance",
    "cross_causal_lag",
    "generate_report",
]
