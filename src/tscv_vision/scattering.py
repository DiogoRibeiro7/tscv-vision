r"""Wavelet scattering encoder, backed by Kymatio.

The scattering transform is a deep convolutional cascade with fixed wavelet
filters:

.. math::

    S_0 x = x \star \phi, \qquad
    S_1 x = |x \star \psi_{\lambda_1}| \star \phi, \qquad
    S_2 x = ||x \star \psi_{\lambda_1}| \star \psi_{\lambda_2}| \star \phi

Averaging by :math:`\phi` buys local time-shift invariance; the second order
recovers the amplitude modulation that averaging destroys.

This module is a thin, validated layer over
:class:`kymatio.numpy.Scattering1D`. It contributes input validation,
deterministic coefficient ordering, a documented image layout derived from the
backend's own coefficient metadata, and axis descriptions — not a
reimplementation. Reproducing a scattering library approximately in NumPy
would produce something that is not the scattering transform while carrying
its name.

.. note::
   **This is time scattering, not joint time-frequency scattering.** The
   original request was for JTFS (Andén, Lostanlen & Mallat, 2019). Kymatio
   exposes ``TimeFrequencyScattering`` only on its development branch: no
   released version (0.3.0 is the latest) provides it. Implementing JTFS by
   hand was ruled out — approximating a complex transform and naming it after
   the paper is exactly the failure this package corrected in 0.2.0. When
   Kymatio releases JTFS, it can be added here as a separate encoder under its
   own name. See ``ROADMAP.md``.

.. warning::
   Kymatio 0.3.0 imports ``scipy.special.sph_harm``, which SciPy removed in
   1.17. The ``scattering`` extra therefore constrains SciPy accordingly; in
   an environment with a newer SciPy the backend imports but cannot be used,
   and this module says so rather than reporting a missing package.
"""

from __future__ import annotations

import math
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .encoders import NanPolicy, _validate_series

Array = NDArray[np.float64]

ScatteringFormat = Literal["tensor", "image", "modulation"]

__all__ = [
    "ScatteringFormat",
    "scattering_transform",
    "scattering_meta",
]


def _require_kymatio() -> Any:
    """Return :class:`kymatio.numpy.Scattering1D` or explain how to get it."""

    try:
        from kymatio.numpy import Scattering1D
    except ImportError as exc:  # pragma: no cover - optional dependency
        import importlib.util

        if importlib.util.find_spec("kymatio") is not None:
            raise ImportError(
                "Kymatio is installed but unusable in this environment: "
                f"{exc}. Kymatio 0.3.0 imports scipy.special.sph_harm, which "
                "SciPy removed in 1.17, so it needs `scipy<1.17`."
            ) from exc
        raise ImportError(
            "Kymatio is required for the scattering transform. The cascade is "
            "the method; a short NumPy approximation would not be it. Install "
            "with `pip install 'tscv-vision[scattering]'`"
        ) from exc
    return Scattering1D


def _build(n: int, J: int, Q: int | tuple[int, ...]) -> Any:
    scattering1d = _require_kymatio()
    return scattering1d(J=J, shape=(n,), Q=Q)


def _ordering(meta: dict[str, Any]) -> NDArray[np.intp]:
    """Deterministic row order: by scattering order, then descending centre frequency.

    Kymatio's native order is an implementation detail of its filter-bank
    construction. Sorting explicitly by ``(order, -xi1, -xi2)`` gives a layout
    that is stable across versions and reads like a spectrogram: low rows are
    high frequency.
    """

    order = np.asarray(meta["order"], dtype=float)
    xi = np.asarray(meta["xi"], dtype=float)
    # NaN centre frequencies (order 0, and xi2 at order 1) sort last within
    # their order group rather than unpredictably.
    first = np.nan_to_num(xi[:, 0], nan=-np.inf)
    second = np.nan_to_num(xi[:, 1], nan=-np.inf)
    return cast(NDArray[np.intp], np.lexsort((-second, -first, order)))


def scattering_meta(
    n_samples: int,
    *,
    J: int = 6,
    Q: int | tuple[int, ...] = 8,
) -> dict[str, Any]:
    """Describe the rows of the image produced by :func:`scattering_transform`.

    Returns the backend's coefficient metadata reordered to match the
    ``"image"`` layout, so a caller can label the axes rather than guess.

    Parameters
    ----------
    n_samples:
        Series length the transform will be applied to; the filter bank, and
        therefore the coefficient set, depends on it.
    J:
        Maximum scale, as a power of two.
    Q:
        Wavelets per octave.

    Returns
    -------
    dict
        ``order`` ``(n_coefficients,)``, ``xi`` and ``j`` ``(n_coefficients, 2)``
        holding the centre frequency and scale of the first and second
        wavelets on each path (``nan`` where the path is shorter), and
        ``n_coefficients``. Row ``i`` of the image corresponds to entry ``i``
        here.

    Raises
    ------
    ImportError
        If Kymatio is not installed.
    ValueError
        If ``n_samples`` is too short for ``J``.

    Examples
    --------
    >>> meta = scattering_meta(2048, J=6, Q=8)      # doctest: +SKIP
    >>> meta["order"][:3]                            # doctest: +SKIP
    array([0, 1, 1])
    """

    _validate_config(n_samples, J, Q)
    backend = _build(n_samples, J, Q)
    raw = backend.meta()
    index = _ordering(raw)
    return {
        "order": np.asarray(raw["order"])[index],
        "xi": np.asarray(raw["xi"], dtype=float)[index],
        "j": np.asarray(raw["j"], dtype=float)[index],
        "n_coefficients": int(index.size),
    }


def _validate_config(n: int, J: int, Q: int | tuple[int, ...]) -> None:
    if J < 1:
        raise ValueError("J must be >= 1")
    counts = (Q,) if isinstance(Q, int) else tuple(Q)
    if not counts or any(int(q) < 1 for q in counts):
        raise ValueError("Q must be a positive integer or a tuple of them")
    if n < 2**J:
        raise ValueError(
            f"series of length {n} is too short for J={J}; the lowest-frequency "
            f"wavelet spans 2**J = {2**J} samples"
        )


def scattering_transform(
    x: Array,
    *,
    J: int = 6,
    Q: int | tuple[int, ...] = 8,
    format: ScatteringFormat = "image",
    log_scale: bool = True,
    nan_policy: NanPolicy = "raise",
) -> Array:
    r"""Wavelet scattering coefficients as an image.

    A thin layer over :class:`kymatio.numpy.Scattering1D`; see the module
    docstring for why the backend is required and why this is *not* joint
    time-frequency scattering.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``. ``N`` must be at least ``2 ** J``.
    J:
        Maximum scale as a power of two. The averaging window is
        ``2 ** J`` samples, which sets the time-shift invariance.
    Q:
        Wavelets per octave, either shared or per order.
    format:
        Which layout to return, all documented below.
    log_scale:
        Apply ``log1p`` to the coefficients before normalising. Scattering
        coefficients span orders of magnitude — second-order energy is
        typically 1e-2 of first-order — so the linear view is dominated by a
        few rows.
    nan_policy:
        How to treat NaNs, see :func:`_validate_series`.

    Returns
    -------
    ndarray
        Depends on ``format``:

        ``"tensor"``
            ``(n_coefficients, n_times)`` in the backend's native coefficient
            order, unnormalised. Use with :func:`scattering_meta` only if you
            reorder it yourself; the metadata function returns the *image*
            order.
        ``"image"``
            ``(n_coefficients, n_times)`` with rows sorted by
            ``(order, -xi1, -xi2)``: order 0 first, then order 1 from high to
            low centre frequency, then order 2 ordered by first then second
            wavelet. Row ``i`` is described by entry ``i`` of
            :func:`scattering_meta`. Normalised to ``[0, 1]``.
        ``"modulation"``
            ``(n_xi1, n_xi2)`` matrix of order-2 coefficients averaged over
            time, indexed by first-wavelet centre frequency (spectral band,
            descending down the rows) against second-wavelet centre frequency
            (modulation rate, descending across the columns). This is the
            "spectral band x modulation rate" view; empty cells, where the
            cascade admits no such path because :math:`\lambda_2 < \lambda_1`
            is required, are zero. Normalised to ``[0, 1]``.

    Raises
    ------
    ImportError
        If Kymatio is not installed.
    ValueError
        If ``J`` or ``Q`` is invalid, ``format`` is unknown, the series is
        shorter than ``2 ** J``, or ``x`` is invalid.

    Notes
    -----
    **Complexity** ``O(P · N log N)`` for ``P`` paths, dominated by the
    backend's FFT convolutions.

    **Invariances** Locally invariant to time shifts up to ``2 ** J`` samples,
    by construction — that is what the averaging is for. Stable to small
    deformations (Mallat, 2012), which is the theoretical property that makes
    scattering useful. Amplitude scaling changes the coefficients
    proportionally, but the normalised image is invariant to it.

    **Information lost** Phase, and time detail finer than ``2 ** J``. The
    cascade is not invertible from ``S_0`` and ``S_1`` alone; the second order
    exists precisely because first-order averaging discards amplitude
    modulation.

    **Use cases** Classification of signals where a spectrogram is too
    variable under small time shifts and deformations — audio, biomedical and
    vibration data.

    References
    ----------
    Mallat (2012), "Group invariant scattering", Communications on Pure and
    Applied Mathematics 65(10):1331-1398.  Andén & Mallat (2014), "Deep scattering
    spectrum", IEEE Transactions on Signal Processing 62(16):4114-4128.
    Andreux et al. (2020), "Kymatio: Scattering transforms in Python", JMLR
    21(60):1-6.
    """

    series = _validate_series(x, nan_policy=nan_policy)
    if format not in {"tensor", "image", "modulation"}:
        raise ValueError("format must be 'tensor', 'image' or 'modulation'")
    _validate_config(series.size, J, Q)

    backend = _build(series.size, J, Q)
    coefficients = np.asarray(backend(series), dtype=float)
    if format == "tensor":
        return cast(Array, coefficients)

    raw = backend.meta()
    if format == "image":
        ordered = coefficients[_ordering(raw)]
        return _normalise(ordered, log_scale)

    order = np.asarray(raw["order"])
    xi = np.asarray(raw["xi"], dtype=float)
    second_order = order == 2
    if not np.any(second_order):
        raise ValueError(
            f"J={J}, Q={Q} produced no second-order paths, so there is no "
            "modulation image; increase J"
        )
    first_bands = np.unique(xi[second_order, 0])[::-1]
    second_bands = np.unique(xi[second_order, 1])[::-1]
    image = np.zeros((first_bands.size, second_bands.size), dtype=float)
    strength = np.abs(coefficients[second_order]).mean(axis=1)
    rows = np.searchsorted(-first_bands, -xi[second_order, 0])
    cols = np.searchsorted(-second_bands, -xi[second_order, 1])
    np.add.at(image, (rows, cols), strength)
    return _normalise(image, log_scale)


def _normalise(image: Array, log_scale: bool) -> Array:
    """Scale to ``[0, 1]``, optionally compressing the dynamic range first."""

    values = np.abs(image)
    if log_scale:
        values = np.log1p(values / (values.max() + 1e-300))
    peak = float(values.max())
    if peak <= 0:
        return cast(Array, np.zeros_like(values))
    normalised: Array = values / peak
    return normalised


def _default_scattering(x: Array, **kwargs: Any) -> Array:
    """Registry entry point with a series-length-aware default for ``J``."""

    series = np.asarray(x, dtype=float)
    if "J" not in kwargs and series.ndim == 1 and series.size > 0:
        # The registry calls encoders with only a series, and the documented
        # default J=6 rejects anything shorter than 64 samples. Pick the
        # largest J the series supports, leaving four octaves of headroom so
        # the backend does not warn about border effects, and capped at the
        # documented default.
        kwargs["J"] = max(1, min(6, int(math.floor(math.log2(series.size))) - 4))
    return scattering_transform(x, **kwargs)
