# cython: boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np
from libc.math cimport cos, sin, fabs


def gaf_polar(np.ndarray[np.float64_t, ndim=1] x, bint summation):
    """Cython helper for the Gramian Angular Field.

    Results are numerically equivalent to the NumPy reference within ``1e-6``
    absolute and relative tolerance.
    """

    cdef Py_ssize_t n = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] phi = np.arccos(np.clip(x, -1.0, 1.0))
    cdef np.ndarray[np.float64_t, ndim=2] out = np.empty((n, n), dtype=np.float64)
    cdef Py_ssize_t i, j
    if summation:
        for i in range(n):
            for j in range(n):
                out[i, j] = cos(phi[i] + phi[j])
    else:
        for i in range(n):
            for j in range(n):
                out[i, j] = sin(phi[i] - phi[j])
    return out


def recurrence_dist(np.ndarray[np.float64_t, ndim=1] x, double eps):
    """Cython helper for recurrence plot distances.

    Matches the NumPy implementation within ``1e-6`` tolerance.
    """

    cdef Py_ssize_t n = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] out = np.empty((n, n), dtype=np.float64)
    cdef Py_ssize_t i, j
    cdef double d, maxd = 0.0
    for i in range(n):
        for j in range(n):
            d = fabs(x[i] - x[j])
            out[i, j] = d
            if d > maxd:
                maxd = d
    maxd = maxd + 1e-12
    if eps >= 0.0:
        for i in range(n):
            for j in range(n):
                out[i, j] = 1.0 if out[i, j] / maxd <= eps else 0.0
    else:
        for i in range(n):
            for j in range(n):
                out[i, j] = 1.0 - out[i, j] / maxd
    return out


def spectrogram_stft(
    np.ndarray[np.float64_t, ndim=1] x,
    np.ndarray[np.float64_t, ndim=1] w,
    int win,
    int hop,
    int n_frames,
):
    """Cython STFT kernel for the spectrogram encoder.

    Produces magnitude spectra matching the NumPy version within ``1e-6``
    tolerance.
    """

    cdef np.ndarray[np.float64_t, ndim=2] out = np.empty((win // 2 + 1, n_frames), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] frame = np.empty(win, dtype=np.float64)
    cdef np.ndarray[np.complex128_t, ndim=1] fft_frame
    cdef Py_ssize_t n, i, k
    for n in range(n_frames):
        for i in range(win):
            frame[i] = x[n * hop + i] * w[i]
        fft_frame = np.fft.rfft(frame)
        for k in range(fft_frame.shape[0]):
            out[k, n] = abs(fft_frame[k])
    return out
