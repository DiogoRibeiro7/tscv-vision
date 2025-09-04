"""GPU-accelerated encoders and feature kernels using CuPy.

This submodule provides optional GPU implementations of core encoders and
feature extraction kernels. The functions lazily import :mod:`cupy` and fall
back to raising ``RuntimeError`` if it is unavailable.
"""
from __future__ import annotations

from .encoders import convolve2d, gaf, lbp, memory_usage, spectrogram

__all__ = ["gaf", "spectrogram", "convolve2d", "lbp", "memory_usage"]
