"""Base class and provenance rules for pretrained representation encoders.

Pretrained backbones bring a contamination risk that no cross-validation
scheme can detect: the weights already saw some corpus, and if that corpus
overlaps the evaluation data the reported score is meaningless. The contract
here therefore forces the checkpoint identity to be recorded in the metadata,
so a results table can always be traced back to what produced it.

No backbone is wired up yet; see ``ROADMAP.md``.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

from .base import FloatArray, PretrainedRepresentation
from .metadata import RepresentationInfo

__all__ = [
    "Device",
    "PretrainedRepresentation",
    "PretrainedBackbone",
    "resolve_device",
    "require_backend",
]

Device = Literal["auto", "cpu", "cuda", "mps"]


def resolve_device(device: Device = "auto") -> str:
    """Resolve ``"auto"`` to the best available torch device.

    Parameters
    ----------
    device:
        Requested device. ``"auto"`` prefers CUDA, then Apple MPS, then CPU.

    Returns
    -------
    str
        A concrete device string.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    ValueError
        If a specific device was requested but is unavailable — silently
        falling back to CPU would turn a configuration error into a
        hundred-fold slowdown that is easy to miss.
    """

    from .learned import require_torch

    torch = require_torch()
    has_cuda = bool(torch.cuda.is_available())
    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    if device == "auto":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
        return "cpu"
    if device == "cuda" and not has_cuda:
        raise ValueError("device='cuda' requested but CUDA is not available")
    if device == "mps" and not has_mps:
        raise ValueError("device='mps' requested but MPS is not available")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unknown device {device!r}")
    return device


def require_backend(module: str, extra: str) -> Any:
    """Import ``module`` or raise an :class:`ImportError` naming the extra.

    Parameters
    ----------
    module:
        Importable module name.
    extra:
        The extra that provides it, quoted back to the user in the message.

    Raises
    ------
    ImportError
        If the module cannot be imported.
    """

    import importlib

    try:
        return importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            f"{module} is required for this representation; install it with "
            f"`pip install 'tscv-vision[{extra}]'`"
        ) from exc


class PretrainedBackbone(PretrainedRepresentation):
    """A representation backed by a named pretrained checkpoint.

    Parameters
    ----------
    model_name:
        Architecture identifier, e.g. ``"ViT-B-32"``.
    checkpoint:
        Weight identifier, e.g. ``"laion2b_s34b_b79k"``. Recorded verbatim in
        :attr:`info` because "pretrained" without a checkpoint name is not a
        reproducible description.
    device:
        See :func:`resolve_device`.
    batch_size:
        Batch size for :meth:`encode`.

    Raises
    ------
    ValueError
        If ``batch_size`` is not positive or ``checkpoint`` is empty.
    """

    def __init__(
        self,
        *,
        model_name: str,
        checkpoint: str,
        device: Device = "auto",
        batch_size: int = 32,
    ) -> None:
        if not checkpoint:
            raise ValueError(
                "checkpoint must name the exact weights used; results from an "
                "unidentified checkpoint cannot be reproduced"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size

    @abstractmethod
    def encode(self, X: Sequence[FloatArray]) -> FloatArray:
        """Encode a batch, returning ``(len(X), embedding_dim)``."""

    def _base_info(self, info: RepresentationInfo) -> RepresentationInfo:
        """Stamp the checkpoint identity onto ``info``.

        Subclasses should return ``self._base_info(...)`` from :attr:`info` so
        that the weights are always part of the recorded provenance.
        """

        detail = f"{self.model_name} @ {self.checkpoint}"
        reference = f"{info.reference}; weights: {detail}" if info.reference else detail
        note = (
            f"Pretrained weights ({detail}). Verify that the pretraining corpus "
            "does not overlap the evaluation data; such contamination is "
            "invisible to cross-validation."
        )
        return info.replace(
            pretrained=True,
            trainable=False,
            reference=reference,
            notes=f"{info.notes} {note}".strip(),
        )
