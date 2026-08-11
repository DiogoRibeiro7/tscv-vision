# mypy: ignore-errors

"""Neural-based encoders and utilities.

This optional module adds PyTorch-powered helpers for training and
using neural encoders while keeping the rest of the package NumPy-first.
It provides small CNN and Vision Transformer adapters, contrastive
learning utilities, attention-based fusion, simple style transfer and a
light-weight variational autoencoder.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float32]

try:  # pragma: no cover - torch is optional
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover - executed when torch missing
    torch = None  # type: ignore
    F = None  # type: ignore

    class _StubModule:  # pragma: no cover - minimal stand-in
        pass

    class _StubNN:  # pragma: no cover - minimal namespace
        Module = _StubModule

    nn = _StubNN()  # type: ignore


def _ensure_torch() -> None:
    """Raise :class:`ImportError` if PyTorch is unavailable."""

    if torch is None:  # pragma: no cover - environment check
        raise ImportError("torch is required for neural encoders")


class TorchCNNEncoder(nn.Module):  # pragma: no cover - thin wrapper
    """Lightweight CNN encoder producing fixed-size feature vectors.

    Parameters
    ----------
    in_channels:
        Number of image channels, usually ``1``.
    features:
        Size of the output feature vector.
    """

    def __init__(self, in_channels: int = 1, features: int = 64) -> None:
        _ensure_torch()
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

    def encode(self, img: Array) -> Array:
        """Encode ``img`` into a feature vector.

        Parameters
        ----------
        img:
            Image array with shape ``(H, W)`` or ``(C, H, W)``.

        Returns
        -------
        Array
            Feature vector with shape ``(features,)``.
        """

        _ensure_torch()
        tens = torch.from_numpy(img.astype(np.float32))
        if tens.ndim == 2:
            tens = tens.unsqueeze(0).unsqueeze(0)
        elif tens.ndim == 3:
            tens = tens.unsqueeze(0)
        else:
            raise ValueError("img must be 2D or 3D array")
        with torch.no_grad():
            out = self.forward(tens).cpu().numpy()
        return out[0]


class ViTAdapter(nn.Module):  # pragma: no cover - heavy ops
    """Minimal Vision Transformer adapter for time-series images."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        patch_size: int = 8,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        _ensure_torch()
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, patch_size, patch_size)
        enc_layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, depth)
        self.cls = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError("image size must be divisible by patch_size")
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, dim)
        h = self.encoder(x).mean(1)
        return self.cls(h)

    def encode(self, img: Array) -> Array:
        _ensure_torch()
        tens = torch.from_numpy(img.astype(np.float32))
        if tens.ndim == 2:
            tens = tens.unsqueeze(0).unsqueeze(0)
        elif tens.ndim == 3:
            tens = tens.unsqueeze(0)
        else:
            raise ValueError("img must be 2D or 3D array")
        with torch.no_grad():
            out = self.forward(tens).cpu().numpy()
        return out[0]


class _SequenceModelEncoder(nn.Module):  # pragma: no cover - optional dependency
    """Shared plumbing for wrappers around ``(batch, length, d_model)`` models.

    A univariate series has one channel, but sequence models such as Mamba and
    RetNet expect ``d_model`` channels per time step. A learned linear
    projection lifts each scalar sample into the model dimension; the encoder
    output is then mean-pooled over time to give one vector per series.
    """

    def __init__(self, dim: int) -> None:
        _ensure_torch()
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self.input_proj = nn.Linear(1, dim)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, length)`` or ``(batch, length, 1)`` to ``(batch, dim)``.

        Raises
        ------
        ValueError
            If the input is not 2D/3D or its channel axis is not ``1``.
        """

        if x.ndim == 2:
            x = x.unsqueeze(-1)
        elif x.ndim != 3:
            raise ValueError("input must have shape (batch, length[, 1])")
        if x.shape[-1] != 1:
            raise ValueError(
                f"expected a univariate series with 1 channel, got {x.shape[-1]}; "
                "project multivariate inputs yourself before calling this encoder"
            )
        hidden = self.input_proj(x)  # (B, L, dim)
        out = self._backbone(hidden)  # (B, L, dim)
        return out.mean(dim=1)

    def encode(self, seq: Array) -> Array:
        """Encode a 1D series into a ``(dim,)`` feature vector."""

        _ensure_torch()
        arr = np.asarray(seq, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("seq must be 1D")
        tens = torch.from_numpy(arr).view(1, -1, 1)
        with torch.no_grad():
            out = self.forward(tens).cpu().numpy()
        return out[0]


class MambaEncoder(_SequenceModelEncoder):  # pragma: no cover - optional dependency
    """Wrapper around a single Mamba block.

    Requires the ``mamba-ssm`` package and PyTorch. The block is constructed
    with the upstream signature ``Mamba(d_model, d_state, d_conv, expand)``
    (state-spaces/mamba) and consumes ``(batch, length, d_model)`` tensors.

    Parameters
    ----------
    dim:
        Model dimension ``d_model``.
    d_state, d_conv, expand:
        Forwarded to the upstream block; defaults match upstream.
    """

    def __init__(
        self,
        dim: int = 64,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        try:  # pragma: no cover - executed only when mamba available
            from mamba_ssm import Mamba  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("mamba-ssm is required for MambaEncoder") from exc
        super().__init__(dim)
        self.model = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class RetNetEncoder(_SequenceModelEncoder):  # pragma: no cover - optional dependency
    """Wrapper around a RetNet backbone.

    Requires a ``retnet`` package exposing ``RetNet(d_model=..., layers=...)``
    (or ``n_layers``) and consuming ``(batch, length, d_model)`` tensors. The
    constructor signature differs between the available RetNet packages, so
    the supported keywords are probed at construction time and a clear error
    is raised if none matches.

    Parameters
    ----------
    dim:
        Model dimension.
    layers:
        Number of stacked blocks.
    """

    def __init__(self, dim: int = 64, *, layers: int = 1) -> None:
        try:  # pragma: no cover - executed when retnet available
            from retnet import RetNet  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("retnet is required for RetNetEncoder") from exc
        super().__init__(dim)
        errors: list[str] = []
        for kwargs in ({"layers": layers}, {"n_layers": layers}, {"num_layers": layers}):
            try:
                self.model = RetNet(d_model=dim, **kwargs)
                break
            except TypeError as exc:  # pragma: no cover - depends on package
                errors.append(f"{kwargs}: {exc}")
        else:  # pragma: no cover - depends on package
            raise TypeError(
                "could not construct RetNet with any known layer-count keyword; "
                "tried " + "; ".join(errors)
            )

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Compute an InfoNCE contrastive loss between two batches of embeddings."""

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def contrastive_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x1: Array,
    x2: Array,
) -> float:
    """Perform a single contrastive training step."""

    _ensure_torch()
    t1 = torch.from_numpy(x1)
    t2 = torch.from_numpy(x2)
    emb1 = model(t1)
    emb2 = model(t2)
    loss = contrastive_loss(emb1, emb2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


class AttentionFusion(nn.Module):  # pragma: no cover - small module
    """Fuse feature vectors from multiple encoders using attention weights."""

    def __init__(self, n_inputs: int, feat_dim: int) -> None:
        _ensure_torch()
        super().__init__()
        self.attn = nn.Linear(feat_dim, 1)

    def forward(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        stack = torch.stack(list(tensors), dim=1)  # (B, M, D)
        scores = self.attn(stack).squeeze(-1)  # (B, M)
        weights = F.softmax(scores, dim=1)
        return torch.sum(stack * weights.unsqueeze(-1), dim=1)


def style_transfer(
    content: Array,
    style: Array,
    *,
    steps: int = 50,
    lr: float = 1e-1,
) -> Array:
    """Run neural style transfer for small grayscale images."""

    _ensure_torch()
    c = torch.from_numpy(content[None, None]).float()
    s = torch.from_numpy(style[None, None]).float()
    target = c.clone().requires_grad_(True)
    opt = torch.optim.Adam([target], lr=lr)

    def gram(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = x.view(b, c, h * w)
        return y @ y.transpose(1, 2) / (c * h * w)

    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(target, c) + F.mse_loss(gram(target), gram(s))
        loss.backward()
        opt.step()
    return target.detach().numpy()[0, 0]


class VAE(nn.Module):  # pragma: no cover - tiny VAE
    """Simple variational autoencoder for 32×32 grayscale images."""

    def __init__(self, latent_dim: int = 16) -> None:
        _ensure_torch()
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32 * 32),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        params = self.encoder(x)
        mu, logvar = params.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        recon = self.decoder(z).view(-1, 1, 32, 32)
        return recon, mu, logvar

    def sample(self, n: int) -> Array:
        _ensure_torch()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim)
            imgs = self.decoder(z).view(n, 1, 32, 32)
        return imgs.cpu().numpy()


def nas_random_search(space: Sequence[int], evaluate: Callable[[int], float]) -> int:
    """Select the best value from ``space`` using ``evaluate`` as the score."""

    best = space[0]
    best_score = evaluate(best)
    for cand in space[1:]:
        score = evaluate(cand)
        if score < best_score:
            best, best_score = cand, score
    return best


__all__ = [
    "TorchCNNEncoder",
    "ViTAdapter",
    "contrastive_loss",
    "contrastive_train_step",
    "AttentionFusion",
    "style_transfer",
    "VAE",
    "nas_random_search",
    "MambaEncoder",
    "RetNetEncoder",
]
