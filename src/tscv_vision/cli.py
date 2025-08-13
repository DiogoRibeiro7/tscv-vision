from __future__ import annotations
import argparse
import numpy as np
from . import encoders, features
from .sliding import encode_sliding


def _load_series(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 1:
        raise ValueError("Input must be 1D time series saved with numpy.save")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CV features from time series.")
    parser.add_argument("--encoder", choices=["gaf", "gadf", "rp", "spec"], default="gaf")
    parser.add_argument("--input", required=True, help="Path to .npy 1D array")
    parser.add_argument("--output", required=True, help="Path to .npz output features")
    parser.add_argument("--bins", type=int, default=32, help="Histogram bins")

    # Sliding window options
    parser.add_argument(
        "--sliding", action="store_true", help="Use sliding windows over the series"
    )
    parser.add_argument(
        "--win-len", type=int, default=128, help="Sliding window length for raw series"
    )
    parser.add_argument(
        "--hop", type=int, default=None, help="Sliding hop (defaults to win_len//2)"
    )

    # RP params
    parser.add_argument("--rp-metric", choices=["euclidean", "manhattan"], default="euclidean")
    parser.add_argument(
        "--rp-eps", type=float, default=None, help="Threshold for binary RP (optional)"
    )

    # Spectrogram params (per-window)
    parser.add_argument("--spec-win", type=int, default=None, help="STFT window for spectrogram")
    parser.add_argument("--spec-hop", type=int, default=None, help="STFT hop for spectrogram")
    parser.add_argument("--spec-window", choices=["hann", "rect"], default="hann")

    args = parser.parse_args()

    x = _load_series(args.input)

    if not args.sliding:
        # Single-image path
        if args.encoder == "gaf":
            img = encoders.gaf(x, method="summation")
        elif args.encoder == "gadf":
            img = encoders.gaf(x, method="difference")
        elif args.encoder == "rp":
            img = encoders.recurrence_plot(x, metric=args.rp_metric, eps=args.rp_eps)
        else:
            img = encoders.spectrogram(x)
        vec = features.extract_feature_vector(img, bins=args.bins)
        np.savez(args.output, features=vec)
        print(f"Saved features to {args.output} with shape {vec.shape}")
        return

    # Sliding path: encode each window, then featureize each image
    images, starts = encode_sliding(
        x,
        encoder=args.encoder,
        size=args.win_len,
        hop=args.hop,
        metric=args.rp_metric,
        eps=args.rp_eps,
        spec_win=args.spec_win,
        spec_hop=args.spec_hop,
        spec_window=args.spec_window,
    )

    # Extract features per image
    feats = []
    for im in images:
        feats.append(features.extract_feature_vector(im, bins=args.bins))
    F = np.vstack(feats) if feats else np.zeros((0, 6 + args.bins + 16 + 256), dtype=float)

    np.savez(args.output, features=F, window_starts=starts, win_len=args.win_len, hop=args.hop)
    print(f"Saved features matrix to {args.output} with shape {F.shape} (n_windows={F.shape[0]})")
