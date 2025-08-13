from __future__ import annotations
import argparse
import numpy as np
from . import encoders, features


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
    args = parser.parse_args()

    x = _load_series(args.input)

    if args.encoder == "gaf":
        img = encoders.gaf(x, method="summation")
    elif args.encoder == "gadf":
        img = encoders.gaf(x, method="difference")
    elif args.encoder == "rp":
        img = encoders.recurrence_plot(x)
    else:
        img = encoders.spectrogram(x)

    vec = features.extract_feature_vector(img, bins=args.bins)
    np.savez(args.output, features=vec)
    print(f"Saved features to {args.output} with shape {vec.shape}")
