"""Command line interface for feature extraction."""

from __future__ import annotations

import argparse
from typing import cast

import numpy as np
from numpy.typing import NDArray

from . import aggregation as _aggregation
from . import encoders
from . import fusion as _fusion
from . import io as _io
from .sliding import encode_sliding, features_for_sliding


def _load_series(path: str) -> NDArray[np.float64]:
    arr = np.load(path)
    if arr.ndim not in (1, 2):
        raise ValueError("Input must be 1D or 2D time series saved with numpy.save")
    return cast(NDArray[np.float64], arr.astype(float))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CV features from time series."
    )
    parser.add_argument("--encoders", default="gaf", help="Comma-separated encoder names")
    parser.add_argument("--input", required=True, help="Path to .npy time series")
    parser.add_argument("--output", required=True, help="Path to .npz output features")
    parser.add_argument("--bins", type=int, default=32, help="Histogram bins")
    parser.add_argument(
        "--features",
        default="all",
        help="Comma-separated feature names or 'all'",
    )
    parser.add_argument(
        "--channel-fusion",
        choices=["stack", "mean", "concat"],
        default="stack",
        help="Channel fusion strategy for multichannel input",
    )
    parser.add_argument(
        "--fusion",
        choices=["concat", "mean", "median", "weighted"],
        default="concat",
        help="Fusion strategy across multiple encoders",
    )
    parser.add_argument(
        "--fusion-weights",
        default=None,
        help="Comma-separated weights for weighted fusion",
    )
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save encoded images to output NPZ",
    )
    parser.add_argument(
        "--save-meta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save metadata (window starts/len/hop) for sliding",
    )
    parser.add_argument(
        "--aggregate",
        default=None,
        help="Comma-separated temporal aggregators (mean,median,var,min,max,skew,kurt)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of worker processes for parallel encoding",
    )

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
    selected = None if args.features == "all" else args.features.split(",")

    enc_names = args.encoders.split(",")
    for name in enc_names:
        if name not in encoders.ENCODER_REGISTRY:
            raise ValueError(f"Unknown encoder '{name}'")

    size = x.shape[0] if not args.sliding else args.win_len
    hop = x.shape[0] if not args.sliding else args.hop

    feats_list = []
    starts = None
    for name in enc_names:
        feats, s = features_for_sliding(
            x,
            encoder=name,
            size=size,
            hop=hop,
            bins=args.bins,
            metric=args.rp_metric,
            eps=args.rp_eps,
            spec_win=args.spec_win,
            spec_hop=args.spec_hop,
            spec_window=args.spec_window,
            channel_fusion=args.channel_fusion,
            feature_names=selected,
            workers=None if args.parallel <= 1 else args.parallel,
        )
        feats_list.append(feats)
        if starts is None:
            starts = s
        elif not np.array_equal(starts, s):
            raise ValueError("Encoders produced mismatched window starts")

    weights = (
        [float(w) for w in args.fusion_weights.split(",")]
        if args.fusion_weights
        else None
    )
    F = _fusion.fuse(feats_list, mode=args.fusion, weights=weights)

    if args.aggregate is not None:
        agg = _aggregation.aggregate(F, args.aggregate.split(","))
        out = {"features": agg}
    elif not args.sliding:
        out = {"features": F[0]}
    else:
        out = {"features": F}

    if not args.sliding and args.save_images and len(enc_names) == 1:
        imgs, _ = encode_sliding(
            x,
            encoder=enc_names[0],
            size=x.shape[0],
            hop=x.shape[0],
            metric=args.rp_metric,
            eps=args.rp_eps,
            spec_win=args.spec_win,
            spec_hop=args.spec_hop,
            spec_window=args.spec_window,
            channel_fusion=args.channel_fusion,
            workers=None if args.parallel <= 1 else args.parallel,
        )
        out["image"] = imgs[0]
    elif args.sliding and args.save_images and len(enc_names) == 1:
        imgs, _ = encode_sliding(
            x,
            encoder=enc_names[0],
            size=args.win_len,
            hop=args.hop,
            metric=args.rp_metric,
            eps=args.rp_eps,
            spec_win=args.spec_win,
            spec_hop=args.spec_hop,
            spec_window=args.spec_window,
            channel_fusion=args.channel_fusion,
            workers=None if args.parallel <= 1 else args.parallel,
        )
        out["images"] = imgs

    hop_val = args.hop if args.hop is not None else args.win_len // 2
    if args.sliding and args.save_meta and starts is not None:
        out["window_starts"] = starts.astype(float)
        out["win_len"] = np.array(float(args.win_len))
        out["hop"] = np.array(float(hop_val))

    metadata = {
        "encoders": enc_names,
        "bins": args.bins,
        "features": selected if selected is not None else "all",
        "channel_fusion": args.channel_fusion,
        "fusion": args.fusion,
        "fusion_weights": weights,
        "aggregate": args.aggregate,
        "sliding": args.sliding,
        "win_len": args.win_len if args.sliding else None,
        "hop": hop_val if args.sliding else None,
        "rp_metric": args.rp_metric,
        "rp_eps": args.rp_eps,
        "spec_win": args.spec_win,
        "spec_hop": args.spec_hop,
        "spec_window": args.spec_window,
    }

    _io.save_npz(args.output, out, metadata=metadata)

    if args.sliding and args.aggregate is None:
        print(
            f"Saved features matrix to {args.output} with shape {F.shape} (n_windows={F.shape[0]})"
        )
    else:
        print(f"Saved features to {args.output} with shape {out['features'].shape}")


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
