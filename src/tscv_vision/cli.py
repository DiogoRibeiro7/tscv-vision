"""Command line interface for feature extraction."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from . import __version__, encoders, features
from . import aggregation as _aggregation
from . import fusion as _fusion
from . import io as _io
from .sliding import encode_sliding, features_for_sliding

T = TypeVar("T")

_OPTIONAL_BACKENDS = {
    "pywavelets": "pywt",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "kymatio": "kymatio",
    "numba": "numba",
    "cupy": "cupy",
    "torch": "torch",
}


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return ivalue


def _progress(it: Iterable[T], total: int) -> Iterator[T]:
    for i, item in enumerate(it, 1):
        pct = int(i / total * 100) if total else 0
        sys.stderr.write(f"\rProcessing {i}/{total} ({pct}%)")
        sys.stderr.flush()
        yield item
    sys.stderr.write("\n")


def _load_series(path: str) -> NDArray[np.float64]:
    arr = np.load(path)
    if arr.ndim not in (1, 2):
        raise ValueError("Input must be 1D or 2D time series saved with numpy.save")
    return cast(NDArray[np.float64], arr.astype(float))


def _optional_backends() -> dict[str, bool]:
    """Return availability flags for optional numeric backends."""

    return {
        name: importlib.util.find_spec(module) is not None
        for name, module in _OPTIONAL_BACKENDS.items()
    }


def main() -> None:
    examples = (
        "Examples:\n"
        "  tscv-features --encoders gaf --input x.npy --output out.npz\n"
        "  tscv-features --encoders gdf --input x.npy --output out.npz\n"
        "  tscv-features --encoders msc,attn --sliding --win-len 128 --hop 64 "
        "--input x.npy --output out.npz"
    )
    parser = argparse.ArgumentParser(
        description="Extract CV features from time series.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="JSON or YAML configuration file")
    parser.add_argument(
        "--encoders",
        default="gaf",
        help="Comma-separated encoder names (e.g. gaf,rp,gdf,attn,msc)",
    )
    parser.add_argument("--input", nargs="+", required=True, help="Path(s) to .npy time series")
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz path or directory when multiple inputs",
    )
    parser.add_argument("--bins", type=_positive_int, default=32, help="Histogram bins (>0)")
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
        type=_positive_int,
        default=1,
        help="Number of worker processes for parallel encoding",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show actions without computing",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity",
    )

    # Sliding window options
    parser.add_argument(
        "--sliding", action="store_true", help="Use sliding windows over the series"
    )
    parser.add_argument(
        "--win-len",
        type=_positive_int,
        default=128,
        help="Sliding window length for raw series",
    )
    parser.add_argument(
        "--hop",
        type=_positive_int,
        default=None,
        help="Sliding hop (defaults to win_len//2)",
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

    pre_args, _ = parser.parse_known_args()
    if pre_args.config is not None:
        cfg_path = Path(pre_args.config)
        with cfg_path.open() as fh:
            if cfg_path.suffix in {".yml", ".yaml"}:
                try:
                    yaml_mod = importlib.import_module("yaml")
                except Exception as exc:  # pragma: no cover - optional dependency
                    raise ValueError("YAML config requires PyYAML") from exc
                yaml = cast(Any, yaml_mod)
                cfg_raw = yaml.safe_load(fh) or {}
            else:
                cfg_raw = json.load(fh)
        if not isinstance(cfg_raw, dict):
            raise ValueError("Config file must define a mapping")
        cfg: dict[str, Any] = cfg_raw
        parser.set_defaults(**cfg)

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s"
    )
    logger = logging.getLogger("tscv_vision.cli")

    inputs = [Path(p) for p in args.input]
    output = Path(args.output)
    multiple = len(inputs) > 1
    if multiple:
        output.mkdir(parents=True, exist_ok=True)

    enc_names = args.encoders.split(",")
    for name in enc_names:
        if name not in encoders.ENCODER_REGISTRY:
            raise ValueError(f"Unknown encoder '{name}'")

    if args.features != "all":
        for name in args.features.split(","):
            if name not in features.FEATURES_REGISTRY:
                raise ValueError(f"Unknown feature '{name}'")

    weights = (
        [float(w) for w in args.fusion_weights.split(",")]
        if args.fusion_weights
        else None
    )
    if weights is not None and len(weights) != len(enc_names):
        raise ValueError("fusion_weights length must match encoders")

    for in_path in inputs:
        x = _load_series(str(in_path))
        if args.sliding and args.win_len > x.shape[0]:
            raise ValueError("win_len cannot exceed series length")
        hop = args.hop if args.hop is not None else args.win_len // 2
        if args.sliding and hop <= 0:
            raise ValueError("hop must be positive")

        out_file = output / f"{in_path.stem}.npz" if multiple else output

        selected = None if args.features == "all" else args.features.split(",")
        size = x.shape[0] if not args.sliding else args.win_len
        hop_val = x.shape[0] if not args.sliding else hop

        if args.dry_run:
            n_windows = 1 if not args.sliding else 1 + (x.shape[0] - args.win_len) // hop
            print(
                f"[DRY RUN] {in_path} -> {out_file} with {n_windows} window(s) "
                f"and encoders {enc_names}"
            )
            continue

        logger.info("Processing %s", in_path)
        feats_list = []
        starts: NDArray[np.int64] | None = None
        for name in enc_names:
            if args.sliding and (x.shape[0] - size) // hop_val > 0:
                gen = features_for_sliding(
                    x,
                    encoder=name,
                    size=size,
                    hop=hop_val,
                    bins=args.bins,
                    metric=args.rp_metric,
                    eps=args.rp_eps,
                    spec_win=args.spec_win,
                    spec_hop=args.spec_hop,
                    spec_window=args.spec_window,
                    channel_fusion=args.channel_fusion,
                    feature_names=selected,
                    workers=None if args.parallel <= 1 else args.parallel,
                    lazy=True,
                )
                feats_i: list[NDArray[np.float64]] = []
                starts_i: list[int] = []
                total = 1 + (x.shape[0] - size) // hop_val
                for feat, st in _progress(gen, total):
                    feats_i.append(feat)
                    starts_i.append(st)
                feats = np.stack(feats_i)
                s = np.array(starts_i, dtype=int)
            else:
                feats, s = features_for_sliding(
                    x,
                    encoder=name,
                    size=size,
                    hop=hop_val,
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
                hop=hop,
                metric=args.rp_metric,
                eps=args.rp_eps,
                spec_win=args.spec_win,
                spec_hop=args.spec_hop,
                spec_window=args.spec_window,
                channel_fusion=args.channel_fusion,
                workers=None if args.parallel <= 1 else args.parallel,
            )
            out["images"] = imgs

        if args.sliding and args.save_meta and starts is not None:
            out["window_starts"] = starts.astype(float)
            out["win_len"] = np.array(float(args.win_len))
            out["hop"] = np.array(float(hop))

        metadata = {
            "tscv_vision_version": __version__,
            "encoders": enc_names,
            "bins": args.bins,
            "features": selected if selected is not None else "all",
            "feature_layout": features.feature_layout(args.bins, selected),
            "optional_backends": _optional_backends(),
            "channel_fusion": args.channel_fusion,
            "fusion": args.fusion,
            "fusion_weights": weights,
            "aggregate": args.aggregate,
            "sliding": args.sliding,
            "win_len": args.win_len if args.sliding else None,
            "hop": hop if args.sliding else None,
            "rp_metric": args.rp_metric,
            "rp_eps": args.rp_eps,
            "spec_win": args.spec_win,
            "spec_hop": args.spec_hop,
            "spec_window": args.spec_window,
        }

        _io.save_npz(str(out_file), out, metadata=metadata)

        if args.sliding and args.aggregate is None:
            print(
                f"Saved features matrix to {out_file} with shape {F.shape} (n_windows={F.shape[0]})"
            )
        else:
            print(f"Saved features to {out_file} with shape {out['features'].shape}")


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()

