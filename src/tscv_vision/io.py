"""Utilities for interoperating with external data formats."""

from __future__ import annotations

import json

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def save_arrow(array: Array, path: str) -> None:
    """Save ``array`` to an Arrow file at ``path``.

    Requires :mod:`pyarrow` to be installed.
    """

    try:
        import pyarrow as pa
        import pyarrow.feather as feather
    except Exception as exc:  # pragma: no cover - import error
        raise ImportError("pyarrow is required for Arrow IO") from exc
    table = pa.table({"data": array})
    feather.write_feather(table, path)


def load_arrow(path: str) -> Array:
    """Load an Arrow file from ``path`` into a NumPy array."""

    try:
        import pyarrow.feather as feather
    except Exception as exc:  # pragma: no cover - import error
        raise ImportError("pyarrow is required for Arrow IO") from exc
    table = feather.read_table(path)
    return np.asarray(table.column("data").to_numpy(), dtype=float)


def save_parquet(array: Array, path: str) -> None:
    """Save ``array`` to a Parquet file."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - import error
        raise ImportError("pyarrow is required for Parquet IO") from exc
    table = pa.table({"data": array})
    pq.write_table(table, path)


def load_parquet(path: str) -> Array:
    """Load a Parquet file into a NumPy array."""

    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - import error
        raise ImportError("pyarrow is required for Parquet IO") from exc
    table = pq.read_table(path)
    return np.asarray(table.column("data").to_numpy(), dtype=float)


def save_hdf5(array: Array, path: str, dataset: str = "data") -> None:
    """Save ``array`` in HDF5 format under ``dataset`` name."""

    try:
        import h5py
    except Exception as exc:  # pragma: no cover - import error
        raise ImportError("h5py is required for HDF5 IO") from exc
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset, data=array)


def load_hdf5(path: str, dataset: str = "data") -> Array:
    """Load ``dataset`` from an HDF5 file."""

    try:
        import h5py
    except Exception as exc:  # pragma: no cover - import error
        raise ImportError("h5py is required for HDF5 IO") from exc
    with h5py.File(path, "r") as f:
        arr = np.asarray(f[dataset][:], dtype=float)
    return arr


def save_npz(path: str, data: dict[str, Array], metadata: dict[str, object] | None = None) -> None:
    """Save arrays and optional ``metadata`` to ``path`` as ``.npz``.

    Parameters
    ----------
    path:
        Output file path.
    data:
        Mapping of array name to NumPy array. ``features`` is typically
        required but arbitrary additional arrays (e.g., ``images``) may be
        provided.
    metadata:
        Optional dictionary describing the encoding/feature parameters. The
        dictionary is serialized to JSON under the ``metadata`` key.
    """

    out = dict(data)
    if metadata is not None:
        out["metadata"] = np.array(json.dumps(metadata))
    np.savez(path, **out)


__all__ = [
    "save_arrow",
    "load_arrow",
    "save_parquet",
    "load_parquet",
    "save_hdf5",
    "load_hdf5",
    "save_npz",
]
