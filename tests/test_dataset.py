from pathlib import Path

import numpy as np
import pytest

from tscv_vision.dataset import WindowedDataset, iter_npz, iter_parquet


def test_windowed_dataset_iter_and_cache(tmp_path: Path) -> None:
    x = np.sin(np.linspace(0.0, 2 * np.pi, 128))
    cache = tmp_path / "cache.npz"
    ds = WindowedDataset(x, size=32, hop=32, cache=str(cache))
    items = list(ds)
    assert len(items) == len(ds)
    assert cache.exists()
    ds_cached = WindowedDataset(x, size=32, hop=32, cache=str(cache))
    items2 = list(ds_cached)
    assert np.allclose(items[0][0], items2[0][0])
    assert items2[0][1]["start"] == 0


def test_windowed_dataset_memmap(tmp_path: Path) -> None:
    x = np.sin(np.linspace(0.0, 2 * np.pi, 128))
    path = tmp_path / "x.npy"
    np.save(path, x)
    ds = WindowedDataset(str(path), size=32, hop=32)
    items = list(ds)
    assert len(items) == len(ds)


def test_iter_npz(tmp_path: Path) -> None:
    path = tmp_path / "arrs.npz"
    np.savez(path, a=np.arange(5), b=np.arange(3))
    arrays = list(iter_npz(str(path)))
    assert len(arrays) == 2
    assert arrays[0].shape[0] == 5


def test_iter_parquet_importerror(tmp_path: Path) -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            list(iter_parquet(str(tmp_path / "x.parquet")))
        return
    # pyarrow is installed — verify that a missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        list(iter_parquet(str(tmp_path / "x.parquet")))
