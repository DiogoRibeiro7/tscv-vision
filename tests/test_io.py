import numpy as np
import pytest

from tscv_vision import io


def test_save_npz_metadata(tmp_path):
    arr = np.arange(4, dtype=float)
    meta = {"encoder": "gaf", "win_len": 8}
    path = tmp_path / "out.npz"
    io.save_npz(path, {"features": arr}, metadata=meta)
    data = np.load(path)
    assert "metadata" in data.files
    import json

    loaded = json.loads(str(data["metadata"]))
    assert loaded["encoder"] == "gaf"


def test_io_import_errors(monkeypatch, tmp_path):
    import sys
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.feather", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(ImportError):
        io.save_arrow(np.zeros(1), tmp_path / "x.arrow")
    with pytest.raises(ImportError):
        io.save_parquet(np.zeros(1), tmp_path / "x.parquet")
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError):
        io.save_hdf5(np.zeros(1), tmp_path / "x.h5")
