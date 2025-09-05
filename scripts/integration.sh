#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"

for PY in 3.9 3.11 3.12; do
  CMD=$(command -v python${PY} 2>/dev/null || true)
  if [ -n "$CMD" ]; then
    echo "[info] testing with python${PY}"
    "$CMD" -m venv "/tmp/tscv_${PY}" || continue
    "/tmp/tscv_${PY}/bin/pip" install -e "$ROOT" >/tmp/pip_${PY}.log || continue
    "/tmp/tscv_${PY}/bin/python" - <<'PY'
import numpy as np
from tscv_vision import encoders, features
x = np.sin(np.linspace(0, 4*np.pi, 128))
img = encoders.gaf(x)
vec = features.extract_feature_vector(img, bins=16)
print('vec', vec.shape)
PY
  else
    echo "[warn] python${PY} not available, skipping"
  fi
done

# Test installation with optional extras (excluding GPU)
python -m venv /tmp/tscv_full || true
/tmp/tscv_full/bin/pip install -e "$ROOT[analytics,cli]" >/tmp/pip_full.log || true
python - "$ROOT" <<'PY'
import numpy as np, pathlib, sys
root = pathlib.Path(sys.argv[1])
root.joinpath('samples').mkdir(exist_ok=True)
np.save(root / 'samples/sine.npy', np.sin(np.linspace(0, 4*np.pi, 128)))
PY
/tmp/tscv_full/bin/tscv-features --encoders gaf --input "$ROOT/samples/sine.npy" --output /tmp/out.npz --features intensity,hist || true
rm -f "$ROOT/samples/sine.npy"

# Build and install from source distribution
python setup.py sdist >/tmp/sdist.log
python -m venv /tmp/tscv_sdist
/tmp/tscv_sdist/bin/pip install dist/*.tar.gz >/tmp/pip_sdist.log
/tmp/tscv_sdist/bin/python - <<'PY'
import numpy as np
from tscv_vision import encoders
x = np.sin(np.linspace(0, 4*np.pi, 128))
encoders.gaf(x)
print('sdist-ok')
PY
