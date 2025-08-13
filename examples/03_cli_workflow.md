# CLI Workflow

End-to-end example using the command line interface.

## Prepare data

```bash
python - <<'PY'
import numpy as np
np.save('series.npy', np.sin(np.linspace(0, 8*np.pi, 256)))
PY
```

## Single image features

```bash
poetry run tscv-features --encoders gaf --input series.npy --output single.npz
```

## Sliding-window batch

```bash
poetry run tscv-features --encoders gaf,spec --fusion mean --sliding --win-len 64 --hop 32 \
    --input series.npy --output batch.npz --aggregate mean --no-save-images
```

Both commands produce `.npz` files with a `features` array. Sliding runs also
include `window_starts`, `win_len` and `hop` metadata.
