# CLI Workflow

End-to-end example using the command line interface.

## Single image features

```bash
python samples/generate.py
poetry run tscv-features --encoders gaf --input samples/sine.npy --output single.npz
```

## Sliding-window batch

```bash
poetry run tscv-features --encoders gaf,spec --fusion mean --sliding --win-len 64 --hop 32 \
    --input samples/sine.npy --output batch.npz --aggregate mean --no-save-images
```

Both commands produce `.npz` files with a `features` array. Sliding runs also
include `window_starts`, `win_len` and `hop` metadata.

