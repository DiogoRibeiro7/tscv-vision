# Sample Data

This directory contains helper scripts for generating small example arrays at
runtime. No binary data is tracked in version control.

To create a sine wave for CLI demos, run:

```bash
python samples/generate.py
```

This writes `samples/sine.npy`, which can be used with `tscv-features`:

```bash
poetry run tscv-features --encoders gaf --input samples/sine.npy --output features.npz
```

The generated `.npy` files are ignored by git and may be safely deleted after
use.
