# Performance and Streaming

This project targets real-time feature extraction on resource-constrained
devices. The streaming utilities offer several knobs to balance accuracy and
throughput:

## Measured cost

That target is an aim, not a measured property, and the committed sweep in
[`results/length-scaling/`](../results/length-scaling/) shows how far the batch
path currently is from it. Reproduce it with
`python benchmarks/scaling/run_length_scaling.py --repeats 3`.

On the recorded hardware, for one series encoded to an `N x N` image and
summarised into 662 features:

| Series length | Encode | Encode peak | Features | Feature peak |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.0001–0.0006 s | 0.3–0.4 MiB | 0.02–0.08 s | 7.1 MiB |
| 512 | 0.002–0.006 s | 2.1–6.0 MiB | 0.46–1.03 s | 112.6 MiB |
| 1024 | 0.004–0.024 s | 8.1–24.0 MiB | 1.79–2.47 s | 450.1 MiB |
| 4096 | 0.07–0.16 s | 128–384 MiB | 26.5–35.2 s | 7200.4 MiB |

Two consequences worth planning around:

- **Feature extraction dominates.** At 4096 it is roughly 200x the encoder. The
  optimised paths in this package — Cython, Numba, CuPy — accelerate the
  encoders, which is the smaller half of the cost.
- **Peak memory, not time, is the binding constraint.** Feature extraction peaks
  at about 56x the image it is given, so a 4096-sample series needs over 7 GiB.
  Memory scales as `N**2.00`, matching the `O(N^2)` recorded in the
  representation metadata, so this is inherent to the dense-image design rather
  than a fixable constant.

For streaming and windowed workloads the relevant length is the window size, not
the length of the record, which is why the streaming path stays practical while
the batch path does not.

## Incremental Updates

`StreamingEncoder` can incrementally update features using a user-supplied
``incremental`` callback. When provided, completed windows reuse the previous
encoding and update it with the outgoing and incoming samples, avoiding full
recomputation.

## GPU Acceleration

Setting ``use_gpu=True`` enables optional CuPy-powered versions of certain
encoders (currently the Gramian Angular Field). If CuPy is not installed a
``RuntimeError`` is raised.

## Parallelism

Many batch helpers accept a `parallel` argument controlling worker processes.
Keep it below the number of physical cores to avoid oversubscription. For
I/O‑bound workloads the CLI can stream from disk using multiple workers.

## Memory mapping

Large `.npy` files can be accessed with `np.memmap` to avoid loading the entire
array into memory:

```python
import numpy as np
series = np.load("huge.npy", mmap_mode="r")
```

## Adaptive Precision

The ``precision`` parameter controls the output data type. ``"adaptive"`` mode
monitors latency and dynamically scales ``float64 → float32 → float16`` when the
per-window processing time exceeds ``latency_threshold``.

## Benchmarking

``benchmark_streaming`` offers a lightweight way to profile throughput, average
latency and memory usage of streaming pipelines. Example:

```python
from tscv_vision.benchmark import benchmark_streaming
from tscv_vision.streaming import StreamingEncoder
import numpy as np

enc = StreamingEncoder(size=128)
samples = np.random.rand(1024)
stats = benchmark_streaming(enc, samples)
```

The returned ``stats`` dictionary contains ``throughput`` (samples/s),
``latency`` (s/sample) and ``memory`` (bytes).

``benchmark_pipeline`` measures the feature extraction rate of a fitted
``AdaptivePipeline``:

```python
from tscv_vision.benchmark import benchmark_pipeline
from tscv_vision.pipeline import AdaptivePipeline
import numpy as np

X = np.random.rand(32, 64)
y = (X.mean(axis=1) > 0.5).astype(int)
pipe = AdaptivePipeline().fit(X, y)
stats = benchmark_pipeline(pipe, X)
```

Only ``throughput`` and ``latency`` are reported since memory usage depends on
the encoder implementations.

For deeper inspection use the standard library's ``cProfile`` or external tools
such as ``perf`` to capture hotspots and cache misses.

