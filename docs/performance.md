# Performance and Streaming

This project targets real-time feature extraction on resource-constrained
devices. The streaming utilities offer several knobs to balance accuracy and
throughput:

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

