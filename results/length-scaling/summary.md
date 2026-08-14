# Length-scaling summary

4 representations x 5 lengths, best of 3 timed runs per cell.

Timing and peak memory are measured in separate passes: `tracemalloc` hooks every allocation and inflates wall-clock on allocation-heavy code, so a number measured under it is not one anybody reproduces without the profiler attached.

## Encode time (seconds)

| Representation | 128 | 256 | 512 | 1024 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gaf` | 0.0002 | 0.0004 | 0.0026 | 0.0088 | 0.1362 |
| `gadf` | 0.0007 | 0.0014 | 0.0031 | 0.0169 | 0.0850 |
| `mtf` | 0.0002 | 0.0003 | 0.0010 | 0.0032 | 0.0458 |
| `rp` | 0.0000 | 0.0001 | 0.0023 | 0.0093 | 0.1573 |

## Encode peak memory (MiB)

| Representation | 128 | 256 | 512 | 1024 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gaf` | 0.3 | 1.0 | 4.0 | 16.0 | 256.1 |
| `gadf` | 0.3 | 1.0 | 4.0 | 16.0 | 256.1 |
| `mtf` | 0.3 | 0.6 | 2.1 | 8.1 | 128.0 |
| `rp` | 0.4 | 1.5 | 6.0 | 24.0 | 384.0 |

## Feature-extraction time (seconds)

| Representation | 128 | 256 | 512 | 1024 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gaf` | 0.0290 | 0.1238 | 0.5039 | 1.9217 | 30.2882 |
| `gadf` | 0.1020 | 0.3530 | 1.1999 | 1.9168 | 25.9264 |
| `mtf` | 0.0222 | 0.0910 | 0.3893 | 1.5859 | 29.2385 |
| `rp` | 0.0231 | 0.0949 | 0.4208 | 1.7637 | 27.1046 |

## Feature-extraction peak memory (MiB)

| Representation | 128 | 256 | 512 | 1024 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gaf` | 7.1 | 28.2 | 112.6 | 450.1 | 7200.4 |
| `gadf` | 7.1 | 28.2 | 112.6 | 450.1 | 7200.4 |
| `mtf` | 7.1 | 28.2 | 112.6 | 450.1 | 7200.4 |
| `rp` | 7.1 | 28.2 | 112.6 | 450.1 | 7200.4 |

## Measured scaling exponents

Fitted as `value ~ length**k` on a log-log scale. Compare `k` against the complexity recorded in the representation metadata: that string is a claim, and this is the measurement of it.

| Representation | encode time | encode memory | feature time | feature memory | documented |
| --- | ---: | ---: | ---: | ---: | --- |
| `gaf` | 1.98 | 2.00 | 2.00 | 2.00 | `O(N^2) time and memory` |
| `gadf` | 1.44 | 2.00 | 1.55 | 2.00 | `O(N^2) time and memory` |
| `mtf` | 1.61 | 1.82 | 2.07 | 2.00 | `O(N^2) time and memory` |
| `rp` | 2.43 | 2.00 | 2.05 | 2.00 | `O(N^2) time and memory` |
