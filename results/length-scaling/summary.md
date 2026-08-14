# Length-scaling summary

4 representations x 5 lengths, best of 3 timed runs per cell.

Timing and peak memory are measured in separate passes: `tracemalloc` hooks every allocation and inflates wall-clock on allocation-heavy code, so a number measured under it is not one anybody reproduces without the profiler attached.

## Encode time (seconds)

| Representation | 128 | 256 | 512 | 1024 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gaf` | 0.0006 | 0.0016 | 0.0056 | 0.0196 | 0.1648 |
| `gadf` | 0.0004 | 0.0006 | 0.0042 | 0.0235 | 0.1117 |
| `mtf` | 0.0004 | 0.0006 | 0.0021 | 0.0045 | 0.0694 |
| `rp` | 0.0001 | 0.0001 | 0.0022 | 0.0088 | 0.1621 |

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
| `gaf` | 0.0805 | 0.2060 | 0.8449 | 2.4666 | 35.1547 |
| `gadf` | 0.0572 | 0.2049 | 1.0310 | 2.1178 | 29.8873 |
| `mtf` | 0.0262 | 0.1153 | 0.4758 | 2.0613 | 28.3130 |
| `rp` | 0.0239 | 0.0993 | 0.4630 | 1.7856 | 26.4607 |

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
| `gaf` | 1.67 | 2.00 | 1.77 | 2.00 | `O(N^2) time and memory` |
| `gadf` | 1.78 | 2.00 | 1.78 | 2.00 | `O(N^2) time and memory` |
| `mtf` | 1.50 | 1.82 | 2.02 | 2.00 | `O(N^2) time and memory` |
| `rp` | 2.41 | 2.00 | 2.03 | 2.00 | `O(N^2) time and memory` |
