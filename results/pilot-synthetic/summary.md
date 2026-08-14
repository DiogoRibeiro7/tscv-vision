# Benchmark summary

5 datasets x 9 methods.

Friedman chi-square = 33.180, p = 5.715e-05. Nemenyi critical difference (alpha=0.05) = 5.373.

| Method | Mean accuracy | Average rank |
| --- | ---: | ---: |
| baseline-rocket-ridge | 1.0000 | 3.00 |
| baseline-1nn-euclidean | 0.9778 | 3.30 |
| mtf-features | 0.9667 | 3.30 |
| rp-features | 0.9111 | 4.30 |
| gasf-features | 0.9000 | 4.60 |
| gadf-features | 0.8889 | 4.60 |
| ablation-gaf-texture-only | 0.8889 | 4.90 |
| baseline-raw-logreg | 0.5444 | 8.10 |
| ablation-gaf-intensity-only | 0.4167 | 8.90 |

## Pairwise Wilcoxon signed-rank (Holm-corrected)

| A | B | mean diff | wins A | wins B | ties | p | p (Holm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ablation-gaf-intensity-only | ablation-gaf-texture-only | -0.4722 | 0 | 5 | 0 | 0.04217 | 1 |
| ablation-gaf-intensity-only | baseline-1nn-euclidean | -0.5611 | 0 | 5 | 0 | 0.04217 | 1 |
| ablation-gaf-intensity-only | baseline-raw-logreg | -0.1278 | 0 | 4 | 1 | 0.0656 | 1 |
| ablation-gaf-intensity-only | baseline-rocket-ridge | -0.5833 | 0 | 5 | 0 | 0.04217 | 1 |
| ablation-gaf-intensity-only | gadf-features | -0.4722 | 0 | 5 | 0 | 0.04217 | 1 |
| ablation-gaf-intensity-only | gasf-features | -0.4833 | 0 | 5 | 0 | 0.04217 | 1 |
| ablation-gaf-intensity-only | mtf-features | -0.5500 | 0 | 5 | 0 | 0.04217 | 1 |
| ablation-gaf-intensity-only | rp-features | -0.4944 | 0 | 5 | 0 | 0.04217 | 1 |
| ablation-gaf-texture-only | baseline-1nn-euclidean | -0.0889 | 0 | 2 | 3 | 0.5 | 1 |
| ablation-gaf-texture-only | baseline-raw-logreg | +0.3444 | 5 | 0 | 0 | 0.0625 | 1 |
| ablation-gaf-texture-only | baseline-rocket-ridge | -0.1111 | 0 | 2 | 3 | 0.5 | 1 |
| ablation-gaf-texture-only | gadf-features | +0.0000 | 1 | 1 | 3 | 1 | 1 |
| ablation-gaf-texture-only | gasf-features | -0.0111 | 0 | 1 | 4 | 1 | 1 |
| ablation-gaf-texture-only | mtf-features | -0.0778 | 0 | 2 | 3 | 0.5 | 1 |
| ablation-gaf-texture-only | rp-features | -0.0222 | 0 | 2 | 3 | 0.1573 | 1 |
| baseline-1nn-euclidean | baseline-raw-logreg | +0.4333 | 5 | 0 | 0 | 0.04217 | 1 |
| baseline-1nn-euclidean | baseline-rocket-ridge | -0.0222 | 0 | 1 | 4 | 1 | 1 |
| baseline-1nn-euclidean | gadf-features | +0.0889 | 2 | 0 | 3 | 0.1573 | 1 |
| baseline-1nn-euclidean | gasf-features | +0.0778 | 2 | 0 | 3 | 0.5 | 1 |
| baseline-1nn-euclidean | mtf-features | +0.0111 | 1 | 1 | 3 | 1 | 1 |
| baseline-1nn-euclidean | rp-features | +0.0667 | 2 | 0 | 3 | 0.5 | 1 |
| baseline-raw-logreg | baseline-rocket-ridge | -0.4556 | 0 | 5 | 0 | 0.04217 | 1 |
| baseline-raw-logreg | gadf-features | -0.3444 | 0 | 5 | 0 | 0.0625 | 1 |
| baseline-raw-logreg | gasf-features | -0.3556 | 0 | 5 | 0 | 0.0625 | 1 |
| baseline-raw-logreg | mtf-features | -0.4222 | 0 | 5 | 0 | 0.0625 | 1 |
| baseline-raw-logreg | rp-features | -0.3667 | 0 | 5 | 0 | 0.0625 | 1 |
| baseline-rocket-ridge | gadf-features | +0.1111 | 2 | 0 | 3 | 0.5 | 1 |
| baseline-rocket-ridge | gasf-features | +0.1000 | 2 | 0 | 3 | 0.5 | 1 |
| baseline-rocket-ridge | mtf-features | +0.0333 | 1 | 0 | 4 | 1 | 1 |
| baseline-rocket-ridge | rp-features | +0.0889 | 2 | 0 | 3 | 0.5 | 1 |
| gadf-features | gasf-features | -0.0111 | 1 | 1 | 3 | 1 | 1 |
| gadf-features | mtf-features | -0.0778 | 0 | 2 | 3 | 0.5 | 1 |
| gadf-features | rp-features | -0.0222 | 1 | 1 | 3 | 1 | 1 |
| gasf-features | mtf-features | -0.0667 | 0 | 2 | 3 | 0.5 | 1 |
| gasf-features | rp-features | -0.0111 | 0 | 1 | 4 | 1 | 1 |
| mtf-features | rp-features | +0.0556 | 2 | 0 | 3 | 0.5 | 1 |

Accuracies come from the archive's predefined train/test split; the test split is used only for the final score.
