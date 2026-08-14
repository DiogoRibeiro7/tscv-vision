# Benchmark summary

38 datasets x 9 methods.

Friedman chi-square = 151.932, p = 7.749e-29. Nemenyi critical difference (alpha=0.05) = 1.949.

| Method | Mean accuracy | Average rank |
| --- | ---: | ---: |
| baseline-rocket-ridge | 0.9005 | 1.14 |
| baseline-raw-logreg | 0.8200 | 3.95 |
| gadf-features | 0.7706 | 4.43 |
| baseline-1nn-euclidean | 0.7906 | 4.57 |
| gasf-features | 0.7612 | 4.83 |
| rp-features | 0.7611 | 4.83 |
| ablation-gaf-texture-only | 0.6924 | 6.95 |
| mtf-features | 0.6701 | 7.04 |
| ablation-gaf-intensity-only | 0.6230 | 7.26 |

## Pairwise Wilcoxon signed-rank (Holm-corrected)

| A | B | mean diff | wins A | wins B | ties | p | p (Holm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline-rocket-ridge | mtf-features | +0.2304 | 38 | 0 | 0 | 7.737e-08 | 2.785e-06 |
| baseline-rocket-ridge | rp-features | +0.1394 | 38 | 0 | 0 | 7.74e-08 | 2.785e-06 |
| ablation-gaf-intensity-only | baseline-rocket-ridge | -0.2775 | 1 | 37 | 0 | 9.087e-08 | 3.09e-06 |
| ablation-gaf-texture-only | baseline-rocket-ridge | -0.2081 | 0 | 37 | 1 | 1.14e-07 | 3.761e-06 |
| baseline-1nn-euclidean | baseline-rocket-ridge | -0.1100 | 0 | 37 | 1 | 1.14e-07 | 3.761e-06 |
| baseline-rocket-ridge | gadf-features | +0.1300 | 37 | 0 | 1 | 1.14e-07 | 3.761e-06 |
| baseline-rocket-ridge | gasf-features | +0.1394 | 37 | 0 | 1 | 1.14e-07 | 3.761e-06 |
| ablation-gaf-intensity-only | baseline-raw-logreg | -0.1971 | 7 | 30 | 1 | 1.435e-06 | 4.162e-05 |
| baseline-raw-logreg | baseline-rocket-ridge | -0.0805 | 2 | 35 | 1 | 3.253e-06 | 9.107e-05 |
| ablation-gaf-texture-only | gasf-features | -0.0687 | 3 | 34 | 1 | 3.5e-06 | 9.451e-05 |
| baseline-raw-logreg | mtf-features | +0.1499 | 31 | 5 | 2 | 5.616e-06 | 0.000146 |
| ablation-gaf-intensity-only | baseline-1nn-euclidean | -0.1676 | 6 | 30 | 2 | 6.05e-06 | 0.0001512 |
| gadf-features | mtf-features | +0.1004 | 31 | 6 | 1 | 7.708e-06 | 0.000185 |
| ablation-gaf-intensity-only | rp-features | -0.1382 | 6 | 31 | 1 | 8.273e-06 | 0.0001903 |
| ablation-gaf-intensity-only | gasf-features | -0.1382 | 6 | 32 | 0 | 1.006e-05 | 0.0002213 |
| ablation-gaf-texture-only | gadf-features | -0.0781 | 4 | 33 | 1 | 1.019e-05 | 0.0002213 |
| ablation-gaf-intensity-only | gadf-features | -0.1476 | 8 | 30 | 0 | 1.15e-05 | 0.0002299 |
| mtf-features | rp-features | -0.0910 | 7 | 30 | 1 | 1.173e-05 | 0.0002299 |
| baseline-1nn-euclidean | mtf-features | +0.1204 | 31 | 7 | 0 | 2.364e-05 | 0.0004255 |
| ablation-gaf-texture-only | rp-features | -0.0687 | 7 | 30 | 1 | 2.651e-05 | 0.0004507 |
| gasf-features | mtf-features | +0.0910 | 29 | 8 | 1 | 5.106e-05 | 0.0008169 |
| ablation-gaf-texture-only | baseline-raw-logreg | -0.1276 | 7 | 29 | 2 | 0.0001264 | 0.001896 |
| ablation-gaf-texture-only | baseline-1nn-euclidean | -0.0981 | 9 | 29 | 0 | 0.0002366 | 0.003313 |
| baseline-raw-logreg | gasf-features | +0.0589 | 25 | 12 | 1 | 0.006178 | 0.08032 |
| ablation-gaf-intensity-only | ablation-gaf-texture-only | -0.0694 | 13 | 24 | 1 | 0.01011 | 0.1213 |
| baseline-raw-logreg | rp-features | +0.0589 | 24 | 12 | 2 | 0.01623 | 0.1785 |
| baseline-raw-logreg | gadf-features | +0.0495 | 23 | 14 | 1 | 0.03533 | 0.3533 |
| ablation-gaf-intensity-only | mtf-features | -0.0471 | 15 | 20 | 3 | 0.04394 | 0.3955 |
| baseline-1nn-euclidean | baseline-raw-logreg | -0.0295 | 14 | 22 | 2 | 0.07329 | 0.5863 |
| ablation-gaf-texture-only | mtf-features | +0.0223 | 20 | 17 | 1 | 0.2807 | 1 |
| baseline-1nn-euclidean | gadf-features | +0.0200 | 19 | 18 | 1 | 0.4196 | 1 |
| baseline-1nn-euclidean | gasf-features | +0.0294 | 21 | 14 | 3 | 0.1639 | 1 |
| baseline-1nn-euclidean | rp-features | +0.0294 | 19 | 17 | 2 | 0.1597 | 1 |
| gadf-features | gasf-features | +0.0094 | 21 | 13 | 4 | 0.3604 | 1 |
| gadf-features | rp-features | +0.0094 | 20 | 14 | 4 | 0.1852 | 1 |
| gasf-features | rp-features | +0.0000 | 19 | 19 | 0 | 0.9942 | 1 |

Accuracies come from the archive's predefined train/test split; the test split is used only for the final score.
