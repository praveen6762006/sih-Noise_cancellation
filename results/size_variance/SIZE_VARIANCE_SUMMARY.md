# Priorities 5 and 6: model size and seed variance

## P5. Fair size comparison (earlier test set, 1,200 clips, PyTorch protocol; mean [95% CI])

Same conditions for the three rows: from scratch, 6,000 steps, batch 16, seed 0, inputs = primary + reference + flag, no reference dropout.

| model | MMACs/s (computed) | dSI-SNR (dB) | STOI | PESQ |
|---|---|---|---|---|
| Tiny (8,389 params), scratch | 15.8 | +7.88 [+7.61, +8.17] | 0.840 [0.831, 0.849] | 2.04 [2.00, 2.09] |
| Small (24,629), scratch  [NEW] | 33.3 | +9.25 [+8.97, +9.53] | 0.866 [0.859, 0.874] | 2.18 [2.13, 2.23] |
| Base (80,725), scratch | 93.9 | +10.22 [+9.95, +10.50] | 0.884 [0.877, 0.891] | 2.35 [2.30, 2.40] |
| *(rows above: like-for-like; rows below: different initialisation or training configuration)* | | | | |
| Small (24,629), warm-started from pretrained GTCRN (C) | 33.3 | +10.32 [+10.02, +10.60] | 0.888 [0.882, 0.895] | 2.40 [2.35, 2.45] |
| Tiny, robust config (I: dropout 0.2, +/-8 dB), scratch | 15.8 | +7.73 [+7.46, +7.99] | 0.846 [0.837, 0.854] | 1.96 [1.92, 2.00] |
| Small, robust config, warm-started (H, deployment) | 33.3 | +10.00 [+9.73, +10.30] | 0.883 [0.875, 0.890] | 2.34 [2.29, 2.38] |

Paired differences (first minus second), same clips:

| comparison | dSI-SNR | STOI | PESQ |
|---|---|---|---|
| Small-scratch - Tiny | +1.365 [+1.279, +1.456] | +0.027 [+0.025, +0.029] | +0.138 [+0.128, +0.148] |
| Base - Small-scratch | +0.971 [+0.934, +1.006] | +0.018 [+0.017, +0.019] | +0.171 [+0.161, +0.181] |
| Base - Tiny | +2.335 [+2.230, +2.447] | +0.044 [+0.042, +0.047] | +0.309 [+0.293, +0.324] |
| Small warm-start (C) - Small-scratch | +1.072 [+1.027, +1.121] | +0.022 [+0.021, +0.023] | +0.220 [+0.208, +0.234] |

## P6. Model H: independent training seeds (same configuration: warm start, reference dropout 0.2, gain jitter +/-8 dB, 6,000 steps)

Seed 0 is the deployment model; seeds 1 and 2 were trained afterwards only to measure training variance and were not used for any selection.

### earlier test set

| seed | dSI-SNR | STOI | PESQ |
|---|---|---|---|
| 0 | +10.00 [+9.72, +10.29] | 0.883 [0.876, 0.890] | 2.34 [2.29, 2.39] |
| 1 | +9.99 [+9.72, +10.29] | 0.882 [0.875, 0.890] | 2.33 [2.28, 2.38] |
| 2 | +9.99 [+9.70, +10.29] | 0.883 [0.875, 0.889] | 2.33 [2.28, 2.38] |
| **mean over 3 seeds** | +9.994 +/- 0.009 (SD); 95% t-interval [+9.971, +10.017]; range +9.987 to +10.005 | 0.8826 +/- 0.0001 (SD); 95% t-interval [0.8823, 0.8829]; range 0.8825 to 0.8827 | 2.3343 +/- 0.0020 (SD); 95% t-interval [2.3293, 2.3392]; range 2.3327 to 2.3365 |

### fresh final_A main

| seed | dSI-SNR | STOI | PESQ |
|---|---|---|---|
| 0 | +9.74 [+9.47, +10.01] | 0.882 [0.875, 0.889] | 2.28 [2.24, 2.33] |
| 1 | +9.73 [+9.47, +10.00] | 0.882 [0.875, 0.889] | 2.28 [2.23, 2.33] |
| 2 | +9.73 [+9.47, +10.00] | 0.882 [0.875, 0.889] | 2.28 [2.23, 2.32] |
| **mean over 3 seeds** | +9.732 +/- 0.007 (SD); 95% t-interval [+9.713, +9.750]; range +9.727 to +9.740 | 0.8820 +/- 0.0002 (SD); 95% t-interval [0.8813, 0.8826]; range 0.8818 to 0.8822 | 2.2788 +/- 0.0023 (SD); 95% t-interval [2.2730, 2.2845]; range 2.2773 to 2.2815 |

### fresh final_B main

| seed | dSI-SNR | STOI | PESQ |
|---|---|---|---|
| 0 | +10.10 [+9.81, +10.38] | 0.866 [0.859, 0.873] | 2.25 [2.21, 2.30] |
| 1 | +10.08 [+9.80, +10.37] | 0.866 [0.858, 0.873] | 2.25 [2.21, 2.30] |
| 2 | +10.08 [+9.80, +10.37] | 0.866 [0.859, 0.873] | 2.25 [2.20, 2.30] |
| **mean over 3 seeds** | +10.089 +/- 0.009 (SD); 95% t-interval [+10.065, +10.112]; range +10.083 to +10.100 | 0.8658 +/- 0.0003 (SD); 95% t-interval [0.8651, 0.8665]; range 0.8655 to 0.8661 | 2.2527 +/- 0.0014 (SD); 95% t-interval [2.2491, 2.2563]; range 2.2517 to 2.2544 |

