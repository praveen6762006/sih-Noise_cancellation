# Priority 4: reference-microphone speech-leakage sweep

Model H (deployment, robust Small) and Model C (same architecture, no dropout/mismatch training) on 480 validation-partition clips (8 classes x 6 SNRs x 10), identical at every level except the speech leaking into the reference mic. PyTorch protocol. This is a sensitivity analysis on validation-partition sources, not a final test.

**Paired-design check.** The primary microphone signal and clean target must be identical across levels. Model H with an all-zero reference on the level 0 clips vs on the level +80 clips: max |difference| of per-clip delta SI-SNR = 3.11e-08 dB (should be ~0).

Reference conditions for context (same clips): no processing = 0 dB; Model H with the reference switched off (zeros): delta SI-SNR +9.11 dB, STOI 0.863, PESQ 2.12.

| level (shadow_db) | measured leakage: median (5-95%) dB | H dSI-SNR | H STOI | H PESQ | C dSI-SNR | C STOI | C PESQ |
|---|---|---|---|---|---|---|---|
| -12 dB | -2.3 (-6.3 to 1.2) | -1.27 [-2.07, -0.44] | 0.681 [0.663, 0.699] | 1.51 [1.45, 1.56] | -15.77 [-16.77, -14.70] | 0.469 [0.458, 0.481] | 1.10 [1.08, 1.11] |
| -6 dB | 3.7 (-0.3 to 7.2) | +8.03 [+7.54, +8.52] | 0.846 [0.832, 0.859] | 2.10 [2.03, 2.18] | +3.11 [+2.15, +4.06] | 0.773 [0.756, 0.789] | 1.85 [1.78, 1.93] |
| +0 dB (training range 0-12) | 9.7 (5.7 to 13.2) | +9.94 [+9.51, +10.42] | 0.875 [0.863, 0.886] | 2.28 [2.20, 2.36] | +10.11 [+9.69, +10.55] | 0.880 [0.868, 0.891] | 2.33 [2.25, 2.41] |
| +6 dB (training range 0-12) | 15.7 (11.7 to 19.2) | +10.19 [+9.76, +10.65] | 0.880 [0.868, 0.891] | 2.30 [2.23, 2.38] | +10.48 [+10.01, +10.94] | 0.885 [0.875, 0.896] | 2.36 [2.28, 2.43] |
| +12 dB (training range 0-12) | 21.7 (17.7 to 25.2) | +10.24 [+9.81, +10.71] | 0.881 [0.870, 0.892] | 2.31 [2.23, 2.38] | +10.52 [+10.10, +10.95] | 0.886 [0.876, 0.896] | 2.35 [2.28, 2.43] |
| +24 dB | 33.7 (29.7 to 37.2) | +10.25 [+9.78, +10.70] | 0.881 [0.870, 0.892] | 2.31 [2.23, 2.39] | +10.51 [+10.07, +10.95] | 0.886 [0.875, 0.896] | 2.35 [2.28, 2.43] |
| +80 dB | 89.7 (85.7 to 93.2) | +10.24 [+9.78, +10.71] | 0.881 [0.870, 0.892] | 2.31 [2.24, 2.39] | +10.51 [+10.05, +10.96] | 0.886 [0.875, 0.897] | 2.35 [2.27, 2.43] |

## Decisions (fixed criteria, delta SI-SNR, paired 95% CI)

| level | median leakage (dB) | H(ref) - H(zero ref) | reference helps? | H(ref) - H(speech-free ref) | degraded vs clean reference? | using the reference harmful? | STOI: H(ref) - H(zero) | PESQ: H(ref) - H(zero) |
|---|---|---|---|---|---|---|---|---|
| -12 dB | -2.3 | -10.38 [-11.02, -9.76] | no | -11.51 [-12.13, -10.89] | yes | YES | -0.182 [-0.194, -0.169] | -0.610 [-0.661, -0.562] |
| -6 dB | 3.7 | -1.08 [-1.29, -0.88] | no | -2.21 [-2.42, -2.02] | yes | YES | -0.017 [-0.021, -0.014] | -0.015 [-0.041, +0.011] |
| +0 dB | 9.7 | +0.83 [+0.73, +0.93] | yes | -0.30 [-0.33, -0.26] | yes | no | +0.012 [+0.010, +0.015] | +0.165 [+0.146, +0.185] |
| +6 dB | 15.7 | +1.08 [+0.98, +1.19] | yes | -0.05 [-0.06, -0.04] | yes | no | +0.017 [+0.015, +0.020] | +0.187 [+0.168, +0.208] |
| +12 dB | 21.7 | +1.13 [+1.02, +1.25] | yes | -0.00 [-0.01, +0.00] | no | no | +0.018 [+0.016, +0.021] | +0.190 [+0.169, +0.211] |
| +24 dB | 33.7 | +1.14 [+1.03, +1.25] | yes | +0.01 [+0.00, +0.01] | no | no | +0.018 [+0.016, +0.021] | +0.191 [+0.170, +0.211] |
| +80 dB | 89.7 | +1.13 [+1.03, +1.24] | yes | +0.00 [+0.00, +0.00] | no | no | +0.018 [+0.016, +0.021] | +0.192 [+0.172, +0.213] |

Model C (no dropout training) against the same criteria:

| level | median leakage (dB) | C(ref) - H(zero ref) | C harmful vs ignoring the reference? |
|---|---|---|---|
| -12 dB | -2.3 | -24.88 [-25.86, -23.87] | YES |
| -6 dB | 3.7 | -6.00 [-6.86, -5.21] | YES |
| +0 dB | 9.7 | +1.00 [+0.88, +1.13] | no |
| +6 dB | 15.7 | +1.37 [+1.25, +1.50] | no |
| +12 dB | 21.7 | +1.41 [+1.28, +1.54] | no |
| +24 dB | 33.7 | +1.40 [+1.28, +1.54] | no |
| +80 dB | 89.7 | +1.40 [+1.26, +1.53] | no |

## Benefit of the reference by input SNR: H(ref) - H(zero ref), delta SI-SNR (dB)

| level | -10 dB | -5 dB | +0 dB | +5 dB | +10 dB | +15 dB |
|---|---|---|---|---|---|---|
| -12 dB | -8.69 | -9.61 | -9.96 | -9.69 | -11.88 | -12.44 |
| -6 dB | -0.64 | -0.74 | -0.80 | -0.94 | -1.45 | -1.92 |
| +0 dB | +1.30 | +0.96 | +0.83 | +0.72 | +0.54 | +0.65 |
| +6 dB | +1.64 | +1.29 | +1.05 | +0.90 | +0.73 | +0.89 |
| +12 dB | +1.71 | +1.35 | +1.09 | +0.93 | +0.75 | +0.93 |
| +24 dB | +1.73 | +1.37 | +1.10 | +0.93 | +0.75 | +0.93 |
| +80 dB | +1.73 | +1.36 | +1.10 | +0.92 | +0.74 | +0.92 |

## What the data show (measured), and what they do not

- Levels where the reference helps (CI above 0 vs ignoring it): shadow_db [0, 6, 12, 24, 80].
- Levels where using the reference is worse than ignoring it: shadow_db [-12, -6].
- The grid is coarse (7 levels) and the leakage here is a simulated, level-and-low-pass model of leakage; the real headset's leakage (frequency-dependent, position-dependent) may behave differently. No claim is made beyond these levels.

## P4b: deployed path (reference guard + ONNX runtime) at the harmful levels, 480 clips each

| level (shadow_db) | dSI-SNR | STOI | PESQ |
|---|---|---|---|
| -12 dB | -0.47 | 0.704 | 1.56 |
| -6 dB | +8.09 | 0.848 | 2.09 |
| +0 dB | +9.82 | 0.875 | 2.27 |

For comparison the reference-off model (H with zeros) gets dSI-SNR +9.11 dB on these clips. Whether the guard fires at heavy leakage is not measured here; the numbers only show that the deployed path is still far below the reference-off result at -12 dB.
