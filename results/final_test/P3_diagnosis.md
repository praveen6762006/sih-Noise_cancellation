# P3 diagnosis: why deployed and PyTorch differ (final_A main)

MEASURED decomposition. Each row: mean over the clips of (variant dSI-SNR minus the PyTorch-path dSI-SNR of the same clip), dB.

## Largest per-clip discrepancies (8 clips)

| variant | mean difference vs PyTorch (dB) | largest |difference| (dB) |
|---|---|---|
| deployed (guard on, AGC) | -4.974 | 11.04 |
| no guard (AGC) | -4.974 | 11.04 |
| fixed gain (guard on) | -0.015 | 0.08 |
| neither (no guard, fixed gain) | -0.015 | 0.08 |

| file | class | input SNR | peak target | clipped share (primary) | PyTorch dSI-SNR | deployed dSI-SNR | no-guard | fixed-gain | guard flagged share of hops |
|---|---|---|---|---|---|---|---|---|---|
| 000612.wav | siren | -10 | None | 0.0000 | +23.7 | +12.7 | +12.7 | +23.7 | 0.000 |
| 000322.wav | drone | -10 | None | 0.0000 | +12.0 | +6.6 | +6.6 | +12.0 | 0.000 |
| 000947.wav | artillery | -5 | None | 0.0000 | +26.0 | +21.2 | +21.2 | +26.0 | 0.000 |
| 000317.wav | drone | -10 | None | 0.0000 | +10.2 | +5.7 | +5.7 | +10.2 | 0.000 |
| 000294.wav | helicopter | +15 | None | 0.0000 | +7.6 | +3.3 | +3.3 | +7.6 | 0.000 |
| 000621.wav | siren | -10 | None | 0.0000 | +26.4 | +22.7 | +22.7 | +26.4 | 0.000 |
| 000308.wav | drone | -10 | None | 0.0000 | +11.3 | +8.2 | +8.2 | +11.3 | 0.000 |
| 000913.wav | artillery | -10 | None | 0.0000 | +15.4 | +12.5 | +12.5 | +15.4 | 0.000 |

## Random sample (60 clips)

| variant | mean difference vs PyTorch (dB) | largest |difference| (dB) |
|---|---|---|
| deployed (guard on, AGC) | -0.045 | 2.13 |
| no guard (AGC) | -0.045 | 2.13 |
| fixed gain (guard on) | +0.006 | 0.26 |
| neither (no guard, fixed gain) | +0.006 | 0.26 |

Reading: if 'fixed gain' removes the discrepancy, the difference comes from the level control (its start-up and slow tracking); if 'no guard' removes it, the guard flagged a healthy reference; if only 'neither' matches PyTorch, both contribute. Any statement about WHY the level control or guard behaves this way on those clips is a hypothesis unless shown above.

