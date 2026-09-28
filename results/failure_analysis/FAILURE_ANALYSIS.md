# Priority 10: failure analysis of Model H

Clips analysed: 2400 main (final_A + final_B, 8 classes x 6 SNRs) and 480 weapon clips (peaks 110-160 dB SPL). PyTorch protocol. Everything below is descriptive; **MEASURED** items are computed from the result files, **HYPOTHESIS** items are untested explanations.

## 1. Which conditions are over-represented among the worst 5% of clips (MEASURED)

Worst = lowest value of the measure over the 2,400 main clips. Enrichment > 1 means the category is more common among the worst clips than overall.

### worst 5% by delta SI-SNR (n = 120)

| factor | category | share among worst | share overall | enrichment | binomial p |
|---|---|---|---|---|---|
| class | wind | 30% | 12% | 2.40 | 4.4e-07 |
| class | drone | 22% | 12% | 1.80 | 2.1e-03 |
| class | gunshot | 3% | 12% | 0.27 | 8.1e-04 |
| class | artillery | 2% | 12% | 0.20 | 2.5e-04 |
| input SNR (dB) | +15 | 90% | 17% | 5.40 | 1.1e-69 |
| input SNR (dB) | +10 | 8% | 17% | 0.50 | 1.4e-02 |
| input SNR (dB) | -10 | 2% | 17% | 0.10 | 2.1e-07 |
| input SNR (dB) | -5 | 0% | 17% | 0.00 | 6.3e-10 |
| input SNR (dB) | +0 | 0% | 17% | 0.00 | 6.3e-10 |
| input SNR (dB) | +5 | 0% | 17% | 0.00 | 6.3e-10 |
| environment | wind | 30% | 12% | 2.40 | 4.4e-07 |
| noise source | real | 2% | 7% | 0.37 | 6.8e-02 |
| primary clipped | clipped | 0% | 7% | 0.00 | 3.9e-04 |

### worst 5% by delta STOI (n = 120)

| factor | category | share among worst | share overall | enrichment | binomial p |
|---|---|---|---|---|---|
| class | engine | 30% | 12% | 2.40 | 4.4e-07 |
| class | wind | 22% | 12% | 1.80 | 2.1e-03 |
| class | drone | 3% | 12% | 0.27 | 8.1e-04 |
| class | siren | 2% | 12% | 0.13 | 4.0e-05 |
| input SNR (dB) | +15 | 52% | 17% | 3.15 | 2.9e-19 |
| input SNR (dB) | +5 | 8% | 17% | 0.45 | 4.7e-03 |
| input SNR (dB) | -5 | 6% | 17% | 0.35 | 5.5e-04 |
| input SNR (dB) | +0 | 3% | 17% | 0.20 | 1.1e-05 |
| environment | wind | 22% | 12% | 1.80 | 2.1e-03 |
| noise source | real | 2% | 7% | 0.24 | 1.8e-02 |

### worst 5% by delta PESQ (n = 120)

| factor | category | share among worst | share overall | enrichment | binomial p |
|---|---|---|---|---|---|
| class | wind | 49% | 12% | 3.93 | 1.7e-22 |
| class | mixed | 23% | 12% | 1.87 | 1.2e-03 |
| class | gunshot | 3% | 12% | 0.27 | 8.1e-04 |
| class | siren | 0% | 12% | 0.00 | 2.2e-07 |
| class | artillery | 0% | 12% | 0.00 | 2.2e-07 |
| input SNR (dB) | -10 | 77% | 17% | 4.60 | 2.9e-47 |
| input SNR (dB) | +0 | 1% | 17% | 0.05 | 1.2e-08 |
| input SNR (dB) | +15 | 1% | 17% | 0.05 | 1.2e-08 |
| input SNR (dB) | +5 | 0% | 17% | 0.00 | 6.3e-10 |
| input SNR (dB) | +10 | 0% | 17% | 0.00 | 6.3e-10 |
| environment | wind | 49% | 12% | 3.93 | 1.7e-22 |
| environment | mixed | 23% | 12% | 1.87 | 1.2e-03 |
| environment | cabin | 8% | 18% | 0.46 | 4.1e-03 |
| environment | open | 9% | 40% | 0.23 | 1.4e-13 |
| noise source | synthetic | 23% | 12% | 1.87 | 1.2e-03 |
| noise source | real | 3% | 7% | 0.49 | 1.5e-01 |

(Only categories with enrichment >= 1.5 or <= 0.5 are listed; the full table is in `enrichment.csv`.)

## 2. Correlation of outcomes with measurable factors (MEASURED)

Spearman rho over all main clips (raw), and within class x SNR cells (removes the class and SNR effects). Speech leakage: the value is primary-minus-reference speech level, so a NEGATIVE rho with a bigger value means heavier leakage helps less.

| factor | outcome | raw rho (p) | n | within-cell rho (p) |
|---|---|---|---|---|
| input SNR | delta SI-SNR | -0.72 (0e+00) | 2400 | n/a |
| input SNR | output STOI | +0.71 (0e+00) | 2400 | n/a |
| input SNR | delta PESQ | +0.60 (2e-230) | 2400 | n/a |
| clipped fraction (primary) | delta SI-SNR | +0.31 (3e-54) | 2400 | +0.01 (9e-01) |
| clipped fraction (primary) | output STOI | -0.05 (8e-03) | 2400 | +0.10 (8e-03) |
| clipped fraction (primary) | delta PESQ | -0.00 (9e-01) | 2400 | +0.11 (3e-03) |
| noise peak level (dB SPL) | delta SI-SNR | +0.58 (9e-213) | 2400 | +0.05 (1e-02) |
| noise peak level (dB SPL) | output STOI | -0.38 (6e-82) | 2400 | +0.10 (4e-07) |
| noise peak level (dB SPL) | delta PESQ | -0.34 (2e-65) | 2400 | +0.07 (1e-03) |
| speech leakage into reference (dB; large = little leakage) | delta SI-SNR | +0.02 (3e-01) | 2400 | +0.09 (2e-05) |
| speech leakage into reference (dB; large = little leakage) | output STOI | +0.03 (1e-01) | 2400 | +0.06 (2e-03) |
| speech leakage into reference (dB; large = little leakage) | delta PESQ | +0.03 (2e-01) | 2400 | +0.05 (1e-02) |
| speech level (dB SPL) | delta SI-SNR | +0.01 (7e-01) | 2400 | -0.04 (5e-02) |
| speech level (dB SPL) | output STOI | -0.02 (3e-01) | 2400 | +0.00 (9e-01) |
| speech level (dB SPL) | delta PESQ | -0.02 (4e-01) | 2400 | -0.01 (7e-01) |
| input STOI | delta SI-SNR | -0.38 (3e-82) | 2400 | +0.21 (7e-25) |
| input STOI | output STOI | +0.93 (0e+00) | 2400 | +0.85 (0e+00) |
| input STOI | delta PESQ | +0.72 (0e+00) | 2400 | +0.26 (7e-38) |

## 3. Environment (a proxy for reverberation) and real vs synthetic gunshots (MEASURED, within class x SNR cells)

Each clip's outcome minus its class x SNR cell mean, averaged by environment. `open` = little reverberation, `room` and `cabin` = reverberant simulated rooms.

| environment | n | mean delta SI-SNR relative to cell mean | mean output STOI relative to cell mean |
|---|---|---|---|
| open | 948 | +0.17 dB | +0.005 |
| room | 421 | -0.72 dB | -0.012 |
| cabin | 431 | +0.33 dB | +0.001 |
| gunshot, real recording | 164 | -1.04 dB | -0.025 |
| gunshot, synth recording | 136 | +1.26 dB | +0.030 |

## 4. Close-range weapons (MEASURED)

| peak level (dB SPL) | n | clipped share of samples (primary) | input STOI | output STOI | delta STOI | output PESQ | delta SI-SNR |
|---|---|---|---|---|---|---|---|
| 110 | 80 | 0.00% | 0.85 | 0.93 | +0.08 | 2.72 | +11.9 |
| 120 | 80 | 0.00% | 0.72 | 0.84 | +0.11 | 2.15 | +16.6 |
| 130 | 80 | 1.19% | 0.64 | 0.74 | +0.10 | 1.63 | +18.9 |
| 140 | 80 | 5.24% | 0.59 | 0.67 | +0.08 | 1.48 | +20.4 |
| 150 | 80 | 13.06% | 0.56 | 0.62 | +0.06 | 1.38 | +23.7 |
| 160 | 80 | 22.04% | 0.53 | 0.57 | +0.03 | 1.32 | +25.3 |

Spearman rho of clipped fraction vs output STOI over weapon clips: -0.75 (p 5e-88, n 480); within (class, peak) cells: -0.44 (p 2e-20).
The energy-based dSI-SNR grows with peak level while STOI falls: at high peaks the burst dominates the energy, so removing it scores well on dSI-SNR without recovering the speech (see the table), which is why weapon results must not be read from SI-SNR alone.

## 5. Possible explanations (HYPOTHESES, not tested here)

- Low-SNR failures concentrated in tonal or broadband continuous classes (wind, helicopter, drone, mixed) may reflect limited model capacity, or that speech and noise overlap heavily in time-frequency at those SNRs.
- Wind being weakest is consistent with its being uncorrelated between the two microphones (so the reference adds nothing), but this was not isolated experimentally here.
- Loss of intelligibility at very high peak levels is consistent with speech being masked or destroyed inside the clipped burst (information absent from the input), but this analysis does not separate that from model limitations.
- Reverberant environments may be harder because the target is the direct-path speech; the environment label is only a coarse proxy.
- Correlations with clipping or leakage could be confounded by class and SNR; the within-cell numbers reduce, but do not remove, that concern.

