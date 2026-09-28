# SIH PS 26052: presentation notes (claims limited to what results/REPORT.md supports)

## 1. The one-paragraph pitch
A lightweight, causal, two-microphone speech-enhancement system for defence communication. A primary microphone at the
mouth and an outward-facing reference microphone feed a 25K-parameter complex-domain network (GTCRN-based) that runs
frame by frame on a Raspberry Pi 4 CPU at 16 kHz with about 24 ms of algorithmic delay. It is trained on physically
calibrated defence noise (engine, helicopter, drone, wind, siren, gunshot, artillery) including the clipping a real
MEMS microphone produces near weapons, and is evaluated per noise class, per SNR, on close-range weapon events, and on
noise it has never seen.

## 2. Headline numbers (held-out speakers and held-out gunshot events; results/REPORT.md)
| | SNR improvement | STOI | PESQ |
|---|---|---|---|
| No processing | 0 | 0.789 | 1.33 |
| Pretrained GTCRN (single mic) | +7.5 dB | 0.847 | 1.91 |
| Ours (Small, reference mic) | **+10.3 dB** | **0.888** | **2.40** |
| Ours, deployment model (robust + guard), PyTorch | +10.0 dB | 0.883 | 2.34 |
| Same model through the deployed ONNX runtime | +9.9 dB | 0.882 | 2.31 |
| Close-range weapons >= 140 dB SPL: unprocessed -> ours | 0 -> +23 dB | 0.557 -> 0.624 | 1.11 -> 1.41 |

Numbers above are from the earlier test set (used in development decisions, so not completely untouched). On the **fresh, verified,
untouched sets** (final_A/final_B; results/FINAL_REPORT.md) the deployment model H gives dSI-SNR +9.7 / +10.1 dB, STOI 0.88 / 0.87, PESQ 2.28 / 2.25
(PyTorch); the deployed ONNX path is 0.14 dB lower on average.

Problem-statement targets (SNR > 15 dB, STOI > 0.85, PESQ > 2.5), **deployed path, fresh sets, per class x input-SNR cell (25 clips,
95% interval must clear the threshold)**: **0 of 48 cells** if "SNR" means SI-SNR improvement > 15 dB; **12 of 48 cells** (the same 12 on both fresh
sets) if it means output SI-SNR > 15 dB: engine +10/+15, helicopter +15, drone +15, siren +10/+15, gunshot +10/+15, artillery +5/+10/+15, mixed +15.
Nothing passes at -10, -5 or 0 dB input SNR. Say clearly that the targets are met only at moderate-to-high input SNR, and only under the looser
reading. (An earlier "19 of 48" statement was mean-based on the earlier test set and is withdrawn.) SI-SNR is against a dereverberated direct-path target.

## 3. What is different (defensible statements)
1. **A second, reference microphone helps a small model, measurably.** +1.0 dB SI-SNR, +0.017 STOI, +0.21 PESQ over the
   same model with one microphone; the two training seeds agree to 0.01 dB. (Prior art: DC-GTCRN uses two microphones with
   spatial separation for far-field speech; ours is a close-talk primary plus outward reference and is tested on defence noise.)
2. **Physically calibrated training data.** Levels are in dB SPL through an INMP441 model (120 dB SPL overload), so clipping
   near weapons comes from the physics. Fine-tuning on it improves close-range-weapon intelligibility over the pretrained
   model (STOI 0.563 -> 0.602 single mic, 0.624 with the reference).
3. **A defence-noise benchmark** with a per-class x per-SNR grid, a peak-level (dB SPL) weapon set, an unseen-noise split
   (shifted parameters + a held-out firearm), and a speech-only distortion test.
4. **Robustness engineering for a real two-microphone headset.** A two-mic model collapses if the reference fails (+2.5 dB
   vs +9.3 dB single-mic). Reference-dropout training plus a runtime copy-detector keeps the worst tested fault at +7.3 dB.
5. **Measured size/quality trade-off** (Tiny 8K / Small 24K / Base 81K parameters, MACs computed): quality rises with size; Small warm-started from the pretrained GTCRN is the compute-efficient choice.

## 4. What we tried that did not help (say this; it shows rigor)
- Impulse/clipping flag as a network input: +0.01 dB, weapon STOI -0.003. Extra loss weight on impulsive frames: no effect.
- Gated NLMS adaptive filter on the reference mic: after the network -3.4 dB; before it, even trained in the loop, -1.2 dB
  with no PESQ gain and -5.9 dB speech fidelity. The problem statement lists the adaptive filter as optional; we evaluated it
  and left it out of the deployed pipeline.
- Model size: from scratch under identical conditions Tiny +7.9, Small +9.3, Base +10.2 dB (Base +0.97 dB over Small for 2.8x the compute). Warm-starting Small from the pretrained GTCRN gives +1.07 dB, so warm-started Small matches from-scratch Base. (An earlier statement that Base gained nothing was a confound and is withdrawn.) Pruning: 30% is free, 50% costs 1.5 dB, 70% collapses.
- INT8 dynamic quantisation: quality unchanged, size 287 -> 229 KB, not faster on x86 (Pi to be measured).

## 5. Claims NOT to make
- "Novel AI military noise cancellation" (prior work exists, including a Springer 2024 paper on battlefield noise cancellation
  that we could not access to verify, and the DRDO review's CRN/GAN examples).
- "Impulse-aware architecture" or "speech-gated adaptive filtering" as contributions: neither improved results.
- That it meets SNR > 15 dB / STOI > 0.85 / PESQ > 2.5 in general: 0/48 cells strict, 12/48 looser reading, only at moderate-to-high input SNR (section 2).
- Anything about real (recorded) engine/helicopter/drone/wind/siren/artillery noise (synthetic), or SI-SNR alone at extreme weapon levels (STOI 0.55-0.65 and PESQ ~1.3-1.5 at 150-160 dB SPL).
- That it cancels noise at the wearer's ear: it enhances the transmitted speech; passive protection is unchanged.
- Real-headset performance: a bench test exists (played-back recordings near real INMP441 mics on the actual Pi:
  speech near-untouched, continuous noise ~-34 dB block-RMS, gunshot -11 to -28 dB peak), but no worn-headset or
  long-duration hardware soak has been done, and the primary-mic channel's intermittent fault needs re-verifying
  (results/FINAL_REPORT.md section 15).

## 6. Likely questions
- **Why a Pi 4 and not a Jetson?** Low cost and scalability; the model needs about 33 MMACs/s. Measured on the laptop CPU:
  about 1.3 ms per 16 ms hop (idle laptop; 32-min soak: 0 exceptions, no memory growth, 9 early missed deadlines).
  **Now also measured on the real Pi 4 (8 GB):** Small fp32 5.31 ms/hop (RTF 0.33, ~3x headroom), 0 missed deadlines in a
  2,000-hop bench, no thermal throttling (52-56 degC); INT8 is slower than fp32 on this board too, so fp32 is deployed.
  TensorRT does not apply; ONNX Runtime does.
- **Latency?** 24 ms mean algorithmic (16-32 ms, verified) plus compute (~5.3 ms measured on the Pi); budget in
  docs/LATENCY_BUDGET.md. Target <= 60 ms; the full audio-device round-trip latency is not yet measured on the Pi.
- **What if the reference microphone fails?** Section 9-10 of the report: dropout training + guard; worst tested case +7.3 dB.
- **Real data?** Speech and gunshots are real recordings; the other noise classes are synthetic; leakage is simulated
  (tools to measure it on the headset: scripts/record_stereo.py, scripts/analyze_two_mic.py).
- **Why is wind the weakest class?** Wind is uncorrelated between the two microphones, so the reference cannot help.
- **Why not a bigger model?** Base is +0.97 dB better than Small when both start from scratch, at 2.8x the compute (94 vs 33 MMACs/s); no pretrained Base exists to warm-start, and Small warm-started already matches it. Whether a warm-started or robust Base would be worth the compute on the Pi is untested.
