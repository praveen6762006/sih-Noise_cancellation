# Final technical report: adaptive noise suppression for defence communication (SIH PS 26052)

This report supersedes the *summary statements* of `results/REPORT.md` (which is kept unchanged as the record of the development phase). Every table below is embedded from the file that produced it; file paths are given so each number can be traced. Statuses use **MEASURED** (from data), **COMPUTED** (deterministic count) and **HYPOTHESIS** (untested explanation).

## 0. Summary of what is and is not established

| Question | Answer (measured, fresh data) |
|---|---|
| Does the deployment model (H) improve over no processing and over the pretrained GTCRN on fresh, never-seen speakers, sources and rooms? | Yes. dSI-SNR +9.74 [9.48, 10.02] (final_A) and +10.10 [9.81, 10.37] (final_B); STOI 0.88 / 0.87; PESQ 2.28 / 2.25. Pretrained GTCRN: +7.1 / +7.3 dB, STOI 0.84 / 0.83, PESQ 1.87 / 1.84. |
| Are the statement's targets (SNR > 15 dB, STOI > 0.85, PESQ > 2.5) met? | **Not in general.** On the deployed path, per class x input-SNR cell: **0 / 48 cells** meet all three when "SNR" means an SI-SNR *improvement* > 15 dB; **12 / 48 cells** (the same 12 on both fresh sets) when it means an *output* SI-SNR > 15 dB. Those 12 are at moderate-to-high input SNR (section 4). |
| Is the deployed path equivalent to the PyTorch evaluation? | Nearly: -0.14 dB dSI-SNR, -0.0007 STOI, -0.02 PESQ on average (all measured with paired CIs); the difference comes from the slow automatic level control (section 4). |
| Does the second (reference) microphone help? | Yes, by about +1.0 dB dSI-SNR (+0.017 STOI, +0.21 PESQ) in a controlled ablation, **provided** speech leakage into it is small (primary at least ~10 dB above reference in the simulated grid); with heavy leakage it hurts (section 5). |
| Is there a stable real-time behaviour? | On this laptop: 32-minute paced run, 120,000 hops, 0 exceptions, no memory growth, streaming output equal to offline within about 52 dB (section 8); **9 missed 16 ms deadlines, all within the first 100 s** (cause not identified). On the physical Pi 4 (8 GB): a 2,000-hop (~32 s) paced benchmark, 0 missed deadlines, no thermal throttling (section 15); no 30-minute-scale soak has been run on the Pi yet. |
| Does it work on the physical Raspberry Pi with real INMP441 microphones? | Partly measured (section 15, 2026-09-22 to 09-27). Compute: Small fp32 mean 5.31 ms/hop (RTF 0.33), about 3x headroom under the 16 ms deadline; INT8 is slower than fp32 on this board's ONNX Runtime build, so fp32 stays the pick. Real dual-mic capture works; with recordings played near the mics, continuous noise (engine/siren) is reduced by about 34 dB block-RMS and impulsive noise (gunshot) by 11-28 dB peak with speech intact; end-to-end live streaming through headphones has not been run yet, and one microphone channel had a recurring intermittent hardware fault that needs re-verifying before it is relied on. |
| What is not established? | Real (recorded) engine/helicopter/drone/wind/siren/artillery noise in the *training/evaluation* data (the Pi bench above uses played-back recordings, not training data); a long (30+ minute) Pi soak; end-to-end live latency; real headset field testing; from-scratch seed variance. |

## 1. Final architecture (deployment candidate)

```
Primary mic  ┐
             ├─> Reference Guard ─> STFT (512, hop 256) + extra features ─> GTCRN-X Model H (robust Small) ─> complex mask ─> iSTFT ─> enhanced speech
Reference mic┘
```

- **Model H** = GTCRN-X *Small* (24,629 trainable parameters), warm-started from the pretrained GTCRN, trained on physically simulated defence noise with reference-channel dropout 0.2 and reference-gain jitter +/-8 dB (6,000 steps, seed 0). Deployed as `models/small_deploy.onnx`; a dynamic-INT8 copy exists.
- **Extra input features** of GTCRN-X: reference magnitude, primary/reference level ratio, coherence, and an impulse flag. The impulse flag is still an *input* of the trained model but was measured to be **negligible** (section 9).
- **Reference Guard** (`src/anc/refguard.py`): runtime detector of a reference that is a copy of the primary (correlation above 1.5 kHz); when it fires the reference input is switched off.
- **Streaming runtime** (`src/anc/runtime.py`): 16 ms hop, 32 ms window; output lags input by one hop (algorithmic delay 16-32 ms, mean 24 ms); slow (~1.5 s) automatic level control.
- **Not part of the pipeline:** NLMS adaptive filtering (evaluated and rejected, section 9), speech-presence head (never built), limiter, ducking, declipping.

## 2. Priority 1: fresh untouched final test sets

Sets `final_A` (utterances of test-clean not used in any earlier set) and `final_B` (test-other, all-new speakers), each with `main` (8 classes x 6 SNRs x 25 = 1,200 clips), `weapon` (240 clips: peaks 110-160 dB SPL) and `speechonly` (50 clips). Master seed 20260921; configuration and checksums are in `results/final_test/final_test_spec.json`, `checksums.sha256`, `LOCK.txt`. **No tuning on these sets was done**; Model H and all thresholds were fixed before they were generated.

### Verification of the fresh final test sets

#### 1. Exact counts

| set | subset | manifest rows | WAV files | distinct speakers | distinct utterances |
|---|---|---|---|---|---|
| final_A | main | 1200 | 1200 | 40 | 549 |
| final_A | weapon | 240 | 240 | 40 | 199 |
| final_A | speechonly | 50 | 50 | 28 | 48 |
| final_B | main | 1200 | 1200 | 33 | 913 |
| final_B | weapon | 240 | 240 | 33 | 225 |
| final_B | speechonly | 50 | 50 | 22 | 49 |

Total mixtures: 2980 (each = one 4 s, 3-channel WAV: noisy primary, reference, clean target). Master seed 20260921, RIR seed 20260921; generation spec in `final_test_spec.json`.

#### 2. Speaker independence

| set | speakers | overlap with TRAINING speakers | overlap with VALIDATION speakers | overlap with EARLIER-TEST speakers |
|---|---|---|---|---|
| final_A | 40 | 0 | 0 | 40 |
| final_B | 33 | 0 | 0 | 0 |

final_A reuses the 40 test-clean speakers of the earlier test (never any training or validation speaker); final_B is speaker-disjoint from everything.

#### 3. Utterance independence

- utterances of final_A used by ANY earlier test/validation set: **0**
- utterances of final_B used by ANY earlier set: **0**
- utterances shared between final_A and final_B: **0**

#### 4. Gunshot file / event independence

| set | real-gunshot draws | distinct files | distinct shots | files used by earlier sets | shots in TRAIN | shots in VALIDATION | shotgun (unseen partition) shots | files outside the never-used list |
|---|---|---|---|---|---|---|---|---|
| final_A | 172 | 34 | 33 | 0 | 0 | 0 | 0 | 0 |
|  | firearm folders drawn: {'glock_17_9mm_caliber': 16, 'ruger_ar_556_dot223_caliber': 13, '38s&ws_dot38_caliber': 5} | | | | | | | |
| final_B | 161 | 35 | 34 | 0 | 0 | 0 | 0 | 0 |
|  | firearm folders drawn: {'glock_17_9mm_caliber': 16, 'ruger_ar_556_dot223_caliber': 13, '38s&ws_dot38_caliber': 6} | | | | | | | |

The never-used pool holds 35 recordings / 34 shots (three firearm types), so both final sets draw from the same small real-gunshot pool. About half of the gunshot-class clips use these real recordings; the rest are synthetic.

#### 5. Room impulse responses

- fresh pool (seed 20260921): 1500 distinct channel responses; identical to any response in the train/val/test/unseen pools: **0**

#### 6. Seeds and audio

- per-clip seeds: 2980 total, 2980 unique, shared with any earlier set: **0**
- checksums: 2980 WAV files recorded; **0** changed since generation

#### 7. Distributions

##### final_A / main

class counts: artillery 150, drone 150, engine 150, gunshot 150, helicopter 150, mixed 150, siren 150, wind 150

| requested SNR (dB) | clips | measured SNR mean | std | min | max |
|---|---|---|---|---|---|
| -10 | 200 | -10.00 | 0.00 | -10.0 | -10.0 |
| -5 | 200 | -5.00 | 0.00 | -5.0 | -5.0 |
| +0 | 200 | -0.00 | 0.00 | -0.0 | +0.0 |
| +5 | 200 | +5.00 | 0.00 | +5.0 | +5.0 |
| +10 | 200 | +10.00 | 0.00 | +10.0 | +10.0 |
| +15 | 200 | +15.00 | 0.00 | +15.0 | +15.0 |

gunshot-class clips: 150 (80 real recordings, 70 synthetic); environments: {'cabin': 223, 'room': 211, 'open': 466, 'wind': 150, 'mixed': 150}
speech level at primary: 75-95 dB SPL; speech leakage into reference: median 15.7 dB (5-95%: 8.8 to 22.2 dB); clips with clipping on the primary: 89

##### final_A / weapon: peak levels {110.0: 40, 120.0: 40, 130.0: 40, 140.0: 40, 150.0: 40, 160.0: 40}, classes {'gunshot': 120, 'artillery': 120}

##### final_A / speechonly: 50 clips at requested SNR [60.0] dB

##### final_B / main

class counts: artillery 150, drone 150, engine 150, gunshot 150, helicopter 150, mixed 150, siren 150, wind 150

| requested SNR (dB) | clips | measured SNR mean | std | min | max |
|---|---|---|---|---|---|
| -10 | 200 | -10.00 | 0.00 | -10.0 | -10.0 |
| -5 | 200 | -5.00 | 0.00 | -5.0 | -5.0 |
| +0 | 200 | -0.00 | 0.00 | -0.0 | +0.0 |
| +5 | 200 | +5.00 | 0.00 | +5.0 | +5.0 |
| +10 | 200 | +10.00 | 0.00 | +10.0 | +10.0 |
| +15 | 200 | +15.00 | 0.00 | +15.0 | +15.0 |

gunshot-class clips: 150 (84 real recordings, 66 synthetic); environments: {'cabin': 208, 'room': 210, 'open': 482, 'wind': 150, 'mixed': 150}
speech level at primary: 75-95 dB SPL; speech leakage into reference: median 15.6 dB (5-95%: 8.9 to 22.6 dB); clips with clipping on the primary: 73

##### final_B / weapon: peak levels {110.0: 40, 120.0: 40, 130.0: 40, 140.0: 40, 150.0: 40, 160.0: 40}, classes {'gunshot': 120, 'artillery': 120}

##### final_B / speechonly: 50 clips at requested SNR [60.0] dB

#### 8. Result

**PASS**: every independence check that must be zero is zero.

Not independent by construction (state this when reporting): the synthetic noise classes come from the same generator as in training (new random draws, same parameter ranges); the gunshot pool is small; final_A's speakers were also in the earlier test set.


### Honest caveats on independence
- `final_A` shares its 40 speakers with the *earlier* test set (utterances are disjoint, speakers are not). `final_B` shares no speaker with anything.
- Only 34 never-used gunshot shots existed (3 firearm types; the shotgun is the held-out weapon in the earlier unseen set) and both fresh sets draw from them.
- Noise for all non-gunshot classes is synthetic, from the same generator family as training.
- **The earlier test set was NOT completely untouched**: it was used for model-selection and guard decisions during development. Results on it are reported only for comparison (size and seed studies) and are labelled "earlier test set".

## 3. Priority 2: Model H on fresh and existing sets (PyTorch protocol)

### Priority 2: Model H (robust Small) evaluation

Protocol for every number here: **PyTorch pipeline, each clip level-normalised by its own RMS** (`scripts/evaluate.py --proc ours`). Values are means over clips with a 95% bootstrap interval in brackets. dSI-SNR = output SI-SNR minus input SI-SNR (both against the dereverberated clean target). PESQ is wideband. The deployed ONNX path is evaluated separately (Priority 3).

#### 1. Fresh untouched final sets, main subset (1,200 clips each)

##### final_A

| model | in SI-SNR | out SI-SNR | dSI-SNR | STOI | PESQ |
|---|---|---|---|---|---|
| Model H | +2.44 [+1.96, +2.91] | +12.18 [+11.84, +12.51] | +9.74 [+9.48, +10.02] | 0.88 [0.87, 0.89] | 2.28 [2.23, 2.33] |
| pretrained GTCRN | +2.44 [+1.98, +2.91] | +9.57 [+9.18, +9.96] | +7.12 [+6.85, +7.41] | 0.84 [0.84, 0.85] | 1.87 [1.83, 1.91] |
| no processing | +2.44 [+1.94, +2.91] | +2.44 [+1.99, +2.91] | +0.00 [+0.00, +0.00] | 0.79 [0.78, 0.80] | 1.33 [1.31, 1.35] |

##### final_B

| model | in SI-SNR | out SI-SNR | dSI-SNR | STOI | PESQ |
|---|---|---|---|---|---|
| Model H | +2.38 [+1.91, +2.85] | +12.48 [+12.14, +12.82] | +10.10 [+9.81, +10.37] | 0.87 [0.86, 0.87] | 2.25 [2.21, 2.30] |
| pretrained GTCRN | +2.38 [+1.91, +2.86] | +9.70 [+9.30, +10.10] | +7.32 [+7.04, +7.63] | 0.83 [0.82, 0.83] | 1.84 [1.81, 1.88] |
| no processing | +2.38 [+1.91, +2.88] | +2.38 [+1.92, +2.86] | +0.00 [+0.00, +0.00] | 0.77 [0.76, 0.78] | 1.30 [1.28, 1.32] |

#### 2a. final_A: Model H per noise class

| class | n | in SI-SNR | out SI-SNR | dSI-SNR | STOI | PESQ |
|---|---|---|---|---|---|---|
| engine | 150 | +2.31 [+0.91, +3.68] | +12.48 [+11.51, +13.40] | +10.17 [+9.48, +10.89] | 0.90 [0.89, 0.92] | 2.28 [2.15, 2.41] |
| helicopter | 150 | +2.36 [+1.04, +3.64] | +10.92 [+9.97, +11.85] | +8.55 [+7.94, +9.19] | 0.87 [0.85, 0.88] | 1.98 [1.87, 2.10] |
| drone | 150 | +2.34 [+0.99, +3.61] | +9.93 [+9.02, +10.84] | +7.59 [+7.08, +8.12] | 0.85 [0.83, 0.87] | 1.97 [1.86, 2.08] |
| wind | 150 | +2.29 [+0.95, +3.63] | +8.52 [+7.44, +9.59] | +6.23 [+5.84, +6.59] | 0.77 [0.74, 0.80] | 1.63 [1.54, 1.73] |
| siren | 150 | +2.33 [+1.01, +3.61] | +15.06 [+14.43, +15.67] | +12.73 [+11.85, +13.66] | 0.96 [0.95, 0.96] | 3.05 [2.94, 3.16] |
| gunshot | 150 | +2.60 [+1.27, +3.92] | +13.34 [+12.49, +14.19] | +10.74 [+9.99, +11.49] | 0.91 [0.90, 0.92] | 2.44 [2.31, 2.57] |
| artillery | 150 | +2.88 [+1.61, +4.12] | +16.09 [+15.30, +16.84] | +13.21 [+12.50, +13.95] | 0.95 [0.94, 0.95] | 2.89 [2.78, 3.00] |
| mixed | 150 | +2.42 [+1.08, +3.72] | +11.13 [+10.20, +12.10] | +8.71 [+8.14, +9.31] | 0.85 [0.83, 0.88] | 2.01 [1.89, 2.13] |

##### final_A: Model H per input SNR

| input SNR (dB) | n | in SI-SNR | out SI-SNR | dSI-SNR | STOI | PESQ |
|---|---|---|---|---|---|---|
| -10 | 200 | -9.56 [-9.73, -9.37] | +4.94 [+4.26, +5.64] | +14.50 [+13.88, +15.12] | 0.74 [0.71, 0.76] | 1.49 [1.42, 1.56] |
| -5 | 200 | -4.86 [-4.94, -4.76] | +7.75 [+7.15, +8.34] | +12.60 [+12.05, +13.21] | 0.81 [0.80, 0.83] | 1.68 [1.61, 1.76] |
| +0 | 200 | -0.02 [-0.05, +0.02] | +10.87 [+10.42, +11.33] | +10.89 [+10.43, +11.37] | 0.88 [0.87, 0.89] | 2.00 [1.92, 2.09] |
| +5 | 200 | +4.90 [+4.87, +4.93] | +13.77 [+13.36, +14.23] | +8.87 [+8.45, +9.30] | 0.93 [0.92, 0.94] | 2.43 [2.33, 2.52] |
| +10 | 200 | +9.80 [+9.76, +9.84] | +16.70 [+16.36, +17.04] | +6.90 [+6.57, +7.25] | 0.96 [0.95, 0.96] | 2.86 [2.78, 2.94] |
| +15 | 200 | +14.40 [+14.29, +14.50] | +19.07 [+18.71, +19.45] | +4.68 [+4.38, +4.97] | 0.97 [0.97, 0.98] | 3.24 [3.16, 3.31] |

#### 2b. final_B: Model H per noise class

| class | n | in SI-SNR | out SI-SNR | dSI-SNR | STOI | PESQ |
|---|---|---|---|---|---|---|
| engine | 150 | +2.36 [+0.95, +3.65] | +12.55 [+11.48, +13.64] | +10.19 [+9.53, +10.84] | 0.88 [0.86, 0.89] | 2.21 [2.08, 2.34] |
| helicopter | 150 | +2.23 [+0.98, +3.60] | +11.34 [+10.46, +12.23] | +9.11 [+8.49, +9.79] | 0.85 [0.83, 0.87] | 2.01 [1.90, 2.12] |
| drone | 150 | +2.33 [+1.03, +3.65] | +10.69 [+9.72, +11.65] | +8.36 [+7.75, +8.98] | 0.84 [0.82, 0.86] | 1.95 [1.84, 2.06] |
| wind | 150 | +2.30 [+1.05, +3.62] | +9.02 [+7.95, +10.07] | +6.72 [+6.30, +7.11] | 0.75 [0.73, 0.78] | 1.63 [1.54, 1.70] |
| siren | 150 | +2.30 [+0.96, +3.64] | +15.08 [+14.41, +15.73] | +12.78 [+11.79, +13.85] | 0.95 [0.94, 0.95] | 2.94 [2.83, 3.05] |
| gunshot | 150 | +2.38 [+1.08, +3.67] | +13.65 [+12.78, +14.50] | +11.27 [+10.45, +12.10] | 0.90 [0.88, 0.91] | 2.49 [2.36, 2.63] |
| artillery | 150 | +2.78 [+1.57, +4.03] | +16.56 [+15.82, +17.31] | +13.78 [+12.97, +14.61] | 0.94 [0.93, 0.95] | 2.92 [2.81, 3.02] |
| mixed | 150 | +2.40 [+1.06, +3.70] | +10.98 [+10.02, +11.91] | +8.58 [+8.01, +9.18] | 0.82 [0.80, 0.84] | 1.88 [1.77, 2.00] |

##### final_B: Model H per input SNR

| input SNR (dB) | n | in SI-SNR | out SI-SNR | dSI-SNR | STOI | PESQ |
|---|---|---|---|---|---|---|
| -10 | 200 | -9.75 [-9.88, -9.59] | +5.07 [+4.35, +5.82] | +14.82 [+14.10, +15.53] | 0.72 [0.69, 0.74] | 1.47 [1.40, 1.55] |
| -5 | 200 | -4.88 [-4.97, -4.79] | +8.08 [+7.47, +8.70] | +12.96 [+12.35, +13.55] | 0.79 [0.77, 0.81] | 1.69 [1.61, 1.78] |
| +0 | 200 | -0.03 [-0.06, +0.01] | +11.51 [+11.02, +12.04] | +11.54 [+11.05, +12.06] | 0.87 [0.86, 0.88] | 2.04 [1.95, 2.13] |
| +5 | 200 | +4.91 [+4.88, +4.94] | +14.29 [+13.88, +14.71] | +9.38 [+8.99, +9.79] | 0.92 [0.91, 0.92] | 2.37 [2.28, 2.45] |
| +10 | 200 | +9.71 [+9.64, +9.77] | +16.87 [+16.48, +17.28] | +7.16 [+6.80, +7.55] | 0.94 [0.94, 0.95] | 2.79 [2.71, 2.88] |
| +15 | 200 | +14.33 [+14.22, +14.44] | +19.07 [+18.69, +19.46] | +4.74 [+4.42, +5.09] | 0.96 [0.96, 0.97] | 3.16 [3.08, 3.23] |

#### 3. Existing unseen-noise set (720 clips: shifted synthetic parameters + held-out shotgun recordings)

| model | in SI-SNR | out SI-SNR | dSI-SNR | STOI | PESQ |
|---|---|---|---|---|---|
| Model H | +2.40 [+1.81, +2.99] | +11.68 [+11.23, +12.13] | +9.28 [+8.94, +9.62] | 0.88 [0.87, 0.88] | 2.22 [2.16, 2.28] |
| pretrained GTCRN | +2.40 [+1.83, +3.01] | +9.26 [+8.77, +9.74] | +6.86 [+6.54, +7.19] | 0.84 [0.83, 0.85] | 1.85 [1.80, 1.89] |
| no processing | +2.40 [+1.76, +2.96] | +2.40 [+1.82, +2.97] | +0.00 [+0.00, +0.00] | 0.78 [0.77, 0.79] | 1.31 [1.28, 1.34] |

Model H per class on the unseen set:

| class | n | dSI-SNR | STOI | PESQ |
|---|---|---|---|---|
| engine | 90 | +7.42 [+6.71, +8.20] | 0.85 [0.82, 0.87] | 1.87 [1.73, 2.01] |
| helicopter | 90 | +8.09 [+7.42, +8.78] | 0.86 [0.84, 0.89] | 1.97 [1.82, 2.12] |
| drone | 90 | +8.57 [+7.84, +9.32] | 0.88 [0.86, 0.90] | 2.16 [2.00, 2.31] |
| wind | 90 | +6.24 [+5.74, +6.72] | 0.78 [0.74, 0.81] | 1.64 [1.53, 1.76] |
| siren | 90 | +11.14 [+10.11, +12.23] | 0.94 [0.92, 0.95] | 2.69 [2.54, 2.84] |
| gunshot | 90 | +11.81 [+10.85, +12.78] | 0.92 [0.91, 0.94] | 2.60 [2.44, 2.76] |
| artillery | 90 | +13.05 [+12.04, +14.11] | 0.95 [0.94, 0.96] | 2.95 [2.81, 3.09] |
| mixed | 90 | +7.89 [+7.26, +8.57] | 0.83 [0.81, 0.86] | 1.88 [1.73, 2.03] |

#### 4. Speech-only sets (no noise; speech leaks into the reference mic)

Input SI-SNR is the ceiling set by the dereverberating target, so the relevant number is how far the output falls below it.

| set | model | in SI-SNR | out SI-SNR | dSI-SNR | STOI | PESQ |
|---|---|---|---|---|---|---|
| final_A | Model H | +25.67 [+24.11, +27.17] | +24.83 [+23.38, +26.38] | -0.83 [-1.39, -0.44] | 1.00 [0.99, 1.00] | 4.25 [4.15, 4.34] |
| final_A | pretrained GTCRN | +25.67 [+24.00, +27.17] | +21.44 [+19.99, +22.93] | -4.22 [-5.24, -3.21] | 0.99 [0.98, 0.99] | 3.69 [3.54, 3.84] |
| final_A | no processing | +25.67 [+24.16, +27.17] | +25.67 [+24.12, +27.14] | +0.00 [+0.00, +0.00] | 1.00 [1.00, 1.00] | 4.20 [4.09, 4.29] |
| final_B | Model H | +26.14 [+24.25, +28.00] | +24.94 [+23.26, +26.72] | -1.20 [-1.80, -0.71] | 1.00 [1.00, 1.00] | 4.27 [4.17, 4.36] |
| final_B | pretrained GTCRN | +26.14 [+24.26, +28.12] | +20.76 [+19.39, +22.16] | -5.38 [-7.23, -3.64] | 0.98 [0.98, 0.99] | 3.61 [3.47, 3.74] |
| final_B | no processing | +26.14 [+24.28, +28.06] | +26.14 [+24.22, +28.11] | +0.00 [+0.00, +0.00] | 1.00 [1.00, 1.00] | 4.18 [4.05, 4.30] |
| earlier test_speechonly | Model H | +29.04 [+27.21, +30.72] | +27.70 [+26.19, +29.16] | -1.34 [-1.80, -0.92] | 1.00 [1.00, 1.00] | 4.34 [4.25, 4.42] |

#### 5. Cross-firearm (gunshot-class clips that use a REAL recording)

The shotgun (12-gauge Remington) was excluded from all training; Glock 17, AR-556 and .38 were in training (different shots). Fresh sets: firearm read from the recording actually drawn. Unseen set: every real gunshot is a shotgun by construction.

| set | firearm | n | model | dSI-SNR | STOI | PESQ |
|---|---|---|---|---|---|---|
| unseen | shotgun (never trained) | 41 | Model H | +9.69 [+8.64, +10.76] | 0.90 [0.87, 0.93] | 2.33 [2.09, 2.57] |
| unseen | shotgun (never trained) | 41 | pretrained GTCRN | +7.68 [+6.52, +8.77] | 0.87 [0.84, 0.90] | 1.93 [1.75, 2.10] |
| unseen | shotgun (never trained) | 41 | no processing | +0.00 [+0.00, +0.00] | 0.80 [0.75, 0.84] | 1.45 [1.31, 1.59] |
| final_A | 38s&ws_dot38_caliber | 13 | Model H | +9.32 [+7.36, +11.37] | 0.89 [0.82, 0.95] | 2.45 [1.98, 2.90] |
| final_A | glock_17_9mm_caliber | 41 | Model H | +10.17 [+8.85, +11.68] | 0.89 [0.86, 0.91] | 2.04 [1.84, 2.23] |
| final_A | ruger_ar_556_dot223_caliber | 26 | Model H | +9.66 [+8.19, +11.10] | 0.87 [0.83, 0.90] | 2.03 [1.73, 2.33] |
| final_A | 38s&ws_dot38_caliber | 13 | pretrained GTCRN | +7.30 [+5.18, +9.54] | 0.85 [0.76, 0.92] | 2.06 [1.71, 2.39] |
| final_A | glock_17_9mm_caliber | 41 | pretrained GTCRN | +8.36 [+6.92, +9.84] | 0.86 [0.83, 0.88] | 1.76 [1.59, 1.93] |
| final_A | ruger_ar_556_dot223_caliber | 26 | pretrained GTCRN | +7.78 [+6.42, +9.16] | 0.82 [0.78, 0.87] | 1.74 [1.53, 1.95] |
| final_A | 38s&ws_dot38_caliber | 13 | no processing | +0.00 [+0.00, +0.00] | 0.80 [0.70, 0.88] | 1.60 [1.32, 1.98] |
| final_A | glock_17_9mm_caliber | 41 | no processing | +0.00 [+0.00, +0.00] | 0.79 [0.75, 0.83] | 1.31 [1.22, 1.40] |
| final_A | ruger_ar_556_dot223_caliber | 26 | no processing | +0.00 [+0.00, +0.00] | 0.77 [0.72, 0.82] | 1.40 [1.23, 1.58] |
| final_B | 38s&ws_dot38_caliber | 13 | Model H | +12.44 [+9.69, +15.84] | 0.91 [0.86, 0.95] | 2.50 [2.08, 2.96] |
| final_B | glock_17_9mm_caliber | 40 | Model H | +9.32 [+7.98, +10.76] | 0.89 [0.86, 0.91] | 2.24 [2.00, 2.47] |
| final_B | ruger_ar_556_dot223_caliber | 31 | Model H | +10.93 [+9.35, +12.64] | 0.83 [0.79, 0.88] | 2.05 [1.76, 2.39] |
| final_B | 38s&ws_dot38_caliber | 13 | pretrained GTCRN | +9.30 [+6.50, +12.24] | 0.87 [0.81, 0.93] | 2.01 [1.64, 2.41] |
| final_B | glock_17_9mm_caliber | 40 | pretrained GTCRN | +7.36 [+5.97, +8.84] | 0.85 [0.81, 0.88] | 1.87 [1.69, 2.04] |
| final_B | ruger_ar_556_dot223_caliber | 31 | pretrained GTCRN | +8.81 [+6.94, +10.68] | 0.79 [0.73, 0.84] | 1.70 [1.49, 1.93] |
| final_B | 38s&ws_dot38_caliber | 13 | no processing | +0.00 [+0.00, +0.00] | 0.80 [0.71, 0.88] | 1.51 [1.24, 1.86] |
| final_B | glock_17_9mm_caliber | 40 | no processing | +0.00 [+0.00, +0.00] | 0.80 [0.75, 0.84] | 1.41 [1.28, 1.54] |
| final_B | ruger_ar_556_dot223_caliber | 31 | no processing | +0.00 [+0.00, +0.00] | 0.70 [0.64, 0.76] | 1.36 [1.22, 1.51] |

#### 6. Close-range weapon sets: metrics by peak level (dB SPL at the primary microphone)

**Do not read dSI-SNR alone at high peaks**: the clipped burst dominates the energy, so a model that only removes the burst scores a large dSI-SNR while intelligibility (STOI) stays low.

##### final_A (gunshot + artillery pooled, 40 clips per peak level)

| model | metric | 110 | 120 | 130 | 140 | 150 | 160 |
|---|---|---|---|---|---|---|---|
| Model H | dSI-SNR | +11.7 | +16.4 | +18.4 | +19.9 | +22.4 | +25.6 |
| Model H | STOI | 0.94 | 0.84 | 0.75 | 0.68 | 0.64 | 0.59 |
| Model H | PESQ | 2.70 | 2.17 | 1.60 | 1.51 | 1.38 | 1.34 |
| pretrained GTCRN | dSI-SNR | +9.8 | +13.9 | +14.3 | +15.2 | +17.0 | +17.0 |
| pretrained GTCRN | STOI | 0.92 | 0.79 | 0.69 | 0.62 | 0.58 | 0.52 |
| pretrained GTCRN | PESQ | 2.35 | 1.88 | 1.44 | 1.42 | 1.27 | 1.26 |
| no processing | dSI-SNR | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |
| no processing | STOI | 0.86 | 0.73 | 0.64 | 0.60 | 0.58 | 0.55 |
| no processing | PESQ | 1.50 | 1.33 | 1.13 | 1.16 | 1.09 | 1.06 |

##### final_B (gunshot + artillery pooled, 40 clips per peak level)

| model | metric | 110 | 120 | 130 | 140 | 150 | 160 |
|---|---|---|---|---|---|---|---|
| Model H | dSI-SNR | +12.1 | +16.8 | +19.3 | +20.8 | +25.1 | +25.0 |
| Model H | STOI | 0.93 | 0.83 | 0.73 | 0.66 | 0.60 | 0.55 |
| Model H | PESQ | 2.74 | 2.14 | 1.66 | 1.45 | 1.38 | 1.29 |
| pretrained GTCRN | dSI-SNR | +10.3 | +14.5 | +15.9 | +14.0 | +17.3 | +15.9 |
| pretrained GTCRN | STOI | 0.90 | 0.79 | 0.68 | 0.58 | 0.54 | 0.51 |
| pretrained GTCRN | PESQ | 2.36 | 1.83 | 1.52 | 1.34 | 1.34 | 1.29 |
| no processing | dSI-SNR | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |
| no processing | STOI | 0.85 | 0.72 | 0.64 | 0.58 | 0.54 | 0.52 |
| no processing | PESQ | 1.57 | 1.25 | 1.14 | 1.13 | 1.09 | 1.12 |

##### earlier weapon set (test_peak, 240 clips; used before, not untouched)

| metric | 110 | 120 | 130 | 140 | 150 | 160 |
|---|---|---|---|---|---|---|
| Model H dSI-SNR | +10.9 | +16.3 | +20.7 | +20.7 | +22.3 | +26.5 |
| Model H STOI | 0.94 | 0.87 | 0.82 | 0.70 | 0.61 | 0.55 |
| Model H PESQ | 2.92 | 2.27 | 1.86 | 1.49 | 1.45 | 1.28 |


## 4. Priority 3: deployed-path target verification (class x input-SNR, ONNX Runtime + guard + level control)

The two SNR readings are shown because the statement only says "SNR > 15 dB": **strict** = SI-SNR improvement > 15 dB; **looser** = output SI-SNR > 15 dB. A cell PASSES only if the 95% bootstrap interval of the 25-clip cell mean lies wholly above the threshold. **The full 48-cell matrices for both paths and both sets are in `results/final_test/P3_summary.md` and `P3_matrix_*.csv`.**

### Verdict on the statement's targets (deployed path, fresh sets)

| | final_A | final_B | PASS on both |
|---|---|---|---|
| ALL, strict reading (dSNR>15 & STOI>0.85 & PESQ>2.5) | 0 / 48 cells | 0 / 48 | 0 |
| ALL, alt reading (oSNR>15 & STOI>0.85 & PESQ>2.5) | 12 / 48 cells | 12 / 48 | 12 |

Cells that PASS ALL (alt reading) on both fresh sets: engine +10 dB, engine +15 dB, helicopter +15 dB, drone +15 dB, siren +10 dB, siren +15 dB, gunshot +10 dB, gunshot +15 dB, artillery +5 dB, artillery +10 dB, artillery +15 dB, mixed +15 dB

Cells that PASS ALL (strict reading) on both fresh sets: none

**All 48 cells do not pass; the statement's targets are met only in the cells listed above.** Wherever a cell is MARG the data cannot distinguish pass from fail at 25 clips.

### Weapon sets: deployed vs PyTorch by peak level (mean over 40 clips per peak)

| set | peak (dB SPL) | STOI PyTorch | STOI deployed | PESQ PyTorch | PESQ deployed | dSI-SNR PyTorch | dSI-SNR deployed |
|---|---|---|---|---|---|---|---|
| final_A | 110 | 0.939 | 0.939 | 2.70 | 2.70 | +11.7 | +11.6 |
| final_A | 120 | 0.839 | 0.838 | 2.17 | 2.17 | +16.4 | +16.1 |
| final_A | 130 | 0.748 | 0.742 | 1.60 | 1.61 | +18.4 | +17.5 |
| final_A | 140 | 0.678 | 0.674 | 1.51 | 1.49 | +19.9 | +18.1 |
| final_A | 150 | 0.640 | 0.629 | 1.38 | 1.34 | +22.4 | +19.6 |
| final_A | 160 | 0.586 | 0.569 | 1.34 | 1.33 | +25.6 | +21.5 |
| final_B | 110 | 0.926 | 0.926 | 2.74 | 2.72 | +12.1 | +12.1 |
| final_B | 120 | 0.835 | 0.836 | 2.14 | 2.15 | +16.8 | +16.7 |
| final_B | 130 | 0.729 | 0.727 | 1.66 | 1.69 | +19.3 | +18.7 |
| final_B | 140 | 0.663 | 0.658 | 1.45 | 1.61 | +20.8 | +19.1 |
| final_B | 150 | 0.602 | 0.597 | 1.38 | 1.35 | +25.1 | +22.8 |
| final_B | 160 | 0.546 | 0.546 | 1.29 | 1.26 | +25.0 | +22.2 |

**Deployed vs PyTorch, final_A:**

#### final_A: deployed minus PyTorch

Paired over 1200 clips (deployed ONNX fp32 minus PyTorch); mean [95% bootstrap CI]:

| metric | difference |
|---|---|
| dSI-SNR (dB) | -0.137 [-0.172, -0.107] (max |per-clip difference| 11.044) |
| STOI | -0.0007 [-0.0011, -0.0004] (max |per-clip difference| 0.0841) |
| PESQ | -0.020 [-0.026, -0.015] (max |per-clip difference| 0.776) |

Cell-level agreement (48 cells): ALL-strict verdict differs in **1**, ALL-alt in **2**, any single-target verdict in **4**; largest cell-mean dSI-SNR difference 1.08 dB.

**INT8 vs fp32:**

#### final_A: INT8 minus fp32 (deployed path)

Paired over 1200 clips (deployed INT8 minus deployed fp32); mean [95% bootstrap CI]:

| metric | difference |
|---|---|
| dSI-SNR (dB) | -0.113 [-0.132, -0.093] (max |per-clip difference| 3.453) |
| STOI | -0.0002 [-0.0006, +0.0001] (max |per-clip difference| 0.0835) |
| PESQ | -0.063 [-0.069, -0.057] (max |per-clip difference| 0.568) |

INT8 ALL-strict PASS 0, ALL-alt PASS 12 (fp32: 0 / 12); ALL verdict differs from fp32 in 2 cells.

### P3 diagnosis: why deployed and PyTorch differ (final_A main)

MEASURED decomposition. Each row: mean over the clips of (variant dSI-SNR minus the PyTorch-path dSI-SNR of the same clip), dB.

#### Largest per-clip discrepancies (8 clips)

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

#### Random sample (60 clips)

| variant | mean difference vs PyTorch (dB) | largest |difference| (dB) |
|---|---|---|
| deployed (guard on, AGC) | -0.045 | 2.13 |
| no guard (AGC) | -0.045 | 2.13 |
| fixed gain (guard on) | +0.006 | 0.26 |
| neither (no guard, fixed gain) | +0.006 | 0.26 |

Reading: if 'fixed gain' removes the discrepancy, the difference comes from the level control (its start-up and slow tracking); if 'no guard' removes it, the guard flagged a healthy reference; if only 'neither' matches PyTorch, both contribute. Any statement about WHY the level control or guard behaves this way on those clips is a hypothesis unless shown above.



## 5. Priority 4: reference-microphone speech-leakage sweep


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

### Decisions (fixed criteria, delta SI-SNR, paired 95% CI)

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

### Benefit of the reference by input SNR: H(ref) - H(zero ref), delta SI-SNR (dB)

| level | -10 dB | -5 dB | +0 dB | +5 dB | +10 dB | +15 dB |
|---|---|---|---|---|---|---|
| -12 dB | -8.69 | -9.61 | -9.96 | -9.69 | -11.88 | -12.44 |
| -6 dB | -0.64 | -0.74 | -0.80 | -0.94 | -1.45 | -1.92 |
| +0 dB | +1.30 | +0.96 | +0.83 | +0.72 | +0.54 | +0.65 |
| +6 dB | +1.64 | +1.29 | +1.05 | +0.90 | +0.73 | +0.89 |
| +12 dB | +1.71 | +1.35 | +1.09 | +0.93 | +0.75 | +0.93 |
| +24 dB | +1.73 | +1.37 | +1.10 | +0.93 | +0.75 | +0.93 |
| +80 dB | +1.73 | +1.36 | +1.10 | +0.92 | +0.74 | +0.92 |

### What the data show (measured), and what they do not

- Levels where the reference helps (CI above 0 vs ignoring it): shadow_db [0, 6, 12, 24, 80].
- Levels where using the reference is worse than ignoring it: shadow_db [-12, -6].
- The grid is coarse (7 levels) and the leakage here is a simulated, level-and-low-pass model of leakage; the real headset's leakage (frequency-dependent, position-dependent) may behave differently. No claim is made beyond these levels.

### P4b: deployed path (reference guard + ONNX runtime) at the harmful levels, 480 clips each

| level (shadow_db) | dSI-SNR | STOI | PESQ |
|---|---|---|---|
| -12 dB | -0.47 | 0.704 | 1.56 |
| -6 dB | +8.09 | 0.848 | 2.09 |
| +0 dB | +9.82 | 0.875 | 2.27 |

For comparison the reference-off model (H with zeros) gets dSI-SNR +9.11 dB on these clips. Whether the guard fires at heavy leakage is not measured here; the numbers only show that the deployed path is still far below the reference-off result at -12 dB.


## 6. Priorities 5 and 6: model size and seed variance


### P5. Fair size comparison (earlier test set, 1,200 clips, PyTorch protocol; mean [95% CI])

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

### P6. Model H: independent training seeds (same configuration: warm start, reference dropout 0.2, gain jitter +/-8 dB, 6,000 steps)

Seed 0 is the deployment model; seeds 1 and 2 were trained afterwards only to measure training variance and were not used for any selection.

#### earlier test set

| seed | dSI-SNR | STOI | PESQ |
|---|---|---|---|
| 0 | +10.00 [+9.72, +10.29] | 0.883 [0.876, 0.890] | 2.34 [2.29, 2.39] |
| 1 | +9.99 [+9.72, +10.29] | 0.882 [0.875, 0.890] | 2.33 [2.28, 2.38] |
| 2 | +9.99 [+9.70, +10.29] | 0.883 [0.875, 0.889] | 2.33 [2.28, 2.38] |
| **mean over 3 seeds** | +9.994 +/- 0.009 (SD); 95% t-interval [+9.971, +10.017]; range +9.987 to +10.005 | 0.8826 +/- 0.0001 (SD); 95% t-interval [0.8823, 0.8829]; range 0.8825 to 0.8827 | 2.3343 +/- 0.0020 (SD); 95% t-interval [2.3293, 2.3392]; range 2.3327 to 2.3365 |

#### fresh final_A main

| seed | dSI-SNR | STOI | PESQ |
|---|---|---|---|
| 0 | +9.74 [+9.47, +10.01] | 0.882 [0.875, 0.889] | 2.28 [2.24, 2.33] |
| 1 | +9.73 [+9.47, +10.00] | 0.882 [0.875, 0.889] | 2.28 [2.23, 2.33] |
| 2 | +9.73 [+9.47, +10.00] | 0.882 [0.875, 0.889] | 2.28 [2.23, 2.32] |
| **mean over 3 seeds** | +9.732 +/- 0.007 (SD); 95% t-interval [+9.713, +9.750]; range +9.727 to +9.740 | 0.8820 +/- 0.0002 (SD); 95% t-interval [0.8813, 0.8826]; range 0.8818 to 0.8822 | 2.2788 +/- 0.0023 (SD); 95% t-interval [2.2730, 2.2845]; range 2.2773 to 2.2815 |

#### fresh final_B main

| seed | dSI-SNR | STOI | PESQ |
|---|---|---|---|
| 0 | +10.10 [+9.81, +10.38] | 0.866 [0.859, 0.873] | 2.25 [2.21, 2.30] |
| 1 | +10.08 [+9.80, +10.37] | 0.866 [0.858, 0.873] | 2.25 [2.21, 2.30] |
| 2 | +10.08 [+9.80, +10.37] | 0.866 [0.859, 0.873] | 2.25 [2.20, 2.30] |
| **mean over 3 seeds** | +10.089 +/- 0.009 (SD); 95% t-interval [+10.065, +10.112]; range +10.083 to +10.100 | 0.8658 +/- 0.0003 (SD); 95% t-interval [0.8651, 0.8665]; range 0.8655 to 0.8661 | 2.2527 +/- 0.0014 (SD); 95% t-interval [2.2491, 2.2563]; range 2.2517 to 2.2544 |



## 7. Priority 7: complexity

| model | parameters (MEASURED) | ONNX size KB (MEASURED) | MACs per hop, executed (COMPUTED) | of which GRU | MMACs/s at 62.5 hops/s (COMPUTED) | independent counter, total (ratio to ours) | textbook offline MACs/frame (COMPUTED) |
|---|---|---|---|---|---|---|---|
| Tiny | 8,389 | 225.1 | 253,096 | 23,904 | 15.8 | 253,096 (1.0) | 253,476 |
| Small | 24,629 | 287.5 | 532,496 | 95,616 | 33.3 | 532,496 (1.0) | 533,256 |
| Base | 80,725 | 504.7 | 1,501,984 | 382,464 | 93.9 | 1,501,984 (1.0) | 1,503,504 |


| model | hops | full hop: mean / p50 / p95 / p99 / max (ms) | ONNX Runtime only: mean / p95 / max (ms) | real-time factor (full mean / 16 ms) | peak RAM (MB) | RAM above imports (MB) |
|---|---|---|---|---|---|---|
| Tiny | 20000 | 1.347 / 1.373 / 1.953 / 2.668 / 11.17 | 0.793 / 1.031 / 3.68 | 0.084 | 280 | 184 |
| Small | 20000 | 1.302 / 1.312 / 1.675 / 1.830 / 2.63 | 0.808 / 1.076 / 4.45 | 0.081 | 280 | 185 |
| Base | 20000 | 1.468 / 1.554 / 1.916 / 2.186 / 3.51 | 0.957 / 1.319 / 5.59 | 0.092 | 280 | 185 |

Labels: parameters, file size, timing, RAM = MEASURED; MACs = COMPUTED (forward hooks on the streaming model, cross-checked against `torch.utils.flop_counter`); FFTs and element-wise operations are not counted. Timing above is single-threaded on the **development laptop CPU**; per-hop time hardly depends on model size because it is dominated by feature computation and ONNX Runtime call overhead. Peak RAM is dominated by the Python + ONNX Runtime baseline. **The actual Raspberry Pi 4 timing is now measured and is in section 15**, not estimated from the laptop.


## 8. Priority 8: 32-minute streaming soak (laptop, real-time paced)

`scripts/soak_test.py`: 120,000 hops of 16 ms fed on a wall-clock schedule through the deployed runtime (`models/small_deploy.onnx`), with memory sampled every 10 s, a 60 s-history control instance and a comparison against offline PyTorch. Files: `results/soak/soak_summary.md|json`, `soak_timeseries.csv`, first/last-60-s audio.

| measure | value |
|---|---|
| model | models/small_deploy.onnx |
| requested_minutes | 32.0 |
| wall_seconds | 1920.0 |
| total_hops_scheduled | 120000 |
| processed_hops | 120000 |
| missed_deadlines | 9 |
| missed_fraction | 7.5e-05 |
| max_lateness_ms | 26.449800491333008 |
| proc_ms_mean | 2.1380271911621094 |
| proc_ms_p50 | 2.135200023651123 |
| proc_ms_p95 | 2.7592999935150146 |
| proc_ms_p99 | 3.1494007110595703 |
| proc_ms_p999 | 3.536400079727173 |
| proc_ms_max | 6.0929999351501465 |
| ram_after_imports_mb | 123.7 |
| ram_start_mb | 127.26 |
| ram_end_mb | 123.8 |
| ram_peak_mb | 153.3 |
| ram_growth_mb_per_hour_after_60s | -24.706133740330664 |
| process_cpu_percent | 14.900826512840522 |
| compute_duty_cycle_percent | 13.362768173217773 |
| exceptions | 0 |
| non_finite_output_hops | 0 |
| silent_gap_hops | 0 |
| guard_flagged_hops | 1 |
| guard_flagged_percent | 0.0008333333333333334 |
| output_peak_abs | 0.47490308929225394 |
| longform_streaming_vs_offline_agreement_db | 52.47224315584216 |
| longform_max_abs_diff | 0.004587674182987955 |
| longform_seconds | 120 |
| state_drift_agreement_db | 48.927628226289855 |
| state_drift_control_60s_history_vs_fresh_db | 61.18706335163792 |
| state_drift_longrun_vs_60s_history_db | 47.88116506680494 |
| state_drift_note | last 30 s of input, first 10 s skipped; drift is indicated only if the long-run instance differs from the fresh one much more than a 60 s-history control does |


Reading (MEASURED):
- All 120,000 hops processed, **0 exceptions, 0 non-finite outputs, 0 silent gaps**; the guard flagged 1 hop.
- Compute per hop: mean 2.14 ms, p95 2.76, p99 3.15, p99.9 3.54, max 6.09 ms against a 16 ms budget (this run measures two instances plus logging, so it is slower than section 7); CPU 15% of one core.
- **9 missed deadlines** (0.0075% of hops; worst lateness 26.4 ms). All occurred in the first ~100 s and none in the remaining ~30 minutes (`soak_timeseries.csv`). The cause was not investigated; **HYPOTHESIS**: operating-system scheduling or background activity on the laptop. Every missed deadline is a real audio glitch on a live device unless buffered.
- Memory: peak 153 MB, end 124 MB vs start 127 MB; no growth (the fitted slope is negative).
- Streaming vs offline (PyTorch, 120 s): 52.5 dB agreement (max sample difference 0.0046). State drift: the long-run instance differs from a fresh instance by an amount (49 dB agreement) comparable to that of a 60 s-history control (61 dB); both are far below audibility, so no drift is indicated.
- **Not tested:** the Pi CPU, real audio-device I/O, clock drift, thermal behaviour.


## 9. Ablations (development phase; earlier test set; PyTorch protocol; from `results/REPORT.md`)

Findings, each as measured there: the **reference microphone** adds +1.00 dB [0.95, 1.06] dSI-SNR (+0.017 STOI, +0.21 PESQ); the **impulse flag** adds +0.01 dB (weapon STOI -0.003), i.e. negligible; extra **loss weight on impulsive frames** has no effect; **NLMS** (a gated adaptive filter on the reference channel) was evaluated in four configurations and **rejected**: after the network -3.4 dB, before the network -2.1 dB (untrained) and -1.2 dB (trained in the loop) with -5.9 dB speech fidelity, and an ungated NLMS destroys speech; **pruning** 30% costs -0.2 dB, 50% costs -1.5 dB, 70% collapses. Sections of the earlier report follow verbatim:

### 4. Ablations (paired differences: first minus second; mean [95% interval])

| question | comparison | test dSI-SNR | test STOI | test PESQ | weapons STOI | weapons PESQ |
|---|---|---|---|---|---|---|
| Does the reference mic help? | B_ref_s0 - A_mono_s0 | +1.00 [+0.95, +1.06]* | +0.017 [+0.015, +0.018]* | +0.209 [+0.196, +0.223]* | +0.025 [+0.017, +0.032]* | +0.021 [-0.025, +0.058] |
| Does the impulse flag help? | C_full_s0 - B_ref_s0 | +0.01 [+0.00, +0.01]* | +0.000 [-0.000, +0.000] | +0.002 [+0.000, +0.003]* | -0.003 [-0.005, -0.001]* | -0.008 [-0.036, +0.019] |
| Does reference dropout help/hurt? | D_fulldrop_s0 - C_full_s0 | -0.17 [-0.20, -0.15]* | -0.003 [-0.003, -0.003]* | -0.033 [-0.037, -0.029]* | -0.001 [-0.004, +0.001] | +0.012 [-0.019, +0.044] |
| Full system vs single mic (seed 0) | C_full_s0 - A_mono_s0 | +1.01 [+0.95, +1.07]* | +0.017 [+0.015, +0.018]* | +0.211 [+0.198, +0.224]* | +0.022 [+0.014, +0.030]* | +0.013 [-0.028, +0.047] |
| Full system vs single mic (seed 1) | C_full_s1 - A_mono_s1 | +1.02 [+0.96, +1.08]* | +0.017 [+0.016, +0.019]* | +0.210 [+0.197, +0.223]* | +0.024 [+0.016, +0.032]* | +0.020 [-0.040, +0.075] |
| Seed-to-seed variation (same config) | C_full_s0 - C_full_s1 | +0.01 [+0.00, +0.02]* | +0.000 [+0.000, +0.000]* | -0.001 [-0.003, +0.001] | +0.003 [+0.001, +0.005]* | -0.018 [-0.042, +0.001] |
| Fine-tuned single mic vs pretrained | A_mono_s0 - gtcrn | +1.78 [+1.66, +1.90]* | +0.025 [+0.023, +0.027]* | +0.281 [+0.261, +0.300]* | +0.039 [+0.032, +0.046]* | +0.064 [+0.013, +0.112]* |
| Full system vs pretrained | C_full_s0 - gtcrn | +2.79 [+2.64, +2.95]* | +0.042 [+0.039, +0.044]* | +0.492 [+0.467, +0.518]* | +0.061 [+0.052, +0.070]* | +0.077 [+0.025, +0.130]* |
| Does the impulse-frame loss weighting matter? | B_ref_s0 - G_noimpw_s0 | -0.03 [-0.04, -0.03]* | -0.000 [-0.001, -0.000]* | -0.005 [-0.006, -0.004]* | -0.001 [-0.002, +0.000] | +0.005 [-0.021, +0.033] |
| Tiny vs Small | T_tiny_s0 - C_full_s0 | -2.44 [-2.55, -2.33]* | -0.049 [-0.051, -0.046]* | -0.358 [-0.378, -0.341]* | -0.041 [-0.047, -0.035]* | -0.124 [-0.178, -0.077]* |
| Base vs Small | L_base_s0 - C_full_s0 | -0.10 [-0.14, -0.07]* | -0.004 [-0.005, -0.004]* | -0.049 [-0.056, -0.042]* | -0.004 [-0.008, +0.000] | -0.017 [-0.060, +0.023] |
| NLMS after the network | C_full_s0_nlmspost - C_full_s0 | -3.43 [-3.71, -3.15]* | -0.023 [-0.024, -0.021]* | -0.428 [-0.456, -0.402]* | -0.028 [-0.036, -0.020]* | -0.230 [-0.302, -0.157]* |
| NLMS before the network (model not retrained) | C_full_s0_nlmspre - C_full_s0 | -2.11 [-2.36, -1.88]* | -0.009 [-0.011, -0.007]* | -0.111 [-0.132, -0.091]* | -0.015 [-0.020, -0.011]* | -0.034 [-0.066, -0.006]* |
| NLMS before the network, trained with it | E_nlmspre_s0 - C_full_s0 | -1.19 [-1.38, -1.00]* | +0.007 [+0.005, +0.008]* | -0.004 [-0.018, +0.009] | +0.006 [+0.002, +0.011]* | -0.001 [-0.046, +0.045] |

### 8. Held-out noise class (dSI-SNR / STOI / PESQ on that class, test set)

'Without it' models never saw the class during training.

| class | trained with it | trained without it | pretrained GTCRN |
|---|---|---|---|
| drone | +8.1 / 0.865 / 2.05 | +7.4 / 0.851 / 1.97 | +2.6 / 0.804 / 1.57 |
| gunshot | +12.6 / 0.934 / 2.78 | +12.5 / 0.932 / 2.76 | +10.5 / 0.908 / 2.27 |

### 9. Reference-microphone faults (dSI-SNR / STOI, every 3rd test clip)

The single-mic model does not use the reference, so it is the control. Dropout training is meant to make the two-mic model survive a failed reference.

| condition | single mic | two mic + flag | + reference dropout |
|---|---|---|---|
| normal | +9.26 / 0.867 | +10.30 / 0.885 | +10.13 / 0.882 |
| reference dead (all zeros) | +9.26 / 0.867 | +2.54 / 0.824 | +8.81 / 0.859 |
| reference = copy of primary | +9.26 / 0.867 | -15.69 / 0.510 | -9.02 / 0.537 |
| reference +10 dB | +9.26 / 0.867 | +4.48 / 0.815 | +4.96 / 0.824 |
| reference -10 dB | +9.26 / 0.867 | +6.27 / 0.859 | +9.36 / 0.870 |
| reference delayed 5 ms | +9.26 / 0.867 | +9.95 / 0.878 | +9.97 / 0.877 |

### 10. Robust deployment model and the reference guard (dSI-SNR / STOI, every 3rd test clip)

Guard = runtime copy-detector (`src/anc/refguard.py`): a reference strongly correlated with the primary above 1.5 kHz is replaced by silence. It needs about 0.4 s to react, and this clip-average includes that first stretch. H = trained with reference dropout 0.2 and +/-8 dB gain mismatch.

| condition | D (dropout), no guard | D + guard | H (robust) + guard |
|---|---|---|---|
| normal | +10.13 / 0.882 | +10.12 / 0.881 | +9.98 / 0.879 |
| reference dead (all zeros) | +8.81 / 0.859 | +8.81 / 0.859 | +8.91 / 0.861 |
| reference = copy of primary | -9.02 / 0.537 | +3.60 / 0.815 | +7.33 / 0.845 |
| reference +10 dB | +4.96 / 0.824 | +4.96 / 0.824 | +9.12 / 0.869 |
| reference -10 dB | +9.36 / 0.870 | +9.36 / 0.870 | +9.54 / 0.872 |
| reference delayed 5 ms | +9.97 / 0.877 | +9.97 / 0.877 | +9.87 / 0.876 |

Normal-condition cost of the robust training (full test set):

| model | dSI-SNR | STOI | PESQ |
|---|---|---|---|
| C (no dropout) | +10.32 | 0.888 | 2.40 |
| D (dropout 0.2) | +10.14 | 0.885 | 2.37 |
| H (dropout + wide mismatch) | +10.00 | 0.883 | 2.34 |

**Two-microphone gain.** In the controlled ablation the two-microphone advantage is about **+1 dB**; the reference microphone is not a large effect, and it carries the leakage risk of section 5.


## 10. Priority 10: failure analysis of Model H


Clips analysed: 2400 main (final_A + final_B, 8 classes x 6 SNRs) and 480 weapon clips (peaks 110-160 dB SPL). PyTorch protocol. Everything below is descriptive; **MEASURED** items are computed from the result files, **HYPOTHESIS** items are untested explanations.

### 1. Which conditions are over-represented among the worst 5% of clips (MEASURED)

Worst = lowest value of the measure over the 2,400 main clips. Enrichment > 1 means the category is more common among the worst clips than overall.

#### worst 5% by delta SI-SNR (n = 120)

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

#### worst 5% by delta STOI (n = 120)

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

#### worst 5% by delta PESQ (n = 120)

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

### 2. Correlation of outcomes with measurable factors (MEASURED)

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

### 3. Environment (a proxy for reverberation) and real vs synthetic gunshots (MEASURED, within class x SNR cells)

Each clip's outcome minus its class x SNR cell mean, averaged by environment. `open` = little reverberation, `room` and `cabin` = reverberant simulated rooms.

| environment | n | mean delta SI-SNR relative to cell mean | mean output STOI relative to cell mean |
|---|---|---|---|
| open | 948 | +0.17 dB | +0.005 |
| room | 421 | -0.72 dB | -0.012 |
| cabin | 431 | +0.33 dB | +0.001 |
| gunshot, real recording | 164 | -1.04 dB | -0.025 |
| gunshot, synth recording | 136 | +1.26 dB | +0.030 |

### 4. Close-range weapons (MEASURED)

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

### 5. Possible explanations (HYPOTHESES, not tested here)

- Low-SNR failures concentrated in tonal or broadband continuous classes (wind, helicopter, drone, mixed) may reflect limited model capacity, or that speech and noise overlap heavily in time-frequency at those SNRs.
- Wind being weakest is consistent with its being uncorrelated between the two microphones (so the reference adds nothing), but this was not isolated experimentally here.
- Loss of intelligibility at very high peak levels is consistent with speech being masked or destroyed inside the clipped burst (information absent from the input), but this analysis does not separate that from model limitations.
- Reverberant environments may be harder because the target is the direct-path speech; the environment label is only a coarse proxy.
- Correlations with clipping or leakage could be confounded by class and SNR; the within-cell numbers reduce, but do not remove, that concern.



## 11. Priority 9: plots

High-resolution PNG and vector PDF in `results/plots/` (regenerate with `python scripts/make_plots.py`): 01 dSI-SNR vs input SNR; 02 STOI vs input SNR; 03 PESQ vs input SNR; 04 per class; 05 single vs two mic; 06 impulse-flag ablation; 07 NLMS ablation; 08/08b leakage robustness; 09 model size trade-off; 10 weapon SPL vs STOI; 11 weapon SPL vs PESQ; 12a-c failure analysis. Plots 05-07 and 09 use the earlier test set (development-phase ablations), the others use the fresh sets.

## 12. Limitations (read before quoting any number)

1. **The statement's targets are not met in general.** Deployed path: 0/48 cells (strict dSI-SNR > 15 dB) and 12/48 cells (output SI-SNR > 15 dB) with all of STOI > 0.85 and PESQ > 2.5; 7-8 more cells are undecided at 25 clips per cell. Low-input-SNR cells (-10 to 0 dB) never pass; wind and drone are the weakest classes.
2. **Model H is the current deployment candidate**, not a final product. Model choices during development (C vs H, guard settings) were made using the earlier test set.
3. **The earlier test set was not fully untouched** (used in model and guard decisions); the fresh sets `final_A`/`final_B` are the only ones not used for any decision. `final_A` shares speakers with the earlier test set.
4. **No real recordings of engine, helicopter, drone, wind, siren or artillery noise.** They are synthetic (physical-model based); only speech and gunshots are real recordings. Performance on real noise of these classes is unknown.
5. **Extreme-gunshot-SPL results must not be judged by SI-SNR alone.** At 140-160 dB SPL, dSI-SNR rises to +20-25 dB while STOI falls to 0.6-0.7 and PESQ to about 1.3-1.5; the burst dominates energy, so removing it scores well on SI-SNR without recovering the speech (section 10).
6. Leakage into the reference microphone is a simulated level-plus-low-pass model; real headsets may differ. The reference is harmful below about 10 dB of primary-over-reference speech level in this model, and the guard only partially mitigates heavy leakage.
7. Seed variance was measured only for warm-started fine-tuning (SD 0.009 dB); from-scratch variance is unknown. Cell verdicts rest on 25 clips per cell.
8. The deployed path loses about 0.14 dB on average to the automatic level control; individual clips differ by up to 11 dB (worst: -10 dB siren/drone/artillery).
9. Sections 1-14 above are laptop-only (PyTorch/ONNX on the development machine, simulated microphones and rooms). **Section 15 now adds real Raspberry Pi 4 and real INMP441 measurements**, but those are a short bench (2,000-hop timing run; short playback clips), not the fresh-test-set scale of sections 2-4, and one microphone channel had a recurring intermittent fault that needs re-verifying before it is relied on.
10. The SNR target is ambiguous in the statement; both readings are reported. SI-SNR is measured against a dereverberated direct-path target.

## 13. Final status table

| Task | Status | Script used | Output files | Key result | Remaining issue |
|---|---|---|---|---|---|
| P1 fresh untouched test sets | Done, verified | `make_final_test.py`, `verify_final_test.py`, `recover_provenance.py` | `results/final_test/{VERIFICATION.md,final_test_spec.json,checksums.sha256,LOCK.txt}`, `data/generated/final_{A,B}` | 2 sets, 2,980 clips in total; 0 speaker/utterance/gunshot/RIR overlap with train/val (final_A shares speakers with earlier test) | Small unused gunshot pool (34 shots); synthetic noise |
| P2 Model H evaluation | Done | `run_p2_eval.sh`, `final_analysis.py` | `results/final_test/P2_*.{md,csv,json}` | +9.7 / +10.1 dB, STOI 0.88 / 0.87, PESQ 2.28 / 2.25; unseen +9.3; shotgun +9.7; speech-only -0.8 to -1.2 dB | Real non-gunshot noise absent |
| P3 deployed-path verification | Done | `run_p3_eval.sh`, `p3_analysis.py`, `targets_matrix.py`, `p3_diagnose.py` | `results/final_test/P3_*` | 0/48 strict, 12/48 looser; ONNX vs PyTorch -0.14 dB | Targets not met at low SNR |
| P4 leakage sweep | Done | `run_p4_leakage.sh`, `run_p4b_guard.sh`, `leakage_analysis.py` | `results/leak_sweep/*`, `plots/` | Helps at >= ~10 dB leakage margin (+~1.1 dB), harmful below | Simulated leakage only |
| P5 size comparison | Done | `run_p5_p6_train.sh`, `run_p5_p6_eval.sh`, `size_variance_analysis.py` | `results/size_variance/*` | Tiny +7.9, Small +9.3, Base +10.2 dB (from scratch); warm start +1.07 dB | No warm-started Base |
| P6 seed variance | Done | same | same | SD 0.009 dB (3 seeds) | Only warm-started fine-tuning |
| P7 complexity | Done (laptop) | `complexity.py` | `results/complexity/*` | Small 24.6K params, 33 MMACs/s, 1.3 ms/hop, 280 MB peak RSS | Pi timing now in section 15; Pi RAM/percentiles still open |
| P8 soak test | Done (laptop, 32 min) | `soak_test.py` | `results/soak/*` | 0 exceptions, no memory growth, 9 early missed deadlines | Cause of the misses; long Pi soak (section 15) still open |
| P9 plots | Done | `make_plots.py` | `results/plots/*.png|pdf` | 15 figures | Several use the earlier test set (labelled) |
| P10 failure analysis | Done | `failure_analysis.py` | `results/failure_analysis/*` | Worst: wind, drone, +15 dB (headroom), real gunshots, clipped bursts | Explanations are hypotheses |
| P11 report | Done | `make_final_report.py` | `results/FINAL_REPORT.md` | this document | - |
| Pi hardware bring-up (2026-09-22 to 09-27) | Partly done, see section 15 | `pi_bundle/benchmark.py`, `live.py`, on-Pi scripts (`persec.py`, `meter.py`, `snr_model.py`, `ref_ablation.py`, `gunshot_peaks.py`, `score_segments.py`) | Section 15 of this report (source: project memory, not files in this repo copy); demo audio in `SIH_Drive_Upload/04_Demo_Audio_RealPiHardware/` | Small fp32 5.31 ms/hop (RTF 0.33), 0 missed in a 2,000-hop run; no throttling; continuous noise ~-34 dB block-RMS, gunshot -11 to -28 dB peak, speech ~-0.2 to -0.3 dB, with played-back recordings | Live end-to-end streaming demo, long soak, exact end-to-end latency, real headset, primary-mic fault re-verification |

## 14. READY FOR RASPBERRY PI CHECKLIST (tasks that genuinely need the Pi or physical hardware)

Updated after real Pi bring-up (2026-09-22 to 09-27, detail in section 15). Actual hardware is a **Pi 4 Model B, 8 GB RAM** (the original constraints assumed 2 GB; this only removes memory pressure as a concern, everything below still applies).

1. ~~Pi 4 CPU timing per hop for FP32 and INT8 ONNX~~ **MEASURED (mean, RTF)** — Small fp32 5.31 ms/hop (RTF 0.33), Small INT8 5.88 ms/hop (RTF 0.37), Tiny 4.80 ms/hop, on a 2,000-hop run. **Still open:** p95/p99/max hop time on the Pi (only mean was recorded).
2. ~~FP32 vs INT8 speed on the Pi~~ **DONE** — INT8 is *slower* than fp32 on this board's ONNX Runtime/ARM64 build (mirrors the laptop finding); fp32 Small stays the deployment pick, INT8 should not be used for speed.
3. Pi RAM use (peak RSS of the runtime) — **still open**, not recorded during the benchmark runs.
4. ~~Temperature and thermal throttling under sustained load~~ **DONE** — 52.5-56 degC across all benchmark runs, no throttling, no heatsink/fan needed for this workload.
5. Sustained real-time operation (long soak, missed deadlines, audio under/overruns) — **partly done**: 0 missed deadlines in the 2,000-hop (~32 s) benchmark and in two 63 s loud-playback stress tests (0 dead seconds on both channels); **a 30+ minute soak on the Pi itself (the on-device equivalent of section 8) has not been run.**
6. ~~ONNX Runtime installation and compatibility on the Pi's OS/architecture~~ **DONE** — installs and runs in a venv on Raspberry Pi OS Lite 64-bit (Debian Trixie, ARM64); system pip is blocked by the OS, use a venv.
7. Live microphone input and speaker/headset output through the audio stack — **partly done**: real I2S dual-mic capture is verified and used throughout; **the full live, real-time loop (`live.py` with simultaneous mic capture and headphone output) has not been run yet** — testing so far played a recording near the mics and scored the captured file afterward, not a live streaming session.
8. Measured end-to-end latency (microphone to output), including audio-device buffers — **still open**; only per-hop compute time is measured (item 1), not a stopwatch/timestamp measurement of the full audio path.
9. Real dual-microphone coherence and speech leakage into the reference microphone (`scripts/record_stereo.py`, `scripts/analyze_two_mic.py`) — **partly done, informally**: an ad hoc reference-ablation bench (`ref_ablation.py`, one speaker, mics adjacent) found the reference added negligible benefit in that specific geometry; **a proper coherence/leakage measurement at multiple real positions and levels (the hardware analogue of section 5) has not been run**, and the formal `record_stereo.py`/`analyze_two_mic.py` tools were not reported as used.
10. I2S or USB clock drift between the two channels — **not applicable in the usual sense**: both INMP441s share one I2S bus (one SCK/WS clock pair, one SD line carrying both channels), so the two channels are sampled synchronously by construction; drift of the kind two independent USB devices would have was not separately measured because the wiring avoids it.
11. INMP441 overload behaviour near real weapon-level sound — **still open**: testing used a gunshot *recording* played through a laptop speaker near the mic, not a genuine high-SPL impulsive source, so real microphone-overload behaviour at 120+ dB SPL is untested (the overload point is built into the training simulation, not verified acoustically).
12. Real headset testing (fit, wind noise, cable/handling noise) — **still open**; testing so far is bench-mounted mics with recordings played from a laptop speaker, not a worn headset.
13. Long-duration hardware soak — **still open** (see item 5).

**New item found during bring-up, not in the original checklist:** the primary microphone channel had a recurring intermittent hardware fault (dropped to all-zero) across the session; it was fixed and passed two stress tests as of 2026-09-27, but **re-verify it is still working (repeat the dead-seconds stress test) before any demo or judging**, since it regressed silently before.

## 15. Raspberry Pi hardware results (measured on the physical device, 2026-09-22 to 2026-09-27)

**Source note:** this bring-up was done over SSH directly on the Pi in separate working sessions; the raw logs live on the Pi / in that session's history, not as files in this repository copy, so the numbers below are transcribed from the record kept at the time rather than embedded from a results file as sections 1-14 are. Nothing here has been re-verified from a saved log in this pass; treat exact figures as approximate and re-run the referenced scripts to regenerate a saved log if an auditable file is needed.

### 15.1 Hardware and OS (as built, corrects earlier planning assumptions)
Raspberry Pi 4 Model B, **8 GB RAM** (the original constraints document assumed 2 GB; the extra RAM only removes memory-pressure planning as a concern). 32 GB microSD. Raspberry Pi OS Lite 64-bit (Debian Trixie). Two INMP441 MEMS microphones on a shared I2S bus (SCK/WS/SD shared; primary mic on the left slot via L/R->GND, reference mic on the right slot via L/R->3.3V), enabled via `dtparam=i2s=on` + `dtoverlay=googlevoicehat-soundcard`, appearing as ALSA card `sndrpigooglevoi`. SSH key-only access; passwordless sudo enabled with the user's consent for setup convenience.

### 15.2 Compute timing (`pi_bundle/benchmark.py`, 2,000 hops, 16 ms deadline each)

| model | mean time/hop | real-time factor | missed deadlines | note |
|---|---|---|---|---|
| Small fp32 (deployment) | 5.31 ms | 0.33 | 0 (paced: 5.27 ms, same) | about 3x headroom under the 16 ms budget |
| Small INT8 | 5.88 ms | 0.37 | 0 | **slower than fp32** on this board's ONNX Runtime/ARM64 build, matching the earlier x86-laptop finding; do not deploy INT8 for speed |
| Tiny | 4.80 ms | 0.30 | 0 |  |

Temperature stayed 52.5-56 degC across all runs with no thermal throttling and no heatsink or fan. **Not measured:** p95/p99/max per-hop time (only the mean and 0-missed count were recorded) and peak RAM.

### 15.3 Real microphone characteristics
- Every capture starts with a roughly 100-200 ms I2S startup burst (peak amplitude around 0.9); this is dropped from analysis (the first ~1 s), not a signal fault.
- Measured noise floor: RMS 0.0008-0.0011 (about -59 dBFS). The INMP441 is weak at range: normal speech at arm's length sits near this floor; close-talk (~5 cm) gives RMS 0.02-0.03 (roughly 14 dB SNR at the mic).
- The primary-mic channel had a recurring intermittent fault (dropping to all-zero for entire recordings), reproduced multiple times; as of the last check (2026-09-27) it passed two consecutive 63 s loud-playback stress tests with 0 dead seconds on both channels. **Not yet trusted without re-verification** — repeat the stress test before relying on it.

### 15.4 Acoustic bench test (recordings played from a laptop speaker near the Pi's mics, scored against the same clean reference used elsewhere in this report)
This is a bench proxy for a live source, not a worn-headset or human-speaker test, and not on the scale of the fresh test sets (sections 2-4): a handful of clips, not hundreds.

- **Both mics, guard on** (`seg_cap3.wav`): speech alone changed by -1.2 / -0.7 / -0.3 dB (near-transparent); noise-only block-RMS reduction: engine -14.2 dB, siren -19.0 dB, gunshot -8.5 dB; speech+noise block-RMS reduction: engine -6.2 dB, siren -7.1 dB, gunshot -3.9 dB.
- **Primary mic alone** scored within about 0.5 dB of the two-mic result in this particular setup (mics adjacent, single playback source) — i.e. the reference microphone added close to nothing here. This is a narrow bench geometry, not a repeat of the controlled two-mic ablation in section 9 (which used many simulated positions and did find about +1 dB), so it does not override that result; it shows the effect can vanish in an unfavourable single-source layout.
- **Best clean run** (+6 dB louder playback, single mic, the number to quote for continuous noise): speech -0.2 to -0.3 dB (untouched), noise-only engine/siren -33.6 / -34.8 dB block-RMS, speech+noise engine/siren -4.5 / -4.8 dB.
- **Gunshot is not weak — the block-RMS numbers understate it.** A 6 s scoring block is mostly near-silence with about 0.6 s of actual shot, so block-average RMS dilutes the result. Scoring only the loudest 10 ms windows (`gunshot_peaks.py`) gives noise-only peak reduction of 11-28 dB and speech+gunshot peak reduction of 30-45 dB. **Use peak-level reduction for impulsive noise (gunshot/artillery) and block-RMS reduction for continuous noise (engine/siren/drone); block-RMS badly understates impulsive-noise performance.**
- Demo recordings from this bench are included in `SIH_Drive_Upload/04_Demo_Audio_RealPiHardware/` (raw two-channel capture and model output, two source-level conditions).

### 15.5 What real-hardware evidence does and does not add
**Adds:** the model runs inside the 16 ms budget on the actual target CPU with headroom to spare, at a stable temperature, with no installation blockers; INT8 is confirmed not worth deploying on this hardware either; real I2S capture works end-to-end into the same runtime code used in sections 1-14; noise suppression is present in a real acoustic (not simulated) bench, both for continuous and impulsive noise, with speech left close to untouched.
**Does not add:** confirmation of the class x SNR target matrix (section 4) on real recordings; a long-duration soak on the Pi; a measured end-to-end latency; a real headset test; a resolved coherence/leakage measurement on real hardware; or full confidence in the primary microphone's reliability, which needs re-verifying before any demo.
