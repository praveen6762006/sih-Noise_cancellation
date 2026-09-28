# Verification of the fresh final test sets

## 1. Exact counts

| set | subset | manifest rows | WAV files | distinct speakers | distinct utterances |
|---|---|---|---|---|---|
| final_A | main | 1200 | 1200 | 40 | 549 |
| final_A | weapon | 240 | 240 | 40 | 199 |
| final_A | speechonly | 50 | 50 | 28 | 48 |
| final_B | main | 1200 | 1200 | 33 | 913 |
| final_B | weapon | 240 | 240 | 33 | 225 |
| final_B | speechonly | 50 | 50 | 22 | 49 |

Total mixtures: 2980 (each = one 4 s, 3-channel WAV: noisy primary, reference, clean target). Master seed 20260921, RIR seed 20260921; generation spec in `final_test_spec.json`.

## 2. Speaker independence

| set | speakers | overlap with TRAINING speakers | overlap with VALIDATION speakers | overlap with EARLIER-TEST speakers |
|---|---|---|---|---|
| final_A | 40 | 0 | 0 | 40 |
| final_B | 33 | 0 | 0 | 0 |

final_A reuses the 40 test-clean speakers of the earlier test (never any training or validation speaker); final_B is speaker-disjoint from everything.

## 3. Utterance independence

- utterances of final_A used by ANY earlier test/validation set: **0**
- utterances of final_B used by ANY earlier set: **0**
- utterances shared between final_A and final_B: **0**

## 4. Gunshot file / event independence

| set | real-gunshot draws | distinct files | distinct shots | files used by earlier sets | shots in TRAIN | shots in VALIDATION | shotgun (unseen partition) shots | files outside the never-used list |
|---|---|---|---|---|---|---|---|---|
| final_A | 172 | 34 | 33 | 0 | 0 | 0 | 0 | 0 |
|  | firearm folders drawn: {'glock_17_9mm_caliber': 16, 'ruger_ar_556_dot223_caliber': 13, '38s&ws_dot38_caliber': 5} | | | | | | | |
| final_B | 161 | 35 | 34 | 0 | 0 | 0 | 0 | 0 |
|  | firearm folders drawn: {'glock_17_9mm_caliber': 16, 'ruger_ar_556_dot223_caliber': 13, '38s&ws_dot38_caliber': 6} | | | | | | | |

The never-used pool holds 35 recordings / 34 shots (three firearm types), so both final sets draw from the same small real-gunshot pool. About half of the gunshot-class clips use these real recordings; the rest are synthetic.

## 5. Room impulse responses

- fresh pool (seed 20260921): 1500 distinct channel responses; identical to any response in the train/val/test/unseen pools: **0**

## 6. Seeds and audio

- per-clip seeds: 2980 total, 2980 unique, shared with any earlier set: **0**
- checksums: 2980 WAV files recorded; **0** changed since generation

## 7. Distributions

### final_A / main

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

### final_A / weapon: peak levels {110.0: 40, 120.0: 40, 130.0: 40, 140.0: 40, 150.0: 40, 160.0: 40}, classes {'gunshot': 120, 'artillery': 120}

### final_A / speechonly: 50 clips at requested SNR [60.0] dB

### final_B / main

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

### final_B / weapon: peak levels {110.0: 40, 120.0: 40, 130.0: 40, 140.0: 40, 150.0: 40, 160.0: 40}, classes {'gunshot': 120, 'artillery': 120}

### final_B / speechonly: 50 clips at requested SNR [60.0] dB

## 8. Result

**PASS**: every independence check that must be zero is zero.

Not independent by construction (state this when reporting): the synthetic noise classes come from the same generator as in training (new random draws, same parameter ranges); the gunshot pool is small; final_A's speakers were also in the earlier test set.
