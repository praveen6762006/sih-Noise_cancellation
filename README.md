# AI/ML-enabled adaptive noise suppression for defence communication (SIH PS 26052, DRDO)

A lightweight, causal, two-microphone speech-enhancement system for impulsive, non-stationary and
stationary defence noise, sized to run on a Raspberry Pi 4 (CPU only, 16 kHz).

Results and all tables: **`results/FINAL_REPORT.md`** (fresh untouched test sets, deployed-path target verification, leakage, size, seeds, complexity, soak, failure analysis, Pi checklist; regenerate with `python scripts/make_final_report.py`). The development-phase record is `results/REPORT.md`.
Deployment model: `models/small_deploy.onnx` (robust Small, 24.6K parameters; reference dropout + gain-mismatch training, with the
runtime reference guard). Deployed-path result: +9.9 dB SNR improvement, STOI 0.882, PESQ 2.31 on held-out clips.
Fallback: `models/tiny_deploy.onnx` (8K parameters, +7.7 dB). Pi package: `pi_bundle/`.

## Pipeline (mirrors the problem statement)

| Statement stage | Where |
|---|---|
| Scalable noisy/clean dataset pipeline, varying SNR, reverb, clipping | `src/anc/data/`, `scripts/make_dataset.py`, `configs/dataset.yaml` |
| STFT + full-band/sub-band complex-domain model | `src/anc/model/gtcrn_x.py` (GTCRN-derived; Tiny 8K / Small 24K / Base 81K parameters) |
| SI-SNR + L1/L2-type + spectral losses, augmentation | `src/anc/losses.py`, `scripts/train.py` |
| SNR / STOI / PESQ evaluation | `scripts/evaluate.py` (PESQ runs in WSL, see `scripts/pesq_score.py`) |
| Optional lightweight adaptive filter (LMS) | `src/anc/nlms.py` (gated block NLMS on the reference mic) |
| Quantization / ONNX conversion | `scripts/export_onnx.py`, `scripts/quantize_onnx.py` (TensorRT does not apply to a Pi) |
| Real-time inference engine on embedded hardware | `src/anc/runtime.py` (NumPy + ONNX Runtime), `scripts/live.py` |
| Prototype with primary + reference microphones | Raspberry Pi 4 + 2x INMP441, see "Hardware steps" |

## What the experiments support (see results/REPORT.md, section 4)
| Design idea | Measured effect (paired, held-out test set) | Verdict |
|---|---|---|
| Reference-microphone conditioning (level ratio, coherence, reference magnitude) | +1.00 dB SI-SNR [0.95, 1.06], STOI +0.017, PESQ +0.21; same in a second seed; also on unseen noise | **Supported** |
| Fine-tuning on calibrated defence noise with physically modelled clipping | +1.8 dB over the pretrained model; weapon STOI 0.563 -> 0.602 | **Supported** |
| Tiny / Small / Base family | Fair comparison (all from scratch, same conditions, earlier test set): Tiny +7.9, Small +9.3, Base +10.2 dB; Base is +0.97 dB over Small for 2.8x the compute. Warm-starting Small from the pretrained GTCRN adds +1.07 dB (no pretrained Base exists). An earlier claim that Base was no better than Small was a confound and is withdrawn | **Small (warm-started) is the deployment choice on compute grounds** |
| Impulse/clipping flag as a network input | +0.01 dB overall; weapon STOI -0.003 (slightly worse) | **Not supported** as an input feature |
| Extra loss weight on impulsive frames | removing it: -0.03 dB, weapon STOI -0.001 (no effect) | **Not supported** |
| Reference dropout training (+ runtime guard) | dead reference: +2.5 dB without, +8.8 dB with; copied reference: -15.7 -> 0.0 dB with the guard | **Supported** (robustness, see section 9-10) |
| Gated NLMS stage on the reference mic | after the network -3.4 dB; before it, trained in the loop: -1.2 dB, PESQ unchanged, speech fidelity -5.9 dB | **Not supported**; kept only as an optional module |
| Defence-noise benchmark (per class, per SNR, peak-SPL weapon set, unseen split) | - | Deliverable |

None of these is claimed as new in itself: neural speech enhancement, dual-microphone processing and adaptive
filtering are established (prior work includes GTCRN, DC-GTCRN, DeepFilterNet, and military noise-cancellation
studies). The contribution is the defence-specific integration, the evaluation protocol and the measured results.

## Reproduce
```
pip install numpy scipy soundfile pyyaml pystoi pyroomacoustics einops matplotlib onnx onnxruntime onnxsim
pip install torch --index-url https://download.pytorch.org/whl/cu124        # GPU training
python scripts/make_dataset.py --split val --per-cell 10 --out data/generated/val        # needs data/speech (LibriSpeech)
python scripts/train.py --name C_full_s0 --size small --pretrained --steps 6000 --workers 10
python scripts/evaluate.py --data data/generated/test --proc ours --ckpt checkpoints/C_full_s0_last.pt
python scripts/export_onnx.py --ckpt checkpoints/C_full_s0_last.pt --out models/small_full.onnx
```
Third-party: `third_party/gtcrn` (MIT, Rong et al.) supplies the base model code and pretrained weights.
Data: LibriSpeech (CC BY 4.0), edge-collected gunshot audio (Zenodo 7004819, CC BY 4.0); other noise classes are synthetic.

## Hardware steps (need the Raspberry Pi and microphones)
1. `python scripts/make_pi_bundle.py models/<model>.onnx`, copy `pi_bundle/` to the Pi, `pip3 install numpy scipy onnxruntime`.
2. `python3 benchmark.py <model>.onnx --ch 16 --paced` for the per-hop time and missed 16 ms deadlines.
3. Record the mic pair (`scripts/record_stereo.py`): one recording of you talking, one of noise only; run
   `scripts/analyze_two_mic.py` to measure real speech leakage and noise coherence, then set `shadow_db` in the config accordingly.
4. `python3 live.py <model>.onnx --ch 16 --in-device N --out-device M` for the live demo.

## Real-hardware validation (Raspberry Pi 4, 8GB, 2x real INMP441 mics)
Everything below was measured on the actual deployed device, not simulated:
- **Compute:** 5.3 ms mean per 16 ms hop (real-time factor 0.33), 0 missed deadlines over 2000 hops, no thermal throttling.
- **Live loop (mic -> model -> headphone):** runs continuously with 46-52 ms average end-to-end latency, 0 overruns/drops.
- **Speech:** preserved to within -0.2 to -0.6 dB (essentially untouched).
- **Continuous noise (engine/siren), acoustically played and captured through the real mics:** cut by 7-34 dB depending on playback level, block-RMS.
- **Impulsive noise (gunshot):** a 6 s block average understates this class badly, since most of a gunfire block is silence between shots. Measured on the actual shot transients: peaks cut 11-28 dB alone, 30-45 dB with speech present.
- Audio evidence (real mic input vs. real model output, same gain applied to both): see `demo/pi_listen/` and `SIH_Drive_Upload/04_Demo_Audio_RealPiHardware/`.
- Full methodology and caveats: `results/FINAL_REPORT.md`, section 15.

## Known limits
- The problem statement's literal target (SNR>15dB **and** STOI>0.85 **and** PESQ>2.5, all at once) is met in 0/48 test cells under a strict reading, 12/48 under a looser one - only at moderate-to-high input SNR, never below 0 dB input SNR. Stated plainly in `results/FINAL_REPORT.md` rather than hidden.
- The primary microphone's breadboard wiring has been intermittently unreliable (loose contact); soldering header pins is the recommended permanent fix before any live judging demo. Always re-run a stress-test capture immediately before a demo - see `results/FINAL_REPORT.md` section 15 for the check.
- Speech leakage into the reference mic and room acoustics are simulated in training; the two mics in the current bench setup sit adjacent to each other, so the reference mic measurably helps very little in this exact layout (it is trained to help more with real physical separation/facing).
- Very low input SNR (<= -5 dB) stays hard for a model this small; results are reported per SNR bin.
- The headphones carry the enhanced transmit-path speech; this is not ambient-noise cancellation at the wearer's ear.
