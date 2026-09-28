"""Assemble results/FINAL_REPORT.md from the generated summaries (nothing is re-computed here, and results/REPORT.md is not modified).

  python scripts/make_final_report.py

Hand-written text (architecture, corrections, limitations, checklist) is in this file; every number table is embedded verbatim from the
file that produced it, so the report cannot drift from the results.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def read(p):
    return (R / p).read_text(encoding="utf-8")


def section(text, start, stops=("## ",)):
    """Text from the line starting with `start` up to the next header of the same level."""
    lines, out, on = text.splitlines(), [], False
    for l in lines:
        if l.startswith(start):
            on = True
            out.append(l)
            continue
        if on and any(l.startswith(s) for s in stops):
            break
        if on:
            out.append(l)
    return "\n".join(out).strip()


def demote(text, n=1):
    return re.sub(r"^(#+) ", lambda m: "#" * (len(m.group(1)) + n) + " ", text, flags=re.M)


def main():
    P2, P3, VER = read("final_test/P2_summary.md"), read("final_test/P3_summary.md"), read("final_test/VERIFICATION.md")
    LEAK, SZ, FA, SOAK = read("leak_sweep/LEAKAGE_SUMMARY.md"), read("size_variance/SIZE_VARIANCE_SUMMARY.md"), read("failure_analysis/FAILURE_ANALYSIS.md"), read("soak/soak_summary.md")
    ST, TM, DIAG, OLD = read("complexity/static.md"), read("complexity/timing.md"), read("final_test/P3_diagnosis.md"), read("REPORT.md")
    soak = json.load(open(R / "soak" / "soak_summary.json"))

    L = []
    L.append("""# Final technical report: adaptive noise suppression for defence communication (SIH PS 26052)

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

""" + demote(VER, 2))

    L.append("""
### Honest caveats on independence
- `final_A` shares its 40 speakers with the *earlier* test set (utterances are disjoint, speakers are not). `final_B` shares no speaker with anything.
- Only 34 never-used gunshot shots existed (3 firearm types; the shotgun is the held-out weapon in the earlier unseen set) and both fresh sets draw from them.
- Noise for all non-gunshot classes is synthetic, from the same generator family as training.
- **The earlier test set was NOT completely untouched**: it was used for model-selection and guard decisions during development. Results on it are reported only for comparison (size and seed studies) and are labelled "earlier test set".

## 3. Priority 2: Model H on fresh and existing sets (PyTorch protocol)

""" + demote(P2, 2))

    L.append("""
## 4. Priority 3: deployed-path target verification (class x input-SNR, ONNX Runtime + guard + level control)

The two SNR readings are shown because the statement only says "SNR > 15 dB": **strict** = SI-SNR improvement > 15 dB; **looser** = output SI-SNR > 15 dB. A cell PASSES only if the 95% bootstrap interval of the 25-clip cell mean lies wholly above the threshold. **The full 48-cell matrices for both paths and both sets are in `results/final_test/P3_summary.md` and `P3_matrix_*.csv`.**

""" + demote(section(P3, "## Verdict on the statement's targets"), 1) + "\n\n" + demote(section(P3, "## Weapon sets"), 1) + "\n\n" +
             "**Deployed vs PyTorch, final_A:**\n\n" + demote(section(P3, "### final_A: deployed minus PyTorch", stops=("### ", "## ")), 1) + "\n\n" +
             "**INT8 vs fp32:**\n\n" + demote(section(P3, "### final_A: INT8 minus fp32", stops=("### ", "## ")), 1) + "\n\n" + demote(DIAG, 2))

    L.append("""
## 5. Priority 4: reference-microphone speech-leakage sweep

""" + demote(LEAK.split("\n", 1)[1], 1))

    L.append("""
## 6. Priorities 5 and 6: model size and seed variance

""" + demote(SZ.split("\n", 1)[1], 1))

    L.append("""
## 7. Priority 7: complexity

""" + ST + "\n\n" + TM + """
Labels: parameters, file size, timing, RAM = MEASURED; MACs = COMPUTED (forward hooks on the streaming model, cross-checked against `torch.utils.flop_counter`); FFTs and element-wise operations are not counted. Timing above is single-threaded on the **development laptop CPU**; per-hop time hardly depends on model size because it is dominated by feature computation and ONNX Runtime call overhead. Peak RAM is dominated by the Python + ONNX Runtime baseline. **The actual Raspberry Pi 4 timing is now measured and is in section 15**, not estimated from the laptop.
""")

    L.append(f"""
## 8. Priority 8: 32-minute streaming soak (laptop, real-time paced)

`scripts/soak_test.py`: 120,000 hops of 16 ms fed on a wall-clock schedule through the deployed runtime (`models/small_deploy.onnx`), with memory sampled every 10 s, a 60 s-history control instance and a comparison against offline PyTorch. Files: `results/soak/soak_summary.md|json`, `soak_timeseries.csv`, first/last-60-s audio.

""" + SOAK.split("\n", 2)[2] + f"""

Reading (MEASURED):
- All {soak['processed_hops']:,} hops processed, **0 exceptions, 0 non-finite outputs, 0 silent gaps**; the guard flagged 1 hop.
- Compute per hop: mean {soak['proc_ms_mean']:.2f} ms, p95 {soak['proc_ms_p95']:.2f}, p99 {soak['proc_ms_p99']:.2f}, p99.9 {soak['proc_ms_p999']:.2f}, max {soak['proc_ms_max']:.2f} ms against a 16 ms budget (this run measures two instances plus logging, so it is slower than section 7); CPU {soak['process_cpu_percent']:.0f}% of one core.
- **{soak['missed_deadlines']} missed deadlines** ({soak['missed_fraction']*100:.4f}% of hops; worst lateness {soak['max_lateness_ms']:.1f} ms). All occurred in the first ~100 s and none in the remaining ~30 minutes (`soak_timeseries.csv`). The cause was not investigated; **HYPOTHESIS**: operating-system scheduling or background activity on the laptop. Every missed deadline is a real audio glitch on a live device unless buffered.
- Memory: peak {soak['ram_peak_mb']:.0f} MB, end {soak['ram_end_mb']:.0f} MB vs start {soak['ram_start_mb']:.0f} MB; no growth (the fitted slope is negative).
- Streaming vs offline (PyTorch, 120 s): {soak['longform_streaming_vs_offline_agreement_db']:.1f} dB agreement (max sample difference {soak['longform_max_abs_diff']:.4f}). State drift: the long-run instance differs from a fresh instance by an amount ({soak['state_drift_agreement_db']:.0f} dB agreement) comparable to that of a 60 s-history control ({soak['state_drift_control_60s_history_vs_fresh_db']:.0f} dB); both are far below audibility, so no drift is indicated.
- **Not tested:** the Pi CPU, real audio-device I/O, clock drift, thermal behaviour.
""")

    L.append("""
## 9. Ablations (development phase; earlier test set; PyTorch protocol; from `results/REPORT.md`)

Findings, each as measured there: the **reference microphone** adds +1.00 dB [0.95, 1.06] dSI-SNR (+0.017 STOI, +0.21 PESQ); the **impulse flag** adds +0.01 dB (weapon STOI -0.003), i.e. negligible; extra **loss weight on impulsive frames** has no effect; **NLMS** (a gated adaptive filter on the reference channel) was evaluated in four configurations and **rejected**: after the network -3.4 dB, before the network -2.1 dB (untrained) and -1.2 dB (trained in the loop) with -5.9 dB speech fidelity, and an ungated NLMS destroys speech; **pruning** 30% costs -0.2 dB, 50% costs -1.5 dB, 70% collapses. Sections of the earlier report follow verbatim:

""" + demote(section(OLD, "## 4. Ablations"), 1) + "\n\n" + demote(section(OLD, "## 8. Held-out noise class"), 1) + "\n\n" +
             demote(section(OLD, "## 9. Reference-microphone faults"), 1) + "\n\n" + demote(section(OLD, "## 10. Robust deployment model"), 1) + """

**Two-microphone gain.** In the controlled ablation the two-microphone advantage is about **+1 dB**; the reference microphone is not a large effect, and it carries the leakage risk of section 5.
""")

    L.append("""
## 10. Priority 10: failure analysis of Model H

""" + demote(FA.split("\n", 1)[1], 1))

    L.append("""
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
""")
    (R / "FINAL_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", R / "FINAL_REPORT.md")


if __name__ == "__main__":
    main()
