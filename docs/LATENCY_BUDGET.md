# Latency budget (mic to headphone) and what is still to be measured

Target set for this project: **<= 60 ms** end to end. The problem statement only asks for latency "suitable for real-time
communication"; 60 ms is our own limit (interactive speech is generally comfortable well beyond this).

## Model path (verified)

The model processes 256-sample hops (16 ms) with a 512-sample (32 ms) sqrt-Hann window and 50% overlap-add.

* Verified by cross-correlation on the deployed runtime: the raw output stream lags the input by exactly **256 samples
  (16 ms)**, correlation 1.000.
* A sample at position *s* of its hop is captured at 16h + s/16 ms; its output is emitted when the next hop is complete,
  so its delay is (256 - s) + 256 samples = **16 to 32 ms, mean 24 ms**. This is a property of the design, not of the hardware.
* Compute per hop (NumPy front end + ONNX Runtime, one thread): **about 1.3 ms mean (p99 1.8 ms) in the P7 measurement, 2.1 ms mean in the 32-min soak** on the development laptop when the
  machine is idle (measured with cProfile: FFTs 0.10, impulse detector 0.04, features 0.13, remaining ORT and glue), up to
  about 2.5 ms when the laptop was busy training. Model size barely matters (Tiny, Small and Base take the same time):
  fixed per-hop overhead dominates, so the Pi figure scales with CPU speed, not with parameter count.
* The model output is one hop late by construction (see above); this is included in the 24 ms mean.

## Audio I/O (design, to be measured on the device)

`scripts/live.py` runs the sound cards in small blocks (default 64 samples = 4 ms) and gathers them into hops, so I/O
does not add a whole hop on each side (with 256-sample blocks it would add about 8 ms in and 32 ms out).

| stage | mean | worst | status |
|---|---|---|---|
| wait for the I/O block to fill (capture) | 2 ms | 4 ms | design value, io block 64 |
| model path: window + overlap-add | 24 ms | 32 ms | **verified** |
| compute per hop | ~1.3 ms here; **Pi 4 (8 GB), measured: 5.31 ms mean, Small fp32, RTF 0.33** | mean only; p99/max not recorded | `pi_bundle/benchmark.py`, 2,000-hop bench, 0 missed deadlines, 52-56 degC, no throttling |
| output queue + sound-card buffer (about 2 blocks) | 8 ms | 12 ms | design value, sound-card dependent |
| I2S / USB device latency | unknown | | measure |
| **total** (Pi compute now measured; device latency still unknown) | **about 39 ms** | **about 52 ms** | |

With the Pi's own compute measured at 5.31 ms mean, the budget leaves roughly **8 ms of slack in the worst case and
about 21 ms in the mean case** before the 60 ms limit, once the still-unmeasured I2S/USB device latency is added.
Only the mean/RTF was recorded on the Pi; p99 and max per-hop time (needed to bound the worst case properly) were not,
and the end-to-end `live.py` capture-to-playback figure below has not been run yet either.

## How to measure on the device

1. `python3 benchmark.py <model>.onnx --ch 16 --paced` gives per-hop time (mean, p99, max) and missed 16 ms deadlines.
2. `python3 live.py <model>.onnx --ch 16 --in-device N --out-device M` prints the measured capture-to-playback time
   (from the moment a hop finished capturing to the moment its output chunk is handed to the sound card), plus overruns,
   underruns and dropped chunks. Add about 8 ms for the average capture wait inside the hop, plus the sound card's own buffering.
3. For a true acoustic figure: click near the primary microphone, record both the direct click and the headphone output
   on a second device with two channels, and read the delay between the two onsets.
4. Run for at least 30 minutes to see thermal throttling (`vcgencmd get_throttled`) and clock drift between the
   I2S capture clock and the USB playback clock (the `dropped` counter).
