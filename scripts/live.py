"""Real-time demo: two INMP441 microphones in, enhanced speech out.

  # process a 2-channel WAV ([primary, reference], 16 kHz) through the exact streaming path (testable anywhere)
  python3 live.py model.onnx --ch 16 --file in.wav out.wav

  # live: list audio devices, then capture and play back in real time
  python3 live.py --list-devices
  python3 live.py model.onnx --ch 16 --in-device 1 --out-device 2 [--io-block 64]

Live mode needs `pip3 install sounddevice` and PortAudio (`sudo apt install libportaudio2`).

Latency design: the model works on 256-sample hops (16 ms), but audio I/O runs in small blocks (default 64
samples = 4 ms) that are gathered into hops here, so buffering does not add a full hop on each side.
Estimated mic-to-headphone delay ~ 24 ms (window/overlap-add, mean) + compute + ~2 io-blocks in + ~2 out.
Capture (I2S mics) and playback (USB sound card) run on separate clocks, so they are decoupled by a queue;
if it grows or runs dry the script drops or repeats a chunk and counts it. The printed "capture->playback"
figure is measured: from the moment a hop finished capturing to the moment its output chunk is handed to
the sound card. Add ~8 ms average for the capture wait inside the hop and the sound card's own buffer.
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))       # repo layout
try:
    from anc.runtime import HOP, StreamProcessor
except ImportError:
    from runtime import HOP, StreamProcessor


def run_file(a):
    import soundfile as sf
    x, sr = sf.read(a.file[0], dtype="float64")
    assert sr == 16000 and x.ndim == 2 and x.shape[1] >= 2, "need a 16 kHz 2-channel WAV: [primary, reference]"
    n = (len(x) // HOP) * HOP
    rt = StreamProcessor(a.model, ch=a.ch)
    outs, t = [], []
    for h in range(n // HOP + 1):
        p = x[h * HOP:(h + 1) * HOP, 0] if h < n // HOP else np.zeros(HOP)
        r = x[h * HOP:(h + 1) * HOP, 1] if h < n // HOP else np.zeros(HOP)
        t0 = time.perf_counter()
        outs.append(rt.process(p, r))
        t.append((time.perf_counter() - t0) * 1000)
    y = np.clip(np.concatenate(outs)[HOP:HOP + n], -1, 1)
    sf.write(a.file[1], y, 16000)
    print(f"wrote {a.file[1]}  |  per-hop compute mean {np.mean(t):.2f} ms, max {np.max(t):.2f} ms (deadline {HOP / 16:.0f} ms)")


def run_live(a):
    import sounddevice as sd
    io = a.io_block
    assert HOP % io == 0, "--io-block must divide 256"
    rt = StreamProcessor(a.model, ch=a.ch)
    q_in, q_out = queue.Queue(maxsize=16), queue.Queue()
    acc, stop = [], threading.Event()
    st = dict(overruns=0, underruns=0, dropped=0, max_ms=0.0, hops=0, lat_sum=0.0, lat_n=0, lat_max=0.0)

    def in_cb(indata, frames, t, status):
        if status:
            st["overruns"] += 1
        acc.append(indata[:, :2].astype(np.float64).copy())
        if len(acc) * io >= HOP:
            hop = np.concatenate(acc)
            acc.clear()
            try:
                q_in.put_nowait((hop, time.perf_counter()))              # stamp: capture of this hop is complete
            except queue.Full:
                st["dropped"] += 1

    def out_cb(outdata, frames, t, status):
        if status:
            st["underruns"] += 1
        try:
            chunk, t_src = q_out.get_nowait()
            outdata[:, 0] = chunk
            lat = (time.perf_counter() - t_src) * 1000
            st["lat_sum"] += lat
            st["lat_n"] += 1
            st["lat_max"] = max(st["lat_max"], lat)
        except queue.Empty:
            outdata.fill(0)
            st["underruns"] += 1
        while q_out.qsize() > 4 * HOP // io:                              # output clock slower than input
            q_out.get_nowait()
            st["dropped"] += 1

    def worker():
        prev_t = time.perf_counter()
        while not stop.is_set():
            try:
                hop, t_cap = q_in.get(timeout=0.5)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            y = np.clip(rt.process(hop[:, 0], hop[:, 1]), -1, 1).astype(np.float32)   # output = PREVIOUS hop
            st["max_ms"] = max(st["max_ms"], (time.perf_counter() - t0) * 1000)
            st["hops"] += 1
            for i in range(0, HOP, io):
                q_out.put((y[i:i + io], prev_t))            # this chunk belongs to the previous hop's capture time
            prev_t = t_cap

    threading.Thread(target=worker, daemon=True).start()
    with sd.InputStream(device=a.in_device, samplerate=16000, blocksize=io, channels=2, dtype="float32", callback=in_cb), \
            sd.OutputStream(device=a.out_device, samplerate=16000, blocksize=io, channels=1, dtype="float32", callback=out_cb):
        print(f"running (io block {io} samples = {io / 16:.0f} ms) - Ctrl+C to stop")
        try:
            while True:
                time.sleep(2)
                lat = st["lat_sum"] / max(st["lat_n"], 1)
                print(f"hops {st['hops']}  compute max {st['max_ms']:.1f} ms  capture->playback mean {lat:.0f} ms max {st['lat_max']:.0f} ms"
                      f"  overruns {st['overruns']} underruns {st['underruns']} dropped {st['dropped']}", flush=True)
        except KeyboardInterrupt:
            stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--ch", type=int, default=16)
    ap.add_argument("--file", nargs=2, metavar=("IN.wav", "OUT.wav"))
    ap.add_argument("--in-device")
    ap.add_argument("--out-device")
    ap.add_argument("--io-block", type=int, default=64)
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args()
    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if not a.model:
        sys.exit("give a model .onnx")
    run_file(a) if a.file else run_live(a)


if __name__ == "__main__":
    main()
