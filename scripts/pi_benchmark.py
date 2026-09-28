"""Benchmark the streaming model on the target device (run this ON the Raspberry Pi).

  python3 benchmark.py model.onnx --ch 16                # timing only, as fast as possible
  python3 benchmark.py model.onnx --ch 16 --paced        # paced at real time (16 ms per hop): shows deadline misses

Needs only numpy, scipy and onnxruntime. Reports per-hop time (mean, p50, p95, p99, max), how many hops
exceeded the 16 ms deadline, CPU temperature and throttling state (if vcgencmd is present), and the
measured mic-independent latency budget of the model path.
"""
import argparse
import subprocess
import time

import numpy as np

try:
    from anc.runtime import StreamProcessor          # inside the repo
except ImportError:
    from runtime import StreamProcessor              # inside the flat Pi bundle

HOP = 256
DEADLINE_MS = HOP / 16.0


def vcgen(*args):
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception:                                # noqa: BLE001
        return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ch", type=int, default=16, help="model width: 8 tiny, 16 small, 32 base")
    ap.add_argument("--hops", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--paced", action="store_true")
    a = ap.parse_args()

    rt = StreamProcessor(a.model, ch=a.ch, threads=a.threads)
    rng = np.random.default_rng(0)
    # speech-like burst noise on the primary, correlated noise on the reference
    prim = rng.standard_normal(a.hops * HOP) * 0.03
    ref = 0.7 * prim + 0.01 * rng.standard_normal(a.hops * HOP)
    for h in range(50):                              # warm-up (not timed)
        rt.process(prim[h * HOP:(h + 1) * HOP], ref[h * HOP:(h + 1) * HOP])
    rt.reset()

    times, t_next = [], time.perf_counter()
    print(f"temp before: {vcgen('measure_temp')}   throttled: {vcgen('get_throttled')}")
    for h in range(a.hops):
        if a.paced:
            t_next += DEADLINE_MS / 1000
            while time.perf_counter() < t_next:
                pass
        t0 = time.perf_counter()
        rt.process(prim[h * HOP:(h + 1) * HOP], ref[h * HOP:(h + 1) * HOP])
        times.append((time.perf_counter() - t0) * 1000)
        if a.paced and time.perf_counter() > t_next + DEADLINE_MS / 1000:     # fell behind: resynchronise
            t_next = time.perf_counter()
    t = np.array(times)
    print(f"temp after:  {vcgen('measure_temp')}   throttled: {vcgen('get_throttled')}")
    print(f"per-hop processing time (ms): mean {t.mean():.2f}  p50 {np.percentile(t, 50):.2f}  p95 {np.percentile(t, 95):.2f}  "
          f"p99 {np.percentile(t, 99):.2f}  max {t.max():.2f}   [deadline {DEADLINE_MS:.0f} ms]")
    print(f"real-time factor {t.mean() / DEADLINE_MS:.3f}   hops over deadline: {(t > DEADLINE_MS).sum()} / {len(t)}")
    print(f"model-path latency = 32 ms window + {t.mean():.1f} ms mean compute (+ audio I/O buffers, measured separately)")


if __name__ == "__main__":
    main()
