"""How good is the impulse/clipping detector? False alarms on speech and continuous noise, detection of weapon events.

  python scripts/detector_analysis.py

Uses the deployed numpy detector (anc.runtime.ImpulseDetector) hop by hop (256 samples) on both microphones.
  * false-alarm rate: share of hops flagged where there is no impulsive event (speech only, continuous noise classes)
  * detection rate: share of gunshot/artillery events whose onset hop or the next one is flagged
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from anc.runtime import HOP, ImpulseDetector   # noqa: E402

IMPULSIVE = {"gunshot", "artillery"}


def hop_flags(x_p, x_r):
    dp, dr = ImpulseDetector(), ImpulseDetector()
    n = len(x_p) // HOP
    return np.array([bool((dp.hop(x_p[h * HOP:(h + 1) * HOP]) | dr.hop(x_r[h * HOP:(h + 1) * HOP])).any()) for h in range(n)])


def main():
    out = {}
    # 1) speech only
    d = ROOT / "data/generated/test_speechonly"
    rows = [json.loads(l) for l in open(d / "manifest.jsonl")]
    fa = [hop_flags(*sf.read(d / m["file"], dtype="float64")[0][:, :2].T).mean() for m in rows]
    print(f"speech only (no noise, {len(rows)} clips): {100 * np.mean(fa):.2f}% of hops flagged (false alarms)")

    # 2) continuous noise classes, by SNR
    d = ROOT / "data/generated/test"
    rows = [json.loads(l) for l in open(d / "manifest.jsonl")]
    fa_cls, det_cls = defaultdict(list), defaultdict(list)
    for m in rows:
        x = sf.read(d / m["file"], dtype="float64")[0]
        f = hop_flags(x[:, 0], x[:, 1])
        if m["cls"] in IMPULSIVE:
            for s in m["impulse_samples"]:
                h = s // HOP
                det_cls[(m["cls"], round(m["snr_db"] / 5) * 5)].append(bool(f[h:h + 2].any()))
        else:
            fa_cls[(m["cls"], round(m["snr_db"] / 5) * 5)].append(f.mean())
    print("\nfalse alarms on continuous noise + speech (share of hops flagged):")
    print(f"{'class':<11}" + "".join(f"{b:>8}" for b in (-10, -5, 0, 5, 10, 15)) + "   (SNR dB)")
    for c in ("engine", "helicopter", "drone", "wind", "siren"):
        print(f"{c:<11}" + "".join(f"{100 * np.mean(fa_cls[(c, b)]):>7.1f}%" for b in (-10, -5, 0, 5, 10, 15)))
    print("\ndetection of gunshot/artillery events at each energy SNR (event onset hop or next hop flagged):")
    for c in sorted(IMPULSIVE):
        print(f"{c:<11}" + "".join(f"{100 * np.mean(det_cls[(c, b)]):>7.0f}%" for b in (-10, -5, 0, 5, 10, 15)))

    # 3) close-range weapon set by peak level
    d = ROOT / "data/generated/test_peak"
    rows = [json.loads(l) for l in open(d / "manifest.jsonl")]
    det = defaultdict(list)
    for m in rows:
        x = sf.read(d / m["file"], dtype="float64")[0]
        f = hop_flags(x[:, 0], x[:, 1])
        for s in m["impulse_samples"]:
            det[m["peak_target"]].append(bool(f[s // HOP:s // HOP + 2].any()))
    print("\nweapon events by peak level (dB SPL):  " + "  ".join(f"{int(k)}: {100 * np.mean(v):.0f}%" for k, v in sorted(det.items())))
    print("(the flag is an input feature to the network, not a hard gate; misses at low peaks are events that are\n"
          " quiet relative to the speech and the running envelope, which is also where they are least harmful)")


if __name__ == "__main__":
    main()
