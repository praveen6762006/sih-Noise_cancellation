"""Measure what the real headset does to the two-microphone signals, and compare with the simulation.

  python scripts/analyze_two_mic.py --speech speech.wav --noise noise.wav

speech.wav : the wearer talking, no other noise (2 channels: primary, reference)
noise.wav  : noise only, nobody talking (same mic positions)

Reports (1) speech leakage into the reference: primary-to-reference level difference during speech,
which is what `shadow_db` in configs/dataset.yaml has to reproduce; (2) the level ratio and gain mismatch
of the two mics on noise; (3) the noise coherence between the mics per frequency band, which limits what
the NLMS stage can do. Also tells you how to adjust the dataset config so training matches your headset.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
BANDS = [(100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 7500)]


def load(path):
    x, sr = sf.read(path, dtype="float64")
    assert sr == 16000 and x.ndim == 2 and x.shape[1] >= 2, "need a 16 kHz 2-channel WAV (primary, reference)"
    return x[:, 0] - x[:, 0].mean(), x[:, 1] - x[:, 1].mean()


def speech_leakage(p, r):
    """Frame-wise primary/reference level difference over the loudest speech frames."""
    n = 512
    fp, fr = [np.mean(v[:len(v) // n * n].reshape(-1, n) ** 2, axis=1) for v in (p, r)]
    active = (fp > np.percentile(fp, 60)) & (fp > 10 * np.percentile(fp, 10))
    d = 10 * np.log10((fp[active] + 1e-12) / (fr[active] + 1e-12))
    return d, active.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech", required=True)
    ap.add_argument("--noise", required=True)
    a = ap.parse_args()

    p, r = load(a.speech)
    d, frac = speech_leakage(p, r)
    med, lo, hi = np.median(d), np.percentile(d, 5), np.percentile(d, 95)
    print(f"== speech leakage into the reference (primary minus reference level) ==")
    print(f"   {len(d)} active frames ({100 * frac:.0f}% of the recording): median {med:.1f} dB, 5-95%: {lo:.1f} to {hi:.1f} dB")
    print(f"   simulation currently gives: median about 15.5 dB, 5-95%: 9 to 22 dB")
    delta = med - 15.5
    cfg = yaml.safe_load(open(ROOT / "configs" / "dataset.yaml"))
    s_lo, s_hi = cfg["shadow_db"]
    new = [float(max(0.0, round(s_lo + delta, 1))), float(round(max(s_hi + delta, s_lo + delta + 6), 1))]
    print(f"   -> shift needed: {delta:+.1f} dB. Suggested shadow_db: {new}  (now {[s_lo, s_hi]})")
    if s_lo + delta < 0:
        print("   NOTE: your headset leaks MORE speech into the reference than the simulated geometry (even with no shadow).")
        print("         Move the reference mic further from the mouth / point it away, or the noise canceller will self-cancel.")

    pn, rn = load(a.noise)
    f, Pp = signal.welch(pn, 16000, nperseg=1024)
    _, Pr = signal.welch(rn, 16000, nperseg=1024)
    _, C = signal.coherence(pn, rn, 16000, nperseg=1024)
    print("\n== noise only: level ratio and coherence between the microphones ==")
    print(f"   {'band (Hz)':>12} {'primary-ref (dB)':>18} {'coherence':>11}")
    for lo_f, hi_f in BANDS:
        m = (f >= lo_f) & (f < hi_f)
        print(f"   {lo_f:>5}-{hi_f:<6} {10 * np.log10(Pp[m].sum() / Pr[m].sum()):>18.1f} {C[m].mean():>11.2f}")
    tot = 10 * np.log10(np.mean(pn ** 2) / np.mean(rn ** 2))
    print(f"   broadband level difference on noise: {tot:+.1f} dB (should be near 0; a constant offset is mic gain mismatch, "
          f"dataset assumes +/-{cfg['ref_gain_jitter_db']} dB)")
    print("   coherence above ~0.5 in a band means a linear (NLMS) canceller can remove noise there; near 0 means it cannot.")
    print(f"   peak levels: primary {np.abs(pn).max():.3f}, reference {np.abs(rn).max():.3f}; full-scale samples "
          f"{int((np.abs(pn) >= 0.999).sum())} / {int((np.abs(rn) >= 0.999).sum())}")


if __name__ == "__main__":
    main()
