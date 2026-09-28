"""Reference-microphone guard: gain calibration and health check, hop by hop.

Motivation (results/REPORT.md, section 9): the two-microphone model is fragile if the reference mic is dead, wired as a
copy of the primary, or badly mismatched in gain. Two protections are active, none needing the model (gain calibration exists but is OFF by default: on simulated data
it was biased by +4.8 dB because speech tails raise the primary's noise floor; gain mismatch is handled by training instead):

  1. Gain calibration. Noise reaches both mics at about the same level, so the quiet-period floors of the two channels
     should match. The reference is scaled so its noise floor (minimum smoothed power over ~3 s) equals the primary's.
     Correction is limited to +/-12 dB and adapts slowly. (Training already covers +/-2 dB of mismatch.)
  2. Copy detection. A real reference is almost uncorrelated with the primary above ~1.5 kHz (different positions,
     diffuse noise, head shadow). If the two are strongly correlated up there the reference is a duplicate of the
     primary or shorted to it, so it is replaced by silence (the model then falls back to single-mic behaviour,
     which needs a model trained with reference dropout).
  3. Dead reference: an all-zero / near-silent reference is already the silence the model was trained to handle.

Everything is causal and cheap (one small filter and a few running sums per hop).
"""
import numpy as np
from scipy import signal

HOP = 256
FS = 16000
FLOOR_HOPS = 188                 # about 3 s of hops for the noise-floor minimum
_HP = signal.butter(2, 1500.0, btype="highpass", fs=FS, output="sos")


class RefGuard:
    def __init__(self, max_gain_db=12.0, copy_thr=0.9, ok_thr=0.7, warmup_hops=24, calibrate=False, smooth=0.95):
        self.a = smooth                  # ~20-hop memory (0.3 s): copy fault caught in ~0.4 s, 0.02% false alarms
        self.calibrate = calibrate       # off by default: biased when speech tails lift the primary's noise floor
        self.max_g = 10 ** (max_gain_db / 20)
        self.copy_thr, self.ok_thr, self.warmup = copy_thr, ok_thr, warmup_hops
        self.zi_p = np.zeros((_HP.shape[0], 2))
        self.zi_r = np.zeros((_HP.shape[0], 2))
        self.sp = self.sr = None                 # smoothed powers
        self.hist_p, self.hist_r = [], []
        self.gain = 1.0
        self.cpp = self.crr = self.cpr = 1e-12   # running sums of the high-passed signals
        self.invalid = False
        self.n = 0

    def process(self, p, r):
        """p, r: one hop each (float). Returns the reference hop to feed the model."""
        self.n += 1
        ep, er = float(np.mean(p ** 2)) + 1e-14, float(np.mean(r ** 2)) + 1e-14
        self.sp = ep if self.sp is None else 0.9 * self.sp + 0.1 * ep
        self.sr = er if self.sr is None else 0.9 * self.sr + 0.1 * er
        self.hist_p.append(self.sp)
        self.hist_r.append(self.sr)
        if len(self.hist_p) > FLOOR_HOPS:
            self.hist_p.pop(0)
            self.hist_r.pop(0)

        # 1. gain calibration from the noise floors (after a warm-up), slow adaptation
        if self.calibrate and self.n > self.warmup and min(self.hist_r) > 1e-12:
            target = np.clip(np.sqrt(min(self.hist_p) / min(self.hist_r)), 1 / self.max_g, self.max_g)
            self.gain = 0.995 * self.gain + 0.005 * target

        # 2. copy detection on the high-passed signals (~1 s memory)
        hp, self.zi_p = signal.sosfilt(_HP, p, zi=self.zi_p)
        hr, self.zi_r = signal.sosfilt(_HP, r, zi=self.zi_r)
        a = self.a
        self.cpp = a * self.cpp + (1 - a) * float(np.mean(hp ** 2))
        self.crr = a * self.crr + (1 - a) * float(np.mean(hr ** 2))
        self.cpr = a * self.cpr + (1 - a) * float(np.mean(hp * hr))
        corr = abs(self.cpr) / np.sqrt(self.cpp * self.crr + 1e-24)
        if self.n > self.warmup:
            if corr > self.copy_thr:
                self.invalid = True
            elif corr < self.ok_thr:
                self.invalid = False
        return np.zeros_like(r) if self.invalid else r * self.gain


def guard_clip(primary, ref):
    """Offline helper: apply the guard hop by hop to a whole clip; returns the corrected reference."""
    g = RefGuard()
    n = (len(primary) // HOP) * HOP
    out = np.array(ref, dtype=np.float64)
    for h in range(n // HOP):
        s = slice(h * HOP, (h + 1) * HOP)
        out[s] = g.process(primary[s], ref[s])
    return out
