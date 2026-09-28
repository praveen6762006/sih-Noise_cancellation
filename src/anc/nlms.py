"""Reference-mic adaptive noise canceller (block NLMS), streaming-capable.

Design choices (from the architecture review):
  * band-limited to 50-1200 Hz: above roughly c/(2d) the noise is incoherent between two mics
    a hand-span apart, so a linear canceller cannot help there and could only add distortion;
  * primary delayed by DELAY samples so the filter can also model acausal reference->primary paths;
  * leaky update, error clipping (robust to impulses), and a norm cap on the weights;
  * adaptation frozen when (a) the primary/reference level ratio says the wearer is talking
    (close-talk cue), or (b) an impulse / clipping is detected. The filter keeps *applying* its
    weights while frozen.

One BlockNLMS instance processes an arbitrarily long stream block by block (BLOCK samples).
"""
import numpy as np
from scipy import signal

FS = 16000
BLOCK = 64
TAPS = 128
DELAY = 16
BAND = (50.0, 1200.0)


def _bandpass():
    return signal.butter(2, BAND, btype="bandpass", fs=FS, output="sos")


class BlockNLMS:
    def __init__(self, taps=TAPS, mu=0.3, leak=1e-4, clip_k=3.0, w_max=2.0, delay=DELAY,
                 speech_db=6.0, imp_hold=4, gate=True):
        self.taps, self.mu, self.leak, self.clip_k, self.w_max = taps, mu, leak, clip_k, w_max
        self.delay, self.speech_db, self.imp_hold, self.gate = delay, speech_db, imp_hold, gate
        self.w = np.zeros(taps)
        self.sos = _bandpass()
        self.zi_d = np.zeros((self.sos.shape[0], 2))
        self.zi_x = np.zeros((self.sos.shape[0], 2))
        self.x_hist = np.zeros(taps - 1 + BLOCK)          # band-limited reference history
        self.d_line = np.zeros(delay)                     # delay line for the full-band primary
        self.p_p = self.p_r = 1e-8
        self.err_rms = 1e-3
        self.hold = 0
        self.frozen_frac = 0.0

    def process_block(self, d, x, impulse=False):
        """d: primary block (BLOCK,), x: reference block (BLOCK,), impulse: bool from the detector.
        Returns the primary with the predicted noise removed (delayed by `delay` samples)."""
        # delay the primary first, so the adaptation target and the output are the same signal
        d_full = np.concatenate([self.d_line, d])
        d_del, self.d_line = d_full[:BLOCK], d_full[BLOCK:]
        d_bp, self.zi_d = signal.sosfilt(self.sos, d_del, zi=self.zi_d)
        x_bp, self.zi_x = signal.sosfilt(self.sos, x, zi=self.zi_x)

        # close-talk level cue on the band-limited signals (smoothed over ~10 blocks)
        self.p_p = 0.9 * self.p_p + 0.1 * (np.mean(d_bp ** 2) + 1e-10)
        self.p_r = 0.9 * self.p_r + 0.1 * (np.mean(x_bp ** 2) + 1e-10)
        ratio_db = 10 * np.log10(self.p_p / self.p_r)

        self.x_hist = np.concatenate([self.x_hist[BLOCK:], x_bp])
        # X[i, k] = x_bp[n_i - k]
        idx = np.arange(BLOCK)[:, None] + (self.taps - 1 - np.arange(self.taps))[None, :]
        X = self.x_hist[idx]
        y = X @ self.w                                     # predicted noise in the band
        # Adapt against the UNFILTERED primary: the output subtracts y from it, so y has to match it.
        # Out-of-band parts of d are uncorrelated with the band-limited reference, so they only add
        # gradient noise and do not bias the solution. d_bp is used for the level gate above only.
        e_ad = d_del - y

        if impulse:
            self.hold = self.imp_hold
        frozen = self.gate and (ratio_db > self.speech_db or self.hold > 0)
        if self.hold > 0:
            self.hold -= 1
        if not frozen:
            lim = self.clip_k * self.err_rms
            e_c = np.clip(e_ad, -lim, lim)
            self.err_rms = 0.95 * self.err_rms + 0.05 * (np.sqrt(np.mean(e_c ** 2)) + 1e-9)
            grad = X.T @ e_c / BLOCK                        # block-averaged gradient
            power = np.mean(np.sum(X ** 2, axis=1)) + 1e-9  # mean ||x_vec||^2 over the block (NLMS normaliser)
            self.w = (1 - self.leak) * self.w + self.mu * grad / power
            wn = np.linalg.norm(self.w)
            if wn > self.w_max:
                self.w *= self.w_max / wn
        self.frozen_frac = 0.99 * self.frozen_frac + 0.01 * float(frozen)

        return d_del - y        # predicted noise removed from the (delayed) full-band primary


def impulse_blocks_np(x, jump_ratio=30.0, min_energy=2e-3):
    """Numpy mirror of anc.features.impulse_blocks (1 ms blocks): energy jump or full-scale samples."""
    nb = len(x) // 16
    seg = x[:nb * 16].reshape(nb, 16)
    e = (seg ** 2).mean(-1)
    slow = np.convolve(np.concatenate([np.zeros(64), e]), np.ones(64) / 64, mode="valid")[:nb]
    return ((e > jump_ratio * (slow + 1e-9)) & (e > min_energy)) | (np.abs(seg).max(-1) >= 0.999)


def run_nlms(primary, ref, **kw):
    """Offline helper: apply BlockNLMS to whole arrays. Output has the same length; the DELAY
    samples of algorithmic latency are compensated so it stays time-aligned with the input."""
    n = len(primary)
    nb = int(np.ceil((n + DELAY) / BLOCK))
    pad = nb * BLOCK - n
    d = np.concatenate([primary, np.zeros(pad)])
    x = np.concatenate([ref, np.zeros(pad)])
    imp16 = impulse_blocks_np(primary) | impulse_blocks_np(ref)
    f = BlockNLMS(**kw)
    out = np.empty(nb * BLOCK)
    for b in range(nb):
        s = slice(b * BLOCK, (b + 1) * BLOCK)
        flag = bool(imp16[b * 4:(b + 1) * 4].any()) if (b + 1) * 4 <= len(imp16) else False
        out[s] = f.process_block(d[s], x[s], flag)
    return out[DELAY:DELAY + n], f
