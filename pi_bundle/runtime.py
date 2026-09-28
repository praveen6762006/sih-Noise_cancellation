"""Frame-by-frame inference runtime (NumPy + ONNX Runtime only, so it runs on a Raspberry Pi).

    rt = StreamProcessor("models/small_full.onnx", ch=16)
    out_hop = rt.process(primary_hop, reference_hop)      # 256 samples in -> 256 samples out

The output is delayed by one hop (256 samples = 16 ms): a frame covers the previous and the current hop,
and the previous hop is complete only once this frame has been overlap-added. With the 32 ms analysis
window this is the algorithmic latency of the model path.

The feature code mirrors anc.features (training) one frame at a time: smoothed power ratio, coherence,
impulse/clipping flag and level normalisation.
"""
import numpy as np
from scipy.signal import get_window

try:
    from anc.refguard import RefGuard
except ImportError:                              # flat Pi bundle
    from refguard import RefGuard

N_FFT, HOP, BINS = 512, 256, 257
BLOCK = 16                       # detector block = 1 ms
TARGET_RMS = 0.05
ALPHA = 0.9                      # power smoothing, as in training


class ImpulseDetector:
    """Per 1 ms block: energy jump over the previous 64 blocks, or samples at full scale
    (numpy mirror of anc.features.impulse_blocks). Vectorised over the 16 blocks of a hop."""

    def __init__(self, jump_ratio=30.0, min_energy=2e-3):
        self.jump_ratio, self.min_energy = jump_ratio, min_energy
        self.hist = np.zeros(64)                      # energies of the previous 64 blocks

    def hop(self, x):
        seg = np.asarray(x).reshape(-1, BLOCK)
        e = (seg ** 2).mean(-1)
        clip = np.abs(seg).max(-1) >= 0.999
        ext = np.concatenate([self.hist, e])
        cs = np.concatenate([[0.0], np.cumsum(ext)])
        slow = (cs[64:64 + len(e)] - cs[:len(e)]) / 64      # mean of the 64 blocks before each block
        self.hist = ext[-64:]
        return clip | ((e > self.jump_ratio * (slow + 1e-9)) & (e > self.min_energy))


class StreamProcessor:
    def __init__(self, onnx_path, ch=16, n_extra=4, use_flag=True, threads=1, fixed_gain=None, guard=True):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3                      # silence harmless shape-merge warnings
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.ch, self.n_extra, self.use_flag, self.fixed_gain = ch, n_extra, use_flag, fixed_gain
        self.use_guard = guard
        self.win = get_window("hann", N_FFT, fftbins=True) ** 0.5        # == torch.hann_window(512).pow(0.5)
        self.reset()

    def reset(self):
        self.conv = np.zeros((2, 1, self.ch, 16, 33), np.float32)
        self.tra = np.zeros((2, 3, 1, 1, self.ch), np.float32)
        self.inter = np.zeros((2, 1, 33, self.ch), np.float32)
        self.prev_p, self.prev_r, self.ola = np.zeros(HOP), np.zeros(HOP), np.zeros(HOP)
        self.ppp, self.prr, self.cross = np.zeros(BINS), np.zeros(BINS), np.zeros(BINS, complex)
        self.det_p, self.det_r = ImpulseDetector(), ImpulseDetector()
        self.prev_flags = np.zeros(HOP // BLOCK, dtype=bool)
        self.power = None
        self.last_flag = 0.0
        self.guard = RefGuard()

    def _gain(self, p_hop):
        """Slow level normalisation (time constant ~1.5 s). Training normalises per 4 s clip instead."""
        if self.fixed_gain is not None:
            return self.fixed_gain
        e = float(np.mean(p_hop ** 2)) + 1e-12
        self.power = e if self.power is None else 0.99 * self.power + 0.01 * e
        return float(np.clip(TARGET_RMS / np.sqrt(self.power), 0.05, 2000.0))

    def features(self, P, R, flag):
        """Extra input channels for one frame. P, R: complex spectra (already level-normalised)."""
        self.ppp = ALPHA * self.ppp + (1 - ALPHA) * np.abs(P) ** 2
        self.prr = ALPHA * self.prr + (1 - ALPHA) * np.abs(R) ** 2
        self.cross = ALPHA * self.cross + (1 - ALPHA) * (P * np.conj(R))
        ratio = np.clip(0.5 * np.log10((self.ppp + 1e-8) / (self.prr + 1e-8)), -2, 2)
        coh = np.clip(np.abs(self.cross) ** 2 / (self.ppp * self.prr + 1e-12), 0, 1)
        flag_ch = np.full(BINS, flag if self.use_flag else 0.0)
        return np.stack([np.abs(R), ratio, coh, flag_ch])[None, :, None, :].astype(np.float32)

    def process(self, p_hop, r_hop):
        p_hop, r_hop = np.asarray(p_hop, np.float64), np.asarray(r_hop, np.float64)
        if self.use_guard:                       # reference-mic health check (dead/copied reference -> silence)
            r_hop = self.guard.process(p_hop, r_hop)
        cur = self.det_p.hop(p_hop) | self.det_r.hop(r_hop)
        flag = float(cur.any() or self.prev_flags.any())      # frame = previous hop + current hop
        self.prev_flags = cur
        g = self._gain(p_hop)

        P = np.fft.rfft(np.concatenate([self.prev_p, p_hop]) * g * self.win)
        R = np.fft.rfft(np.concatenate([self.prev_r, r_hop]) * g * self.win)
        self.prev_p, self.prev_r = p_hop, r_hop

        spec = np.stack([P.real, P.imag], axis=-1)[None, :, None, :].astype(np.float32)   # (1,257,1,2)
        out, self.conv, self.tra, self.inter = self.sess.run(None, {
            "spec": spec, "extra": self.features(P, R, flag),
            "conv_cache": self.conv, "tra_cache": self.tra, "inter_cache": self.inter})
        Y = out[0, :, 0, 0] + 1j * out[0, :, 0, 1]
        frame = np.fft.irfft(Y, N_FFT) * self.win / g
        hop_out = self.ola + frame[:HOP]
        self.ola = frame[HOP:]
        return hop_out                                        # enhanced version of the PREVIOUS input hop
