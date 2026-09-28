"""GPU feature extraction shared by training, evaluation and (mirrored in numpy) the Pi runtime.

Input channels per STFT bin, beyond the primary mic's magnitude/real/imag that GTCRN already uses:
  ref_mag    magnitude of the reference mic
  ratio      0.5*log10 of smoothed primary/reference power  (close-talk level cue; >0 when the
             primary is louder, i.e. during speech)
  coherence  smoothed magnitude-squared coherence between the two mics
  flag       impulse/clipping flag for the frame (same value in every bin)
"""
import torch
import torch.nn.functional as F

N_FFT, HOP = 512, 256
N_EXTRA = 4
TARGET_RMS = 0.05        # input level the network is trained at; a slow AGC does this on the device
SMOOTH_ALPHA, SMOOTH_K = 0.9, 40
BLOCK = 16               # 1 ms detector blocks


def window(dev):
    return torch.hann_window(N_FFT, device=dev).pow(0.5)


def stft(x):
    """(B, N) -> (B, F, T, 2)"""
    return torch.view_as_real(torch.stft(x, N_FFT, HOP, N_FFT, window(x.device), return_complex=True))


def istft(spec, length=None):
    return torch.istft(torch.view_as_complex(spec.contiguous()), N_FFT, HOP, N_FFT, window(spec.device), length=length)


def _smooth(x, alpha=SMOOTH_ALPHA, k=SMOOTH_K):
    """Causal exponential smoothing along the last (time) axis. x: (B, F, T)."""
    b, f, t = x.shape
    w = (1 - alpha) * alpha ** torch.arange(k - 1, -1, -1, device=x.device, dtype=x.dtype)
    y = F.conv1d(F.pad(x.reshape(b * f, 1, t), (k - 1, 0)), w.view(1, 1, k))
    return y.reshape(b, f, t)


def impulse_blocks(x, jump_ratio=30.0, min_energy=2e-3):
    """Per 1 ms block: energy jump over the previous ~64 ms, or samples at full scale. x: (B, N) raw."""
    b, n = x.shape
    nb = n // BLOCK
    seg = x[:, :nb * BLOCK].reshape(b, nb, BLOCK)
    e = seg.pow(2).mean(-1)
    kernel = torch.ones(1, 1, 64, device=x.device) / 64
    slow = F.conv1d(F.pad(e[:, None], (64, 0)), kernel)[:, 0, :nb]     # mean of the 64 previous blocks
    jump = (e > jump_ratio * (slow + 1e-9)) & (e > min_energy)
    clip = seg.abs().amax(-1) >= 0.999
    return (jump | clip).float()


def impulse_frame_flags(noisy, ref):
    """(B, T) flag per STFT frame: 1 if either channel has an impulse or clipping in the frame."""
    blocks = torch.maximum(impulse_blocks(noisy), impulse_blocks(ref))
    per = N_FFT // BLOCK
    fr = F.max_pool1d(blocks[:, None], per, HOP // BLOCK, padding=per // 2)[:, 0]
    t = noisy.shape[-1] // HOP + 1
    return fr[:, :t] if fr.shape[-1] >= t else F.pad(fr, (0, t - fr.shape[-1]))


def extra_features(p, r, flags):
    """p, r: (B, F, T, 2) primary/reference spectra; flags: (B, T). Returns (B, N_EXTRA, T, F)."""
    pc, rc = torch.view_as_complex(p.contiguous()), torch.view_as_complex(r.contiguous())
    ppp, prr = _smooth(pc.abs() ** 2), _smooth(rc.abs() ** 2)
    cross = pc * rc.conj()
    cr, ci = _smooth(cross.real), _smooth(cross.imag)
    ratio = (0.5 * torch.log10((ppp + 1e-8) / (prr + 1e-8))).clamp(-2, 2)
    coh = ((cr ** 2 + ci ** 2) / (ppp * prr + 1e-12)).clamp(0, 1)
    ref_mag = rc.abs()
    flag = flags[:, None, :].expand(-1, p.shape[1], -1)
    feats = torch.stack([ref_mag, ratio, coh, flag], dim=1)             # (B, 4, F, T)
    return feats.permute(0, 1, 3, 2).contiguous()                        # (B, 4, T, F)


def prepare_batch(noisy, ref, clean=None):
    """Raw waveforms (B, N) -> network inputs. Flags are computed on the raw (un-normalised) samples,
    then the level is normalised so absolute gain does not matter to the network."""
    flags = impulse_frame_flags(noisy, ref)
    g = TARGET_RMS / (noisy.pow(2).mean(-1, keepdim=True).sqrt() + 1e-9)
    noisy, ref = noisy * g, ref * g
    p, r = stft(noisy), stft(ref)
    out = dict(spec=p, extra=extra_features(p, r, flags), flags=flags, gain=g, noisy=noisy)
    if clean is not None:
        out["clean"] = clean * g
        out["clean_spec"] = stft(out["clean"])
    return out
