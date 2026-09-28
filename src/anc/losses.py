"""Training losses: SI-SNR, power-law-compressed complex spectral loss (magnitude + complex),
multi-resolution STFT loss (log-magnitude L1 + spectral convergence), with optional extra weight
on frames flagged as impulsive."""
import torch

COMPRESS = 0.3


def si_snr(est, ref, eps=1e-8):
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    t = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps) * ref
    return 10 * torch.log10(t.pow(2).sum(-1) / ((est - t).pow(2).sum(-1) + eps) + eps)


def spectral_loss(est, ref, frame_w=None, c=COMPRESS):
    """est, ref: (B, F, T, 2). Compressed magnitude MSE + compressed complex MSE."""
    em = torch.sqrt(est.pow(2).sum(-1) + 1e-12)
    rm = torch.sqrt(ref.pow(2).sum(-1) + 1e-12)
    mag = (em ** c - rm ** c).pow(2)
    ec = est * (em ** (c - 1))[..., None]
    rc = ref * (rm ** (c - 1))[..., None]
    cpx = (ec - rc).pow(2).sum(-1)
    per_frame = (mag + cpx).mean(1)                       # (B, T)
    if frame_w is not None:
        per_frame = per_frame * frame_w
    return per_frame.mean()


def mr_stft_loss(est, ref, sizes=((256, 64), (512, 128), (1024, 256))):
    total = 0.0
    for n_fft, hop in sizes:
        w = torch.hann_window(n_fft, device=est.device)
        e = torch.stft(est, n_fft, hop, n_fft, w, return_complex=True).abs() + 1e-7
        r = torch.stft(ref, n_fft, hop, n_fft, w, return_complex=True).abs() + 1e-7
        sc = torch.linalg.norm(r - e, dim=(-2, -1)) / torch.linalg.norm(r, dim=(-2, -1))
        total = total + sc.mean() + (torch.log(e) - torch.log(r)).abs().mean()
    return total / len(sizes)


def total_loss(est_spec, est_wav, clean_spec, clean_wav, flags=None, impulse_weight=2.0,
               w_sisnr=0.05, w_spec=30.0, w_mr=0.5):
    fw = None if flags is None else 1.0 + impulse_weight * flags
    l_si = -si_snr(est_wav, clean_wav).mean()
    l_sp = spectral_loss(est_spec, clean_spec, fw)
    l_mr = mr_stft_loss(est_wav, clean_wav)
    loss = w_sisnr * l_si + w_spec * l_sp + w_mr * l_mr
    return loss, dict(si=float(l_si), spec=float(l_sp), mr=float(l_mr))
