"""GTCRN-X: GTCRN (Rong et al., ICASSP 2024, MIT licence, third_party/gtcrn) with
  * configurable width, giving the Tiny / Small / Base family, and
  * extra input feature channels (reference mic, level ratio, coherence, impulse flag).

Small (ch=16, n_extra=0) is architecturally identical to the original GTCRN, so the pretrained
checkpoint loads directly; with n_extra>0 the new input-conv weights start at zero, so the model
initially behaves exactly like single-mic GTCRN and learns to use the extra cues.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party" / "gtcrn"))
import gtcrn as _g  # noqa: E402

SIZES = {"tiny": 8, "small": 16, "base": 32}


class Encoder(nn.Module):
    def __init__(self, in_ch, ch):
        super().__init__()
        self.en_convs = nn.ModuleList([
            _g.ConvBlock(in_ch, ch, (1, 5), stride=(1, 2), padding=(0, 2)),
            _g.ConvBlock(ch, ch, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
            _g.GTConvBlock(ch, ch, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
            _g.GTConvBlock(ch, ch, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
            _g.GTConvBlock(ch, ch, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
        ])

    def forward(self, x):
        outs = []
        for m in self.en_convs:
            x = m(x)
            outs.append(x)
        return x, outs


class Decoder(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.de_convs = nn.ModuleList([
            _g.GTConvBlock(ch, ch, (3, 3), stride=(1, 1), padding=(2 * 5, 1), dilation=(5, 1), use_deconv=True),
            _g.GTConvBlock(ch, ch, (3, 3), stride=(1, 1), padding=(2 * 2, 1), dilation=(2, 1), use_deconv=True),
            _g.GTConvBlock(ch, ch, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True),
            _g.ConvBlock(ch, ch, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True),
            _g.ConvBlock(ch, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True),
        ])

    def forward(self, x, en_outs):
        n = len(self.de_convs)
        for i, m in enumerate(self.de_convs):
            x = m(x + en_outs[n - 1 - i])
        return x


class GTCRNX(nn.Module):
    def __init__(self, ch=16, n_extra=4):
        super().__init__()
        self.n_extra = n_extra
        self.use_flag = True       # False: zero the impulse-flag channel (ablation)
        self.erb = _g.ERB(65, 64)
        self.sfe = _g.SFE(3, 1)
        self.encoder = Encoder((3 + n_extra) * 3, ch)
        self.dpgrnn1 = _g.DPGRNN(ch, 33, ch)
        self.dpgrnn2 = _g.DPGRNN(ch, 33, ch)
        self.decoder = Decoder(ch)
        self.mask = _g.Mask()

    def forward(self, spec, extra=None):
        """spec: (B, F, T, 2) primary STFT; extra: (B, n_extra, T, F) or None. Returns (B, F, T, 2)."""
        real, imag = spec[..., 0].permute(0, 2, 1), spec[..., 1].permute(0, 2, 1)
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-12)
        feat = torch.stack([mag, real, imag], dim=1)                     # (B, 3, T, F)
        if self.n_extra:
            if not self.use_flag:
                extra = torch.cat([extra[:, :3], torch.zeros_like(extra[:, 3:])], dim=1)
            feat = torch.cat([feat, extra], dim=1)
        feat = self.sfe(self.erb.bm(feat))
        feat, en_outs = self.encoder(feat)
        feat = self.dpgrnn2(self.dpgrnn1(feat))
        m = self.erb.bs(self.decoder(feat, en_outs))
        out = self.mask(m, spec.permute(0, 3, 2, 1))                     # (B, 2, T, F)
        return out.permute(0, 3, 2, 1)


def count_params(model):
    """Trainable parameters (the fixed ERB filter banks are stored as non-trainable parameters)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_pretrained(model, ckpt_path, device="cpu"):
    """Load original GTCRN weights into a ch=16 GTCRN-X, zero-filling the extra input channels."""
    sd = torch.load(ckpt_path, map_location=device)["model"]
    key = "encoder.en_convs.0.conv.weight"
    if model.n_extra:
        old = sd[key]                                   # (16, 9, 1, 5)
        new = torch.zeros(old.shape[0], (3 + model.n_extra) * 3, *old.shape[2:])
        new[:, :old.shape[1]] = old                     # SFE keeps the 3 primary channels first
        sd[key] = new
    model.load_state_dict(sd)
    return model
