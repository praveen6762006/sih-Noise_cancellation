"""Streaming (one frame per call) version of GTCRN-X, for ONNX export and the Pi runtime.

Same structure as third_party/gtcrn/stream/gtcrn_stream.py (MIT licence, Rong et al.), generalised to
variable width and extra input channels, with the caches returned as fresh tensors instead of being
updated in place. State tensors, for width `ch`:

    conv_cache  (2, B, ch, 16, 33)   temporal history of the dilated convolutions [encoder, decoder]
    tra_cache   (2, 3, 1, B, ch)     GRU state of the temporal-recurrent-attention blocks
    inter_cache (2, 1, B*33, ch)     GRU state of the two inter-frame RNNs

Weights are converted from an offline GTCRNX with `build_stream_model`.
"""
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from . import gtcrn_x  # noqa: F401  (puts third_party/gtcrn on sys.path)
import gtcrn as _g

_MOD = Path(__file__).resolve().parents[3] / "third_party" / "gtcrn" / "stream" / "modules"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_gtcrn_stream_{name}", _MOD / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_conv, _cvt = _load("convolution"), _load("convert")
StreamConv2d, StreamConvTranspose2d, convert_to_stream = _conv.StreamConv2d, _conv.StreamConvTranspose2d, _cvt.convert_to_stream


class StreamTRA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x, h_cache):
        zt = torch.mean(x.pow(2), dim=-1)
        at, h_cache = self.att_gru(zt.transpose(1, 2), h_cache)
        at = self.att_act(self.att_fc(at).transpose(1, 2))
        return x * at[..., None], h_cache


class StreamGTConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        conv = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        sconv = StreamConvTranspose2d if use_deconv else StreamConv2d
        self.sfe = _g.SFE(kernel_size=3, stride=1)
        self.point_conv1 = conv(in_channels // 2 * 3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()
        self.depth_conv = sconv(hidden_channels, hidden_channels, kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()
        self.point_conv2 = conv(hidden_channels, in_channels // 2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels // 2)
        self.tra = StreamTRA(in_channels // 2)

    def forward(self, x, conv_cache, tra_cache):
        x1, x2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]
        h = self.point_act(self.point_bn1(self.point_conv1(self.sfe(x1))))
        h, conv_cache = self.depth_conv(h, conv_cache)
        h = self.point_bn2(self.point_conv2(self.depth_act(self.depth_bn(h))))
        h, tra_cache = self.tra(h, tra_cache)
        out = torch.stack([h, x2], dim=1).transpose(1, 2).contiguous()      # channel shuffle
        return out.view(out.shape[0], -1, out.shape[3], out.shape[4]), conv_cache, tra_cache


class StreamDPGRNN(nn.Module):
    def __init__(self, input_size, width, hidden_size):
        super().__init__()
        self.width, self.hidden_size = width, hidden_size
        self.intra_rnn = _g.GRNN(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
        self.inter_rnn = _g.GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x, inter_cache):
        x = x.permute(0, 2, 3, 1)                                            # (B,T,F,C), T == 1
        intra = self.intra_rnn(x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))[0]
        intra = self.intra_ln(self.intra_fc(intra).reshape(x.shape[0], -1, self.width, self.hidden_size))
        intra_out = x + intra
        x = intra_out.permute(0, 2, 1, 3)                                    # (B,F,T,C)
        inter, inter_cache = self.inter_rnn(x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]), inter_cache)
        inter = self.inter_ln(self.inter_fc(inter).reshape(x.shape[0], self.width, -1, self.hidden_size).permute(0, 2, 1, 3))
        return (intra_out + inter).permute(0, 3, 1, 2), inter_cache


class StreamEncoder(nn.Module):
    def __init__(self, in_ch, ch):
        super().__init__()
        self.en_convs = nn.ModuleList([
            _g.ConvBlock(in_ch, ch, (1, 5), stride=(1, 2), padding=(0, 2)),
            _g.ConvBlock(ch, ch, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
            StreamGTConvBlock(ch, ch, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
            StreamGTConvBlock(ch, ch, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
            StreamGTConvBlock(ch, ch, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
        ])

    def forward(self, x, conv_cache, tra_cache):
        outs = []
        for i in range(2):
            x = self.en_convs[i](x)
            outs.append(x)
        x, c0, t0 = self.en_convs[2](x, conv_cache[:, :, :2], tra_cache[0]); outs.append(x)
        x, c1, t1 = self.en_convs[3](x, conv_cache[:, :, 2:6], tra_cache[1]); outs.append(x)
        x, c2, t2 = self.en_convs[4](x, conv_cache[:, :, 6:16], tra_cache[2]); outs.append(x)
        return x, outs, torch.cat([c0, c1, c2], dim=2), torch.stack([t0, t1, t2], dim=0)


class StreamDecoder(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.de_convs = nn.ModuleList([
            StreamGTConvBlock(ch, ch, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1), use_deconv=True),
            StreamGTConvBlock(ch, ch, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), use_deconv=True),
            StreamGTConvBlock(ch, ch, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=True),
            _g.ConvBlock(ch, ch, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True),
            _g.ConvBlock(ch, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True),
        ])

    def forward(self, x, en_outs, conv_cache, tra_cache):
        x, c2, t2 = self.de_convs[0](x + en_outs[4], conv_cache[:, :, 6:16], tra_cache[0])
        x, c1, t1 = self.de_convs[1](x + en_outs[3], conv_cache[:, :, 2:6], tra_cache[1])
        x, c0, t0 = self.de_convs[2](x + en_outs[2], conv_cache[:, :, :2], tra_cache[2])
        for i in (3, 4):
            x = self.de_convs[i](x + en_outs[4 - i])
        return x, torch.cat([c0, c1, c2], dim=2), torch.stack([t2, t1, t0], dim=0)   # index i <- de_convs[i]


class StreamGTCRNX(nn.Module):
    def __init__(self, ch=16, n_extra=4):
        super().__init__()
        self.ch, self.n_extra = ch, n_extra
        self.erb = _g.ERB(65, 64)
        self.sfe = _g.SFE(3, 1)
        self.encoder = StreamEncoder((3 + n_extra) * 3, ch)
        self.dpgrnn1 = StreamDPGRNN(ch, 33, ch)
        self.dpgrnn2 = StreamDPGRNN(ch, 33, ch)
        self.decoder = StreamDecoder(ch)
        self.mask = _g.Mask()

    def init_state(self, batch=1):
        ch = self.ch
        return (torch.zeros(2, batch, ch, 16, 33), torch.zeros(2, 3, 1, batch, ch), torch.zeros(2, 1, batch * 33, ch))

    def forward(self, spec, extra, conv_cache, tra_cache, inter_cache):
        """spec (B,257,1,2), extra (B,n_extra,1,257) -> enhanced spec (B,257,1,2) and the new caches."""
        real, imag = spec[..., 0].permute(0, 2, 1), spec[..., 1].permute(0, 2, 1)
        feat = torch.stack([torch.sqrt(real ** 2 + imag ** 2 + 1e-12), real, imag], dim=1)
        if self.n_extra:
            feat = torch.cat([feat, extra], dim=1)
        feat = self.sfe(self.erb.bm(feat))
        feat, en_outs, c_en, t_en = self.encoder(feat, conv_cache[0], tra_cache[0])
        feat, i1 = self.dpgrnn1(feat, inter_cache[0])
        feat, i2 = self.dpgrnn2(feat, inter_cache[1])
        m, c_de, t_de = self.decoder(feat, en_outs, conv_cache[1], tra_cache[1])
        out = self.mask(self.erb.bs(m), spec.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return out, torch.stack([c_en, c_de]), torch.stack([t_en, t_de]), torch.stack([i1, i2])


def build_stream_model(offline):
    """Stream model with the weights of a trained offline GTCRNX. (Flag ablation: bake it into the input.)"""
    model = StreamGTCRNX(ch=offline.encoder.en_convs[0].conv.out_channels, n_extra=offline.n_extra).eval()
    convert_to_stream(model, offline)
    model.use_flag = getattr(offline, "use_flag", True)
    return model
