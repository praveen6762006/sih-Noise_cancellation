"""Export a trained GTCRN-X checkpoint to a streaming ONNX model and verify it.

  python scripts/export_onnx.py --ckpt checkpoints/C_full_s0_best.pt --out models/small_full.onnx

Checks (1) ONNX Runtime output == PyTorch streaming output frame by frame, and
(2) the full NumPy runtime (features + STFT + ONNX + overlap-add) == the offline PyTorch pipeline on a real
clip from the validation set, using the same framing (zero history at the start, fixed gain).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anc.features import HOP, N_FFT, extra_features, impulse_blocks, window   # noqa: E402
from anc.model.gtcrn_x import SIZES, GTCRNX                                    # noqa: E402
from anc.model.stream import build_stream_model                                # noqa: E402
from anc.runtime import StreamProcessor                                        # noqa: E402


def export(stream, out_path):
    spec = torch.randn(1, 257, 1, 2)
    extra = torch.rand(1, stream.n_extra, 1, 257)
    state = stream.init_state(1)
    torch.onnx.export(
        stream, (spec, extra, *state), str(out_path), dynamo=False, opset_version=13,
        input_names=["spec", "extra", "conv_cache", "tra_cache", "inter_cache"],
        output_names=["enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"])
    try:                                              # constant folding / fusion; optional
        import onnx
        from onnxsim import simplify
        m, ok = simplify(onnx.load(str(out_path)))
        if ok:
            onnx.save(m, str(out_path))
    except Exception as e:                            # noqa: BLE001
        print("  (onnxsim skipped:", type(e).__name__, ")")


def offline_reference(off, x_p, x_r, gain):
    """Offline pipeline with the streaming framing: frame t covers samples [256(t-1), 256(t+1))."""
    n = len(x_p)
    pad = lambda v: torch.cat([torch.zeros(HOP), v, torch.zeros(HOP)])
    w = window(torch.device("cpu"))
    xp, xr = pad(torch.from_numpy(x_p).float()), pad(torch.from_numpy(x_r).float())
    P = torch.view_as_real(torch.stft(xp * gain, N_FFT, HOP, N_FFT, w, center=False, return_complex=True))[None]
    R = torch.view_as_real(torch.stft(xr * gain, N_FFT, HOP, N_FFT, w, center=False, return_complex=True))[None]
    T = P.shape[2]
    blocks = torch.maximum(impulse_blocks(xp[None]), impulse_blocks(xr[None]))[0]     # raw samples, before gain
    per = N_FFT // 16
    flags = torch.stack([blocks[16 * (t - 1) + 16:16 * (t - 1) + 16 + per].max() if t >= 0 else torch.tensor(0.0)
                         for t in range(T)])[None]
    with torch.no_grad():
        Y = off(P, extra_features(P, R, flags))
        frames = torch.fft.irfft(torch.view_as_complex(Y.contiguous())[0], n=N_FFT, dim=0) * w[:, None]   # (512, T)
        y = torch.zeros((T + 1) * HOP)
        for t in range(T):                                    # plain overlap-add, as in the runtime
            y[t * HOP:t * HOP + N_FFT] += frames[:, t]
    return (y[HOP:HOP + n] / gain).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", default="data/generated/val/000200.wav")
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu")
    off = GTCRNX(SIZES[st["size"]], st["n_extra"]).eval()
    off.load_state_dict(st["model"])
    off.use_flag = st.get("use_flag", True)
    stream = build_stream_model(off)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    export(stream, a.out)
    print(f"exported {a.out}  ({Path(a.out).stat().st_size / 1024:.0f} KB)  size={st['size']} n_extra={st['n_extra']} use_flag={off.use_flag}")

    # (2) full runtime vs offline pipeline on a real clip
    x, _ = sf.read(ROOT / a.clip, dtype="float32")
    x_p, x_r = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    n = (len(x_p) // HOP) * HOP
    x_p, x_r = x_p[:n], x_r[:n]
    gain = 0.05 / np.sqrt(np.mean(x_p ** 2))
    rt = StreamProcessor(a.out, ch=SIZES[st["size"]], n_extra=st["n_extra"], use_flag=off.use_flag, fixed_gain=gain)
    outs, times = [], []
    for h in range(n // HOP + 1):                                   # one extra hop flushes the last output
        p = x_p[h * HOP:(h + 1) * HOP] if h < n // HOP else np.zeros(HOP)
        r = x_r[h * HOP:(h + 1) * HOP] if h < n // HOP else np.zeros(HOP)
        t0 = time.perf_counter()
        outs.append(rt.process(p, r))
        times.append(time.perf_counter() - t0)
    y_stream = np.concatenate(outs)[HOP:HOP + n]
    y_off = offline_reference(off, x_p, x_r, gain)
    inner = slice(2 * HOP, n - 2 * HOP)
    err = np.abs(y_stream[inner] - y_off[inner])
    rel = 10 * np.log10(np.sum(y_off[inner] ** 2) / np.sum((y_stream[inner] - y_off[inner]) ** 2))
    print(f"runtime vs offline pipeline: max abs diff {err.max():.2e} (signal peak {np.abs(y_off).max():.3f}), agreement {rel:.1f} dB")
    t = np.array(times[1:]) * 1000
    print(f"per-hop time on THIS machine, 1 thread: mean {t.mean():.2f} ms, p99 {np.percentile(t, 99):.2f} ms, max {t.max():.2f} ms "
          f"(hop = {HOP / 16:.0f} ms, so real-time factor {t.mean() / (HOP / 16):.3f})")


if __name__ == "__main__":
    main()
