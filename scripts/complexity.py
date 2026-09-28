"""Model complexity for Tiny / Small / Base.  Every number is labelled MEASURED, COMPUTED or ESTIMATED.

  python scripts/complexity.py static                     # parameters, file size, MACs (no load sensitivity)
  python scripts/complexity.py timing                     # per-hop time and peak RAM (run on an IDLE machine)

static
  parameters      MEASURED  (trainable parameters of the model; the fixed ERB filter banks are not trained and not counted)
  file size       MEASURED  (size of the exported .onnx on disk)
  MACs per hop    COMPUTED  by forward hooks on every Conv2d / Linear / GRU of the STREAMING model for one frame, i.e. exactly
                            the operations the exported graph executes (the streaming transposed convolutions run as ordinary
                            convolutions on zero-stuffed input, so they cost more than a textbook transposed conv; both are shown).
                            The total is cross-checked against torch.utils.flop_counter (an independent counter that also counts the GRU matmuls).
                            GRU MACs use the exact per-step formula 3*(in*h + h*h)*directions from the layer sizes.
                            Not counted: element-wise ops (BatchNorm, PReLU, tanh, sigmoid, mask multiply) and the FFTs.
  MACs/second     COMPUTED  = MACs per hop x 62.5 hops per second
timing (uses the numpy runtime with the ONNX model, one thread)
  per-hop time    MEASURED  full process() call (features + ONNX Runtime + overlap-add), and the ONNX Runtime call alone
  peak RAM        MEASURED  peak working set of a fresh process (Windows), minus the same process after imports
"""
import ctypes
import json
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "complexity"
MODELS = [("Tiny", 8, "tiny_deploy.onnx"), ("Small", 16, "small_deploy.onnx"), ("Base", 32, "base_full.onnx")]
HOPS_PER_S = 16000 / 256


def static():
    import torch
    import torch.nn as nn
    from torch.utils.flop_counter import FlopCounterMode
    from anc.model.gtcrn_x import GTCRNX, count_params
    from anc.model.stream import build_stream_model

    rows = []
    for name, ch, onnx in MODELS:
        off = GTCRNX(ch, 4).eval()
        stream = build_stream_model(off)
        macs = {"conv": 0, "convT_as_conv": 0, "linear": 0, "gru": 0}

        def hook(m, inp, out):
            if isinstance(m, nn.Conv2d):
                macs["conv"] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            elif isinstance(m, nn.ConvTranspose2d):
                macs["convT_as_conv"] += inp[0].numel() * (m.out_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            elif isinstance(m, nn.Linear):
                macs["linear"] += out.numel() * m.in_features
            elif isinstance(m, nn.GRU):
                x = inp[0]                                            # (batch, seq, in) with batch_first
                b, s, i = x.shape
                h, d = m.hidden_size, 2 if m.bidirectional else 1
                macs["gru"] += b * s * d * 3 * (i * h + h * h)

        hs = [mod.register_forward_hook(hook) for mod in stream.modules() if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.GRU))]
        spec, extra = torch.randn(1, 257, 1, 2), torch.rand(1, 4, 1, 257)
        state = stream.init_state(1)
        with torch.no_grad():
            stream(spec, extra, *state)
        for h_ in hs:
            h_.remove()
        counted = macs["conv"] + macs["convT_as_conv"] + macs["linear"]
        # independent cross-check of conv + linear
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            stream(spec, extra, *state)
        cross = fc.get_total_flops() / 2
        # textbook cost of the transposed convolutions in the OFFLINE model (per frame, T=100)
        offm = {"convT": 0, "conv": 0, "linear": 0, "gru": 0}

        def hook2(m, inp, out, T=100):
            if isinstance(m, nn.Conv2d):
                offm["conv"] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1] / T
            elif isinstance(m, nn.ConvTranspose2d):
                offm["convT"] += inp[0].numel() * (m.out_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1] / T
            elif isinstance(m, nn.Linear):
                offm["linear"] += out.numel() * m.in_features / T
        hs = [mod.register_forward_hook(hook2) for mod in off.modules() if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))]
        with torch.no_grad():
            off(torch.randn(1, 257, 100, 2), torch.rand(1, 4, 100, 257))
        for h_ in hs:
            h_.remove()
        total = counted + macs["gru"]
        size = (ROOT / "models" / onnx).stat().st_size
        rows.append(dict(model=name, ch=ch, parameters=count_params(off), onnx_file=onnx, onnx_kb=round(size / 1024, 1),
                         macs_conv_linear_executed=int(counted), macs_gru=int(macs["gru"]), macs_per_hop_executed=int(total),
                         mmacs_per_s_executed=round(total * HOPS_PER_S / 1e6, 1),
                         crosscheck_total_flopcounter=int(cross), crosscheck_ratio=round(cross / max(total, 1), 4),
                         offline_textbook_macs_per_frame=int(sum(offm.values()) + macs["gru"])))
        print(f"{name}: {rows[-1]}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT / "static.json", "w"), indent=1)
    L = ["| model | parameters (MEASURED) | ONNX size KB (MEASURED) | MACs per hop, executed (COMPUTED) | of which GRU | MMACs/s at 62.5 hops/s (COMPUTED) | independent counter, total (ratio to ours) | textbook offline MACs/frame (COMPUTED) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['parameters']:,} | {r['onnx_kb']} | {r['macs_per_hop_executed']:,} | {r['macs_gru']:,} | {r['mmacs_per_s_executed']} | "
                 f"{r['crosscheck_total_flopcounter']:,} ({r['crosscheck_ratio']}) | {r['offline_textbook_macs_per_frame']:,} |")
    (OUT / "static.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ------------------------------------------------------------------------------------------------ timing / RAM
class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


_k32, _psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
_k32.GetCurrentProcess.restype = wintypes.HANDLE                      # 64-bit safe handle handling
_psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def mem():
    """(current working set, peak working set) of this process in MB."""
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if not _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return pmc.WorkingSetSize / 2 ** 20, pmc.PeakWorkingSetSize / 2 ** 20


def worker(onnx, ch, hops):
    from anc.runtime import StreamProcessor
    base_ws, _ = mem()                                        # after imports (numpy, scipy, onnxruntime)
    rt = StreamProcessor(str(ROOT / "models" / onnx), ch=ch, threads=1)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256 * (hops + 300)) * 0.03
    r = 0.6 * x + 0.01 * rng.standard_normal(len(x))
    for h in range(300):
        rt.process(x[h * 256:(h + 1) * 256], r[h * 256:(h + 1) * 256])
    full, ortonly = [], []
    spec = np.zeros((1, 257, 1, 2), np.float32)
    extra = np.zeros((1, 4, 1, 257), np.float32)
    for h in range(300, 300 + hops):
        t0 = time.perf_counter()
        rt.process(x[h * 256:(h + 1) * 256], r[h * 256:(h + 1) * 256])
        full.append((time.perf_counter() - t0) * 1000)
    for _ in range(hops):
        t0 = time.perf_counter()
        rt.sess.run(None, {"spec": spec, "extra": extra, "conv_cache": rt.conv, "tra_cache": rt.tra, "inter_cache": rt.inter})
        ortonly.append((time.perf_counter() - t0) * 1000)
    ws, peak = mem()
    f, o = np.array(full), np.array(ortonly)
    print(json.dumps(dict(model=onnx, hops=hops, full_mean=f.mean(), full_p50=np.percentile(f, 50), full_p95=np.percentile(f, 95), full_p99=np.percentile(f, 99),
                          full_max=f.max(), ort_mean=o.mean(), ort_p95=np.percentile(o, 95), ort_max=o.max(), ram_after_imports_mb=base_ws,
                          ram_peak_mb=peak, ram_increment_mb=peak - base_ws)))


def timing(hops=20000):
    rows = []
    for name, ch, onnx in MODELS:
        p = subprocess.run([sys.executable, str(Path(__file__)), "worker", onnx, str(ch), str(hops)], capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("{")][-1]
        d = json.loads(line)
        d["name"] = name
        rows.append(d)
        print(name, d, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT / "timing.json", "w"), indent=1)
    L = ["| model | hops | full hop: mean / p50 / p95 / p99 / max (ms) | ONNX Runtime only: mean / p95 / max (ms) | real-time factor (full mean / 16 ms) | peak RAM (MB) | RAM above imports (MB) |", "|---|---|---|---|---|---|---|"]
    for d in rows:
        L.append(f"| {d['name']} | {d['hops']} | {d['full_mean']:.3f} / {d['full_p50']:.3f} / {d['full_p95']:.3f} / {d['full_p99']:.3f} / {d['full_max']:.2f} | "
                 f"{d['ort_mean']:.3f} / {d['ort_p95']:.3f} / {d['ort_max']:.2f} | {d['full_mean'] / 16:.3f} | {d['ram_peak_mb']:.0f} | {d['ram_increment_mb']:.0f} |")
    (OUT / "timing.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "static":
        static()
    elif sys.argv[1] == "timing":
        timing(int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
    elif sys.argv[1] == "worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
