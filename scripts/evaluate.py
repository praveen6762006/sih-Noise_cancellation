"""Evaluate a processor on a generated dataset and print the per-class x SNR table.

  python scripts/evaluate.py --data data/generated/val --proc none
  python scripts/evaluate.py --data data/generated/val --proc gtcrn
  python scripts/evaluate.py --data data/generated/val --proc gtcrn --peak   # group by peak SPL

Metrics: output SI-SNR, SI-SNR improvement over the unprocessed noisy primary, STOI, wideband
PESQ (through WSL). The last columns mark which of the statement's targets are met on
average in that cell: SNR>15 dB (SI-SNR out), STOI>0.85, PESQ>2.5.

Processors take the noisy primary (and optionally the reference) and return the enhanced signal:
  none   identity (the 'no processing' baseline row)
  gtcrn  pretrained single-mic GTCRN from third_party/gtcrn (DNS3 checkpoint)
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from pystoi import stoi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def si_snr(est, ref, eps=1e-8):
    est, ref = est - est.mean(), ref - ref.mean()
    t = (est @ ref) / (ref @ ref + eps) * ref
    return 10 * np.log10((t @ t + eps) / ((est - t) @ (est - t) + eps))


def wsl_path(p: Path) -> str:
    p = p.resolve()
    drive, rest = str(p).split(":", 1)
    return f"/mnt/{drive.lower()}{rest.replace(chr(92), '/')}"


# ---------------------------------------------------------------- processors
class Identity:
    name = "none"

    def __call__(self, noisy, ref):
        return noisy


class PretrainedGTCRN:
    """Single-mic pretrained GTCRN. The network has fixed BatchNorm statistics, so the input is
    normalised to a typical level first and the gain is undone afterwards."""
    name = "gtcrn"

    def __init__(self, ckpt="third_party/gtcrn/checkpoints/model_trained_on_dns3.tar", device=None):
        import torch
        sys.path.insert(0, str(ROOT / "third_party" / "gtcrn"))
        from gtcrn import GTCRN
        self.torch = torch
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = GTCRN().eval().to(self.dev)
        self.model.load_state_dict(torch.load(ROOT / ckpt, map_location=self.dev)["model"])
        self.win = torch.hann_window(512, device=self.dev).pow(0.5)

    def __call__(self, noisy, ref):
        t = self.torch
        g = 0.05 / (np.sqrt(np.mean(noisy ** 2)) + 1e-9)
        x = t.from_numpy((noisy * g).astype(np.float32)).to(self.dev)
        with t.no_grad():
            spec = t.view_as_real(t.stft(x, 512, 256, 512, self.win, return_complex=True))
            out = self.model(spec[None])[0]
            y = t.istft(t.view_as_complex(out.contiguous()), 512, 256, 512, self.win)
        y = y.cpu().numpy()
        y = np.pad(y, (0, max(0, len(noisy) - len(y))))[:len(noisy)]
        return y / g


class OurModel:
    """A GTCRN-X checkpoint from scripts/train.py; uses the reference mic if the model has extra inputs."""
    name = "ours"

    def __init__(self, ckpt, device=None):
        import torch
        from anc.model.gtcrn_x import GTCRNX, SIZES
        self.torch = torch
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(ckpt, map_location=self.dev)
        self.model = GTCRNX(ch=SIZES[state["size"]], n_extra=state["n_extra"]).to(self.dev).eval()
        self.model.load_state_dict(state["model"])
        self.model.use_flag = state.get("use_flag", True)

    def __call__(self, noisy, ref):
        from anc.features import istft, prepare_batch
        t = self.torch
        n = t.from_numpy(noisy.astype(np.float32))[None].to(self.dev)
        r = t.from_numpy(ref.astype(np.float32))[None].to(self.dev)
        with t.no_grad():
            b = prepare_batch(n, r)
            est = istft(self.model(b["spec"], b["extra"] if self.model.n_extra else None), n.shape[-1])
        return (est / b["gain"])[0].cpu().numpy().astype(np.float64)


class OnnxStream:
    """The DEPLOYED path: NumPy front end + streaming ONNX model + overlap-add, with the slow AGC
    (unlike the PyTorch evaluation, which normalises each clip by its own RMS)."""
    name = "onnx"

    def __init__(self, path, ch=16, n_extra=4, use_flag=True):
        from anc.runtime import HOP, StreamProcessor
        self.HOP = HOP
        self.rt = StreamProcessor(path, ch=ch, n_extra=n_extra, use_flag=use_flag)

    def __call__(self, noisy, ref):
        H = self.HOP
        n = (len(noisy) // H) * H
        self.rt.reset()
        outs = [self.rt.process(noisy[h * H:(h + 1) * H], ref[h * H:(h + 1) * H]) for h in range(n // H)]
        outs.append(self.rt.process(np.zeros(H), np.zeros(H)))              # flush the last hop
        y = np.concatenate(outs)[H:H + n]
        return np.pad(y, (0, len(noisy) - n))


class WithRefGuard:
    """Reference-mic gain calibration + copy detection (anc.refguard) applied before any processor."""

    def __init__(self, proc):
        self.proc = proc

    def __call__(self, noisy, ref):
        from anc.refguard import guard_clip
        return self.proc(noisy, guard_clip(noisy, ref))


class WithNLMS:
    """Wrap a processor with the reference-mic NLMS stage.
    pre : NLMS on the raw primary, then the network (network sees the NLMS residual; reference unchanged)
    post: network first, then NLMS on the enhanced signal using the reference mic"""

    def __init__(self, proc, mode, **kw):
        self.proc, self.mode, self.kw = proc, mode, kw

    def __call__(self, noisy, ref):
        from anc.nlms import run_nlms
        if self.mode == "pre":
            return self.proc(run_nlms(noisy, ref, **self.kw)[0], ref)
        return run_nlms(self.proc(noisy, ref), ref, **self.kw)[0]


PROCESSORS = {"none": Identity, "gtcrn": PretrainedGTCRN, "ours": OurModel, "onnx": OnnxStream}


# ---------------------------------------------------------------- evaluation
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--proc", default="none", choices=list(PROCESSORS))
    ap.add_argument("--ckpt", help="checkpoint for --proc ours")
    ap.add_argument("--onnx", help="model file for --proc onnx")
    ap.add_argument("--ch", type=int, default=16, help="width of the --onnx model (8/16/32)")
    ap.add_argument("--no-extra", action="store_true", help="--onnx model is single-mic")
    ap.add_argument("--tag", help="name for the output folder (default: the processor name)")
    ap.add_argument("--nlms", choices=["pre", "post"], help="add the reference-mic NLMS stage before/after the processor")
    ap.add_argument("--ref-fault", choices=["zero", "copy", "loud", "quiet", "shifted"],
                    help="corrupt the reference mic before processing (robustness test)")
    ap.add_argument("--stride", type=int, default=1, help="use every N-th example")
    ap.add_argument("--refguard", action="store_true", help="apply the reference-mic guard (gain calibration, copy detection)")
    ap.add_argument("--no-gate", action="store_true", help="NLMS without the speech/impulse gate (ablation)")
    ap.add_argument("--peak", action="store_true", help="group by target peak SPL (close-range weapon set)")
    ap.add_argument("--no-pesq", action="store_true")
    ap.add_argument("--keep-audio", action="store_true", help="keep the enhanced WAVs (default: delete them once PESQ scoring is done; results are in the JSON)")
    ap.add_argument("--limit", type=int, help="only the first N examples")
    ap.add_argument("--out", help="write per-example results to this JSON file")
    a = ap.parse_args()

    d = Path(a.data)
    rows = [json.loads(l) for l in open(d / "manifest.jsonl")]
    if a.limit:
        rows = rows[:a.limit]
    rows = rows[::a.stride]
    if a.proc == "ours":
        proc = PROCESSORS["ours"](a.ckpt)
    elif a.proc == "onnx":
        proc = OnnxStream(a.onnx, ch=a.ch, n_extra=0 if a.no_extra else 4)
    else:
        proc = PROCESSORS[a.proc]()
    if a.nlms:
        proc = WithNLMS(proc, a.nlms, gate=not a.no_gate)
    if a.refguard:
        proc = WithRefGuard(proc)
    enh_dir = ROOT / "results" / "enhanced" / f"{a.tag or a.proc}_{d.name}"
    enh_dir.mkdir(parents=True, exist_ok=True)

    per = []
    for i, m in enumerate(rows):
        x, sr = sf.read(d / m["file"], dtype="float64")
        noisy, ref, clean = x[:, 0], x[:, 1], x[:, 2]
        if a.ref_fault:
            ref = {"zero": np.zeros_like(ref), "copy": noisy.copy(), "loud": ref * 10 ** (10 / 20),
                   "quiet": ref * 10 ** (-10 / 20), "shifted": np.roll(ref, 80)}[a.ref_fault]
        enh = np.clip(proc(noisy, ref), -1, 1)
        sf.write(enh_dir / m["file"], np.stack([enh, ref, clean], axis=1).astype(np.float32), sr, subtype="PCM_16")
        per.append(dict(
            file=m["file"], cls=m["cls"],
            key=m["peak_target"] if a.peak else round(m["snr_db"] / 5) * 5,
            sisnr=si_snr(enh, clean), sisnr_in=si_snr(noisy, clean), stoi=stoi(clean, enh, sr, extended=False)))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    pesq_scores = {}
    if not a.no_pesq:
        (enh_dir / "manifest.jsonl").write_text("".join(json.dumps({"file": p["file"]}) + "\n" for p in per))
        script = wsl_path(ROOT / "scripts" / "pesq_score.py")
        out = subprocess.run(["wsl", "-d", "Ubuntu-22.04", "--", "python3", script, wsl_path(enh_dir)],
                             capture_output=True, text=True, check=True).stdout
        pesq_scores = json.loads(out)
    for p in per:
        p["pesq"] = pesq_scores.get(p["file"]) if pesq_scores.get(p["file"]) is not None else np.nan
    if not a.keep_audio:                       # the enhanced audio is only needed for PESQ scoring
        import shutil
        shutil.rmtree(enh_dir, ignore_errors=True)

    cells = defaultdict(list)
    for p in per:
        cells[(p["cls"], p["key"])].append(p)

    print(f"\nprocessor: {a.proc}   data: {d}   n={len(per)}")
    print(f"{'class':<11}{'peak' if a.peak else 'snr':>5}{'n':>4}{'SI-SNR':>8}{'dSI-SNR':>9}{'STOI':>7}{'PESQ':>6}   targets(SNR>15,STOI>.85,PESQ>2.5)")
    for (c, k), v in sorted(cells.items()):
        mean = lambda f: float(np.nanmean([e[f] for e in v])) if not all(np.isnan(e[f]) for e in v) else np.nan
        s, dS, t, q = mean("sisnr"), mean("sisnr") - mean("sisnr_in"), mean("stoi"), mean("pesq")
        ok = "".join("Y" if cond else "-" for cond in (s > 15, t > 0.85, q > 2.5))
        print(f"{c:<11}{k:>5.0f}{len(v):>4}{s:>8.1f}{dS:>9.1f}{t:>7.2f}{q:>6.2f}   {ok}")
    if a.out:
        Path(a.out).write_text(json.dumps(per))


if __name__ == "__main__":
    main()
