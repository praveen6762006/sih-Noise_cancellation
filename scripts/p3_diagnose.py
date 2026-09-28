"""Why does the deployed path differ from PyTorch on some clips?  (measured decomposition, not a guess)

  python scripts/p3_diagnose.py

Two things differ between the PyTorch evaluation and the deployed runtime besides numerics: the slow automatic level control
(AGC, ~1.5 s) instead of per-clip RMS normalisation, and the reference guard. For the clips with the largest discrepancy plus a random
sample of 60 clips of final_A/main, the runtime is run in four variants and scored against the same clean target:
    deployed        guard on,  AGC on      (what runs on the target)
    no guard        guard off, AGC on
    fixed gain      guard on,  gain fixed to the PyTorch per-clip value
    neither         guard off, gain fixed  (should reproduce PyTorch up to numerics)
Reported: mean dSI-SNR of each variant minus the PyTorch value on the same clips, and the same for the largest-discrepancy clips.
Output: results/final_test/P3_diagnosis.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from anc.runtime import HOP, StreamProcessor        # noqa: E402
from evaluate import si_snr                          # noqa: E402

FT = ROOT / "results" / "final_test"


def run(model, noisy, ref, guard, gain):
    rt = StreamProcessor(str(ROOT / "models" / model), ch=16, guard=guard, fixed_gain=gain)
    n = (len(noisy) // HOP) * HOP
    outs = [rt.process(noisy[h * HOP:(h + 1) * HOP], ref[h * HOP:(h + 1) * HOP]) for h in range(n // HOP)] + [rt.process(np.zeros(HOP), np.zeros(HOP))]
    y = np.concatenate(outs)[HOP:HOP + n]
    return np.pad(y, (0, len(noisy) - n)), rt


def main():
    S = "A"
    man = {json.loads(l)["file"]: json.loads(l) for l in open(ROOT / f"data/generated/final_{S}/main/manifest.jsonl")}
    pt = {e["file"]: e for e in json.load(open(FT / f"P2_H_final{S}_main.json"))}
    on = {e["file"]: e for e in json.load(open(FT / f"P3_onnx_fp32_final{S}_main.json"))}
    d = {f: (on[f]["sisnr"] - on[f]["sisnr_in"]) - (pt[f]["sisnr"] - pt[f]["sisnr_in"]) for f in pt}
    worst = sorted(d, key=lambda f: -abs(d[f]))[:8]
    rng = np.random.default_rng(0)
    sample = [f for f in rng.permutation(sorted(pt))[:60] if f not in worst]
    variants = [("deployed (guard on, AGC)", True, False), ("no guard (AGC)", False, False), ("fixed gain (guard on)", True, True), ("neither (no guard, fixed gain)", False, True)]

    def score(f):
        x, _ = sf.read(ROOT / f"data/generated/final_{S}/main/{f}", dtype="float64")
        noisy, ref, clean = x[:, 0], x[:, 1], x[:, 2]
        g = 0.05 / (np.sqrt(np.mean(noisy ** 2)) + 1e-9)
        res = {}
        for name, guard, fixed in variants:
            y, rt = run("small_deploy.onnx", noisy, ref, guard, g if fixed else None)
            res[name] = si_snr(np.clip(y, -1, 1), clean) - si_snr(noisy, clean)
            if name.startswith("deployed"):
                res["guard_flagged_share"] = float(rt.guard.invalid)   # state at the end; the share is tracked below
        return res

    def guard_share(f):
        x, _ = sf.read(ROOT / f"data/generated/final_{S}/main/{f}", dtype="float64")
        rt = StreamProcessor(str(ROOT / "models" / "small_deploy.onnx"), ch=16)
        n = len(x) // HOP
        flagged = 0
        for h in range(n):
            rt.process(x[h * HOP:(h + 1) * HOP, 0], x[h * HOP:(h + 1) * HOP, 1])
            flagged += rt.guard.invalid
        return flagged / n

    L = ["# P3 diagnosis: why deployed and PyTorch differ (final_A main)\n",
         "MEASURED decomposition. Each row: mean over the clips of (variant dSI-SNR minus the PyTorch-path dSI-SNR of the same clip), dB.\n"]
    for title, files in (("Largest per-clip discrepancies (8 clips)", worst), ("Random sample (60 clips)", sample)):
        R = {f: score(f) for f in files}
        pt_d = {f: pt[f]["sisnr"] - pt[f]["sisnr_in"] for f in files}
        L += [f"## {title}\n", "| variant | mean difference vs PyTorch (dB) | largest |difference| (dB) |", "|---|---|---|"]
        for name, _, _ in variants:
            diffs = np.array([R[f][name] - pt_d[f] for f in files])
            L.append(f"| {name} | {diffs.mean():+.3f} | {np.abs(diffs).max():.2f} |")
        L.append("")
        if title.startswith("Largest"):
            L += ["| file | class | input SNR | peak target | clipped share (primary) | PyTorch dSI-SNR | deployed dSI-SNR | no-guard | fixed-gain | guard flagged share of hops |", "|---|---|---|---|---|---|---|---|---|---|"]
            for f in files:
                m = man[f]
                L.append(f"| {f} | {m['cls']} | {m['snr_requested']:+.0f} | {m.get('peak_target')} | {m['clip_frac_primary']:.4f} | {pt_d[f]:+.1f} | {R[f]['deployed (guard on, AGC)']:+.1f} | "
                         f"{R[f]['no guard (AGC)']:+.1f} | {R[f]['fixed gain (guard on)']:+.1f} | {guard_share(f):.3f} |")
            L.append("")
    L += ["Reading: if 'fixed gain' removes the discrepancy, the difference comes from the level control (its start-up and slow tracking); "
          "if 'no guard' removes it, the guard flagged a healthy reference; if only 'neither' matches PyTorch, both contribute. "
          "Any statement about WHY the level control or guard behaves this way on those clips is a hypothesis unless shown above.\n"]
    (FT / "P3_diagnosis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
