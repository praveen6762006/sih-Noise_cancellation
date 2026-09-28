"""Before/after audio and spectrogram figures for the presentation.

  python scripts/make_demo.py --ckpt checkpoints/C_full_s0_last.pt --out demo

For each class, one clip at 0 dB SNR plus close-range weapon clips: writes noisy / enhanced / clean WAVs and a
spectrogram figure. Uses the held-out test sets, so these are clips the model has never seen.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import OurModel, si_snr   # noqa: E402
from pystoi import stoi                  # noqa: E402


def pick(rows, cls, key, field):
    c = [m for m in rows if m["cls"] == cls and (m[field] == key if field == "peak_target" else abs(m["snr_db"] - key) < 2.5)]
    return c[0] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="demo")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proc = OurModel(a.ckpt)
    out = ROOT / a.out
    out.mkdir(exist_ok=True)
    sel = []
    rows = [json.loads(l) for l in open(ROOT / "data/generated/test/manifest.jsonl")]
    for c in ("engine", "helicopter", "drone", "wind", "siren", "gunshot", "artillery", "mixed"):
        m = pick([r for r in rows if r.get("peak_target") is None], c, 0, "snr")
        if m:
            sel.append(("test", m, f"{c}_0dB"))
    rows_p = [json.loads(l) for l in open(ROOT / "data/generated/test_peak/manifest.jsonl")]
    for c, pk in (("gunshot", 140), ("artillery", 150)):
        m = pick(rows_p, c, pk, "peak_target")
        if m:
            sel.append(("test_peak", m, f"{c}_peak{pk}dB"))

    fig, ax = plt.subplots(len(sel), 3, figsize=(13, 2.0 * len(sel)))
    lines = []
    for i, (d, m, name) in enumerate(sel):
        x, _ = sf.read(ROOT / "data/generated" / d / m["file"], dtype="float64")
        noisy, ref, clean = x[:, 0], x[:, 1], x[:, 2]
        enh = np.clip(proc(noisy, ref), -1, 1)
        for tag, sig in (("noisy", noisy), ("enhanced", enh), ("clean", clean)):
            sf.write(out / f"{name}_{tag}.wav", sig, 16000, subtype="PCM_16")
        lines.append(f"{name:22s} SI-SNR {si_snr(noisy, clean):6.1f} -> {si_snr(enh, clean):6.1f} dB   "
                     f"STOI {stoi(clean, noisy, 16000):.2f} -> {stoi(clean, enh, 16000):.2f}")
        for j, (t, sig) in enumerate((("noisy (primary mic)", noisy), ("enhanced", enh), ("clean target", clean))):
            ax[i, j].specgram(sig + 1e-9, NFFT=512, Fs=16000, noverlap=256, vmin=-140, vmax=-30)
            ax[i, j].set_title(f"{name}: {t}", fontsize=8)
            ax[i, j].tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(out / "spectrograms.png", dpi=90)
    (out / "examples.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
