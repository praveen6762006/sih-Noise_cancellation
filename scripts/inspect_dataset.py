"""Sanity report for a generated dataset, including the 'no processing' baseline row.

  python scripts/inspect_dataset.py data/generated/smoke [--plot out.png]

Per class x SNR bin: unprocessed SI-SNR and STOI (noisy primary vs clean target), measured
SNR after the mic model, clipping, peak level, and speech leakage into the reference.
PESQ is not computed here yet (the `pesq` package needs a C++ compiler on this machine).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from pystoi import stoi


def si_snr(est, ref, eps=1e-8):
    est, ref = est - est.mean(), ref - ref.mean()
    t = (est @ ref) / (ref @ ref + eps) * ref
    return 10 * np.log10((t @ t + eps) / ((est - t) @ (est - t) + eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--plot")
    ap.add_argument("--pesq", action="store_true", help="also compute wideband PESQ (runs in WSL)")
    a = ap.parse_args()
    d = Path(a.dir)
    rows = [json.loads(l) for l in open(d / "manifest.jsonl")]
    pesq_scores = {}
    if a.pesq:
        import subprocess
        drive, rest = str(d.resolve()).split(":", 1)
        wsl_dir = f"/mnt/{drive.lower()}{rest.replace(chr(92), '/')}"
        script = f"/mnt/{drive.lower()}{str(Path(__file__).resolve().parent.joinpath('pesq_score.py'))[2:].replace(chr(92), '/')}"
        out = subprocess.run(["wsl", "-d", "Ubuntu-22.04", "--", "python3", script, wsl_dir],
                             capture_output=True, text=True, check=True).stdout
        pesq_scores = json.loads(out)
    cells = defaultdict(list)
    for m in rows:
        x, sr = sf.read(d / m["file"], dtype="float64")
        noisy, ref, clean = x[:, 0], x[:, 1], x[:, 2]
        post_snr = 10 * np.log10(np.sum(clean ** 2) / (np.sum((noisy - clean) ** 2) + 1e-12))
        key = (m["cls"], m["peak_target"] if m.get("peak_target") is not None else round(m["snr_db"] / 5) * 5)
        cells[key].append(dict(
            sisnr=si_snr(noisy, clean), stoi=stoi(clean, noisy, sr, extended=False), post_snr=post_snr,
            clip_p=100 * m["clip_frac_primary"], clip_r=100 * m["clip_frac_ref"],
            pesq=pesq_scores.get(m["file"]) if pesq_scores.get(m["file"]) is not None else np.nan,
            peak=m["noise_peak_spl"], leak=m["leak_db"], n_imp=len(m["impulse_samples"]),
            real=m["noise_src"] == "real"))

    peak_mode = any(m.get("peak_target") is not None for m in rows)
    print(f"{'class':<11}{'peak' if peak_mode else 'snr':>5}{'n':>4}{'SI-SNR':>8}{'STOI':>7}{'PESQ':>6}{'postSNR':>9}{'clip%P':>8}{'clip%R':>8}{'peakSPL':>9}{'leak dB':>9}")
    for (c, s), v in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        g = lambda k: np.nanmean([e[k] for e in v]) if not all(np.isnan(e[k]) for e in v) else np.nan
        print(f"{c:<11}{s:>5.0f}{len(v):>4}{g('sisnr'):>8.1f}{g('stoi'):>7.2f}{g('pesq'):>6.2f}{g('post_snr'):>9.1f}"
              f"{g('clip_p'):>8.2f}{g('clip_r'):>8.2f}{g('peak'):>9.0f}{g('leak'):>9.1f}")
    leaks = [m["leak_db"] for m in rows]
    print(f"\nspeech leakage into reference: median {np.median(leaks):.1f} dB, 5-95% {np.percentile(leaks, 5):.1f} to {np.percentile(leaks, 95):.1f} dB")
    print(f"speech SPL at primary: {np.min([m['speech_spl'] for m in rows]):.0f} to {np.max([m['speech_spl'] for m in rows]):.0f} dB")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pick = {}
        for m in rows:
            if m["snr_db"] == 0 and m["cls"] not in pick:
                pick[m["cls"]] = m
        fig, ax = plt.subplots(len(pick), 3, figsize=(13, 2.1 * len(pick)))
        for i, (c, m) in enumerate(pick.items()):
            x, sr = sf.read(d / m["file"], dtype="float64")
            for j, (nm, ch) in enumerate([("noisy primary", 0), ("reference", 1), ("clean target", 2)]):
                ax[i, j].specgram(x[:, ch] + 1e-9, NFFT=512, Fs=sr, noverlap=256, vmin=-140, vmax=-30)
                ax[i, j].set_title(f"{c}: {nm}", fontsize=8)
                ax[i, j].tick_params(labelsize=6)
        plt.tight_layout()
        plt.savefig(a.plot, dpi=90)
        print("plot ->", a.plot)


if __name__ == "__main__":
    main()
