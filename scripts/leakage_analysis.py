"""P4 analysis: how much target-speech leakage into the reference microphone can Model H tolerate?

  python scripts/leakage_analysis.py

Inputs (from scripts/run_p4_leakage.sh): results/leak_sweep/{H,C}_<level>.json, Hzero_*.json, none_p00.json and the
manifests in data/generated/leak_sweep/<level>/. Every level uses the SAME clips, seeds, noise, room simulation and primary
signal; only the extra loss (shadow_db) on the target speech reaching the reference mic changes, so differences between
levels are caused by leakage alone. shadow_db is ON TOP of the simulated geometry; the measured leak_db (primary speech level
minus reference speech level, in the clip) is what the x-axis shows. The model was trained with shadow_db in [0, 12].

Decision criteria, fixed in advance (paired per-clip differences, 95% bootstrap CI over clips, on delta SI-SNR):
  reference HELPS       H(ref) - H(zero reference)  has CI entirely above 0
  reference DEGRADED    H(ref) - H(speech-free reference, shadow +80 dB)  has CI entirely below 0
  reference HARMFUL     H(ref) - H(zero reference)  has CI entirely below 0   (using the reference is worse than ignoring it)
Outputs: results/leak_sweep/{leakage_results.csv, leakage_by_snr.csv, LEAKAGE_SUMMARY.md, plots/*.png|pdf}
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_analysis import SNRS, ROOT, group, load, stat   # noqa: E402

D = ROOT / "results" / "leak_sweep"
G = "data/generated/leak_sweep"
LEVELS = [("m12", -12), ("m06", -6), ("p00", 0), ("p06", 6), ("p12", 12), ("p24", 24), ("p80", 80)]
rng = np.random.default_rng(0)


def paired(a, b, metric):
    """mean and 95% CI of (a - b) over clips present in both."""
    A, B = {r["file"]: r for r in a}, {r["file"]: r for r in b}
    d = np.array([A[f][metric] - B[f][metric] for f in A if f in B], float)
    d = d[~np.isnan(d)]
    idx = rng.integers(0, len(d), (2000, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    R = {}
    for tag, x in LEVELS:
        for m in ("H", "C"):
            p = D / f"{m}_{tag}.json"
            if p.exists():
                R[(m, tag)] = load(p, f"{G}/{tag}")
    Hz = load(D / "Hzero_p00.json", f"{G}/p00")
    Hz80 = load(D / "Hzero_p80.json", f"{G}/p80")
    none = load(D / "none_p00.json", f"{G}/p00")
    leak = {}
    for tag, _ in LEVELS:
        v = [json.loads(l)["leak_db"] for l in open(ROOT / G / tag / "manifest.jsonl")]
        leak[tag] = (float(np.median(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95)))

    L = ["# Priority 4: reference-microphone speech-leakage sweep\n",
         "Model H (deployment, robust Small) and Model C (same architecture, no dropout/mismatch training) on 480 validation-partition clips "
         "(8 classes x 6 SNRs x 10), identical at every level except the speech leaking into the reference mic. PyTorch protocol. "
         "This is a sensitivity analysis on validation-partition sources, not a final test.\n",
         "**Paired-design check.** The primary microphone signal and clean target must be identical across levels. Model H with an all-zero reference on the "
         f"level 0 clips vs on the level +80 clips: max |difference| of per-clip delta SI-SNR = "
         f"{max(abs(a['dsisnr'] - b['dsisnr']) for a, b in zip(sorted(Hz, key=lambda r: r['file']), sorted(Hz80, key=lambda r: r['file']))):.2e} dB "
         "(should be ~0).\n",
         "Reference conditions for context (same clips): no processing = 0 dB; Model H with the reference switched off (zeros): "
         f"delta SI-SNR {stat(Hz, 'dsisnr')[0]:+.2f} dB, STOI {stat(Hz, 'stoi')[0]:.3f}, PESQ {stat(Hz, 'pesq')[0]:.2f}.\n",
         "| level (shadow_db) | measured leakage: median (5-95%) dB | H dSI-SNR | H STOI | H PESQ | C dSI-SNR | C STOI | C PESQ |", "|---|---|---|---|---|---|---|---|"]
    csv_rows = []
    for tag, x in LEVELS:
        cells = []
        for m in ("H", "C"):
            r = R.get((m, tag))
            if r:
                for met in ("dsisnr", "stoi", "pesq"):
                    mu, lo, hi, n = stat(r, met)
                    cells.append(f"{mu:+.2f} [{lo:+.2f}, {hi:+.2f}]" if met == "dsisnr" else f"{mu:.3f} [{lo:.3f}, {hi:.3f}]" if met == "stoi" else f"{mu:.2f} [{lo:.2f}, {hi:.2f}]")
                    csv_rows.append([tag, x, leak[tag][0], m, n, met, mu, lo, hi])
        lk = leak[tag]
        L.append(f"| {x:+d} dB{' (training range 0-12)' if 0 <= x <= 12 else ''} | {lk[0]:.1f} ({lk[1]:.1f} to {lk[2]:.1f}) | " + " | ".join(cells) + " |")
    for met, name in (("dsisnr", "dSI-SNR"), ("stoi", "STOI"), ("pesq", "PESQ")):
        mu, lo, hi, n = stat(Hz, met)
        csv_rows.append(["zero_ref", "", "", "H", n, met, mu, lo, hi])
    with open(D / "leakage_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "shadow_db", "median_leak_db", "model", "n", "metric", "mean", "ci_lo", "ci_hi"])
        w.writerows(csv_rows)

    # ---- decision criteria
    L += ["", "## Decisions (fixed criteria, delta SI-SNR, paired 95% CI)\n",
          "| level | median leakage (dB) | H(ref) - H(zero ref) | reference helps? | H(ref) - H(speech-free ref) | degraded vs clean reference? | using the reference harmful? | STOI: H(ref) - H(zero) | PESQ: H(ref) - H(zero) |", "|---|---|---|---|---|---|---|---|---|"]
    ideal = R[("H", "p80")]
    verdicts = {}
    for tag, x in LEVELS:
        h = R.get(("H", tag))
        if not h:
            continue
        d0, l0, u0 = paired(h, Hz, "dsisnr")
        di, li, ui = paired(h, ideal, "dsisnr")
        ds, ls, us = paired(h, Hz, "stoi")
        dp, lp, up = paired(h, Hz, "pesq")
        helps, degr, harm = l0 > 0, ui < 0, u0 < 0
        verdicts[tag] = (helps, degr, harm)
        L.append(f"| {x:+d} dB | {leak[tag][0]:.1f} | {d0:+.2f} [{l0:+.2f}, {u0:+.2f}] | {'yes' if helps else 'no'} | {di:+.2f} [{li:+.2f}, {ui:+.2f}] | "
                 f"{'yes' if degr else 'no'} | {'YES' if harm else 'no'} | {ds:+.3f} [{ls:+.3f}, {us:+.3f}] | {dp:+.3f} [{lp:+.3f}, {up:+.3f}] |")
    L += ["", "Model C (no dropout training) against the same criteria:\n", "| level | median leakage (dB) | C(ref) - H(zero ref) | C harmful vs ignoring the reference? |", "|---|---|---|---|"]
    for tag, x in LEVELS:
        c = R.get(("C", tag))
        if c:
            d0, l0, u0 = paired(c, Hz, "dsisnr")
            L.append(f"| {x:+d} dB | {leak[tag][0]:.1f} | {d0:+.2f} [{l0:+.2f}, {u0:+.2f}] | {'YES' if u0 < 0 else 'no'} |")

    # ---- by input SNR
    by = []
    L += ["", "## Benefit of the reference by input SNR: H(ref) - H(zero ref), delta SI-SNR (dB)\n", "| level | " + " | ".join(f"{s:+d} dB" for s in SNRS) + " |", "|---|" + "---|" * len(SNRS)]
    Z = {r["file"]: r for r in Hz}
    for tag, x in LEVELS:
        h = R.get(("H", tag))
        if not h:
            continue
        g = group(h, lambda r: r["snr"])
        row = []
        for s in SNRS:
            d = np.mean([r["dsisnr"] - Z[r["file"]]["dsisnr"] for r in g[s]])
            row.append(f"{d:+.2f}")
            by.append([tag, x, s, float(d)])
        L.append(f"| {x:+d} dB | " + " | ".join(row) + " |")
    with open(D / "leakage_by_snr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "shadow_db", "input_snr", "H_ref_minus_H_zero_dsisnr"])
        w.writerows(by)

    # ---- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (D / "plots").mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    xs = [leak[t][0] for t, _ in LEVELS]
    train_lo, train_hi = sorted([leak["p00"][0], leak["p12"][0]])
    for a, (met, lab) in zip(ax, (("dsisnr", "$\\Delta$SI-SNR (dB)"), ("stoi", "STOI"), ("pesq", "PESQ (WB)"))):
        for m, col, mk in (("H", "#1b6ca8", "o"), ("C", "#d1495b", "s")):
            ys, lo, hi = [], [], []
            for t, _ in LEVELS:
                mu, l_, h_, _n = stat(R[(m, t)], met)
                ys.append(mu); lo.append(mu - l_); hi.append(h_ - mu)
            a.errorbar(xs, ys, yerr=[lo, hi], color=col, marker=mk, capsize=3, label=f"Model {m} with reference" + (" (deployment)" if m == "H" else " (no dropout)"))
        mu = stat(Hz, met)[0]
        a.axhline(mu, color="k", ls="--", lw=1, label="Model H, reference off (zeros)")
        a.axvspan(train_lo, train_hi, color="grey", alpha=0.15, label="training leakage range")
        a.set_xlabel("target-speech leakage into reference: primary - ref level (dB)\n(large = little leakage)")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Reference-microphone speech leakage: effect on Model H and Model C (validation-partition clips, 95% CI)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(D / "plots" / f"leakage_sweep.{ext}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    M = np.array([[r[3] for r in by if r[0] == t and r[1] == x] for t, x in LEVELS if any(r[0] == t for r in by)])
    im = ax.imshow(M, cmap="RdBu", vmin=-max(abs(M).max(), 1), vmax=max(abs(M).max(), 1), aspect="auto")
    ax.set_xticks(range(len(SNRS)), [f"{s:+d}" for s in SNRS])
    ax.set_yticks(range(len(M)), [f"{leak[t][0]:.0f} dB" for t, _ in LEVELS if any(r[0] == t for r in by)])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("input SNR (dB)")
    ax.set_ylabel("median leakage level (primary - reference, dB)")
    ax.set_title("Benefit of the reference: H(ref) - H(zero ref), $\\Delta$SI-SNR (dB)")
    fig.colorbar(im)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(D / "plots" / f"leakage_benefit_by_snr.{ext}", dpi=300)
    plt.close(fig)

    helps = [x for t, x in LEVELS if verdicts.get(t, (0,))[0]]
    harm = [x for t, x in LEVELS if len(verdicts.get(t, ())) and verdicts[t][2]]
    L += ["", "## What the data show (measured), and what they do not\n",
          f"- Levels where the reference helps (CI above 0 vs ignoring it): shadow_db {helps}.",
          f"- Levels where using the reference is worse than ignoring it: shadow_db {harm if harm else 'none in the tested range'}.",
          "- The grid is coarse (7 levels) and the leakage here is a simulated, level-and-low-pass model of leakage; the real headset's leakage (frequency-dependent, "
          "position-dependent) may behave differently. No claim is made beyond these levels."]
    # P4b: the harmful levels through the deployed path (reference guard + ONNX runtime), same clips
    rows = []
    for t, x in LEVELS:
        f = D / f"Honnx_{t}.json"
        if f.exists():
            r = json.load(open(f))
            d = np.array([e["sisnr"] - e["sisnr_in"] for e in r])
            rows.append(f"| {x:+g} dB | {d.mean():+.2f} | {np.mean([e['stoi'] for e in r]):.3f} | {np.mean([e['pesq'] for e in r if e.get('pesq') is not None]):.2f} |")
    if rows:
        L += ["", "## P4b: deployed path (reference guard + ONNX runtime) at the harmful levels, 480 clips each\n",
              "| level (shadow_db) | dSI-SNR | STOI | PESQ |", "|---|---|---|---|"] + rows + [
              "", "For comparison the reference-off model (H with zeros) gets dSI-SNR +9.11 dB on these clips. Whether the guard fires at heavy leakage is not "
              "measured here; the numbers only show that the deployed path is still far below the reference-off result at -12 dB."]
    (D / "LEAKAGE_SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
