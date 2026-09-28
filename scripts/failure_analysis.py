"""P10: where and why does Model H fail?  MEASURED results are kept apart from HYPOTHESES.

  python scripts/failure_analysis.py

Data: Model H per-clip results on the fresh final sets (results/final_test/P2_H_final{A,B}_{main,weapon}.json), the no-processing
per-clip results on the same clips (input STOI / PESQ / SI-SNR), and the manifests (class, SNR, peak level, clipping, leakage,
environment, source). Descriptive only: nothing here is used to change the model.

Outputs (results/failure_analysis/): FAILURE_ANALYSIS.md, worst_clips.csv, enrichment.csv, correlations.csv, plots/*.png|pdf.
Measures: dSI-SNR, STOI (output), dSTOI (= output - input STOI), dPESQ. "Worst" = worst 5% of clips by that measure.
Enrichment = share of a category among the worst clips divided by its share overall (1 = no association);
p-value = two-sided binomial test of that share (many categories are tested, so treat single p-values with caution).
Correlations are Spearman; "within-cell" removes the class x input-SNR effect by z-scoring outcome and feature inside each cell.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "failure_analysis"
FT = ROOT / "results" / "final_test"


def load_set(sid, sub):
    man = {}
    for l in open(ROOT / f"data/generated/final_{sid}/{sub}/manifest.jsonl"):
        m = json.loads(l)
        man[m["file"]] = m
    H = {e["file"]: e for e in json.load(open(FT / f"P2_H_final{sid}_{sub}.json"))}
    N = {e["file"]: e for e in json.load(open(FT / f"P2_none_final{sid}_{sub}.json"))}
    rows = []
    for f, m in man.items():
        h, n = H[f], N[f]
        rows.append(dict(set=f"final_{sid}", sub=sub, file=f, cls=m["cls"], snr=m["snr_requested"] if sub == "main" else None,
                         peak=m.get("peak_target"), env=m["env"], src=m["noise_src"], leak_db=m["leak_db"], speech_spl=m["speech_spl"],
                         clip_p=m["clip_frac_primary"], clip_r=m["clip_frac_ref"], noise_peak_spl=m["noise_peak_spl"],
                         in_sisnr=h["sisnr_in"], in_stoi=n["stoi"], in_pesq=n["pesq"], out_sisnr=h["sisnr"], dsisnr=h["sisnr"] - h["sisnr_in"],
                         out_stoi=h["stoi"], dstoi=h["stoi"] - n["stoi"], out_pesq=h["pesq"], dpesq=h["pesq"] - n["pesq"],
                         speaker=m["speaker"]))
    return rows


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 10 or np.std(x[ok]) == 0:
        return float("nan"), float("nan"), int(ok.sum())
    r, p = stats.spearmanr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def within_cell(rows, feat, outc, cellkey):
    xs, ys = [], []
    g = defaultdict(list)
    for r in rows:
        if r[feat] is not None and r[outc] == r[outc]:
            g[cellkey(r)].append(r)
    for cell in g.values():
        if len(cell) < 5:
            continue
        x, y = np.array([c[feat] for c in cell], float), np.array([c[outc] for c in cell], float)
        if x.std() == 0 or y.std() == 0:
            continue
        xs += list((x - x.mean()) / x.std())
        ys += list((y - y.mean()) / y.std())
    return spearman(xs, ys)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    main_rows = load_set("A", "main") + load_set("B", "main")
    weap_rows = load_set("A", "weapon") + load_set("B", "weapon")
    L = ["# Priority 10: failure analysis of Model H\n",
         f"Clips analysed: {len(main_rows)} main (final_A + final_B, 8 classes x 6 SNRs) and {len(weap_rows)} weapon clips (peaks 110-160 dB SPL). "
         "PyTorch protocol. Everything below is descriptive; **MEASURED** items are computed from the result files, **HYPOTHESIS** items are untested explanations.\n"]

    # ---- 1. worst clips
    csv_w = []
    for outc, lab, worst_low in (("dsisnr", "delta SI-SNR", True), ("dstoi", "delta STOI (output - input)", True), ("out_stoi", "output STOI", True), ("dpesq", "delta PESQ", True)):
        v = np.array([r[outc] for r in main_rows], float)
        thr = np.nanpercentile(v, 5)
        worst = [r for r in main_rows if r[outc] <= thr]
        for r in sorted(worst, key=lambda r: r[outc])[:30]:
            csv_w.append([outc, r["set"], r["file"], r["cls"], r["snr"], r["env"], r["src"], round(r["leak_db"], 1), round(r["in_sisnr"], 1), round(r["out_sisnr"], 1),
                          round(r["in_stoi"], 3), round(r["out_stoi"], 3), round(r["in_pesq"], 2), round(r["out_pesq"], 2)])
    with open(OUT / "worst_clips.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ranked_by", "set", "file", "class", "input_snr", "env", "noise_source", "leak_db", "in_sisnr", "out_sisnr", "in_stoi", "out_stoi", "in_pesq", "out_pesq"])
        w.writerows(csv_w)

    # ---- 2. enrichment of categories among the worst 5%
    L += ["## 1. Which conditions are over-represented among the worst 5% of clips (MEASURED)\n",
          "Worst = lowest value of the measure over the 2,400 main clips. Enrichment > 1 means the category is more common among the worst clips than overall.\n"]
    enr_rows = []
    cats = (("class", lambda r: r["cls"]), ("input SNR (dB)", lambda r: f"{r['snr']:+.0f}"), ("environment", lambda r: r["env"]),
            ("noise source", lambda r: r["src"] or "synthetic"), ("primary clipped", lambda r: "clipped" if r["clip_p"] > 0 else "not clipped"))
    for outc, lab in (("dsisnr", "delta SI-SNR"), ("dstoi", "delta STOI"), ("dpesq", "delta PESQ")):
        v = np.array([r[outc] for r in main_rows], float)
        worst = [r for r in main_rows if r[outc] <= np.nanpercentile(v, 5)]
        L += [f"### worst 5% by {lab} (n = {len(worst)})\n", "| factor | category | share among worst | share overall | enrichment | binomial p |", "|---|---|---|---|---|---|"]
        for cname, f in cats:
            allc = defaultdict(int)
            wc = defaultdict(int)
            for r in main_rows:
                allc[f(r)] += 1
            for r in worst:
                wc[f(r)] += 1
            for k in sorted(allc, key=lambda k: -wc[k] / allc[k]):
                share_o = allc[k] / len(main_rows)
                share_w = wc[k] / len(worst)
                p = stats.binomtest(wc[k], len(worst), share_o).pvalue
                enr_rows.append([lab, cname, k, wc[k], len(worst), allc[k], len(main_rows), share_w / share_o, p])
                if share_w / share_o >= 1.5 or share_w / share_o <= 0.5:
                    L.append(f"| {cname} | {k} | {100 * share_w:.0f}% | {100 * share_o:.0f}% | {share_w / share_o:.2f} | {p:.1e} |")
        L.append("")
    with open(OUT / "enrichment.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["measure", "factor", "category", "n_worst", "worst_total", "n_all", "all_total", "enrichment", "binomial_p"])
        w.writerows(enr_rows)
    L.append("(Only categories with enrichment >= 1.5 or <= 0.5 are listed; the full table is in `enrichment.csv`.)\n")

    # ---- 3. correlations
    feats = [("input SNR", "snr"), ("clipped fraction (primary)", "clip_p"), ("noise peak level (dB SPL)", "noise_peak_spl"),
             ("speech leakage into reference (dB; large = little leakage)", "leak_db"), ("speech level (dB SPL)", "speech_spl"), ("input STOI", "in_stoi")]
    L += ["## 2. Correlation of outcomes with measurable factors (MEASURED)\n",
          "Spearman rho over all main clips (raw), and within class x SNR cells (removes the class and SNR effects). "
          "Speech leakage: the value is primary-minus-reference speech level, so a NEGATIVE rho with a bigger value means heavier leakage helps less.\n",
          "| factor | outcome | raw rho (p) | n | within-cell rho (p) |", "|---|---|---|---|---|"]
    cor_rows = []
    for fl, fk in feats:
        for outc, ol in (("dsisnr", "delta SI-SNR"), ("out_stoi", "output STOI"), ("dpesq", "delta PESQ")):
            r, p, n = spearman([x[fk] for x in main_rows], [x[outc] for x in main_rows])
            rw, pw, _ = within_cell(main_rows, fk, outc, lambda x: (x["cls"], x["snr"])) if fk != "snr" else (float("nan"), float("nan"), 0)
            cor_rows.append([fl, ol, r, p, n, rw, pw])
            L.append(f"| {fl} | {ol} | {r:+.2f} ({p:.0e}) | {n} | " + (f"{rw:+.2f} ({pw:.0e})" if rw == rw else "n/a") + " |")
    L.append("")
    with open(OUT / "correlations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["factor", "outcome", "raw_rho", "raw_p", "n", "within_cell_rho", "within_cell_p"])
        w.writerows(cor_rows)

    # ---- 4. by-environment (reverberation proxy), synthetic vs real, within class x SNR
    L += ["## 3. Environment (a proxy for reverberation) and real vs synthetic gunshots (MEASURED, within class x SNR cells)\n",
          "Each clip's outcome minus its class x SNR cell mean, averaged by environment. `open` = little reverberation, `room` and `cabin` = reverberant simulated rooms.\n",
          "| environment | n | mean delta SI-SNR relative to cell mean | mean output STOI relative to cell mean |", "|---|---|---|---|"]
    cellmean = defaultdict(lambda: defaultdict(list))
    for r in main_rows:
        for o in ("dsisnr", "out_stoi"):
            cellmean[(r["cls"], r["snr"])][o].append(r[o])
    for env in ("open", "room", "cabin"):
        rs = [r for r in main_rows if r["env"] == env]
        d1 = np.mean([r["dsisnr"] - np.mean(cellmean[(r["cls"], r["snr"])]["dsisnr"]) for r in rs])
        d2 = np.mean([r["out_stoi"] - np.mean(cellmean[(r["cls"], r["snr"])]["out_stoi"]) for r in rs])
        L.append(f"| {env} | {len(rs)} | {d1:+.2f} dB | {d2:+.3f} |")
    gs = [r for r in main_rows if r["cls"] == "gunshot"]
    for src in ("real", "synth"):
        rs = [r for r in gs if (r["src"] == "real") == (src == "real")]
        d1 = np.mean([r["dsisnr"] - np.mean(cellmean[(r["cls"], r["snr"])]["dsisnr"]) for r in rs])
        d2 = np.mean([r["out_stoi"] - np.mean(cellmean[(r["cls"], r["snr"])]["out_stoi"]) for r in rs])
        L.append(f"| gunshot, {src} recording | {len(rs)} | {d1:+.2f} dB | {d2:+.3f} |")
    L.append("")

    # ---- 5. weapons
    L += ["## 4. Close-range weapons (MEASURED)\n",
          "| peak level (dB SPL) | n | clipped share of samples (primary) | input STOI | output STOI | delta STOI | output PESQ | delta SI-SNR |", "|---|---|---|---|---|---|---|---|"]
    for pk in (110, 120, 130, 140, 150, 160):
        rs = [r for r in weap_rows if r["peak"] == pk]
        L.append(f"| {pk} | {len(rs)} | {100 * np.mean([r['clip_p'] for r in rs]):.2f}% | {np.mean([r['in_stoi'] for r in rs]):.2f} | {np.mean([r['out_stoi'] for r in rs]):.2f} | "
                 f"{np.mean([r['dstoi'] for r in rs]):+.2f} | {np.mean([r['out_pesq'] for r in rs]):.2f} | {np.mean([r['dsisnr'] for r in rs]):+.1f} |")
    r1, p1, n1 = spearman([r["clip_p"] for r in weap_rows], [r["out_stoi"] for r in weap_rows])
    r2, p2, _ = within_cell(weap_rows, "clip_p", "out_stoi", lambda x: (x["cls"], x["peak"]))
    L += ["", f"Spearman rho of clipped fraction vs output STOI over weapon clips: {r1:+.2f} (p {p1:.0e}, n {n1}); within (class, peak) cells: {r2:+.2f} (p {p2:.0e}).",
          "The energy-based dSI-SNR grows with peak level while STOI falls: at high peaks the burst dominates the energy, so removing it scores well on dSI-SNR "
          "without recovering the speech (see the table), which is why weapon results must not be read from SI-SNR alone.\n"]

    # ---- 6. plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    classes = ["engine", "helicopter", "drone", "wind", "siren", "gunshot", "artillery", "mixed"]
    snrs = [-10, -5, 0, 5, 10, 15]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, (o, t, cm) in zip(ax, (("dsisnr", "mean $\\Delta$SI-SNR (dB)", "viridis"), ("out_stoi", "mean output STOI", "viridis"), ("out_pesq", "mean output PESQ", "viridis"))):
        M = np.array([[np.mean([r[o] for r in main_rows if r["cls"] == c and r["snr"] == s]) for s in snrs] for c in classes])
        im = a.imshow(M, cmap=cm, aspect="auto")
        a.set_xticks(range(6), [f"{s:+d}" for s in snrs])
        a.set_yticks(range(8), classes)
        for i in range(8):
            for j in range(6):
                a.text(j, i, f"{M[i, j]:.2f}" if o != "dsisnr" else f"{M[i, j]:.1f}", ha="center", va="center", color="w", fontsize=7)
        a.set_title(t)
        a.set_xlabel("input SNR (dB)")
        fig.colorbar(im, ax=a)
    fig.suptitle("Model H on the fresh final sets: performance by class and input SNR (final_A + final_B)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "plots" / f"failure_heatmaps.{ext}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, (fk, fl) in zip(ax, (("leak_db", "speech leakage: primary - reference level (dB)"), ("clip_p", "clipped fraction of primary samples"), ("noise_peak_spl", "noise peak level (dB SPL)"))):
        for c, col in zip(classes, plt.cm.tab10.colors):
            rs = [r for r in main_rows if r["cls"] == c]
            a.scatter([r[fk] for r in rs], [r["dsisnr"] for r in rs], s=6, alpha=0.4, color=col, label=c)
        a.set_xlabel(fl)
        a.set_ylabel("$\\Delta$SI-SNR (dB)")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=6, markerscale=2)
    fig.suptitle("Failure analysis: per-clip $\\Delta$SI-SNR vs candidate factors (raw, not adjusted for class/SNR)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "plots" / f"failure_scatter.{ext}", dpi=300)
    plt.close(fig)

    # gallery of the worst clips (enhanced audio regenerated with Model H; the evaluation deleted its temporary audio)
    import soundfile as sf
    from evaluate import OurModel
    proc = OurModel(str(ROOT / "checkpoints" / "H_robust_s0_last.pt"))
    worst = sorted(main_rows, key=lambda r: r["out_stoi"])[:6]
    fig, ax = plt.subplots(len(worst), 3, figsize=(13, 2.0 * len(worst)))
    for i, r in enumerate(worst):
        sid = r["set"].split("_")[1]
        x, _ = sf.read(ROOT / f"data/generated/final_{sid}/main/{r['file']}", dtype="float64")
        enh = np.clip(proc(x[:, 0], x[:, 1]), -1, 1)
        for j, (t, sig) in enumerate((("noisy (primary)", x[:, 0]), ("enhanced (Model H)", enh), ("clean target", x[:, 2]))):
            ax[i, j].specgram(sig + 1e-9, NFFT=512, Fs=16000, noverlap=256, vmin=-140, vmax=-30)
            ax[i, j].set_title(f"{r['cls']} {r['snr']:+.0f} dB, {r['env']}: {t}" + (f"  STOI {r['out_stoi']:.2f}" if j == 1 else ""), fontsize=7)
            ax[i, j].tick_params(labelsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "plots" / f"failure_worst_clips.{ext}", dpi=300)
    plt.close(fig)

    L += ["## 5. Possible explanations (HYPOTHESES, not tested here)\n",
          "- Low-SNR failures concentrated in tonal or broadband continuous classes (wind, helicopter, drone, mixed) may reflect limited model capacity, or that "
          "speech and noise overlap heavily in time-frequency at those SNRs.",
          "- Wind being weakest is consistent with its being uncorrelated between the two microphones (so the reference adds nothing), but this was not isolated experimentally here.",
          "- Loss of intelligibility at very high peak levels is consistent with speech being masked or destroyed inside the clipped burst (information absent from the input), "
          "but this analysis does not separate that from model limitations.",
          "- Reverberant environments may be harder because the target is the direct-path speech; the environment label is only a coarse proxy.",
          "- Correlations with clipping or leakage could be confounded by class and SNR; the within-cell numbers reduce, but do not remove, that concern.\n"]
    (OUT / "FAILURE_ANALYSIS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
