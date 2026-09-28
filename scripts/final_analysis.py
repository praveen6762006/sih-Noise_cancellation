"""Tables from per-clip evaluation results, with bootstrap 95% confidence intervals. One protocol per table, never mixed.

  python scripts/final_analysis.py p2          # Priority 2: Model H on the fresh, unseen, speech-only, cross-firearm, weapon sets

Reads results/final_test/P2_*.json (written by evaluate.py) joined with the set manifests (class, SNR bin, peak level,
firearm). Writes CSVs and P2_summary.md to results/final_test/. Metrics per clip: input SI-SNR, output SI-SNR,
delta SI-SNR (= output - input), STOI, PESQ (wideband). CI = 2.5-97.5 percentile of 2000 bootstrap resamples of the clips.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "results" / "final_test"
CLASSES = ["engine", "helicopter", "drone", "wind", "siren", "gunshot", "artillery", "mixed"]
SNRS = [-10, -5, 0, 5, 10, 15]
PEAKS = [110, 120, 130, 140, 150, 160]
METRICS = ["in_sisnr", "out_sisnr", "dsisnr", "stoi", "pesq"]
_rng = np.random.default_rng(0)


def load(result_json, manifest_dir):
    """Per-clip records: evaluation metrics joined with manifest fields."""
    man = {}
    for l in open(ROOT / manifest_dir / "manifest.jsonl"):
        m = json.loads(l)
        man[m["file"]] = m
    out = []
    for e in json.load(open(result_json)):
        m = man[e["file"]]
        fw = sorted({Path(p).parent.name for p in m.get("gunshot_files", [])})
        out.append(dict(file=e["file"], cls=m["cls"], snr=m.get("snr_requested", e["key"]), peak=m.get("peak_target"),
                        src=m.get("noise_src", ""), firearm=fw[0] if len(fw) == 1 else ("+".join(fw) if fw else ""),
                        in_sisnr=e["sisnr_in"], out_sisnr=e["sisnr"], dsisnr=e["sisnr"] - e["sisnr_in"], stoi=e["stoi"],
                        pesq=e["pesq"] if e["pesq"] == e["pesq"] else float("nan")))
    return out


def stat(rows, metric):
    v = np.array([r[metric] for r in rows], float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    idx = _rng.integers(0, len(v), (2000, len(v)))
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(v)


def fmt(rows, metric, prec=2):
    m, lo, hi, n = stat(rows, metric)
    return "-" if n == 0 else f"{m:+.{prec}f} [{lo:+.{prec}f}, {hi:+.{prec}f}]" if metric in ("dsisnr", "in_sisnr", "out_sisnr") else f"{m:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"


def write_csv(path, header, body):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)


def group(rows, keyf):
    d = defaultdict(list)
    for r in rows:
        d[keyf(r)].append(r)
    return d


def table_rows(rows, keyf, keys):
    g = group(rows, keyf)
    return [(k, len(g[k]), *[stat(g[k], m)[:3] for m in METRICS]) for k in keys if g.get(k)]


def p2():
    L = ["# Priority 2: Model H (robust Small) evaluation\n",
         "Protocol for every number here: **PyTorch pipeline, each clip level-normalised by its own RMS** (`scripts/evaluate.py --proc ours`). "
         "Values are means over clips with a 95% bootstrap interval in brackets. dSI-SNR = output SI-SNR minus input SI-SNR (both against the "
         "dereverberated clean target). PESQ is wideband. The deployed ONNX path is evaluated separately (Priority 3).\n"]
    models = [("H", "Model H"), ("gtcrn", "pretrained GTCRN"), ("none", "no processing")]
    sets = {}
    for S in "AB":
        for sub in ("main", "weapon", "speechonly"):
            for t, _ in models:
                p = FT / f"P2_{t}_final{S}_{sub}.json"
                if p.exists():
                    sets[(t, f"final_{S}", sub)] = load(p, f"data/generated/final_{S}/{sub}")
    for t, name, d in (("H", "unseen", "unseen"), ("H", "test_speechonly", "test_speechonly")):
        p = FT / f"P2_H_{name}.json"
        if p.exists():
            sets[(t, name, "-")] = load(p, f"data/generated/{d}")
    for t in ("none", "gtcrn"):                                    # earlier same-protocol results, for context on the existing sets
        p = ROOT / "results" / f"final_{t}_unseen.json"
        if p.exists():
            sets[(t, "unseen", "-")] = load(p, "data/generated/unseen")
    csv_over = []

    # ---- 1. overall, fresh sets
    L.append("## 1. Fresh untouched final sets, main subset (1,200 clips each)\n")
    for S in "AB":
        L += [f"### final_{S}\n", "| model | in SI-SNR | out SI-SNR | dSI-SNR | STOI | PESQ |", "|---|---|---|---|---|---|"]
        for t, name in models:
            r = sets.get((t, f"final_{S}", "main"))
            if r:
                L.append(f"| {name} | " + " | ".join(fmt(r, m_) for m_ in METRICS) + " |")
                csv_over.append([f"final_{S}", "main", t, len(r)] + [x for m_ in METRICS for x in stat(r, m_)[:3]])
        L.append("")

    # ---- 2. per class and per SNR (Model H, fresh sets)
    for S in "AB":
        r = sets.get(("H", f"final_{S}", "main"))
        if not r:
            continue
        L += [f"## 2{'ab'[S == 'B']}. final_{S}: Model H per noise class\n", "| class | n | in SI-SNR | out SI-SNR | dSI-SNR | STOI | PESQ |", "|---|---|---|---|---|---|---|"]
        g = group(r, lambda x: x["cls"])
        for c in CLASSES:
            L.append(f"| {c} | {len(g[c])} | " + " | ".join(fmt(g[c], m_) for m_ in METRICS) + " |")
        write_csv(FT / f"P2_final{S}_H_per_class.csv", ["class", "n"] + [f"{m_}_{k}" for m_ in METRICS for k in ("mean", "lo", "hi")],
                  [[k, n] + [x for s in stats for x in s] for k, n, *stats in table_rows(r, lambda x: x["cls"], CLASSES)])
        L += ["", f"### final_{S}: Model H per input SNR\n", "| input SNR (dB) | n | in SI-SNR | out SI-SNR | dSI-SNR | STOI | PESQ |", "|---|---|---|---|---|---|---|"]
        g = group(r, lambda x: x["snr"])
        for s in SNRS:
            L.append(f"| {s:+d} | {len(g[s])} | " + " | ".join(fmt(g[s], m_) for m_ in METRICS) + " |")
        write_csv(FT / f"P2_final{S}_H_per_snr.csv", ["snr", "n"] + [f"{m_}_{k}" for m_ in METRICS for k in ("mean", "lo", "hi")],
                  [[k, n] + [x for s in stats for x in s] for k, n, *stats in table_rows(r, lambda x: x["snr"], SNRS)])
        L.append("")
    write_csv(FT / "P2_overall.csv", ["set", "subset", "model", "n"] + [f"{m_}_{k}" for m_ in METRICS for k in ("mean", "lo", "hi")], csv_over)

    # ---- 3. unseen noise
    r = sets.get(("H", "unseen", "-"))
    if r:
        L += ["## 3. Existing unseen-noise set (720 clips: shifted synthetic parameters + held-out shotgun recordings)\n",
              "| model | in SI-SNR | out SI-SNR | dSI-SNR | STOI | PESQ |", "|---|---|---|---|---|---|"]
        for t, name in models:
            x = sets.get((t, "unseen", "-"))
            if x:
                L.append(f"| {name} | " + " | ".join(fmt(x, m_) for m_ in METRICS) + " |")
        L += ["", "Model H per class on the unseen set:\n", "| class | n | dSI-SNR | STOI | PESQ |", "|---|---|---|---|---|"]
        g = group(r, lambda x: x["cls"])
        for c in CLASSES:
            L.append(f"| {c} | {len(g[c])} | {fmt(g[c], 'dsisnr')} | {fmt(g[c], 'stoi')} | {fmt(g[c], 'pesq')} |")
        L.append("")

    # ---- 4. speech-only
    L += ["## 4. Speech-only sets (no noise; speech leaks into the reference mic)\n",
          "Input SI-SNR is the ceiling set by the dereverberating target, so the relevant number is how far the output falls below it.\n",
          "| set | model | in SI-SNR | out SI-SNR | dSI-SNR | STOI | PESQ |", "|---|---|---|---|---|---|---|"]
    for key, label in ((("final_A", "speechonly"), "final_A"), (("final_B", "speechonly"), "final_B"), (("test_speechonly", "-"), "earlier test_speechonly")):
        for t, name in models:
            x = sets.get((t, key[0], key[1]))
            if x:
                L.append(f"| {label} | {name} | " + " | ".join(fmt(x, m_) for m_ in METRICS) + " |")
    L.append("")

    # ---- 5. cross-firearm
    L += ["## 5. Cross-firearm (gunshot-class clips that use a REAL recording)\n",
          "The shotgun (12-gauge Remington) was excluded from all training; Glock 17, AR-556 and .38 were in training (different shots). "
          "Fresh sets: firearm read from the recording actually drawn. Unseen set: every real gunshot is a shotgun by construction.\n",
          "| set | firearm | n | model | dSI-SNR | STOI | PESQ |", "|---|---|---|---|---|---|---|"]
    for t, name in models:
        x = sets.get((t, "unseen", "-"))
        if x:
            xs = [r for r in x if r["cls"] == "gunshot" and r["src"] == "real"]
            L.append(f"| unseen | shotgun (never trained) | {len(xs)} | {name} | {fmt(xs, 'dsisnr')} | {fmt(xs, 'stoi')} | {fmt(xs, 'pesq')} |")
    for S in "AB":
        for t, name in models:
            x = sets.get((t, f"final_{S}", "main"))
            if not x:
                continue
            g = group([r for r in x if r["cls"] == "gunshot" and r["src"] == "real"], lambda r: r["firearm"])
            for fam in sorted(g):
                L.append(f"| final_{S} | {fam} | {len(g[fam])} | {name} | {fmt(g[fam], 'dsisnr')} | {fmt(g[fam], 'stoi')} | {fmt(g[fam], 'pesq')} |")
    L.append("")

    # ---- 6. weapon / extreme SPL
    L += ["## 6. Close-range weapon sets: metrics by peak level (dB SPL at the primary microphone)\n",
          "**Do not read dSI-SNR alone at high peaks**: the clipped burst dominates the energy, so a model that only removes the burst "
          "scores a large dSI-SNR while intelligibility (STOI) stays low.\n"]
    for S in "AB":
        L += [f"### final_{S} (gunshot + artillery pooled, 40 clips per peak level)\n", "| model | metric | " + " | ".join(str(p) for p in PEAKS) + " |", "|---|---|" + "---|" * len(PEAKS)]
        for t, name in models:
            x = sets.get((t, f"final_{S}", "weapon"))
            if not x:
                continue
            g = group(x, lambda r: r["peak"])
            for m_, lab in (("dsisnr", "dSI-SNR"), ("stoi", "STOI"), ("pesq", "PESQ")):
                L.append(f"| {name} | {lab} | " + " | ".join(f"{stat(g[p], m_)[0]:+.1f}" if m_ == "dsisnr" else f"{stat(g[p], m_)[0]:.2f}" for p in PEAKS) + " |")
        L.append("")
    # earlier weapon set (Model H had been run on it)
    ew = ROOT / "results" / "final_H_robust_s0_testpeak.json"
    if ew.exists():
        x = load(ew, "data/generated/test_peak")
        g = group(x, lambda r: r["peak"])
        L += ["### earlier weapon set (test_peak, 240 clips; used before, not untouched)\n", "| metric | " + " | ".join(str(p) for p in PEAKS) + " |", "|---|" + "---|" * len(PEAKS)]
        for m_, lab in (("dsisnr", "dSI-SNR"), ("stoi", "STOI"), ("pesq", "PESQ")):
            L.append(f"| Model H {lab} | " + " | ".join(f"{stat(g[p], m_)[0]:+.1f}" if m_ == "dsisnr" else f"{stat(g[p], m_)[0]:.2f}" for p in PEAKS) + " |")
    rows_csv = []
    for S in "AB":
        for t, _ in models:
            x = sets.get((t, f"final_{S}", "weapon"))
            if x:
                g = group(x, lambda r: r["peak"])
                for p in PEAKS:
                    rows_csv.append([f"final_{S}", t, p, len(g[p])] + [stat(g[p], m_)[0] for m_ in METRICS])
    write_csv(FT / "P2_weapon_per_peak.csv", ["set", "model", "peak_dB_SPL", "n"] + METRICS, rows_csv)
    (FT / "P2_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", FT / "P2_summary.md")


if __name__ == "__main__":
    {"p2": p2}[sys.argv[1]]()
