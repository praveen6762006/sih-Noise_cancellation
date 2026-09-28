"""P5 (fair model-size comparison) and P6 (Model H seed variance) tables.

  python scripts/size_variance_analysis.py

P5: Tiny (from scratch), Small (from scratch, NEW), Base (from scratch): identical training conditions (6,000 steps, batch 16,
    lr 1e-3, no reference dropout, seed 0); PyTorch protocol on the EARLIER test set. Small warm-started from the pretrained
    GTCRN (C) is shown separately to isolate the effect of initialisation. Robust-configuration models (I, H) are shown
    separately because their training conditions differ (dropout 0.2, +/-8 dB mismatch).
P6: Model H seeds 0, 1, 2 (same configuration): per-seed means on the earlier test set and on the fresh final sets,
    then mean, sample standard deviation and a t-based 95% interval across seeds (n = 3, so the interval is wide).
    The extra seeds were NOT used to select anything: seed 0 remains the deployment model.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_analysis import ROOT, load, stat   # noqa: E402

R = ROOT / "results"
SV = R / "size_variance"
T_CRIT_2DOF = 4.303


def paired(a, b, metric, rng=np.random.default_rng(0)):
    A, B = {r["file"]: r for r in a}, {r["file"]: r for r in b}
    d = np.array([A[f][metric] - B[f][metric] for f in A if f in B], float)
    d = d[~np.isnan(d)]
    idx = rng.integers(0, len(d), (2000, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cell(rows, m):
    mu, lo, hi, n = stat(rows, m)
    return f"{mu:+.2f} [{lo:+.2f}, {hi:+.2f}]" if m == "dsisnr" else (f"{mu:.3f} [{lo:.3f}, {hi:.3f}]" if m == "stoi" else f"{mu:.2f} [{lo:.2f}, {hi:.2f}]")


def main():
    static = {r["model"]: r for r in json.load(open(R / "complexity" / "static.json"))}
    L = ["# Priorities 5 and 6: model size and seed variance\n"]

    # ------------------------------------------------------------------ P5
    tst = "data/generated/test"
    P5 = [("Tiny (8,389 params), scratch", "Tiny", R / "final_T_tiny_s0_test.json"),
          ("Small (24,629), scratch  [NEW]", "Small", SV / "J_small_scratch_s0_test.json"),
          ("Base (80,725), scratch", "Base", R / "final_L_base_s0_test.json")]
    ref = [("Small (24,629), warm-started from pretrained GTCRN (C)", "Small", R / "final_C_full_s0_test.json"),
           ("Tiny, robust config (I: dropout 0.2, +/-8 dB), scratch", "Tiny", R / "final_I_tinyrobust_s0_test.json"),
           ("Small, robust config, warm-started (H, deployment)", "Small", R / "final_H_robust_s0_test.json")]
    data = {}
    L += ["## P5. Fair size comparison (earlier test set, 1,200 clips, PyTorch protocol; mean [95% CI])\n",
          "Same conditions for the three rows: from scratch, 6,000 steps, batch 16, seed 0, inputs = primary + reference + flag, no reference dropout.\n",
          "| model | MMACs/s (computed) | dSI-SNR (dB) | STOI | PESQ |", "|---|---|---|---|---|"]
    csv_rows = []
    for lab, key, p in P5 + ref:
        if not p.exists():
            L.append(f"| {lab} | - | *missing: {p.name}* | | |")
            continue
        rows = load(p, tst)
        data[lab] = rows
        L.append(f"| {lab} | {static[key]['mmacs_per_s_executed']} | {cell(rows, 'dsisnr')} | {cell(rows, 'stoi')} | {cell(rows, 'pesq')} |")
        for m in ("dsisnr", "stoi", "pesq"):
            mu, lo, hi, n = stat(rows, m)
            csv_rows.append([lab, key, static[key]["parameters"], static[key]["mmacs_per_s_executed"], m, n, mu, lo, hi])
        if lab.startswith("Base"):
            L.append("| *(rows above: like-for-like; rows below: different initialisation or training configuration)* | | | | |")
    with open(SV / "P5_size_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "size", "parameters", "mmacs_per_s", "metric", "n", "mean", "ci_lo", "ci_hi"])
        w.writerows(csv_rows)
    tiny, small, base = (data.get(P5[i][0]) for i in range(3))
    warm = data.get(ref[0][0])
    if tiny and small and base:
        L += ["", "Paired differences (first minus second), same clips:\n", "| comparison | dSI-SNR | STOI | PESQ |", "|---|---|---|---|"]
        for name, a, b in (("Small-scratch - Tiny", small, tiny), ("Base - Small-scratch", base, small), ("Base - Tiny", base, tiny), ("Small warm-start (C) - Small-scratch", warm, small)):
            if a and b:
                d = [paired(a, b, m) for m in ("dsisnr", "stoi", "pesq")]
                L.append(f"| {name} | " + " | ".join(f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]" for x in d) + " |")
    L.append("")

    # ------------------------------------------------------------------ P6
    seeds = {0: dict(test=R / "final_H_robust_s0_test.json", A=R / "final_test" / "P2_H_finalA_main.json", B=R / "final_test" / "P2_H_finalB_main.json"),
             1: dict(test=SV / "H_robust_s1_test.json", A=R / "final_test" / "P6_H_s1_finalA_main.json", B=R / "final_test" / "P6_H_s1_finalB_main.json"),
             2: dict(test=SV / "H_robust_s2_test.json", A=R / "final_test" / "P6_H_s2_finalA_main.json", B=R / "final_test" / "P6_H_s2_finalB_main.json")}
    man = dict(test="data/generated/test", A="data/generated/final_A/main", B="data/generated/final_B/main")
    names = dict(test="earlier test set", A="fresh final_A main", B="fresh final_B main")
    L += ["## P6. Model H: independent training seeds (same configuration: warm start, reference dropout 0.2, gain jitter +/-8 dB, 6,000 steps)\n",
          "Seed 0 is the deployment model; seeds 1 and 2 were trained afterwards only to measure training variance and were not used for any selection.\n"]
    for k in ("test", "A", "B"):
        per = {}
        for sd, ps in seeds.items():
            if ps[k].exists():
                per[sd] = load(ps[k], man[k])
        if len(per) < 2:
            L.append(f"### {names[k]}: fewer than two seeds available yet\n")
            continue
        L += [f"### {names[k]}\n", "| seed | dSI-SNR | STOI | PESQ |", "|---|---|---|---|"]
        vals = {m: [] for m in ("dsisnr", "stoi", "pesq")}
        for sd, rows in sorted(per.items()):
            L.append(f"| {sd} | " + " | ".join(cell(rows, m) for m in vals) + " |")
            for m in vals:
                vals[m].append(stat(rows, m)[0])
        n = len(per)
        tc = {2: 12.706, 3: 4.303}[n]
        summ = []
        for m in vals:
            v = np.array(vals[m])
            sd_ = v.std(ddof=1)
            half = tc * sd_ / np.sqrt(n)
            summ.append(f"{v.mean():+.3f} +/- {sd_:.3f} (SD); 95% t-interval [{v.mean() - half:+.3f}, {v.mean() + half:+.3f}]; range {v.min():+.3f} to {v.max():+.3f}" if m == "dsisnr"
                        else f"{v.mean():.4f} +/- {sd_:.4f} (SD); 95% t-interval [{v.mean() - half:.4f}, {v.mean() + half:.4f}]; range {v.min():.4f} to {v.max():.4f}")
        L += [f"| **mean over {n} seeds** | " + " | ".join(summ) + " |", ""]
    (SV / "SIZE_VARIANCE_SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
