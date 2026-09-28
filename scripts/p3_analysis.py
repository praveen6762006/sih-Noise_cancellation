"""P3: target verification on the DEPLOYED path, and PyTorch-vs-deployment comparison.

  python scripts/p3_analysis.py

Inputs: results/final_test/P2_H_final{A,B}_{main,weapon}.json (PyTorch protocol) and P3_onnx_*.json (deployed: numpy front end +
reference guard + ONNX Runtime + slow level control). Same clips in both, so comparisons are paired per clip.
Outputs: results/final_test/P3_summary.md, P3_matrix_<set>_<path>.csv.
Targets and verdict rules: see scripts/targets_matrix.py (PASS/MARGINAL/FAIL from the 95% bootstrap interval of the cell mean over 25 clips).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_analysis import CLASSES, PEAKS, ROOT, SNRS, group, load, stat   # noqa: E402
from targets_matrix import compare, grid, matrix, render, write_csv      # noqa: E402

FT = ROOT / "results" / "final_test"


def main():
    L = ["# Priority 3: target verification, deployed path vs PyTorch\n",
         "Model H. **Deployed path** = NumPy front end + reference guard + streaming ONNX Runtime + ~1.5 s automatic level control (what runs on the target). "
         "**PyTorch path** = per-clip RMS normalisation, offline STFT. The two are never averaged together; they are compared clip by clip.\n",
         "Target verdicts per class x input-SNR cell (25 clips): PASS = the 95% bootstrap interval of the cell mean is entirely above the threshold; "
         "FAIL = entirely below; MARG = the interval contains the threshold. Two readings of the SNR target are shown because the statement says only "
         "\"SNR > 15 dB\": **dSNR** = SI-SNR improvement > 15 dB (strict, primary), **oSNR** = output SI-SNR > 15 dB (looser, used earlier in this project). "
         "SI-SNR is measured against the dereverberated direct-path target.\n"]
    summary = {}
    for S in "AB":
        pt_json, on_json = FT / f"P2_H_final{S}_main.json", FT / f"P3_onnx_fp32_final{S}_main.json"
        if not (pt_json.exists() and on_json.exists()):
            L.append(f"## final_{S}: deployed-path results not available yet\n")
            continue
        man = f"data/generated/final_{S}/main"
        pt, on = load(pt_json, man), load(on_json, man)
        cp, co = matrix(pt), matrix(on)
        write_csv(FT / f"P3_matrix_final{S}_pytorch.csv", cp)
        write_csv(FT / f"P3_matrix_final{S}_onnx_fp32.csv", co)
        L += [f"## final_{S} ({'unused test-clean utterances' if S == 'A' else 'test-other speakers, all new'})\n",
              render(co, f"final_{S}: DEPLOYED (ONNX fp32) full target matrix"),
              "Compact grid, deployed path (letters = target PASSED: D = dSNR>15, O = oSNR>15, T = STOI>0.85, P = PESQ>2.5; '.' = not passed):\n", grid(co), "",
              render(cp, f"final_{S}: PyTorch full target matrix (for comparison)"), "Compact grid, PyTorch path:\n", grid(cp), ""]
        txt, files = compare(pt, on, "PyTorch", "deployed ONNX fp32")
        L += [f"### final_{S}: deployed minus PyTorch\n", txt, ""]
        # cell-level agreement
        diff_all_strict = sum(a["all_strict"] != b["all_strict"] for a, b in zip(cp, co))
        diff_all_alt = sum(a["all_alt"] != b["all_alt"] for a, b in zip(cp, co))
        diff_any = sum((a["v_dsnr"], a["v_osnr"], a["v_stoi"], a["v_pesq"]) != (b["v_dsnr"], b["v_osnr"], b["v_stoi"], b["v_pesq"]) for a, b in zip(cp, co))
        maxd = max(abs(a["dsisnr"] - b["dsisnr"]) for a, b in zip(cp, co))
        L += [f"Cell-level agreement ({len(cp)} cells): ALL-strict verdict differs in **{diff_all_strict}**, ALL-alt in **{diff_all_alt}**, any single-target verdict in **{diff_any}**; "
              f"largest cell-mean dSI-SNR difference {maxd:.2f} dB.\n"]
        summary[S] = dict(pass_strict=sum(c["all_strict"] == "PASS" for c in co), marg_strict=sum(c["all_strict"] == "MARG" for c in co),
                          pass_alt=sum(c["all_alt"] == "PASS" for c in co), marg_alt=sum(c["all_alt"] == "MARG" for c in co), cells=len(co),
                          pass_pt_strict=sum(c["all_strict"] == "PASS" for c in cp), pass_pt_alt=sum(c["all_alt"] == "PASS" for c in cp),
                          diff_strict=diff_all_strict, diff_alt=diff_all_alt)
        # INT8 on set A
        i8 = FT / f"P3_onnx_int8_final{S}_main.json"
        if i8.exists():
            q = load(i8, man)
            t2, _ = compare(on, q, "deployed fp32", "deployed INT8")
            cq = matrix(q)
            L += [f"### final_{S}: INT8 minus fp32 (deployed path)\n", t2,
                  f"\nINT8 ALL-strict PASS {sum(c['all_strict'] == 'PASS' for c in cq)}, ALL-alt PASS {sum(c['all_alt'] == 'PASS' for c in cq)} (fp32: {summary[S]['pass_strict']} / {summary[S]['pass_alt']}); "
                  f"ALL verdict differs from fp32 in {sum(a['all_strict'] != b['all_strict'] or a['all_alt'] != b['all_alt'] for a, b in zip(co, cq))} cells.\n"]
    # cells passing on both fresh sets
    if len(summary) == 2:
        both_strict, both_alt = [], []
        A = {(c["cls"], c["snr"]): c for c in matrix(load(FT / "P3_onnx_fp32_finalA_main.json", "data/generated/final_A/main"))}
        B = {(c["cls"], c["snr"]): c for c in matrix(load(FT / "P3_onnx_fp32_finalB_main.json", "data/generated/final_B/main"))}
        for k in A:
            if A[k]["all_strict"] == "PASS" and B[k]["all_strict"] == "PASS":
                both_strict.append(k)
            if A[k]["all_alt"] == "PASS" and B[k]["all_alt"] == "PASS":
                both_alt.append(k)
        L += ["## Verdict on the statement's targets (deployed path, fresh sets)\n",
              "| | final_A | final_B | PASS on both |", "|---|---|---|---|",
              f"| ALL, strict reading (dSNR>15 & STOI>0.85 & PESQ>2.5) | {summary['A']['pass_strict']} / {summary['A']['cells']} cells | {summary['B']['pass_strict']} / {summary['B']['cells']} | {len(both_strict)} |",
              f"| ALL, alt reading (oSNR>15 & STOI>0.85 & PESQ>2.5) | {summary['A']['pass_alt']} / {summary['A']['cells']} cells | {summary['B']['pass_alt']} / {summary['B']['cells']} | {len(both_alt)} |", "",
              "Cells that PASS ALL (alt reading) on both fresh sets: " + (", ".join(f"{c} {s:+d} dB" for c, s in both_alt) if both_alt else "none"),
              "", "Cells that PASS ALL (strict reading) on both fresh sets: " + (", ".join(f"{c} {s:+d} dB" for c, s in both_strict) if both_strict else "none"), "",
              "**All 48 cells do not pass; the statement's targets are met only in the cells listed above.** Wherever a cell is MARG the data cannot distinguish pass from fail at 25 clips.\n"]

    # weapon sets: deployed vs PyTorch by peak
    L += ["## Weapon sets: deployed vs PyTorch by peak level (mean over 40 clips per peak)\n", "| set | peak (dB SPL) | STOI PyTorch | STOI deployed | PESQ PyTorch | PESQ deployed | dSI-SNR PyTorch | dSI-SNR deployed |", "|---|---|---|---|---|---|---|---|"]
    for S in "AB":
        pj, oj = FT / f"P2_H_final{S}_weapon.json", FT / f"P3_onnx_fp32_final{S}_weapon.json"
        if pj.exists() and oj.exists():
            gp = group(load(pj, f"data/generated/final_{S}/weapon"), lambda r: r["peak"])
            go = group(load(oj, f"data/generated/final_{S}/weapon"), lambda r: r["peak"])
            for p in PEAKS:
                L.append(f"| final_{S} | {p} | {stat(gp[p], 'stoi')[0]:.3f} | {stat(go[p], 'stoi')[0]:.3f} | {stat(gp[p], 'pesq')[0]:.2f} | {stat(go[p], 'pesq')[0]:.2f} | "
                         f"{stat(gp[p], 'dsisnr')[0]:+.1f} | {stat(go[p], 'dsisnr')[0]:+.1f} |")
    (FT / "P3_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-40:]))


if __name__ == "__main__":
    main()
