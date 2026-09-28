"""Class x input-SNR target matrix, and PyTorch-vs-ONNX comparison, from per-clip results.

  python scripts/targets_matrix.py matrix <results.json> <manifest_dir> <label>
  python scripts/targets_matrix.py compare <pytorch.json> <deployed.json> <manifest_dir> <label>

Targets (statement: SNR > 15 dB, STOI > 0.85, PESQ > 2.5). The SNR target is shown under BOTH readings because the
statement is ambiguous:
    dSNR : delta SI-SNR (output minus input) > 15 dB   <- the strict reading, used for the primary ALL column
    oSNR : output SI-SNR > 15 dB                        <- the reading used earlier in this project
A cell (25 clips) is judged on the mean with a 95% bootstrap interval:
    PASS      the whole interval is above the threshold
    FAIL      the whole interval is below the threshold
    MARGINAL  the interval contains the threshold (the mean may be above or below it)
ALL = PASS if every target passes, FAIL if any target fails, otherwise MARGINAL.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_analysis import CLASSES, SNRS, ROOT, group, load, stat   # noqa: E402

THR = dict(dsisnr=15.0, out_sisnr=15.0, stoi=0.85, pesq=2.5)


def verdict(rows, metric):
    m, lo, hi, n = stat(rows, metric)
    t = THR[metric]
    return ("PASS" if lo > t else "FAIL" if hi < t else "MARG"), m, lo, hi


def combine(vs):
    if all(v == "PASS" for v in vs):
        return "PASS"
    if any(v == "FAIL" for v in vs):
        return "FAIL"
    return "MARG"


def matrix(rows):
    g = group(rows, lambda r: (r["cls"], r["snr"]))
    out = []
    for c in CLASSES:
        for s in SNRS:
            cell = g.get((c, s), [])
            if not cell:
                continue
            vd, md, ld, hd = verdict(cell, "dsisnr")
            vo, mo, lo_, ho = verdict(cell, "out_sisnr")
            vs, ms, ls, hs = verdict(cell, "stoi")
            vp, mp, lp, hp = verdict(cell, "pesq")
            out.append(dict(cls=c, snr=s, n=len(cell), dsisnr=md, dsisnr_lo=ld, dsisnr_hi=hd, v_dsnr=vd,
                            out_sisnr=mo, v_osnr=vo, stoi=ms, stoi_lo=ls, v_stoi=vs, pesq=mp, pesq_lo=lp, v_pesq=vp,
                            all_strict=combine([vd, vs, vp]), all_alt=combine([vo, vs, vp])))
    return out


def render(cells, label):
    sym = {"PASS": "PASS", "FAIL": "fail", "MARG": "marg"}
    L = [f"### {label}\n",
         "| class | input SNR | n | dSI-SNR | dSNR>15 | out SI-SNR | oSNR>15 | STOI | >0.85 | PESQ | >2.5 | **ALL (strict, dSNR)** | ALL (alt, oSNR) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        L.append(f"| {c['cls']} | {c['snr']:+d} | {c['n']} | {c['dsisnr']:+.1f} | {sym[c['v_dsnr']]} | {c['out_sisnr']:+.1f} | {sym[c['v_osnr']]} | "
                 f"{c['stoi']:.3f} | {sym[c['v_stoi']]} | {c['pesq']:.2f} | {sym[c['v_pesq']]} | **{sym[c['all_strict']]}** | {sym[c['all_alt']]} |")
    n = len(cells)
    cnt = lambda k, v: sum(c[k] == v for c in cells)
    L += ["", f"Cells: {n}. ALL strict (dSNR): PASS {cnt('all_strict', 'PASS')}, MARGINAL {cnt('all_strict', 'MARG')}, FAIL {cnt('all_strict', 'FAIL')}. "
          f"ALL alt (oSNR): PASS {cnt('all_alt', 'PASS')}, MARGINAL {cnt('all_alt', 'MARG')}, FAIL {cnt('all_alt', 'FAIL')}.\n",
          "Per input SNR (cells passing ALL, strict / alt): " + ", ".join(
              f"{s:+d} dB: {sum(c['all_strict'] == 'PASS' for c in cells if c['snr'] == s)}/{sum(c['all_alt'] == 'PASS' for c in cells if c['snr'] == s)} of {sum(c['snr'] == s for c in cells)}" for s in SNRS) + "\n"]
    return "\n".join(L)


def grid(cells):
    """Compact class x SNR grid, letters: D/O = dSNR/oSNR, T = STOI, P = PESQ ('.' not PASS)."""
    L = ["| class | " + " | ".join(f"{s:+d}" for s in SNRS) + " |", "|---|" + "---|" * len(SNRS)]
    for c in CLASSES:
        row = []
        for s in SNRS:
            x = next((k for k in cells if k["cls"] == c and k["snr"] == s), None)
            row.append("-" if not x else ("D" if x["v_dsnr"] == "PASS" else ".") + ("O" if x["v_osnr"] == "PASS" else ".") + ("T" if x["v_stoi"] == "PASS" else ".") + ("P" if x["v_pesq"] == "PASS" else "."))
        L.append(f"| {c} | " + " | ".join(row) + " |")
    return "\n".join(L)


def write_csv(path, cells):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(cells)


def compare(a, b, label_a="PyTorch", label_b="deployed"):
    """Paired per-clip differences (b - a) overall and per cell."""
    A, B = {r["file"]: r for r in a}, {r["file"]: r for r in b}
    files = [f for f in A if f in B]
    L = [f"Paired over {len(files)} clips ({label_b} minus {label_a}); mean [95% bootstrap CI]:\n", "| metric | difference |", "|---|---|"]
    for m_, lab, prec in (("dsisnr", "dSI-SNR (dB)", 3), ("stoi", "STOI", 4), ("pesq", "PESQ", 3)):
        d = np.array([B[f][m_] - A[f][m_] for f in files], float)
        d = d[~np.isnan(d)]
        idx = np.random.default_rng(0).integers(0, len(d), (2000, len(d)))
        mm = d[idx].mean(1)
        L.append(f"| {lab} | {d.mean():+.{prec}f} [{np.percentile(mm, 2.5):+.{prec}f}, {np.percentile(mm, 97.5):+.{prec}f}] (max |per-clip difference| {np.abs(d).max():.{prec}f}) |")
    return "\n".join(L), files


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "matrix":
        rows = load(sys.argv[2], sys.argv[3])
        cells = matrix(rows)
        print(render(cells, sys.argv[4]))
        print(grid(cells))
    elif cmd == "compare":
        a, b = load(sys.argv[2], sys.argv[4]), load(sys.argv[3], sys.argv[4])
        txt, _ = compare(a, b)
        print(txt)
