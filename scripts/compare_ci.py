"""Paired comparison of two evaluation runs with bootstrap 95% confidence intervals.

  python scripts/compare_ci.py B=results/val_smoke.json A=results/val_smokemono.json

Prints, per class and overall, the mean difference (first minus second) in dSI-SNR, STOI and PESQ, with a
95% bootstrap interval over examples (paired by file name). An interval that excludes 0 means the
difference is unlikely to be evaluation noise for THIS test set; it says nothing about run-to-run training
variance, which needs several seeds.
"""
import argparse
import json

import numpy as np


def load(path):
    return {p["file"]: p for p in json.load(open(path))}


def ci(diff, rng, n=2000):
    idx = rng.integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(1)
    return diff.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--min-key", type=float)
    ap.add_argument("--max-key", type=float)
    x = ap.parse_args()
    (la, pa), (lb, pb) = x.a.split("=", 1), x.b.split("=", 1)
    A, B = load(pa), load(pb)
    files = [f for f in A if f in B and (x.min_key is None or A[f]["key"] >= x.min_key) and (x.max_key is None or A[f]["key"] <= x.max_key)]
    rng = np.random.default_rng(0)
    print(f"{la} minus {lb}   (n={len(files)} paired examples; * = 95% interval excludes 0)")
    print(f"{'class':<11}{'n':>5} | {'dSI-SNR (dB)':^24} | {'STOI':^24} | {'PESQ':^24}")
    for cls in sorted({A[f]["cls"] for f in files}) + ["ALL"]:
        fs = [f for f in files if cls == "ALL" or A[f]["cls"] == cls]
        cells = []
        for k in ("dsisnr", "stoi", "pesq"):
            get = (lambda p: p["sisnr"] - p["sisnr_in"]) if k == "dsisnr" else (lambda p, k=k: p[k])
            d = np.array([get(A[f]) - get(B[f]) for f in fs], float)
            d = d[~np.isnan(d)]
            m, lo, hi = ci(d, rng)
            cells.append(f"{m:+6.3f} [{lo:+6.3f},{hi:+6.3f}]{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"{cls:<11}{len(fs):>5} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
