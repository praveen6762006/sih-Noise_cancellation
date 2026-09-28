"""Side-by-side comparison of several evaluate.py result files (per-class means over all bins,
plus means restricted to a range of the grouping key).

  python scripts/compare.py none=results/val_none.json gtcrn=results/val_gtcrn.json ours=results/val_ours.json
  python scripts/compare.py a=results/valpeak_gtcrn.json b=results/valpeak_ours.json --min-key 140
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=path.json")
    ap.add_argument("--min-key", type=float, help="only cells with grouping key >= this (SNR or peak SPL)")
    ap.add_argument("--max-key", type=float)
    a = ap.parse_args()

    data = {}
    for r in a.runs:
        label, path = r.split("=", 1)
        rows = [p for p in json.load(open(path))
                if (a.min_key is None or p["key"] >= a.min_key) and (a.max_key is None or p["key"] <= a.max_key)]
        by = defaultdict(list)
        for p in rows:
            by[p["cls"]].append(p)
            by["ALL"].append(p)
        data[label] = by

    labels = list(data)
    classes = sorted(set().union(*[set(d) for d in data.values()]) - {"ALL"}) + ["ALL"]
    head = f"{'class':<11}" + "".join(f"| {l:^22} " for l in labels)
    print(head)
    print(f"{'':<11}" + "".join("| dSI-SNR  STOI   PESQ  " for _ in labels))
    for c in classes:
        line = f"{c:<11}"
        for l in labels:
            v = data[l].get(c, [])
            if not v:
                line += "|" + " " * 24
                continue
            m = lambda f: np.nanmean([e[f] for e in v])
            line += f"| {m('sisnr') - m('sisnr_in'):>6.1f}  {m('stoi'):>5.2f}  {m('pesq'):>5.2f}  "
        print(line)


if __name__ == "__main__":
    main()
