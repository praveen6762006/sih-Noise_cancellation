"""Wideband PESQ (ITU-T P.862.2) for a generated dataset. Runs inside WSL, where the `pesq`
package builds; called from inspect_dataset.py via `wsl`.

  python3 pesq_score.py <dataset_dir> [--channel 0]

Prints one JSON object {filename: pesq or null}. The clean target is channel 2 of each WAV;
`--channel` picks the degraded signal (0 = noisy primary). To score enhanced audio, pass
--deg-dir with WAVs of the same names holding the enhanced signal in channel 0.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from pesq import pesq, NoUtterancesError


def score(args):
    ref_path, deg_path, channel = args
    ref = sf.read(ref_path, dtype="float32")[0]
    clean = ref[:, 2]
    deg = sf.read(deg_path, dtype="float32", always_2d=True)[0][:, channel]
    n = min(len(clean), len(deg))
    try:
        return Path(ref_path).name, float(pesq(16000, clean[:n], deg[:n], "wb"))
    except (NoUtterancesError, Exception):
        return Path(ref_path).name, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--deg-dir")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    d = Path(a.dir)
    dd = Path(a.deg_dir) if a.deg_dir else d
    files = [json.loads(l)["file"] for l in open(d / "manifest.jsonl")]
    jobs = [(str(d / f), str(dd / f), a.channel) for f in files]
    with Pool(a.workers) as p:
        res = dict(p.map(score, jobs, chunksize=4))
    json.dump(res, sys.stdout)


if __name__ == "__main__":
    main()
