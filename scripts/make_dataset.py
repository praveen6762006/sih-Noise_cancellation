"""Generate a noisy/reference/clean dataset.

  python scripts/make_dataset.py --split test  --per-cell 50  --out data/generated/test
  python scripts/make_dataset.py --split train --n 20000      --out data/generated/train

Each example is a 3-channel 16-bit WAV: [noisy primary, reference, clean target],
plus a line in manifest.jsonl with the scene parameters. The `test`, `val` and `unseen`
splits use a fixed class x SNR grid; `train` samples classes and SNR at random.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anc.data.mixer import Context, make_example          # noqa: E402
from anc.data.noise_bank import NoiseBank                  # noqa: E402
from anc.data.scene import RirPool                         # noqa: E402

from anc.data.speech import list_speech, load_speech      # noqa: E402

_CTX = None
_SPEECH = None


def _init(cfg, split, pool_path, speech_files):
    global _CTX, _SPEECH
    bank = NoiseBank(cfg["noise_dir"], cfg["use_real_noise_prob"], cfg["holdout_tokens"])
    _CTX = Context(cfg=cfg, bank=bank, rirs=RirPool.load(pool_path), split=split)
    _SPEECH = speech_files


def _work(job):
    idx, cls, snr, peak, seed, out_dir = job
    rng = np.random.default_rng(seed)
    n = int(_CTX.cfg["clip_seconds"] * _CTX.cfg["fs"])
    speech, path, spk = load_speech(_SPEECH, n, rng)
    ex = make_example(speech, cls, snr, _CTX, rng, peak_spl_db=peak)
    name = f"{idx:06d}.wav"
    sf.write(Path(out_dir) / name, np.stack([ex["noisy"], ex["ref"], ex["clean"]], axis=1), 16000, subtype="PCM_16")
    m = ex["meta"]
    m.update(file=name, speech_file=path, speaker=spk, seed=int(seed))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "dataset.yaml"))
    ap.add_argument("--split", required=True, choices=["train", "val", "test", "unseen"])
    ap.add_argument("--n", type=int, help="random examples (train)")
    ap.add_argument("--per-cell", type=int, help="examples per class x SNR bin (fixed grid)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--speech-root", nargs="*", help="override the speech directories")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", nargs="*", help="restrict to these classes")
    ap.add_argument("--snr-bins", nargs="*", type=float, help="override the fixed SNR grid, e.g. 60 for a speech-only set")
    ap.add_argument("--peak-grid", nargs="*", type=float,
                    help="peak SPL bins (dB) for gunshot/artillery, e.g. 110 120 130 140 150 160")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split_seed = {"train": 1, "val": 2, "test": 3, "unseen": 4}[a.split]

    speech = list_speech(a.speech_root or cfg["speech_dirs"][a.split])
    if not speech:
        sys.exit(f"no speech files found for split '{a.split}'")

    pool_path = ROOT / "data" / f"rir_pool_{a.split}.npz"
    if not pool_path.exists():
        print(f"building RIR pool -> {pool_path}")
        RirPool.build(cfg["rir"]["n_speech"], cfg["rir"]["n_noise_per_env"], split_seed, a.workers).save(pool_path)

    classes = a.classes or cfg["classes"]
    base = np.random.default_rng(a.seed * 1000 + split_seed)
    jobs = []
    if a.peak_grid:
        # close-range weapon scenario: level set by peak SPL at the primary mic, not by energy SNR
        assert a.per_cell, "--peak-grid needs --per-cell"
        for c in a.classes or ["gunshot", "artillery"]:
            for pk in a.peak_grid:
                for _ in range(a.per_cell):
                    jobs.append((len(jobs), c, 0.0, float(pk), int(base.integers(2 ** 31)), str(out)))
    elif a.per_cell:
        for c in classes:
            for s in (a.snr_bins or cfg["snr_bins"]):
                for _ in range(a.per_cell):
                    jobs.append((len(jobs), c, float(s), None, int(base.integers(2 ** 31)), str(out)))
    else:
        assert a.n, "give --n or --per-cell"
        lo, hi = cfg["snr_db"]
        for _ in range(a.n):
            jobs.append((len(jobs), str(base.choice(classes)), float(base.uniform(lo, hi)), None,
                         int(base.integers(2 ** 31)), str(out)))

    print(f"{len(jobs)} examples, {len(speech)} speech files, split={a.split}")
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(cfg, a.split, pool_path, speech)) as ex, \
            open(out / "manifest.jsonl", "w") as mf:
        for i, m in enumerate(ex.map(_work, jobs, chunksize=8)):
            mf.write(json.dumps(m) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
