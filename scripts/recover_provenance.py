"""Recover exactly which source files every earlier test-type clip used, so a new test set can exclude them.

  python scripts/recover_provenance.py

The manifests record the speech file and a per-clip seed but not which gunshot recording was drawn. Generation is
deterministic, so each clip is regenerated from its seed and its audio compared with the saved WAV; the gunshot files
it draws are logged. Sets covered: the earlier `test` and `test_peak` (test partition), `unseen` (shotgun partition),
`val`, `val_peak` (validation partition) and the speech-only sets.

Output: results/final_test/earlier_sources.json  (speech utterances used per set, gunshot files used per set, and the
sizes of what is left over). Fails loudly if any clip does not regenerate exactly (then provenance would be unreliable).
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from anc.data.mixer import Context, make_example          # noqa: E402
from anc.data.noise_bank import NoiseBank, _event_id       # noqa: E402
from anc.data.scene import RirPool                         # noqa: E402
from anc.data.speech import list_speech, load_speech       # noqa: E402

cfg = yaml.safe_load(open(ROOT / "configs" / "dataset.yaml"))
LSB = 1.5 / 32767


class LoggingList(list):
    log = []

    def __getitem__(self, i):
        v = super().__getitem__(i)
        LoggingList.log.append(str(v))
        return v


# (dataset dir, rir/noise split, per-cell, peak grid, classes, snr bins) exactly as the datasets were generated
SETS = [
    ("test", "test", 25, None, cfg["classes"], cfg["snr_bins"]),
    ("test_peak", "test", 20, [110, 120, 130, 140, 150, 160], ["gunshot", "artillery"], None),
    ("test_speechonly", "test", 50, None, ["engine"], [60]),
    ("unseen", "unseen", 15, None, cfg["classes"], cfg["snr_bins"]),
    ("val", "val", 10, None, cfg["classes"], cfg["snr_bins"]),
    ("val_peak", "val", 20, [110, 120, 130, 140, 150, 160], ["gunshot", "artillery"], None),
    ("val_speechonly", "val", 100, None, ["engine"], [60]),
]


def main():
    out = dict(speech_used={}, gunshot_files_used={}, exact_reproduction={})
    for dset, split, pc, pg, classes, bins in SETS:
        rows = [json.loads(l) for l in open(ROOT / f"data/generated/{dset}/manifest.jsonl")]
        bank = NoiseBank(cfg["noise_dir"], cfg["use_real_noise_prob"], cfg["holdout_tokens"])
        for k in list(bank.files):
            bank.files[k] = LoggingList(bank.files[k])
        ctx = Context(cfg=cfg, bank=bank, rirs=RirPool.load(ROOT / f"data/rir_pool_{split}.npz"), split=split)
        files = list_speech(cfg["speech_dirs"][split])
        used, bad = set(), 0
        for idx, m in enumerate(rows):
            rng = np.random.default_rng(m["seed"])
            speech, path, _ = load_speech(files, int(cfg["clip_seconds"] * cfg["fs"]), rng)
            if pg:
                c, snr, peak = classes[idx // (len(pg) * pc)], 0.0, float(pg[(idx // pc) % len(pg)])
            else:
                c, snr, peak = classes[idx // (len(bins) * pc)], float(bins[(idx // pc) % len(bins)]), None
            LoggingList.log = []
            ex = make_example(speech, c, snr, ctx, rng, peak_spl_db=peak)
            stored = sf.read(ROOT / f"data/generated/{dset}/{m['file']}", dtype="float64")[0][:, 0]
            q = np.round(np.clip(ex["noisy"].astype(np.float64), -1, 1) * 32767) / 32767
            bad += (np.abs(q - stored).max() > LSB) or (Path(path) != Path(m["speech_file"]))
            used.update(u for u in LoggingList.log if "gunshot" in u)
        out["speech_used"][dset] = sorted({m["speech_file"] for m in rows})
        out["gunshot_files_used"][dset] = sorted(used)
        out["exact_reproduction"][dset] = dict(clips=len(rows), mismatches=int(bad))
        print(f"{dset:16s} {len(rows):5d} clips, mismatches {bad}, gunshot files drawn {len(used)}", flush=True)
        if bad:
            sys.exit(f"{dset}: {bad} clips did not regenerate exactly; provenance would be unreliable")

    # what is left over in the TEST partition (the only partition a final test may draw from)
    pool = NoiseBank(cfg["noise_dir"], 0.5, cfg["holdout_tokens"]).files[("gunshot", "test")]
    used_test = set(out["gunshot_files_used"]["test"]) | set(out["gunshot_files_used"]["test_peak"]) | set(out["gunshot_files_used"]["test_speechonly"])
    used_events = {_event_id(Path(u)) for u in used_test}
    free = [str(p) for p in pool if _event_id(p) not in used_events]
    out["gunshot_test_partition"] = dict(files=len(pool), shots=len({_event_id(p) for p in pool}), files_used=len(used_test),
                                         shots_used=len(used_events), free_files=free,
                                         free_shots=sorted({_event_id(Path(p)) for p in free}))
    used_speech = set()
    for d in ("test", "test_peak", "test_speechonly", "unseen"):
        used_speech |= set(out["speech_used"][d])
    out["test_clean_utterances"] = dict(total=2620, used_by_earlier_test_sets=len(used_speech))
    (ROOT / "results" / "final_test").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "results" / "final_test" / "earlier_sources.json", "w"))
    g = out["gunshot_test_partition"]
    print(f"\ntest-partition gunshots: {g['files']} files / {g['shots']} shots; earlier test sets used {g['files_used']} files / {g['shots_used']} shots; "
          f"never used: {len(g['free_files'])} files / {len(g['free_shots'])} shots")
    print(f"test-clean utterances used by earlier test-type sets: {len(used_speech)} of 2620")
    print("wrote results/final_test/earlier_sources.json")


if __name__ == "__main__":
    main()
