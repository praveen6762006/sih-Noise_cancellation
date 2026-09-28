"""Generate the FRESH final test sets, frozen and checksummed. Never used for tuning or model selection.

  python scripts/make_final_test.py            # builds final_A and final_B (skips a set that already exists)

  final_A : speech = test-clean utterances NEVER used by any earlier test-type set
            (same 40 speakers as the earlier test, independent of training and validation)
  final_B : speech = LibriSpeech test-other (speakers that appear nowhere else)
  gunshots (both): only the test-partition recordings NEVER drawn by any earlier set (results/final_test/earlier_sources.json)
  room impulse responses (new pool, new seed), synthetic noise draws, SNR/peak draws and speech crops: new seeds

Each set has three subsets, mirroring the earlier protocol so results are comparable:
  main      8 classes x 6 SNR bins (-10..+15 dB) x 25 clips = 1200
  weapon    gunshot/artillery x 6 peak levels (110..160 dB SPL) x 20 clips = 240
  speechonly  engine at +60 dB SNR (i.e. speech only) x 50 clips

Outputs under data/generated/final_{A,B}/{main,weapon,speechonly}/ (WAV = [noisy primary, reference, clean] + manifest.jsonl)
and results/final_test/{final_test_spec.json, checksums.sha256, LOCK.txt}.
"""
import hashlib
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
from anc.data.speech import list_speech, load_speech       # noqa: E402

MASTER_SEED = 20260921
RIR_SEED = 20260921
POOL = ROOT / "data" / "rir_pool_final.npz"
OUT = ROOT / "results" / "final_test"
PEAKS = [110, 120, 130, 140, 150, 160]
SUBSETS = {                        # name -> (classes, snr bins or None, per cell, peak grid or None)
    "main": (None, None, 25, None),
    "weapon": (["gunshot", "artillery"], None, 20, PEAKS),
    "speechonly": (["engine"], [60.0], 50, None),
}
SET_CODE = {"A": 1, "B": 2}
SUB_CODE = {"main": 1, "weapon": 2, "speechonly": 3}


class LoggingList(list):
    log = []

    def __getitem__(self, i):
        v = super().__getitem__(i)
        LoggingList.log.append(str(v))
        return v


_CTX = _FILES = None


def _init(cfg, speech_files, allowed_gunshots):
    global _CTX, _FILES
    bank = NoiseBank(cfg["noise_dir"], cfg["use_real_noise_prob"], cfg["holdout_tokens"])
    bank.files[("gunshot", "test")] = LoggingList([Path(p) for p in allowed_gunshots])      # ONLY never-used recordings
    _CTX = Context(cfg=cfg, bank=bank, rirs=RirPool.load(POOL), split="test")
    _FILES = speech_files


def _work(job):
    idx, cls, snr, peak, seed, out_dir = job
    rng = np.random.default_rng(seed)
    n = int(_CTX.cfg["clip_seconds"] * _CTX.cfg["fs"])
    speech, path, spk = load_speech(_FILES, n, rng)
    LoggingList.log = []
    ex = make_example(speech, cls, snr, _CTX, rng, peak_spl_db=peak)
    name = f"{idx:06d}.wav"
    sf.write(Path(out_dir) / name, np.stack([ex["noisy"], ex["ref"], ex["clean"]], axis=1), 16000, subtype="PCM_16")
    m = ex["meta"]
    m.update(file=name, speech_file=Path(path).as_posix(), speaker=spk, seed=int(seed), snr_requested=snr,
             gunshot_files=sorted({Path(u).as_posix() for u in LoggingList.log if "gunshot" in u}))
    return m


def sha256(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "dataset.yaml"))
    src = json.load(open(OUT / "earlier_sources.json"))
    used_speech = set()
    for d in ("test", "test_peak", "test_speechonly", "unseen", "val", "val_peak", "val_speechonly"):
        used_speech |= {Path(p).as_posix() for p in src["speech_used"][d]}
    free_gun = src["gunshot_test_partition"]["free_files"]

    speech = {
        "A": [(p, s) for p, s in list_speech(["data/speech/test-clean"]) if Path(p).as_posix() not in used_speech],
        "B": list_speech(["data/speech/test-other"]),
    }
    assert speech["A"] and speech["B"], "missing speech (is test-other extracted?)"
    print(f"speech pools: final_A {len(speech['A'])} utterances ({len({s for _, s in speech['A']})} speakers); "
          f"final_B {len(speech['B'])} utterances ({len({s for _, s in speech['B']})} speakers); gunshot files allowed: {len(free_gun)}")

    if not POOL.exists():
        print("building the fresh RIR pool (new seed) ...")
        RirPool.build(cfg["rir"]["n_speech"], cfg["rir"]["n_noise_per_env"], RIR_SEED, 10).save(POOL)

    spec = dict(master_seed=MASTER_SEED, rir_seed=RIR_SEED, config=cfg, config_sha256=sha256(ROOT / "configs" / "dataset.yaml"),
                source_sha256={f: sha256(ROOT / "src" / "anc" / "data" / f) for f in ("mixer.py", "noise_bank.py", "noise_synth.py", "scene.py", "mic.py", "speech.py")},
                subsets={k: dict(per_cell=v[2], peaks=v[3], classes=v[0] or cfg["classes"], snr_bins=v[1] or cfg["snr_bins"]) for k, v in SUBSETS.items()},
                speech_pool_sizes={k: len(v) for k, v in speech.items()}, gunshot_files_allowed=len(free_gun),
                earlier_sources_file="results/final_test/earlier_sources.json")
    checks = []
    for sid in ("A", "B"):
        with ProcessPoolExecutor(10, initializer=_init, initargs=(cfg, speech[sid], free_gun)) as ex:
            for sub, (classes, bins, pc, grid) in SUBSETS.items():
                out = ROOT / "data" / "generated" / f"final_{sid}" / sub
                if (out / "manifest.jsonl").exists():
                    print(f"final_{sid}/{sub}: exists, skipped")
                    continue
                out.mkdir(parents=True, exist_ok=True)
                base = np.random.default_rng([MASTER_SEED, SET_CODE[sid], SUB_CODE[sub]])
                classes = classes or cfg["classes"]
                jobs = []
                if grid:
                    for c in classes:
                        for pk in grid:
                            for _ in range(pc):
                                jobs.append((len(jobs), c, 0.0, float(pk), int(base.integers(2 ** 31)), str(out)))
                else:
                    for c in classes:
                        for s in (bins or cfg["snr_bins"]):
                            for _ in range(pc):
                                jobs.append((len(jobs), c, float(s), None, int(base.integers(2 ** 31)), str(out)))
                with open(out / "manifest.jsonl", "w") as mf:
                    for m in ex.map(_work, jobs, chunksize=8):
                        mf.write(json.dumps(m) + "\n")
                print(f"final_{sid}/{sub}: {len(jobs)} clips", flush=True)
        for sub in SUBSETS:
            for p in sorted((ROOT / "data" / "generated" / f"final_{sid}" / sub).glob("*.wav")):
                checks.append(f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}")
    if checks:
        (OUT / "checksums.sha256").write_text("\n".join(checks) + "\n")
    json.dump(spec, open(OUT / "final_test_spec.json", "w"), indent=1)
    (OUT / "LOCK.txt").write_text(
        "FINAL TEST SETS: data/generated/final_A and final_B (frozen; checksums in checksums.sha256).\n"
        "Rules: never tune, select, or set thresholds using these sets. Evaluate finished models once and report everything.\n"
        "If any result here influences a design decision, the set is no longer untouched and this must be stated.\n")
    print("spec, checksums and lock written to results/final_test/")


if __name__ == "__main__":
    main()
