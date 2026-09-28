"""Independent verification of the fresh final test sets. Writes results/final_test/VERIFICATION.md and verification.json.

  python scripts/verify_final_test.py

Checks: counts, speaker / utterance / gunshot-file / gunshot-event / room-impulse-response independence from training,
validation and every earlier test-type set, seed uniqueness, checksums, and the class / SNR / peak distributions.
Exits non-zero if any independence check that must be zero is not zero.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from anc.data.noise_bank import NoiseBank, _event_id       # noqa: E402
from anc.data.speech import list_speech                    # noqa: E402
import yaml                                                # noqa: E402

FT = ROOT / "results" / "final_test"
cfg = yaml.safe_load(open(ROOT / "configs" / "dataset.yaml"))
posix = lambda p: Path(p).as_posix()
src = json.load(open(FT / "earlier_sources.json"))
spec = json.load(open(FT / "final_test_spec.json"))


def rows(sid, sub):
    return [json.loads(l) for l in open(ROOT / "data/generated" / f"final_{sid}" / sub / "manifest.jsonl")]


def main():
    fail, L, J = [], ["# Verification of the fresh final test sets\n"], {}
    R = {(sid, sub): rows(sid, sub) for sid in "AB" for sub in ("main", "weapon", "speechonly")}

    # ---- counts
    L += ["## 1. Exact counts\n", "| set | subset | manifest rows | WAV files | distinct speakers | distinct utterances |", "|---|---|---|---|---|---|"]
    for (sid, sub), r in R.items():
        n_wav = len(list((ROOT / "data/generated" / f"final_{sid}" / sub).glob("*.wav")))
        L.append(f"| final_{sid} | {sub} | {len(r)} | {n_wav} | {len({m['speaker'] for m in r})} | {len({m['speech_file'] for m in r})} |")
        if len(r) != n_wav:
            fail.append(f"row/file count mismatch in final_{sid}/{sub}")
    L.append(f"\nTotal mixtures: {sum(len(r) for r in R.values())} (each = one 4 s, 3-channel WAV: noisy primary, reference, clean target). "
             f"Master seed {spec['master_seed']}, RIR seed {spec['rir_seed']}; generation spec in `final_test_spec.json`.\n")

    # ---- speakers
    train_spk = {s for _, s in list_speech(["data/speech/train-clean-100"])}
    val_spk = {s for _, s in list_speech(["data/speech/dev-clean"])}
    earlier_test_spk = set()
    for d in ("test", "unseen", "test_peak", "test_speechonly"):
        earlier_test_spk |= {json.loads(l)["speaker"] for l in open(ROOT / f"data/generated/{d}/manifest.jsonl")}
    L += ["## 2. Speaker independence\n", "| set | speakers | overlap with TRAINING speakers | overlap with VALIDATION speakers | overlap with EARLIER-TEST speakers |", "|---|---|---|---|---|"]
    for sid in "AB":
        sp = {m["speaker"] for k in ("main", "weapon", "speechonly") for m in R[(sid, k)]}
        o_tr, o_va, o_te = len(sp & train_spk), len(sp & val_spk), len(sp & earlier_test_spk)
        L.append(f"| final_{sid} | {len(sp)} | {o_tr} | {o_va} | {o_te} |")
        J[f"speakers_{sid}"] = dict(n=len(sp), overlap_train=o_tr, overlap_val=o_va, overlap_earlier_test=o_te)
        if o_tr or o_va:
            fail.append(f"final_{sid}: speaker overlap with training/validation")
        if sid == "B" and o_te:
            fail.append("final_B shares speakers with earlier test")
    L.append("\nfinal_A reuses the 40 test-clean speakers of the earlier test (never any training or validation speaker); "
             "final_B is speaker-disjoint from everything.\n")

    # ---- utterances
    earlier_utt = set()
    for d, lst in src["speech_used"].items():
        earlier_utt |= {posix(p) for p in lst}
    utt = {sid: {m["speech_file"] for k in ("main", "weapon", "speechonly") for m in R[(sid, k)]} for sid in "AB"}
    oa, ob, oab = len(utt["A"] & earlier_utt), len(utt["B"] & earlier_utt), len(utt["A"] & utt["B"])
    L += ["## 3. Utterance independence\n", f"- utterances of final_A used by ANY earlier test/validation set: **{oa}**",
          f"- utterances of final_B used by ANY earlier set: **{ob}**", f"- utterances shared between final_A and final_B: **{oab}**\n"]
    J["utterance_overlap"] = dict(A_vs_earlier=oa, B_vs_earlier=ob, A_vs_B=oab)
    if oa or ob or oab:
        fail.append("utterance overlap")

    # ---- gunshots
    bank = NoiseBank("data/noise", 0.5, cfg["holdout_tokens"])
    ev = lambda files: {_event_id(Path(p)) for p in files}
    train_ev, val_ev, unseen_ev = (ev(bank.files[("gunshot", s)]) for s in ("train", "val", "unseen"))
    earlier_files = set()
    for d, lst in src["gunshot_files_used"].items():
        earlier_files |= {posix(p) for p in lst}
    allowed = {posix(p) for p in src["gunshot_test_partition"]["free_files"]}
    L += ["## 4. Gunshot file / event independence\n", "| set | real-gunshot draws | distinct files | distinct shots | files used by earlier sets | shots in TRAIN | shots in VALIDATION | shotgun (unseen partition) shots | files outside the never-used list |", "|---|---|---|---|---|---|---|---|---|"]
    for sid in "AB":
        drawn = collections.Counter(posix(f) for k in ("main", "weapon") for m in R[(sid, k)] for f in m["gunshot_files"])
        files = set(drawn)
        e = ev(files)
        a, b, c, d, f_ = len(files & earlier_files), len(e & train_ev), len(e & val_ev), len(e & unseen_ev), len(files - allowed)
        L.append(f"| final_{sid} | {sum(drawn.values())} | {len(files)} | {len(e)} | {a} | {b} | {c} | {d} | {f_} |")
        J[f"gunshots_{sid}"] = dict(draws=sum(drawn.values()), files=len(files), shots=len(e), overlap_earlier_files=a, overlap_train_shots=b, overlap_val_shots=c, shotgun=d, outside_allowed=f_)
        if a or b or c or d or f_:
            fail.append(f"final_{sid}: gunshot independence violated")
        fam = collections.Counter(Path(p).parent.name for p in files)
        L.append(f"|  | firearm folders drawn: {dict(fam)} | | | | | | | |")
    L.append("\nThe never-used pool holds 35 recordings / 34 shots (three firearm types), so both final sets draw from the same small "
             "real-gunshot pool. About half of the gunshot-class clips use these real recordings; the rest are synthetic.\n")

    # ---- RIRs
    def rowhash(a):
        return {hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest() for x in a}
    new = np.load(ROOT / "data/rir_pool_final.npz")
    new_h = set()
    for k in new.files:
        new_h |= rowhash(new[k].reshape(-1, new[k].shape[-1]))
    dup = 0
    for s in ("train", "val", "test", "unseen"):
        old = np.load(ROOT / f"data/rir_pool_{s}.npz")
        for k in old.files:
            dup += len(new_h & rowhash(old[k].reshape(-1, old[k].shape[-1])))
    L += ["## 5. Room impulse responses\n", f"- fresh pool (seed {spec['rir_seed']}): {len(new_h)} distinct channel responses; identical to any response in the train/val/test/unseen pools: **{dup}**\n"]
    J["rir_duplicates"] = dup
    if dup:
        fail.append("RIR duplicates")

    # ---- seeds and audio
    seeds = [m["seed"] for r in R.values() for m in r]
    earlier_seeds = set()
    for d in ("test", "unseen", "test_peak", "test_speechonly", "val", "val_peak", "val_speechonly"):
        earlier_seeds |= {json.loads(l)["seed"] for l in open(ROOT / f"data/generated/{d}/manifest.jsonl")}
    L += ["## 6. Seeds and audio\n", f"- per-clip seeds: {len(seeds)} total, {len(set(seeds))} unique, shared with any earlier set: **{len(set(seeds) & earlier_seeds)}**"]
    if len(set(seeds)) != len(seeds) or (set(seeds) & earlier_seeds):
        fail.append("seed collision")
    chk = {l.split("  ", 1)[1]: l.split("  ", 1)[0] for l in open(FT / "checksums.sha256").read().splitlines() if l}
    bad = 0
    for rel, h in chk.items():
        bad += hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != h
    L.append(f"- checksums: {len(chk)} WAV files recorded; **{bad}** changed since generation\n")
    if bad:
        fail.append("checksum mismatch")

    # ---- distributions
    L += ["## 7. Distributions\n"]
    for sid in "AB":
        m_ = R[(sid, "main")]
        L.append(f"### final_{sid} / main\n")
        cc = collections.Counter(m["cls"] for m in m_)
        L.append("class counts: " + ", ".join(f"{c} {n}" for c, n in sorted(cc.items())))
        L.append("\n| requested SNR (dB) | clips | measured SNR mean | std | min | max |\n|---|---|---|---|---|---|")
        for b in spec["subsets"]["main"]["snr_bins"]:
            v = np.array([m["snr_db"] for m in m_ if m["snr_requested"] == b])
            L.append(f"| {b:+.0f} | {len(v)} | {v.mean():+.2f} | {v.std():.2f} | {v.min():+.1f} | {v.max():+.1f} |")
        gun = [m for m in m_ if m["cls"] == "gunshot"]
        L.append(f"\ngunshot-class clips: {len(gun)} ({sum(m['noise_src'] == 'real' for m in gun)} real recordings, {sum(m['noise_src'] != 'real' for m in gun)} synthetic); "
                 f"environments: {dict(collections.Counter(m['env'] for m in m_))}")
        L.append(f"speech level at primary: {min(m['speech_spl'] for m in m_):.0f}-{max(m['speech_spl'] for m in m_):.0f} dB SPL; "
                 f"speech leakage into reference: median {np.median([m['leak_db'] for m in m_]):.1f} dB (5-95%: {np.percentile([m['leak_db'] for m in m_], 5):.1f} to {np.percentile([m['leak_db'] for m in m_], 95):.1f} dB); "
                 f"clips with clipping on the primary: {sum(m['clip_frac_primary'] > 0 for m in m_)}\n")
        w = R[(sid, "weapon")]
        L.append(f"### final_{sid} / weapon: peak levels {dict(sorted(collections.Counter(m['peak_target'] for m in w).items()))}, classes {dict(collections.Counter(m['cls'] for m in w))}\n")
        so = R[(sid, "speechonly")]
        L.append(f"### final_{sid} / speechonly: {len(so)} clips at requested SNR {sorted({m['snr_requested'] for m in so})} dB\n")

    L += ["## 8. Result\n", "**PASS**: every independence check that must be zero is zero." if not fail else "**FAIL**: " + "; ".join(fail),
          "\nNot independent by construction (state this when reporting): the synthetic noise classes come from the same generator "
          "as in training (new random draws, same parameter ranges); the gunshot pool is small; final_A's speakers were also in the earlier test set.\n"]
    (FT / "VERIFICATION.md").write_text("\n".join(L), encoding="utf-8")
    json.dump(J, open(FT / "verification.json", "w"), indent=1)
    print("\n".join(L))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
