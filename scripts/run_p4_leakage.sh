#!/usr/bin/env bash
# P4: reference-mic speech-leakage sweep. Same seeds/sources for every level (paired): only the extra loss applied to the
# target speech reaching the reference mic (shadow_db, on top of the simulated geometry) changes. Validation partition
# sources (dev-clean, validation gunshots, validation RIRs) are used: this is a sensitivity analysis, not a final test.
# shadow_db = -12 ... +80 dB. The training distribution used shadow_db in [0, 12]; +80 dB means an (almost) speech-free reference.
cd "$(dirname "$0")/.."
D=data/generated/leak_sweep; O=results/leak_sweep; mkdir -p $D $O
for tag in m12 m06 p00 p06 p12 p24 p80; do
  [ -f $D/$tag/manifest.jsonl ] || python scripts/make_dataset.py --config configs/leakage_sweep/shadow_$tag.yaml --split val \
      --per-cell 10 --seed 7 --out $D/$tag --workers 8 > $O/gen_$tag.log 2>&1
done
ev() { out=$1; shift; [ -f "$out" ] && { echo "skip $out"; return; }; echo "== $out ($(date +%H:%M))"
  python scripts/evaluate.py --out "$out" "$@" > "${out%.json}.log" 2>&1 || echo "   FAILED"; }
for tag in m12 m06 p00 p06 p12 p24 p80; do
  ev $O/H_$tag.json --data $D/$tag --proc ours --ckpt checkpoints/H_robust_s0_last.pt --tag P4_H_$tag
  ev $O/C_$tag.json --data $D/$tag --proc ours --ckpt checkpoints/C_full_s0_last.pt --tag P4_C_$tag
done
# reference switched off (all zeros) on the identical primary signals: what the model does when it ignores the reference
ev $O/Hzero_p00.json --data $D/p00 --proc ours --ckpt checkpoints/H_robust_s0_last.pt --ref-fault zero --tag P4_Hzero
ev $O/Hzero_p80.json --data $D/p80 --proc ours --ckpt checkpoints/H_robust_s0_last.pt --ref-fault zero --tag P4_Hzero80
ev $O/none_p00.json --data $D/p00 --proc none --tag P4_none
echo P4_DONE
