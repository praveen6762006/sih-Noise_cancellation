#!/usr/bin/env bash
# P2: evaluate the deployment model H (and the no-processing / pretrained references) ONCE on the fresh final sets and on
# the existing unseen, speech-only and weapon sets. PyTorch protocol (each clip level-normalised by its own RMS).
# Nothing here is used to tune or select anything.
cd "$(dirname "$0")/.."
O=results/final_test; mkdir -p $O
H="--proc ours --ckpt checkpoints/H_robust_s0_last.pt"
ev() { out=$1; shift; [ -f "$out" ] && { echo "skip $out"; return; }; echo "== $out ($(date +%H:%M))"
  python scripts/evaluate.py --out "$out" "$@" > "${out%.json}.log" 2>&1 || echo "   FAILED"; }
for S in A B; do
  for sub in main speechonly; do
    d=data/generated/final_$S/$sub
    ev $O/P2_H_final${S}_$sub.json     --data $d $H     --tag P2_H_${S}_$sub
    ev $O/P2_none_final${S}_$sub.json  --data $d --proc none  --tag P2_none_${S}_$sub
    ev $O/P2_gtcrn_final${S}_$sub.json --data $d --proc gtcrn --tag P2_gtcrn_${S}_$sub
  done
  d=data/generated/final_$S/weapon
  ev $O/P2_H_final${S}_weapon.json     --data $d $H --peak    --tag P2_H_${S}_weapon
  ev $O/P2_none_final${S}_weapon.json  --data $d --proc none --peak --tag P2_none_${S}_weapon
  ev $O/P2_gtcrn_final${S}_weapon.json --data $d --proc gtcrn --peak --tag P2_gtcrn_${S}_weapon
done
# existing sets that Model H had not been run on
ev $O/P2_H_unseen.json          --data data/generated/unseen          $H --tag P2_H_unseen
ev $O/P2_H_test_speechonly.json --data data/generated/test_speechonly $H --tag P2_H_tso
echo P2_EVAL_DONE
