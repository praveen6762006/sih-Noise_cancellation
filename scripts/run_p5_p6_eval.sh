#!/usr/bin/env bash
# Evaluate the P5/P6 models. Model-size comparison and seed variance are measured on the EARLIER test set (no final-set use for
# experiments). The extra H seeds are additionally scored on the fresh final sets ONCE, only to report seed variance:
# H seed 0 remains the deployment model whatever these numbers show (no selection).
cd "$(dirname "$0")/.."
O=results/size_variance
ev() { out=$1; shift; [ -f "$out" ] && { echo "skip $out"; return; }; echo "== $out ($(date +%H:%M))"
  python scripts/evaluate.py --out "$out" "$@" > "${out%.json}.log" 2>&1 || echo "   FAILED"; }
ev $O/J_small_scratch_s0_test.json --data data/generated/test --proc ours --ckpt checkpoints/J_small_scratch_s0_last.pt --tag P5_J
for s in 1 2; do
  ev $O/H_robust_s${s}_test.json --data data/generated/test --proc ours --ckpt checkpoints/H_robust_s${s}_last.pt --tag P6_H$s
  for S in A B; do
    ev results/final_test/P6_H_s${s}_final${S}_main.json --data data/generated/final_$S/main --proc ours --ckpt checkpoints/H_robust_s${s}_last.pt --tag P6_H${s}_$S
  done
done
echo P5P6_EVAL_DONE
