#!/usr/bin/env bash
# P5: Small from scratch (same conditions as the Tiny/Base runs: scratch, no dropout, seed 0)
# P6: two more independent seeds of the deployment configuration H (warm start, reference dropout 0.2, gain jitter +/-8 dB)
# Checkpoints are chosen by the LAST step (no selection on any test set); validation set only for monitoring.
cd "$(dirname "$0")/.."
run() { name=$1; shift
  if grep -q "^done" "results/train_${name}.log" 2>/dev/null; then echo "skip $name"; return; fi
  echo "=== $name ($(date +%H:%M))"
  python scripts/train.py --name "$name" --steps 6000 --batch 16 --workers 10 --val-every 1000 "$@" > "results/train_${name}.log" 2>&1
  tail -1 "results/train_${name}.log"
}
run J_small_scratch_s0 --size small --seed 0
run H_robust_s1 --size small --pretrained --ref-dropout 0.2 --ref-jitter-db 8 --seed 1
run H_robust_s2 --size small --pretrained --ref-dropout 0.2 --ref-jitter-db 8 --seed 2
echo P5P6_TRAIN_DONE
