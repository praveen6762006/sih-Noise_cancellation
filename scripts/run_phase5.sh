#!/usr/bin/env bash
# Phase 5: robust Tiny (fallback if the Pi is too slow for Small) + evaluation.
cd "$(dirname "$0")/.."
[ -f checkpoints/I_tinyrobust_s0_last.pt ] || python scripts/train.py --name I_tinyrobust_s0 --steps 6000 --batch 16 --workers 10 \
    --val-every 1000 --size tiny --ref-dropout 0.2 --ref-jitter-db 8 --seed 0 > results/train_I_tinyrobust_s0.log 2>&1
for cfg in "test:test:" "testpeak:test_peak:--peak"; do IFS=: read s d x <<< "$cfg"
  [ -f results/final_I_tinyrobust_s0_$s.json ] || python scripts/evaluate.py --data data/generated/$d --proc ours --ckpt checkpoints/I_tinyrobust_s0_last.pt \
      --tag I_tinyrobust_s0 --out results/final_I_tinyrobust_s0_$s.json $x > results/log_final_I_tinyrobust_s0_$s.txt 2>&1
done
python scripts/export_onnx.py --ckpt checkpoints/I_tinyrobust_s0_last.pt --out models/tiny_deploy.onnx > results/export_tiny_deploy.log 2>&1
echo PHASE5_DONE
