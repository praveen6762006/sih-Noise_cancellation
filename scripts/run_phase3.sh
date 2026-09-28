#!/usr/bin/env bash
# Phase 3: does the impulse-frame loss weighting matter? (B_ref_s0 = same config WITH weight 2.0)
cd "$(dirname "$0")/.."
[ -f checkpoints/G_noimpw_s0_last.pt ] || python scripts/train.py --name G_noimpw_s0 --steps 6000 --batch 16 --workers 10 \
    --val-every 1000 --size small --pretrained --no-flag --impulse-weight 0 --seed 0 > results/train_G_noimpw_s0.log 2>&1
for cfg in "test:test:" "testpeak:test_peak:--peak"; do IFS=: read s d x <<< "$cfg"
  [ -f results/final_G_noimpw_s0_$s.json ] || python scripts/evaluate.py --data data/generated/$d --proc ours --ckpt checkpoints/G_noimpw_s0_last.pt \
      --tag G_noimpw_s0 --out results/final_G_noimpw_s0_$s.json $x > results/log_final_G_noimpw_s0_$s.txt 2>&1
done
python scripts/make_report.py
echo PHASE3_DONE
