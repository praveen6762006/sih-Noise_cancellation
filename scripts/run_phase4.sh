#!/usr/bin/env bash
# Phase 4: robust deployment model (reference dropout + wide gain-mismatch training) and guarded fault tests.
cd "$(dirname "$0")/.."
[ -f checkpoints/H_robust_s0_last.pt ] || python scripts/train.py --name H_robust_s0 --steps 6000 --batch 16 --workers 10 \
    --val-every 1000 --size small --pretrained --ref-dropout 0.2 --ref-jitter-db 8 --seed 0 > results/train_H_robust_s0.log 2>&1
for cfg in "test:test:" "testpeak:test_peak:--peak"; do IFS=: read s d x <<< "$cfg"
  [ -f results/final_H_robust_s0_$s.json ] || python scripts/evaluate.py --data data/generated/$d --proc ours --ckpt checkpoints/H_robust_s0_last.pt \
      --tag H_robust_s0 --out results/final_H_robust_s0_$s.json $x > results/log_final_H_robust_s0_$s.txt 2>&1
done
for m in D_fulldrop_s0 H_robust_s0; do
  for f in none zero copy loud quiet shifted; do
    fa=""; [ "$f" != none ] && fa="--ref-fault $f"
    out=results/p4_fault_${m}_${f}.json
    [ -f "$out" ] || python scripts/evaluate.py --data data/generated/test --proc ours --ckpt checkpoints/${m}_last.pt --tag p4f_${m}_$f \
        --stride 3 --no-pesq --refguard $fa --out "$out" > "${out%.json}.log" 2>&1
  done
done
python scripts/make_report.py
echo PHASE4_DONE
