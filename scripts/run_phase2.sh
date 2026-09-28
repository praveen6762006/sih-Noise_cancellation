#!/usr/bin/env bash
# Phase 2 (no Pi needed): pruning study, held-out-class generalisation, reference-mic fault robustness.
cd "$(dirname "$0")/.."
EV="python scripts/evaluate.py"
ev() { out=$1; shift; [ -f "$out" ] && { echo "skip $out"; return; }; echo "== $out ($(date +%H:%M))"; $EV --out "$out" "$@" > "${out%.json}.log" 2>&1 || echo "   FAILED"; }

# 1. pruning: Small model at 30/50/70% sparsity, fine-tuned 1500 steps
for s in 30 50 70; do
  [ -f checkpoints/P${s}_C_full_s0_last.pt ] || python scripts/prune_finetune.py --ckpt checkpoints/C_full_s0_last.pt \
      --sparsity 0.$s --name P${s}_C_full_s0 > results/prune_P${s}.log 2>&1
  ev results/p2_prune${s}_test.json --data data/generated/test --proc ours --ckpt checkpoints/P${s}_C_full_s0_last.pt --tag p2_prune$s --stride 2
done

# 2. held-out class: train without drone / without gunshot, test on everything (the excluded class is the point)
for cls in drone gunshot; do
  [ -f checkpoints/F_no_${cls}_s0_last.pt ] || python scripts/train.py --name F_no_${cls}_s0 --steps 6000 --batch 16 --workers 10 \
      --val-every 1000 --size small --pretrained --exclude-classes $cls --seed 0 > results/train_F_no_${cls}_s0.log 2>&1
  ev results/p2_no${cls}_test.json --data data/generated/test --proc ours --ckpt checkpoints/F_no_${cls}_s0_last.pt --tag p2_no$cls
  ev results/p2_no${cls}_testpeak.json --data data/generated/test_peak --proc ours --ckpt checkpoints/F_no_${cls}_s0_last.pt --tag p2_no${cls}p --peak
done

# 3. reference mic faults, on every 3rd test clip (no PESQ): zero, copy of primary, +10 dB, -10 dB, 5 ms shift
for m in A_mono_s0 C_full_s0 D_fulldrop_s0; do
  for f in none zero copy loud quiet shifted; do
    fa=""; [ "$f" != none ] && fa="--ref-fault $f"
    ev results/p2_fault_${m}_${f}.json --data data/generated/test --proc ours --ckpt checkpoints/${m}_last.pt --tag p2f_${m}_$f --stride 3 --no-pesq $fa
  done
done
python scripts/make_report.py
echo PHASE2_DONE
