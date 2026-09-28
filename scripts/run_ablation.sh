#!/usr/bin/env bash
# Sequential training chain for the ablation study. Each run warm-starts from the pretrained
# GTCRN (Small only); Tiny and Base start from scratch. Logs: results/train_<name>.log
cd "$(dirname "$0")/.."
STEPS=${STEPS:-6000}
COMMON="--steps $STEPS --batch 16 --workers 10 --val-every 1000"

run() {   # run <name> <args...>
  name=$1; shift
  if [ -f "checkpoints/${name}_last.pt" ] && grep -q "^done" "results/train_${name}.log" 2>/dev/null; then
    echo "skip $name (finished)"; return; fi
  echo "=== $name: $* ($(date +%H:%M))"
  python scripts/train.py --name "$name" $COMMON "$@" > "results/train_${name}.log" 2>&1
  tail -2 "results/train_${name}.log" | grep -E "done|Error|Traceback"
}

# Ablation ladder on the Small model (seed 0)
run A_mono_s0       --size small --pretrained --no-extra --seed 0                       # single mic
run B_ref_s0        --size small --pretrained --no-flag --seed 0                        # + reference mic
run C_full_s0       --size small --pretrained --seed 0                                  # + impulse flag
run D_fulldrop_s0   --size small --pretrained --ref-dropout 0.2 --seed 0                # + reference dropout
# Second seed for the key comparison, to see whether differences exceed run-to-run noise
run A_mono_s1       --size small --pretrained --no-extra --seed 1
run C_full_s1       --size small --pretrained --seed 1
# Size family (from scratch, full inputs)
run T_tiny_s0       --size tiny --seed 0
run L_base_s0       --size base --seed 0
echo CHAIN_DONE
