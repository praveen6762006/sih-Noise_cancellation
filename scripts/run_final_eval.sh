#!/usr/bin/env bash
# Final evaluation of every trained model on the fixed test sets, then the report.
# Result files: results/final_<model>_<set>.json   (set = test | unseen | testpeak | speechonly)
cd "$(dirname "$0")/.."
EV="python scripts/evaluate.py"

ev() {   # ev <model-tag> <set-name> <data-dir> <extra args...>
  tag=$1; set=$2; dir=$3; shift 3
  out=results/final_${tag}_${set}.json
  [ -f "$out" ] && { echo "skip $out"; return; }
  echo "== $tag on $set ($(date +%H:%M))"
  $EV --data data/generated/$dir --tag ${tag} --out "$out" "$@" > results/log_final_${tag}_${set}.txt 2>&1 \
     || echo "   FAILED, see results/log_final_${tag}_${set}.txt"
}

sets() {  # sets <model-tag> <args...>
  tag=$1; shift
  ev $tag test       test       "$@"
  ev $tag unseen     unseen     "$@"
  ev $tag testpeak   test_peak  "$@" --peak
  ev $tag speechonly test_speechonly "$@" --no-pesq
}

sets none  --proc none
sets gtcrn --proc gtcrn
for m in A_mono_s0 B_ref_s0 C_full_s0 D_fulldrop_s0 A_mono_s1 C_full_s1 T_tiny_s0 L_base_s0; do
  [ -f checkpoints/${m}_last.pt ] && sets $m --proc ours --ckpt checkpoints/${m}_last.pt
done
# NLMS ablations on the best-performing Small model, and the model trained with NLMS in the loop
[ -f checkpoints/C_full_s0_last.pt ] && {
  sets C_full_s0_nlmspost --proc ours --ckpt checkpoints/C_full_s0_last.pt --nlms post
  sets C_full_s0_nlmspre  --proc ours --ckpt checkpoints/C_full_s0_last.pt --nlms pre
}
[ -f checkpoints/E_nlmspre_s0_last.pt ] && sets E_nlmspre_s0 --proc ours --ckpt checkpoints/E_nlmspre_s0_last.pt --nlms pre
# NLMS alone (Baseline 2) and the ungated variant (to show what the gate protects against)
sets nlms_only          --proc none --nlms pre
sets nlms_only_nogate   --proc none --nlms pre --no-gate

python scripts/make_report.py && echo "report -> results/REPORT.md"
echo FINAL_EVAL_DONE
