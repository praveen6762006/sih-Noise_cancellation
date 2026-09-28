#!/usr/bin/env bash
# P4b: the harmful leakage levels through the DEPLOYED path (reference guard + ONNX runtime), same clips as run_p4_leakage.sh.
cd "$(dirname "$0")/.."
D=data/generated/leak_sweep; O=results/leak_sweep
for tag in m12 m06 p00; do
  out=$O/Honnx_$tag.json; [ -f "$out" ] && continue; echo "== $out ($(date +%H:%M))"
  python scripts/evaluate.py --out "$out" --data $D/$tag --proc onnx --onnx models/small_deploy.onnx --tag P4b_$tag > "${out%.json}.log" 2>&1 || echo FAILED
done
echo P4B_DONE
