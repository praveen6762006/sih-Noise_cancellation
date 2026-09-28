#!/usr/bin/env bash
# P3: the DEPLOYED path (numpy front end + reference guard + ONNX Runtime + slow AGC) on the fresh final sets.
# fp32 on both sets, INT8 on final_A; the PyTorch numbers come from P2 (same clips, different protocol, compared pairwise).
cd "$(dirname "$0")/.."
O=results/final_test
ev() { out=$1; shift; [ -f "$out" ] && { echo "skip $out"; return; }; echo "== $out ($(date +%H:%M))"
  python scripts/evaluate.py --out "$out" "$@" > "${out%.json}.log" 2>&1 || echo "   FAILED"; }
ev $O/P3_onnx_fp32_finalA_main.json --data data/generated/final_A/main --proc onnx --onnx models/small_deploy.onnx --tag P3_fp32_A
ev $O/P3_onnx_fp32_finalB_main.json --data data/generated/final_B/main --proc onnx --onnx models/small_deploy.onnx --tag P3_fp32_B
ev $O/P3_onnx_int8_finalA_main.json --data data/generated/final_A/main --proc onnx --onnx models/small_deploy_int8.onnx --tag P3_int8_A
ev $O/P3_onnx_fp32_finalA_weapon.json --data data/generated/final_A/weapon --proc onnx --onnx models/small_deploy.onnx --peak --tag P3_fp32_Aw
ev $O/P3_onnx_fp32_finalB_weapon.json --data data/generated/final_B/weapon --proc onnx --onnx models/small_deploy.onnx --peak --tag P3_fp32_Bw
echo P3_EVAL_DONE
