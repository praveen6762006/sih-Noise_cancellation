#!/usr/bin/env bash
# Score the DEPLOYED path (numpy front end + ONNX Runtime + guard + slow AGC), fp32 and int8.
cd "$(dirname "$0")/.."
ev() { out=$1; shift; [ -f "$out" ] || python scripts/evaluate.py --out "$out" "$@" > "${out%.json}.log" 2>&1; }
ev results/deployed_fp32_test.json     --data data/generated/test      --proc onnx --onnx models/small_deploy.onnx      --tag dep_fp32 --stride 2
ev results/deployed_int8_test.json     --data data/generated/test      --proc onnx --onnx models/small_deploy_int8.onnx --tag dep_int8 --stride 2
ev results/deployed_fp32_testpeak.json --data data/generated/test_peak --proc onnx --onnx models/small_deploy.onnx      --tag dep_fp32p --peak
echo DEPLOYED_DONE
