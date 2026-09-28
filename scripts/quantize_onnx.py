"""INT8 weight quantisation of the streaming ONNX model (ONNX Runtime dynamic quantisation).

  python scripts/quantize_onnx.py models/small_full.onnx models/small_full_int8.onnx

Quantises the MatMul/Gemm weights to int8 (activations are quantised on the fly). The recurrent GRUs and the
convolutions stay in float, because GRUs are where post-training INT8 loses most quality (see the review).
Always re-evaluate afterwards (scripts/evaluate.py --proc onnx --onnx <file>) and time it on the Pi: on a
Cortex-A72 (no int8 dot-product instructions) INT8 may be no faster than float32, so this is only kept if the
measured speed or size justifies it.
"""
import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

src, dst = sys.argv[1], sys.argv[2]
quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
print(f"{src}: {Path(src).stat().st_size / 1024:.0f} KB  ->  {dst}: {Path(dst).stat().st_size / 1024:.0f} KB")
