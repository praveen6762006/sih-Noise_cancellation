"""Create pi_bundle/: a flat folder to copy to the Raspberry Pi (runtime + benchmark + live demo + models).

  python scripts/make_pi_bundle.py models/smoke_small.onnx [more.onnx ...]

On the Pi:   pip3 install numpy scipy onnxruntime     then     python3 benchmark.py <model>.onnx --ch 16
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "pi_bundle"
out.mkdir(exist_ok=True)
shutil.copy(ROOT / "src" / "anc" / "runtime.py", out / "runtime.py")
shutil.copy(ROOT / "src" / "anc" / "refguard.py", out / "refguard.py")
shutil.copy(ROOT / "scripts" / "pi_benchmark.py", out / "benchmark.py")
shutil.copy(ROOT / "scripts" / "live.py", out / "live.py")
for m in sys.argv[1:]:
    shutil.copy(m, out / Path(m).name)

README = """Copy this folder to the Raspberry Pi 4.

  pip3 install numpy scipy onnxruntime
  python3 benchmark.py <model>.onnx --ch 16            # small model, timing only
  python3 benchmark.py <model>.onnx --ch 16 --paced    # real-time pacing, counts missed 16 ms deadlines

  python3 live.py <model>.onnx --ch 16 --file in.wav out.wav       # offline test of the same path
  python3 live.py --list-devices                                    # (needs: pip3 install sounddevice, apt install libportaudio2)
  python3 live.py <model>.onnx --ch 16 --in-device N --out-device M # live demo

Model widths: --ch 8 tiny, --ch 16 small, --ch 32 base.
For clean numbers: heatsink/fan, CPU governor set to performance, and no other heavy processes.
"""
(out / "README.txt").write_text(README, encoding="utf-8")
print("bundle:", out, [p.name for p in sorted(out.iterdir())])
