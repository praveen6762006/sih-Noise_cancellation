"""Run /tmp/demo_cap.wav through the deployed model, writing /tmp/demo_raw.wav and /tmp/demo_enhanced.wav."""
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
from runtime import StreamProcessor, HOP

x, sr = sf.read("/tmp/demo_cap.wav", dtype="float64")
x = x[sr:]                                       # drop the ~1s I2S startup burst
p, r = x[:, 0], x[:, 1]

rt = StreamProcessor("small_deploy.onnx", ch=16)
n = (len(p) // HOP) * HOP
out = np.concatenate([rt.process(p[h * HOP:(h + 1) * HOP], r[h * HOP:(h + 1) * HOP]) for h in range(n // HOP)])[HOP:]

sf.write("/tmp/demo_raw.wav", p, sr, subtype="PCM_16")
sf.write("/tmp/demo_enhanced.wav", np.clip(out, -1, 1), sr, subtype="PCM_16")
print("done: /tmp/demo_raw.wav and /tmp/demo_enhanced.wav")
