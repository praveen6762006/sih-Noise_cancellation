Copy this folder to the Raspberry Pi 4.

  pip3 install numpy scipy onnxruntime
  python3 benchmark.py <model>.onnx --ch 16            # small model, timing only
  python3 benchmark.py <model>.onnx --ch 16 --paced    # real-time pacing, counts missed 16 ms deadlines

  python3 live.py <model>.onnx --ch 16 --file in.wav out.wav       # offline test of the same path
  python3 live.py --list-devices                                    # (needs: pip3 install sounddevice, apt install libportaudio2)
  python3 live.py <model>.onnx --ch 16 --in-device N --out-device M # live demo

Model widths: --ch 8 tiny, --ch 16 small, --ch 32 base.
For clean numbers: heatsink/fan, CPU governor set to performance, and no other heavy processes.
