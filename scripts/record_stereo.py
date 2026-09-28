"""Record the two INMP441 microphones to a stereo WAV (run on the Pi, or any machine with a 2-channel input).

  python3 record_stereo.py --list-devices
  python3 record_stereo.py speech.wav --seconds 20 --device 1
  python3 record_stereo.py noise.wav  --seconds 30 --device 1

Channel 1 = primary (close to the mouth), channel 2 = reference (outward). Prints the peak level and the
number of full-scale samples per channel, so overload is obvious. Needs `pip3 install sounddevice soundfile`.
"""
import argparse

import numpy as np
import sounddevice as sd
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--device")
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args()
    if a.list_devices or not a.out:
        print(sd.query_devices())
        return
    print(f"recording {a.seconds:.0f} s ... (channel 1 = primary, channel 2 = reference)")
    x = sd.rec(int(a.seconds * 16000), samplerate=16000, channels=2, dtype="float32", device=a.device)
    sd.wait()
    sf.write(a.out, x, 16000, subtype="PCM_24")
    for i, name in enumerate(["primary", "reference"]):
        ch = x[:, i]
        print(f"{name:9s} peak {np.abs(ch).max():.3f} ({20 * np.log10(np.abs(ch).max() + 1e-9):.1f} dBFS), "
              f"rms {np.sqrt(np.mean(ch ** 2)):.4f}, full-scale samples {(np.abs(ch) >= 0.999).sum()}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
