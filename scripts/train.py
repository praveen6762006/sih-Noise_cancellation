"""Train GTCRN-X on on-the-fly defence-noise mixtures.

  python scripts/train.py --size small --steps 2000 --name small_ref --pretrained
  python scripts/train.py --size small --steps 2000 --name small_mono --no-extra   # single-mic ablation

Validation uses the fixed generated val set (data/generated/val): mean output SI-SNR on every 5th
example. Checkpoints go to checkpoints/<name>_{best,last}.pt.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anc.data.speech import list_speech                    # noqa: E402
from anc.data.torch_dataset import MixStream               # noqa: E402
from anc.features import N_EXTRA, istft, prepare_batch     # noqa: E402
from anc.losses import si_snr, total_loss                  # noqa: E402
from anc.model.gtcrn_x import SIZES, GTCRNX, count_params, load_pretrained  # noqa: E402


def forward(model, batch, n_samples):
    est_spec = model(batch["spec"], batch["extra"] if model.n_extra else None)
    return est_spec, istft(est_spec, n_samples)


@torch.no_grad()
def validate(model, val_rows, val_dir, dev, nlms_pre=False):
    model.eval()
    gains = []
    for m in val_rows:
        x, _ = sf.read(val_dir / m["file"], dtype="float32")
        if nlms_pre:                                    # validate the same pipeline the model is trained for
            from anc.nlms import run_nlms
            x[:, 0] = run_nlms(x[:, 0].astype(np.float64), x[:, 1].astype(np.float64))[0].astype(np.float32)
        noisy, ref, clean = (torch.from_numpy(x[:, i].copy())[None].to(dev) for i in range(3))
        b = prepare_batch(noisy, ref, clean)
        _, est = forward(model, b, noisy.shape[-1])
        gains.append(float(si_snr(est, b["clean"]).mean() - si_snr(b["noisy"], b["clean"]).mean()))
    model.train()
    return float(np.mean(gains))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small", choices=list(SIZES))
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pretrained", action="store_true", help="warm-start from the DNS3 GTCRN checkpoint (small only)")
    ap.add_argument("--no-extra", action="store_true", help="single-mic model (ablation)")
    ap.add_argument("--speech-root", nargs="*", help="override training speech directories")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--nlms-pre", action="store_true", help="train on the output of the NLMS stage (NLMS before the network)")
    ap.add_argument("--impulse-weight", type=float, default=2.0, help="extra loss weight on frames flagged as impulsive (0 = none)")
    ap.add_argument("--ref-jitter-db", type=float, help="override the reference-mic gain mismatch range (+/- dB) used for training data")
    ap.add_argument("--no-flag", action="store_true", help="zero the impulse-flag input (ablation)")
    ap.add_argument("--ref-dropout", type=float, default=0.0, help="probability of zeroing the reference mic per example")
    ap.add_argument("--exclude-classes", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(ROOT / "configs" / "dataset.yaml"))
    if a.ref_jitter_db is not None:
        cfg["ref_gain_jitter_db"] = a.ref_jitter_db
    files = list_speech(a.speech_root or cfg["speech_dirs"]["train"])
    assert files, "no training speech found"
    pool = ROOT / "data" / "rir_pool_train.npz"
    assert pool.exists(), "build the RIR pool first: python scripts/make_dataset.py --split train --n 8 --out data/generated/_tmp"

    model = GTCRNX(ch=SIZES[a.size], n_extra=0 if a.no_extra else N_EXTRA).to(dev)
    model.use_flag = not a.no_flag
    if a.pretrained:
        assert a.size == "small", "pretrained weights exist only for the small (ch=16) size"
        load_pretrained(model, ROOT / "third_party/gtcrn/checkpoints/model_trained_on_dns3.tar", dev)
    print(f"{a.name}: {a.size}, {count_params(model):,} parameters, device {dev}, {len(files)} speech files", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    loader = DataLoader(MixStream(cfg, "train", files, pool, seed=a.seed, exclude_classes=a.exclude_classes, nlms_pre=a.nlms_pre),
                        batch_size=a.batch, num_workers=a.workers, prefetch_factor=4, persistent_workers=True)

    val_dir = ROOT / "data" / "generated" / "val"
    val_rows = [json.loads(l) for l in open(val_dir / "manifest.jsonl")][::5]
    ck = ROOT / "checkpoints"
    ck.mkdir(exist_ok=True)
    best, t0, ema = -1e9, time.time(), None

    for step, (noisy, ref, clean) in enumerate(loader, start=1):
        noisy, ref, clean = noisy.to(dev), ref.to(dev), clean.to(dev)
        if a.ref_dropout > 0:                       # simulate a failed/absent reference mic
            ref = ref * (torch.rand(ref.shape[0], 1, device=dev) >= a.ref_dropout).float()
        b = prepare_batch(noisy, ref, clean)
        est_spec, est = forward(model, b, noisy.shape[-1])
        loss, parts = total_loss(est_spec, est, b["clean_spec"], b["clean"], b["flags"], impulse_weight=a.impulse_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        ema = float(loss) if ema is None else 0.98 * ema + 0.02 * float(loss)
        if step % 50 == 0:
            print(f"step {step:5d}  loss {ema:.4f}  si {parts['si']:.2f}  spec {parts['spec']:.4f}  "
                  f"mr {parts['mr']:.3f}  {(time.time() - t0) / step:.2f}s/step", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            g = validate(model, val_rows, val_dir, dev, a.nlms_pre)
            print(f"  >> val SI-SNR improvement {g:+.2f} dB", flush=True)
            state = dict(model=model.state_dict(), size=a.size, n_extra=model.n_extra, use_flag=model.use_flag,
                         ref_dropout=a.ref_dropout, step=step, val=g, args=vars(a))
            torch.save(state, ck / f"{a.name}_last.pt")
            if g > best:
                best = g
                torch.save(state, ck / f"{a.name}_best.pt")
        if step >= a.steps:
            break
    print(f"done. best val SI-SNR improvement {best:+.2f} dB")


if __name__ == "__main__":
    main()
