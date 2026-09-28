"""Global magnitude pruning + fine-tuning of a trained GTCRN-X (the statement lists pruning as an optimisation).

  python scripts/prune_finetune.py --ckpt checkpoints/C_full_s0_last.pt --sparsity 0.5 --name P50_C_full_s0

Prunes the weights of all convolution, transposed-convolution, linear and GRU layers (the fixed ERB filter banks,
BatchNorm and PReLU parameters are left alone) by global magnitude, then fine-tunes with the mask enforced.
The checkpoint stores dense (zeroed) weights, so evaluation works unchanged with `evaluate.py --proc ours`.
NOTE: unstructured sparsity reduces the parameter count but does not speed up ONNX Runtime on a CPU; the
result is reported as quality versus sparsity, not as a speed-up.
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anc.data.speech import list_speech                    # noqa: E402
from anc.data.torch_dataset import MixStream               # noqa: E402
from anc.features import istft, prepare_batch              # noqa: E402
from anc.losses import total_loss                          # noqa: E402
from anc.model.gtcrn_x import SIZES, GTCRNX                # noqa: E402


def prunable(model):
    """(name, parameter) of weight tensors that are pruned."""
    out = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() > 1 and "erb" not in name:
            out.append((name, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sparsity", type=float, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(a.ckpt, map_location=dev)
    model = GTCRNX(SIZES[st["size"]], st["n_extra"]).to(dev)
    model.load_state_dict(st["model"])
    model.use_flag = st.get("use_flag", True)

    params = prunable(model)
    allw = torch.cat([p.detach().abs().flatten() for _, p in params])
    thr = torch.quantile(allw.float().cpu(), a.sparsity).item()
    masks = {n: (p.detach().abs() > thr).float() for n, p in params}
    kept = sum(int(m.sum()) for m in masks.values())
    total = sum(m.numel() for m in masks.values())
    print(f"pruning {a.sparsity:.0%}: keeping {kept:,} of {total:,} prunable weights (threshold {thr:.4f})", flush=True)

    def apply_masks():
        with torch.no_grad():
            for n, p in params:
                p.mul_(masks[n])
    apply_masks()

    cfg = yaml.safe_load(open(ROOT / "configs" / "dataset.yaml"))
    files = list_speech(cfg["speech_dirs"]["train"])
    loader = DataLoader(MixStream(cfg, "train", files, ROOT / "data" / "rir_pool_train.npz", seed=7),
                        batch_size=16, num_workers=a.workers, prefetch_factor=4, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    model.train()
    for step, (noisy, ref, clean) in enumerate(loader, start=1):
        b = prepare_batch(noisy.to(dev), ref.to(dev), clean.to(dev))
        est_spec = model(b["spec"], b["extra"] if model.n_extra else None)
        loss, _ = total_loss(est_spec, istft(est_spec, noisy.shape[-1]), b["clean_spec"], b["clean"], b["flags"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        apply_masks()
        if step % 250 == 0:
            print(f"step {step} loss {float(loss):.3f}", flush=True)
        if step >= a.steps:
            break
    nz = sum(int((p != 0).sum()) for _, p in params)
    print(f"final: {nz:,} non-zero prunable weights ({100 * (1 - nz / total):.1f}% sparse)")
    out = dict(st)
    out.update(model=model.state_dict(), sparsity=a.sparsity, step=a.steps)
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    torch.save(out, ROOT / "checkpoints" / f"{a.name}_last.pt")
    print("saved", a.name)


if __name__ == "__main__":
    main()
