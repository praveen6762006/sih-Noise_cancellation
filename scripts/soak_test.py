"""Continuous real-time streaming soak test of the deployed runtime (run on an IDLE machine).

  python scripts/soak_test.py --minutes 32 --model models/small_deploy.onnx --ch 16 --out results/soak
  python scripts/soak_test.py --minutes 0.5 --out results/soak_smoke      # quick logic check (timings not meaningful under load)

Input: clips from the validation sets (data/generated/val + val_peak), read from disk and fed hop by hop (256 samples =
16 ms) at REAL-TIME PACE: hop h becomes available at t0 + h*16 ms and must be finished by t0 + (h+1)*16 ms, otherwise it is a
missed deadline (a late hop delays the following ones, so backlog is counted, not hidden).
Recorded: per-hop processing time (around runtime.process() only, not disk reads), lateness, memory every 10 s,
process CPU time, exceptions, non-finite outputs, silent gaps, guard flags. After the run two equivalence checks:
  (1) long-form: 120 s streamed with a fixed gain vs the offline PyTorch pipeline (same framing) -> agreement in dB
  (2) state drift: the run's final instance vs a fresh instance on the last 30 s of input (first 10 s skipped) -> agreement in dB
Output: <out>/soak_summary.json, soak_summary.md, soak_timeseries.csv, first/last 60 s of output audio.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from anc.runtime import HOP, StreamProcessor            # noqa: E402
from complexity import mem                              # noqa: E402

HOP_S = HOP / 16000


def clip_stream(paths, seconds):
    """Yield (n_clip_index, primary, reference) arrays until enough audio is queued."""
    total = 0.0
    i = 0
    while total < seconds:
        x, _ = sf.read(paths[i % len(paths)], dtype="float64")
        yield i, x[:, 0], x[:, 1]
        total += len(x) / 16000
        i += 1


def agreement_db(ref, test):
    return 10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - test) ** 2) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=32)
    ap.add_argument("--model", default="models/small_deploy.onnx")
    ap.add_argument("--ch", type=int, default=16)
    ap.add_argument("--out", default="results/soak")
    ap.add_argument("--ckpt", default="checkpoints/H_robust_s0_last.pt", help="offline model for the equivalence check")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / "data/generated/val").glob("*.wav")) + sorted((ROOT / "data/generated/val_peak").glob("*.wav"))
    seconds = a.minutes * 60
    n_hops = int(seconds / HOP_S)
    rt = StreamProcessor(str(ROOT / a.model), ch=a.ch, threads=1)
    base_ws, _ = mem()

    proc_ms = np.zeros(n_hops, np.float32)
    late_ms = np.zeros(n_hops, np.float32)
    missed = exceptions = nonfinite = gap_hops = guard_invalid = 0
    out_first, out_last = [], []
    series = []
    last_sample_t, cpu0, wall0 = -1e9, time.process_time(), time.perf_counter()
    inp_tail = []                                        # inputs of the last 30 s, for the drift check
    print(f"soak: {a.minutes} min = {n_hops} hops of 16 ms, model {a.model}", flush=True)

    h = 0
    t0 = time.perf_counter() + 1.0
    prev_end = t0
    gen = clip_stream(paths, seconds + 8)
    for ci, p_clip, r_clip in gen:
        for s in range(0, len(p_clip) - HOP + 1, HOP):
            if h >= n_hops:
                break
            p, r = p_clip[s:s + HOP], r_clip[s:s + HOP]
            arrival = t0 + h * HOP_S
            now = time.perf_counter()
            if arrival - now > 0.0007:
                time.sleep(arrival - now - 0.0005)
            while time.perf_counter() < arrival:
                pass
            ts = time.perf_counter()
            try:
                y = rt.process(p, r)
            except Exception as e:                       # noqa: BLE001
                exceptions += 1
                y = np.zeros(HOP)
                print(f"EXCEPTION at hop {h}: {type(e).__name__}: {e}", flush=True)
            te = time.perf_counter()
            proc_ms[h] = (te - ts) * 1000
            late = (te - (arrival + HOP_S)) * 1000       # > 0 : finished after the next hop had already arrived
            late_ms[h] = late
            missed += late > 0
            if not np.all(np.isfinite(y)):
                nonfinite += 1
            if np.sqrt(np.mean(p ** 2)) > 1e-3 and np.sqrt(np.mean(y ** 2)) < 1e-9 and h > 2:
                gap_hops += 1                            # silent output although the input has signal
            guard_invalid += bool(rt.guard.invalid)
            if h < 60 * 62:
                out_first.append(y.copy())
            out_last.append(y.copy())
            if len(out_last) > 60 * 62:
                out_last.pop(0)
            inp_tail.append((p.copy(), r.copy()))
            if len(inp_tail) > 30 * 62 + 8:
                inp_tail.pop(0)
            if te - last_sample_t >= 10:
                ws, peak = mem()
                series.append(dict(t_s=round(te - t0, 1), hop=h, ws_mb=round(ws, 2), peak_ws_mb=round(peak, 2),
                                   cpu_s=round(time.process_time() - cpu0, 2), hops_missed=int(missed)))
                last_sample_t = te
                if len(series) % 30 == 1:
                    print(f"  t={te - t0:6.0f}s hop {h}  RSS {ws:.1f} MB  missed {int(missed)}  exceptions {exceptions}", flush=True)
            h += 1
        if h >= n_hops:
            break
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0
    ws_end, peak = mem()

    pm = proc_ms[:h]
    # memory growth: linear fit of working set vs time after a 60 s warm-up
    ser = [s for s in series if s["t_s"] >= 60]
    slope = float(np.polyfit([s["t_s"] for s in ser], [s["ws_mb"] for s in ser], 1)[0] * 3600) if len(ser) > 5 else float("nan")
    with open(out / "soak_timeseries.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(series[0]))
        w.writeheader()
        w.writerows(series)
    sf.write(out / "output_first60s.wav", np.clip(np.concatenate(out_first), -1, 1), 16000)
    sf.write(out / "output_last60s.wav", np.clip(np.concatenate(out_last), -1, 1), 16000)

    # ---- equivalence check 2: state drift (final instance vs fresh instance, last 30 s of input)
    tail = inp_tail[-30 * 62:]
    fresh = StreamProcessor(str(ROOT / a.model), ch=a.ch, threads=1)
    y_long, y_fresh = [], []
    for p, r in tail:
        y_long.append(rt.process(p, r))
        y_fresh.append(fresh.process(p, r))
    y_long, y_fresh = np.concatenate(y_long), np.concatenate(y_fresh)
    skip = 10 * 16000
    drift_db = agreement_db(y_fresh[skip:], y_long[skip:])
    # control: an instance with only 60 s of history. Recurrent state keeps some memory, so "fresh vs long-run" differs a little
    # even without any drift; the fair reading is the long-run difference compared with this natural difference.
    ctrl = StreamProcessor(str(ROOT / a.model), ch=a.ch, threads=1)
    for ci, pc, rc in clip_stream(paths[300:], 60):
        for i in range(0, len(pc) - HOP + 1, HOP):
            ctrl.process(pc[i:i + HOP], rc[i:i + HOP])
    y_ctrl = np.concatenate([ctrl.process(p, r) for p, r in tail])
    ctrl_vs_fresh_db = agreement_db(y_fresh[skip:], y_ctrl[skip:])
    long_vs_ctrl_db = agreement_db(y_ctrl[skip:], y_long[skip:])

    # ---- equivalence check 1: long-form streaming (fixed gain, no guard) vs offline PyTorch
    import torch
    from export_onnx import offline_reference
    from anc.model.gtcrn_x import SIZES, GTCRNX
    st = torch.load(ROOT / a.ckpt, map_location="cpu")
    off = GTCRNX(SIZES[st["size"]], st["n_extra"]).eval()
    off.load_state_dict(st["model"])
    off.use_flag = st.get("use_flag", True)
    seg_p, seg_r = [], []
    for ci, pc, rc in clip_stream(paths[200:], 120):
        seg_p.append(pc)
        seg_r.append(rc)
    seg_p, seg_r = np.concatenate(seg_p)[:120 * 16000], np.concatenate(seg_r)[:120 * 16000]
    gain = 0.05 / np.sqrt(np.mean(seg_p ** 2))
    rt2 = StreamProcessor(str(ROOT / a.model), ch=a.ch, use_flag=off.use_flag, fixed_gain=gain, guard=False, threads=1)
    outs = [rt2.process(seg_p[i:i + HOP], seg_r[i:i + HOP]) for i in range(0, len(seg_p) - HOP + 1, HOP)] + [rt2.process(np.zeros(HOP), np.zeros(HOP))]
    n = (len(seg_p) // HOP) * HOP
    y_str = np.concatenate(outs)[HOP:HOP + n]
    y_off = offline_reference(off, seg_p[:n], seg_r[:n], gain)
    inner = slice(2 * HOP, n - 2 * HOP)
    long_db = agreement_db(y_off[inner], y_str[inner])
    long_max = float(np.abs(y_off[inner] - y_str[inner]).max())

    S = dict(model=a.model, requested_minutes=a.minutes, wall_seconds=round(wall, 1), total_hops_scheduled=int(n_hops), processed_hops=int(h),
             missed_deadlines=int(missed), missed_fraction=float(missed / max(h, 1)), max_lateness_ms=float(late_ms[:h].max()),
             proc_ms_mean=float(pm.mean()), proc_ms_p50=float(np.percentile(pm, 50)), proc_ms_p95=float(np.percentile(pm, 95)),
             proc_ms_p99=float(np.percentile(pm, 99)), proc_ms_p999=float(np.percentile(pm, 99.9)), proc_ms_max=float(pm.max()),
             ram_after_imports_mb=round(base_ws, 1), ram_start_mb=series[0]["ws_mb"], ram_end_mb=round(ws_end, 1), ram_peak_mb=round(peak, 1),
             ram_growth_mb_per_hour_after_60s=slope, process_cpu_percent=float(100 * cpu / wall),
             compute_duty_cycle_percent=float(100 * pm.sum() / 1000 / wall),
             exceptions=int(exceptions), non_finite_output_hops=int(nonfinite), silent_gap_hops=int(gap_hops),
             guard_flagged_hops=int(guard_invalid), guard_flagged_percent=float(100 * guard_invalid / max(h, 1)),
             output_peak_abs=float(max(np.abs(np.concatenate(out_last)).max(), np.abs(np.concatenate(out_first)).max())),
             longform_streaming_vs_offline_agreement_db=float(long_db), longform_max_abs_diff=long_max, longform_seconds=120,
             state_drift_agreement_db=float(drift_db), state_drift_control_60s_history_vs_fresh_db=float(ctrl_vs_fresh_db),
             state_drift_longrun_vs_60s_history_db=float(long_vs_ctrl_db),
             state_drift_note="last 30 s of input, first 10 s skipped; drift is indicated only if the long-run instance differs from the fresh one much more than a 60 s-history control does")
    json.dump(S, open(out / "soak_summary.json", "w"), indent=1)
    md = [f"# Soak test: {a.model}\n", "| measure | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in S.items()]
    (out / "soak_summary.md").write_text("\n".join(md) + "\n")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
