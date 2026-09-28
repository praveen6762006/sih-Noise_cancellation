"""P9: publication-quality plots (PNG 300 dpi + vector PDF) into results/plots/. Every plot reads saved result files.

  python scripts/make_plots.py [--only 1 5 9]

 1 dSI-SNR vs input SNR            2 STOI vs input SNR            3 PESQ vs input SNR        (fresh final sets, Model H vs references)
 4 per-noise-class performance     (fresh final_A and final_B)
 5 single-mic vs two-mic           6 impulse-flag ablation        7 NLMS ablation            (earlier test set: ablation protocol)
 8 reference-leakage robustness    (copied from leakage_analysis.py output)
 9 model size vs performance       (P5 data + complexity)
10 weapon SPL vs STOI              11 weapon SPL vs PESQ          (fresh final weapon sets + earlier weapon set)
12 failure analysis               (copied from failure_analysis.py output)
Error bars / bands are 95% bootstrap intervals over clips. Plots whose inputs are missing are skipped with a message.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_analysis import CLASSES, PEAKS, ROOT, SNRS, group, load, stat   # noqa: E402

import matplotlib   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 100, "savefig.bbox": "tight", "legend.fontsize": 8})
OUT = ROOT / "results" / "plots"
FT = ROOT / "results" / "final_test"
R = ROOT / "results"
COL = {"H": "#1b6ca8", "gtcrn": "#e08a1e", "none": "#7a7a7a", "A": "#d1495b", "B": "#2a9d8f", "C": "#6a4c93", "G": "#8ab17d", "E": "#e9c46a"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("  saved", name)


def have(*paths):
    miss = [p for p in paths if not Path(p).exists()]
    if miss:
        print("  skipped (missing:", ", ".join(Path(m).name for m in miss), ")")
    return not miss


def errbar(ax, x, rows, metric, **kw):
    ys, lo, hi = [], [], []
    for r in rows:
        m, l, h, _ = stat(r, metric)
        ys.append(m); lo.append(m - l); hi.append(h - m)
    ax.errorbar(x, ys, yerr=[lo, hi], capsize=2.5, **kw)


LAB = {"dsisnr": "$\\Delta$SI-SNR (dB)", "stoi": "STOI", "pesq": "PESQ (wideband)"}
THR = {"stoi": 0.85, "pesq": 2.5}


def snr_curves(metric, name):
    """1-3: metric vs input SNR, Model H vs pretrained GTCRN vs no processing, both fresh sets."""
    P = {(t, s): FT / f"P2_{t}_final{s}_main.json" for t in ("H", "gtcrn", "none") for s in "AB"}
    if not have(*P.values()):
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, s in zip(ax, "AB"):
        for t, lab, ls in (("H", "Model H (deployment)", "-"), ("gtcrn", "pretrained GTCRN", "--"), ("none", "no processing", ":")):
            rows = load(P[(t, s)], f"data/generated/final_{s}/main")
            g = group(rows, lambda r: r["snr"])
            errbar(a, SNRS, [g[x] for x in SNRS], metric, color=COL[t], marker="o", ls=ls, label=lab)
        if metric in THR:
            a.axhline(THR[metric], color="r", lw=0.8, ls="-.", label=f"statement target ({THR[metric]})")
        if metric == "dsisnr":
            a.axhline(15, color="r", lw=0.8, ls="-.", label="strict target ($\\Delta$SI-SNR 15 dB)")
        a.set_title(f"final_{s}: " + ("unused test-clean utterances" if s == "A" else "test-other speakers (new)"))
        a.set_xlabel("input SNR (dB)")
        a.set_xticks(SNRS)
    ax[0].set_ylabel(LAB[metric])
    ax[0].legend()
    fig.suptitle(f"{LAB[metric]} vs input SNR, all noise classes pooled (mean, 95% CI)")
    save(fig, name)


def per_class():
    P = {(t, s): FT / f"P2_{t}_final{s}_main.json" for t in ("H", "gtcrn", "none") for s in "AB"}
    if not have(*P.values()):
        return
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for row, s in enumerate("AB"):
        for a, metric in zip(ax[row], ("dsisnr", "stoi", "pesq")):
            for k, (t, lab) in enumerate((("none", "no processing"), ("gtcrn", "pretrained GTCRN"), ("H", "Model H"))):
                g = group(load(P[(t, s)], f"data/generated/final_{s}/main"), lambda r: r["cls"])
                errbar(a, np.arange(8) + (k - 1) * 0.25, [g[c] for c in CLASSES], metric, color=COL[t], marker="s", ls="none", label=lab)
            a.set_xticks(range(8), CLASSES, rotation=35)
            a.set_ylabel(LAB[metric])
            if metric in THR:
                a.axhline(THR[metric], color="r", lw=0.8, ls="-.")
        ax[row][0].set_title(f"final_{s}")
    ax[0][0].legend()
    fig.suptitle("Per-noise-class performance on the fresh final sets (all SNRs pooled; red = statement target)")
    fig.tight_layout()
    save(fig, "04_per_class")


def bars(specs, files, title, name, sets="test", man="data/generated/test", metrics=("dsisnr", "stoi", "pesq")):
    """Grouped bars over models for the earlier-test ablations."""
    if not have(*[p for _, p in specs]):
        return
    fig, ax = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.2))
    for a, metric in zip(np.atleast_1d(ax), metrics):
        for k, (lab, p) in enumerate(specs):
            rows = load(p, man)
            m, lo, hi, _ = stat(rows, metric)
            a.bar(k, m, yerr=[[m - lo], [hi - m]], color=list(COL.values())[k % len(COL)], capsize=3)
            a.text(k, m, f"{m:.3f}" if metric == "stoi" else f"{m:.2f}", ha="center", va="bottom", fontsize=7)
        a.set_xticks(range(len(specs)), [s[0] for s in specs], rotation=25, ha="right")
        a.set_ylabel(LAB[metric])
        if metric == "stoi":
            a.set_ylim(min(0.7, a.get_ylim()[0]), 0.95)
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, name)


def single_vs_two():
    specs = [("no processing", R / "final_none_test.json"), ("pretrained GTCRN", R / "final_gtcrn_test.json"),
             ("single mic (A)", R / "final_A_mono_s0_test.json"), ("two mic (B)", R / "final_B_ref_s0_test.json"),
             ("two mic, seed 1", R / "final_C_full_s1_test.json"), ("single mic, seed 1", R / "final_A_mono_s1_test.json")]
    if not have(*[p for _, p in specs]):
        return
    # per-class paired benefit of the reference: (two mic B) - (single mic A)
    A = {r["file"]: r for r in load(R / "final_A_mono_s0_test.json", "data/generated/test")}
    B = load(R / "final_B_ref_s0_test.json", "data/generated/test")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    rng = np.random.default_rng(0)
    for a, metric in zip(ax[:2], ("dsisnr", "stoi")):
        for k, (lab, p) in enumerate(specs[2:]):
            rows = load(p, "data/generated/test")
            errbar(a, [k], [rows], metric, color=list(COL.values())[k], marker="o", ls="none", label=lab)
        a.set_xticks([])
        a.set_ylabel(LAB[metric])
        a.legend(fontsize=7)
    diffs, los, his = [], [], []
    for c in CLASSES:
        d = np.array([b["dsisnr"] - A[b["file"]]["dsisnr"] for b in B if b["cls"] == c])
        idx = rng.integers(0, len(d), (2000, len(d)))
        m = d[idx].mean(1)
        diffs.append(d.mean()); los.append(d.mean() - np.percentile(m, 2.5)); his.append(np.percentile(m, 97.5) - d.mean())
    ax[2].bar(range(8), diffs, yerr=[los, his], color=COL["B"], capsize=3)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xticks(range(8), CLASSES, rotation=35)
    ax[2].set_ylabel("two-mic minus single-mic $\\Delta$SI-SNR (dB)")
    ax[2].set_title("benefit of the reference mic per class (paired)")
    fig.suptitle("Single-mic vs two-mic (controlled ablation, earlier test set, both warm-started, same training)")
    fig.tight_layout()
    save(fig, "05_single_vs_two_mic")


def flag_ablation():
    bars([("two mic, no flag (B)", R / "final_B_ref_s0_test.json"), ("+ impulse flag (C)", R / "final_C_full_s0_test.json"),
          ("no impulse-frame loss weight (G)", R / "final_G_noimpw_s0_test.json")], None,
         "Impulse-flag ablation (earlier test set; differences are within a few hundredths of a dB)", "06_impulse_flag_ablation")


def nlms():
    specs = [("no processing", R / "final_none_test.json"), ("NLMS only (gated)", R / "final_nlms_only_test.json"), ("NLMS only, gate OFF", R / "final_nlms_only_nogate_test.json"),
             ("model (C)", R / "final_C_full_s0_test.json"), ("model + NLMS after", R / "final_C_full_s0_nlmspost_test.json"),
             ("model + NLMS before", R / "final_C_full_s0_nlmspre_test.json"), ("trained with NLMS before (E)", R / "final_E_nlmspre_s0_test.json")]
    bars(specs, None, "NLMS ablation (earlier test set, 95% CI): every NLMS variant has lower dSI-SNR than the model alone; STOI differs by <= 0.007", "07_nlms_ablation")


def leakage_and_failure():
    for src, name in ((R / "leak_sweep" / "plots" / "leakage_sweep", "08_leakage_robustness"), (R / "leak_sweep" / "plots" / "leakage_benefit_by_snr", "08b_leakage_benefit_by_snr"),
                      (R / "failure_analysis" / "plots" / "failure_heatmaps", "12a_failure_heatmaps"), (R / "failure_analysis" / "plots" / "failure_scatter", "12b_failure_scatter"),
                      (R / "failure_analysis" / "plots" / "failure_worst_clips", "12c_failure_worst_clips")):
        if Path(f"{src}.png").exists():
            OUT.mkdir(parents=True, exist_ok=True)
            for ext in ("png", "pdf"):
                shutil.copy(f"{src}.{ext}", OUT / f"{name}.{ext}")
            print("  copied", name)
        else:
            print("  skipped", name, "(source not generated yet)")


def size_tradeoff():
    st = R / "complexity" / "static.json"
    P5 = [("Tiny", "Tiny", R / "final_T_tiny_s0_test.json"), ("Small (scratch)", "Small", R / "size_variance" / "J_small_scratch_s0_test.json"), ("Base", "Base", R / "final_L_base_s0_test.json")]
    extra = [("Small (warm start, C)", "Small", R / "final_C_full_s0_test.json"), ("Small robust (H)", "Small", R / "final_H_robust_s0_test.json")]
    if not have(st, *[p for *_, p in P5]):
        return
    stat_ = {r["model"]: r for r in json.load(open(st))}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for a, metric in zip(ax, ("dsisnr", "stoi", "pesq")):
        xs = [stat_[k]["mmacs_per_s_executed"] for _, k, _ in P5]
        errbar(a, xs, [load(p, "data/generated/test") for *_, p in P5], metric, color=COL["A"], marker="o", ls="-", label="from scratch, same conditions")
        for (lab, k, p) in extra:
            if Path(p).exists():
                errbar(a, [stat_[k]["mmacs_per_s_executed"]], [load(p, "data/generated/test")], metric, marker="D", ls="none", label=lab,
                       color=COL["C"] if "warm" in lab else COL["H"])
        for (lab, k, _), x in zip(P5, xs):
            a.annotate(lab.split(" ")[0], (x, a.get_ylim()[0]), fontsize=7, ha="center", va="bottom")
        a.set_xscale("log")
        a.set_xlabel("compute (MMACs/s, computed)")
        a.set_ylabel(LAB[metric])
    ax[0].legend(fontsize=7)
    fig.suptitle("Model size vs performance (earlier test set, 95% CI). Parameters: Tiny 8,389, Small 24,629, Base 80,725")
    fig.tight_layout()
    save(fig, "09_model_size_tradeoff")


def weapons(metric, name):
    P = {(t, s): FT / f"P2_{t}_final{s}_weapon.json" for t in ("H", "gtcrn", "none") for s in "AB"}
    if not have(*P.values()):
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, s in zip(ax, "AB"):
        for t, lab, ls in (("H", "Model H", "-"), ("gtcrn", "pretrained GTCRN", "--"), ("none", "no processing", ":")):
            g = group(load(P[(t, s)], f"data/generated/final_{s}/weapon"), lambda r: r["peak"])
            errbar(a, PEAKS, [g[p] for p in PEAKS], metric, color=COL[t], marker="o", ls=ls, label=lab)
        a.axvline(120, color="grey", lw=1, ls="-.")
        a.text(120.5, a.get_ylim()[0], " INMP441 overload point (120 dB SPL)", fontsize=7, va="bottom", rotation=90)
        a.set_xlabel("peak level at the primary microphone (dB SPL)")
        a.set_title(f"final_{s} (gunshot + artillery pooled)")
        a.set_xticks(PEAKS)
    ax[0].set_ylabel(LAB[metric])
    ax[0].legend()
    fig.suptitle(f"Close-range weapons: {LAB[metric]} vs peak level (mean, 95% CI). Do not judge intelligibility from SI-SNR alone")
    save(fig, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int)
    a = ap.parse_args()
    todo = {1: lambda: snr_curves("dsisnr", "01_dsisnr_vs_input_snr"), 2: lambda: snr_curves("stoi", "02_stoi_vs_input_snr"), 3: lambda: snr_curves("pesq", "03_pesq_vs_input_snr"),
            4: per_class, 5: single_vs_two, 6: flag_ablation, 7: nlms, 8: leakage_and_failure, 9: size_tradeoff,
            10: lambda: weapons("stoi", "10_weapon_spl_vs_stoi"), 11: lambda: weapons("pesq", "11_weapon_spl_vs_pesq")}
    for k, fn in todo.items():
        if not a.only or k in a.only:
            print(f"plot {k}")
            fn()


if __name__ == "__main__":
    main()
