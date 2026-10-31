#!/usr/bin/env python3
"""Per-method Core-7 plots for the final presentation (M1-M7, four models).

Inputs (read-only):
- consolidation/report/data/qwen25vl_7b_core7_wandb (audited Qwen W&B tables);
- consolidation/core7_release_results/{internvl35_8b,llava_ov_7b_si}_core7;
- consolidation/report/data/tuned_lens_full_01 (Logit Lens and Tuned Lens, all four models);
- presentation/final_beamer/data/paligemma_core7 (PaliGemma CLEVR Core-7, exploratory);
- the corrected pair-balanced Qwen M3 values used by the final report.
Outputs: presentation/final_beamer/assets/core7_m{1..7}.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DECK = Path(__file__).resolve().parents[1]
ROOT = DECK.parents[1]
DATA = ROOT / "consolidation" / "report" / "data"
RELEASE = ROOT / "consolidation" / "core7_release_results"
TL = DATA / "tuned_lens_full_01"
OUT = DECK / "assets"

MODELS = {
    "qwen": {"name": "Qwen2.5-VL", "layers": 28, "color": "#2855b8", "tl": "qwen25vl_7b",
             "conf": DATA / "qwen25vl_7b_core7_wandb" / "confirmation"},
    "intern": {"name": "InternVL3.5", "layers": 36, "color": "#d65f35", "tl": "internvl35_8b",
               "conf": RELEASE / "internvl35_8b_core7" / "confirmation" / "tables",
               "heads": RELEASE / "internvl35_8b_core7" / "confirmation" / "heads_confirmation_summary.json"},
    "llava": {"name": "LLaVA-OV", "layers": 28, "color": "#2a9d55", "tl": "llava_ov_7b_si",
              "conf": RELEASE / "llava_ov_7b_si_core7" / "confirmation" / "tables",
              "heads": RELEASE / "llava_ov_7b_si_core7" / "confirmation" / "heads_confirmation_summary.json"},
    "pali": {"name": "PaliGemma2", "layers": 42, "color": "#7a4fa3", "tl": "paligemma2_10b_mix",
             "conf": DECK / "data" / "paligemma_core7"},
}

# Corrected pair-balanced Qwen M3 confirmation (k = 1, 2, 4, 8, 16), as in the final report.
QWEN_M3 = {"mean": [0.115560, 0.244082, 0.417044, 0.683652, 0.894855],
           "low": [0.104577, 0.230427, 0.397349, 0.664095, 0.879359],
           "high": [0.127635, 0.258873, 0.437539, 0.702845, 0.909297],
           "p95": [0.013711, 0.069404, 0.069131, 0.012863, 0.004914]}

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "legend.fontsize": 6.2, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.labelcolor": "#333333",
                     "xtick.color": "#555555", "ytick.color": "#555555"})


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def curve(table, **match):
    picked = [r for r in table if all(r.get(k) == v for k, v in match.items())]
    key = "stage" if "stage" in picked[0] else "layer"
    picked.sort(key=lambda r: int(r[key]))
    return (np.array([int(r[key]) for r in picked]), np.array([float(r["mean"]) for r in picked]),
            np.array([float(r["ci_low"]) for r in picked]), np.array([float(r["ci_high"]) for r in picked]))


def depth(x, spec, stages=False):
    return 100 * x / (spec["layers"] if stages else spec["layers"] - 1)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def lens_panel(ax, metric, scheme, ref=None, xmin=0):
    for spec in MODELS.values():
        for lens, ls in (("logit", "--"), ("tuned", "-")):
            x, y, _, _ = curve(rows(TL / spec["tl"] / f"m4_m5_m7_{lens}.csv"), metric=metric, scheme=scheme)
            ax.plot(depth(x, spec, stages=True), y, color=spec["color"], ls=ls, lw=1.1,
                    alpha=0.6 if spec is MODELS["pali"] else 1.0)
    if ref is not None:
        ax.axhline(ref, color="black", lw=0.5, alpha=0.4)
    ax.set_xlim(xmin, 101)
    ax.set_xlabel("relative depth (% of residual stages)")


def lens_legend(fig):
    handles = [plt.Line2D([], [], color=s["color"], lw=1.4, label=s["name"]) for s in MODELS.values()]
    handles += [plt.Line2D([], [], color="black", ls="--", lw=1, label="Logit Lens"),
                plt.Line2D([], [], color="black", ls="-", lw=1, label="Tuned Lens")]
    fig.legend(handles=handles, frameon=False, loc="outside lower center", ncol=6, handlelength=1.6)


def m4():
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.55), constrained_layout=True)
    lens_panel(axes[0], "correct_restricted_prob", "letters", ref=0.25, xmin=40)
    axes[0].set_title("correct-label probability (A/B/C/D)")
    axes[0].set_ylabel("mean over 1,024 prompts")
    lens_panel(axes[1], "label_mass_full", "letters", xmin=40)
    axes[1].set_title("probability mass on the four labels")
    lens_legend(fig)
    save(fig, "core7_m4.png")


def m5():
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.55), constrained_layout=True)
    lens_panel(axes[0], "margin", "letters", ref=0.0, xmin=40)
    axes[0].set_title("label margin (A/B/C/D)")
    axes[0].set_ylabel("logits")
    lens_panel(axes[1], "content_margin", "letters", ref=0.0, xmin=40)
    axes[1].set_title("content-word margin")
    axes[1].set_ylim(-8, 4)
    lens_legend(fig)
    save(fig, "core7_m5.png")


def m7():
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.55), constrained_layout=True)
    lens_panel(axes[0], "requested_mass_8way", "numbers", ref=0.5, xmin=40)
    axes[0].set_title("requested-alphabet share, 1/2/3/4")
    axes[0].set_ylabel("share vs. default A/B/C/D")
    lens_panel(axes[1], "requested_mass_8way", "rare_letters", ref=0.5, xmin=40)
    axes[1].set_title("requested-alphabet share, Q/Z/R/X")
    lens_legend(fig)
    save(fig, "core7_m7.png")


def grid(plot_one, name, ylabel):
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 3.05), sharex=True, constrained_layout=True)
    for ax, spec in zip(axes.flat, MODELS.values()):
        plot_one(ax, spec)
        ax.set_title(spec["name"], color=spec["color"])
        ax.axhline(0, color="black", lw=0.4, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("relative depth (% of decoder layers)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False, loc="upper left", handlelength=1.6)
    save(fig, name)


def m1():
    style = {"position": ("#2878b5", "-", "position"), "content": ("#d94b3d", "--", "content"),
             "random_pair": ("#7f7f7f", ":", "random pair")}

    def one(ax, spec):
        table = rows(spec["conf"] / "m1_residual_confirmation.csv")
        for contrast, (color, ls, label) in style.items():
            x, y, lo, hi = curve(table, contrast_type=contrast, metric="normalized_recovery")
            ax.plot(depth(x, spec), y, color=color, ls=ls, lw=1.2, label=label)
            ax.fill_between(depth(x, spec), lo, hi, color=color, alpha=0.15, lw=0)
        ax.axhline(0.5, color="black", lw=0.5, alpha=0.4)
        ax.set_ylim(-0.15, 1.15)
    grid(one, "core7_m1.png", "recovery")


def m2():
    style = {"attn": ("#2878b5", "attention"), "mlp": ("#ee8a28", "MLP")}

    def one(ax, spec):
        table = rows(spec["conf"] / "m2_components_confirmation.csv")
        for site, (color, label) in style.items():
            x, y, lo, hi = curve(table, contrast_type="position", site=site, metric="normalized_recovery")
            ax.plot(depth(x, spec), y, color=color, lw=1.2, label=label)
            ax.fill_between(depth(x, spec), lo, hi, color=color, alpha=0.15, lw=0)
        ax.set_ylim(-0.1, 0.75)
    grid(one, "core7_m2.png", "recovery")


def m6():
    style = {"image": ("#2878b5", "image"), "question": ("#d94b3d", "question"),
             "options": ("#2a9d55", "options"), "labels": ("#ee8a28", "labels")}

    def one(ax, spec):
        table = rows(spec["conf"] / "m6_token_groups_confirmation.csv")
        for group, (color, label) in style.items():
            x, y, lo, hi = curve(table, contrast_type="content", group=group, metric="normalized_recovery")
            ax.plot(depth(x, spec), y, color=color, lw=1.1, marker="o", ms=1.6, label=label)
            ax.fill_between(depth(x, spec), lo, hi, color=color, alpha=0.12, lw=0)
        ax.set_ylim(-0.3, 1.6)
    grid(one, "core7_m6.png", "recovery")


def head_sets(key):
    if key == "qwen":
        return QWEN_M3
    if key == "pali":
        picked = {int(r["k"]): r for r in rows(MODELS["pali"]["conf"] / "head_sets.csv")
                  if r["phase"] == "confirmation"}
        ks = [1, 2, 4, 8, 16]
        return {f: [float(picked[k][c]) for k in ks]
                for f, c in (("mean", "mean"), ("low", "ci_low"), ("high", "ci_high"), ("p95", "random_p95"))}
    per = json.loads(MODELS[key]["heads"].read_text())["per_k"]
    ks = ["1", "2", "4", "8", "16"]
    return {"mean": [per[k]["top_k"]["mean"] for k in ks], "low": [per[k]["top_k"]["ci_low"] for k in ks],
            "high": [per[k]["top_k"]["ci_high"] for k in ks], "p95": [per[k]["random_p95"] for k in ks]}


def m3():
    fig, ax = plt.subplots(figsize=(4.8, 2.8), constrained_layout=True)
    k = np.array([1, 2, 4, 8, 16])
    for key, spec in MODELS.items():
        m = head_sets(key)
        mean, low, high = map(np.array, (m["mean"], m["low"], m["high"]))
        ax.errorbar(k, mean, yerr=[mean - low, high - mean], color=spec["color"], marker="o", ms=3,
                    lw=1.3, capsize=2, label=spec["name"])
        ax.plot(k, m["p95"], color=spec["color"], ls="--", lw=0.9, marker="s", ms=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(k, [str(v) for v in k])
    ax.set_xlabel("number of frozen top heads patched")
    ax.set_ylabel("held-out recovery")
    ax.set_title("frozen top-$k$ heads (solid) vs. random-set 95th pct. (dashed)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "core7_m3.png")


if __name__ == "__main__":
    for make in (m4, m5, m7, m1, m2, m3, m6):
        make()
