#!/usr/bin/env python3
"""Per-experiment plots for the extended studies (E20-E24) and the IconQA results slide.

Inputs (read-only):
- consolidation/stretch_results/report_metrics_2026-10-30.csv (audited stretch scalars);
- the corrected E22 single-option discovery values and InternVL E20 random-head
  differences quoted in the final report (same constants as plot_final_report.py);
- presentation/final_beamer/data/iconqa_four_model (Yashav's four-model IconQA tables).
Outputs: presentation/final_beamer/assets/ext_e20.png, ext_e22.png, ext_e23.png,
ext_e24.png and iconqa_results.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DECK = Path(__file__).resolve().parents[1]
ROOT = DECK.parents[1]
STRETCH = ROOT / "consolidation" / "stretch_results" / "report_metrics_2026-10-30.csv"
ICONQA = DECK / "data" / "iconqa_four_model"
OUT = DECK / "assets"

COLORS = {"qwen": "#2855b8", "intern": "#d65f35", "llava": "#2a9d55", "pali": "#7a4fa3"}
NAMES = {"qwen": "Qwen2.5-VL", "intern": "InternVL3.5", "llava": "LLaVA-OV", "pali": "PaliGemma2"}
ICONQA_KEY = {"qwen": "qwen", "intern": "internvl", "llava": "llava", "pali": "pali"}

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "legend.fontsize": 6.4, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.labelcolor": "#333333",
                     "xtick.color": "#555555", "ytick.color": "#555555"})


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def metric(model, study, name):
    for r in rows(STRETCH):
        if r["model"] == model and r["study"] == study and r["metric"] == name:
            low = float(r["ci_low"]) if r["ci_low"] else None
            high = float(r["ci_high"]) if r["ci_high"] else None
            return float(r["value"]), low, high
    raise KeyError((model, study, name))


def point(ax, x, value, low, high, color, marker="o"):
    err = None if low is None else [[value - low], [high - value]]
    ax.errorbar([x], [value], yerr=err, color=color, marker=marker, ms=4.5, capsize=2.5, lw=1.3)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def e20():
    fig, ax = plt.subplots(figsize=(4.6, 2.5), constrained_layout=True)
    labels = ["image\nrestoration", "top-4 heads\ntransfer", "random heads\ntransfer",
              "top-4 heads\nclamp", "random heads\nclamp"]
    qwen = [metric("qwen25vl_7b", "E20", m) for m in (
        "image_restoration_raw_effect", "selected_four_head_transfer", "matched_random_four_head_transfer",
        "selected_four_head_clamp_attenuation", "matched_random_clamp_attenuation")]
    # InternVL random-head values are implied by the reported selected-minus-random
    # differences (17.8199 and 17.8281), as in the final report.
    it = metric("internvl35_8b", "E20", "selected_four_head_transfer")
    ic = metric("internvl35_8b", "E20", "selected_four_head_clamp_attenuation")
    intern = [metric("internvl35_8b", "E20", "image_restoration_raw_effect"), it,
              (it[0] - 17.8199, None, None), ic, (ic[0] - 17.8281, None, None)]
    x = np.arange(len(labels))
    for offset, values, key in ((-0.19, qwen, "qwen"), (0.19, intern, "intern")):
        heights = [v[0] for v in values]
        errs = np.array([[v[0] - v[1], v[2] - v[0]] if v[1] is not None else [0, 0] for v in values]).T
        ax.bar(x + offset, heights, 0.38, color=COLORS[key], label=NAMES[key], yerr=errs, capsize=2,
               error_kw={"lw": 0.8})
    ax.set_xticks(x, labels)
    ax.set_ylabel("effect on donor answer (logits)")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "ext_e20.png")


def e22():
    fig, ax = plt.subplots(figsize=(4.6, 2.5), constrained_layout=True)
    single = [(-0.1283, -0.2261, -0.0444), (-0.2689, -0.3594, -0.1845), (-0.0383, -0.0557, -0.0222)]
    for i, (v, lo, hi) in enumerate(single):
        point(ax, i, v, lo, hi, "#7f7f7f", "s")
    v, lo, hi = metric("qwen25vl_7b", "E22", "all_option_truth_margin_gain")
    point(ax, 3.5, v, lo, hi, COLORS["qwen"])
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.axvline(2.75, color="#bbbbbb", lw=0.6, ls=":")
    ax.set_xticks([0, 1, 2, 3.5], ["L20\ndiscovery", "L23\ndiscovery", "L25\ndiscovery", "L20\nconfirmation"])
    ax.set_xlim(-0.6, 4.2)
    ax.set_ylabel("Qwen: change in truth margin (logits)")
    ax.set_ylim(-0.42, 0.42)
    save(fig, "ext_e22.png")


def e23():
    fig, ax = plt.subplots(figsize=(4.6, 2.35), constrained_layout=True)
    x = np.arange(2)
    regions = (("question", "question object", "#7b3fa0"), ("option", "option objects", "#20a39e"))
    for offset, (region, label, color) in zip((-0.19, 0.19), regions):
        for xi, model in ((0, "internvl35_8b"), (1, "llava_ov_7b_si")):
            v, lo, hi = metric(model, "E23", f"{region}_region_specificity")
            ax.bar(xi + offset, v, 0.36, color=color, yerr=[[v - lo], [hi - v]], capsize=2.5,
                   error_kw={"lw": 0.8}, label=label if xi == 0 else None)
    ax.axhline(0, color="black", lw=0.6, alpha=0.6)
    ax.set_xticks(x, ["InternVL", "LLaVA"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.055, 0.1)
    ax.set_ylabel("extra drop in answer lead\nvs. control patch (logits)")
    ax.legend(frameon=False, loc="upper left", title="region blocked", title_fontsize=6.4)
    save(fig, "ext_e23.png")


def e24():
    fig, ax = plt.subplots(figsize=(4.6, 2.5), constrained_layout=True)
    items = [("top4_minus_random", "top heads moved C\nminus random heads"),
             ("content_minus_literal_copy_change", "patch moved C\nminus moved A"),
             ("patched_content_minus_literal_preference", "after patch:\nC minus A")]
    for i, (name, _) in enumerate(items):
        for offset, model, key in ((-0.15, "qwen25vl_7b", "qwen"), (0.15, "internvl35_8b", "intern")):
            v, lo, hi = metric(model, "E24", name)
            point(ax, i + offset, v, lo, hi, COLORS[key])
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xticks(range(3), [m[1] for m in items])
    ax.set_ylabel("logits")
    ax.set_ylim(-5.8, 3.6)
    ax.set_title("Head patch, new scenes")
    ax.legend(handles=[plt.Line2D([], [], color=COLORS[k], marker="o", ls="", label=NAMES[k])
                       for k in ("qwen", "intern")], frameon=False, loc="upper right", ncol=2)
    save(fig, "ext_e24.png")


def iconqa():
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    heads = rows(ICONQA / "heads.csv")
    ax = axes[0]
    k = [1, 2, 4, 8, 16]
    for key in ("qwen", "intern", "llava", "pali"):
        for arm, ls in (("iconqa", "-"), ("clevr", "--")):
            picked = {int(r["k"]): float(r["fraction_all"]) for r in heads
                      if r["model"] == ICONQA_KEY[key] and r["arm"] == arm and r["phase"] == "confirmation"}
            ks = [v for v in k if v in picked]  # Qwen's CLEVR heads were tested at k = 4 only
            ax.plot(ks, [100 * picked[v] for v in ks], color=COLORS[key], ls=ls, marker="o", ms=2.5, lw=1.2)
    ax.axhline(50, color="black", lw=0.5, alpha=0.4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(k, [str(v) for v in k])
    ax.set_xlabel("number of heads patched")
    ax.set_ylabel("% of all-head effect")
    ax.set_title("IconQA-chosen (solid) vs. CLEVR (dashed) heads")
    ax = axes[1]
    controls = rows(ICONQA / "fixed_text_controls.csv")
    x = np.arange(4)
    for offset, cond, alpha, label in ((-0.18, "counterfactual", 1.0, "answer-changing edit"),
                                        (0.18, "same_answer", 0.4, "same-answer edit")):
        vals = [next(float(r["mean"]) for r in controls if r["model"] == ICONQA_KEY[key] and r["condition"] == cond
                     and r["phase"] == "confirmation") for key in ("qwen", "intern", "llava", "pali")]
        ax.bar(x + offset, vals, 0.36, color=[COLORS[k] for k in ("qwen", "intern", "llava", "pali")],
               alpha=alpha, label=label)
    ax.set_xticks(x, ["Qwen", "InternVL", "LLaVA", "PaliGemma"], rotation=20)
    ax.set_ylabel("answer shift (logits)")
    ax.set_title("CLEVR heads, image-only change")
    ax.set_ylim(0, 17.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=1.0, label="answer-changing edit"),
               plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=0.4, label="same-answer edit")]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    handles = [plt.Line2D([], [], color=COLORS[k], lw=1.4, label=NAMES[k]) for k in COLORS]
    fig.legend(handles=handles, frameon=False, loc="outside lower center", ncol=4)
    save(fig, "iconqa_results.png")


if __name__ == "__main__":
    for make in (e20, e22, e23, e24, iconqa):
        make()
