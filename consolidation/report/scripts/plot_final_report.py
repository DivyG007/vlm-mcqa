#!/usr/bin/env python3
"""Create the figures for the final report from audited, versioned inputs.

Inputs (all read-only):
- ``consolidation/core7_release_results/{internvl35_8b,llava_ov_7b_si}_core7``;
- ``consolidation/report/data/qwen25vl_7b_core7_wandb`` (audited W&B tables);
- corrected pair-balanced Qwen M3 values from
  ``docs/ai_handoff/qwen7b_m3_pairbalanced_rerun_audit_2026-10-01.md``;
- ``consolidation/report/data/tuned_lens_full_01`` (Tuned Lens branch);
- ``consolidation/stretch_results/report_metrics_2026-10-30.csv`` plus the
  values quoted in ``consolidation/stretch_results/README.md`` and
  ``docs/rachit_qwen_e20_e24_results_2026-10-27.md``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "consolidation" / "report"
OUT = REPORT / "figures"
RELEASE = ROOT / "consolidation" / "core7_release_results"
DATA = REPORT / "data"
STRETCH = ROOT / "consolidation" / "stretch_results" / "report_metrics_2026-10-30.csv"

MODELS = {
    "qwen": {"name": "Qwen2.5-VL-7B", "layers": 28, "color": "#2855b8",
             "conf": DATA / "qwen25vl_7b_core7_wandb" / "confirmation",
             "tl": "qwen25vl_7b"},
    "intern": {"name": "InternVL3.5-8B", "layers": 36, "color": "#d65f35",
               "conf": RELEASE / "internvl35_8b_core7" / "confirmation" / "tables",
               "heads": RELEASE / "internvl35_8b_core7" / "confirmation" / "heads_confirmation_summary.json",
               "tl": "internvl35_8b"},
    "llava": {"name": "LLaVA-OV-7B-SI", "layers": 28, "color": "#2a9d55",
              "conf": RELEASE / "llava_ov_7b_si_core7" / "confirmation" / "tables",
              "heads": RELEASE / "llava_ov_7b_si_core7" / "confirmation" / "heads_confirmation_summary.json",
              "tl": "llava_ov_7b_si"},
}
PALI = {"name": "PaliGemma2-10B (provisional)", "layers": 42, "color": "#7a4fa3",
        "tl": "paligemma2_10b_mix"}

# Corrected pair-balanced Qwen M3 confirmation (audit table, k = 1, 2, 4, 8, 16).
QWEN_M3 = {"mean": [0.115560, 0.244082, 0.417044, 0.683652, 0.894855],
           "low": [0.104577, 0.230427, 0.397349, 0.664095, 0.879359],
           "high": [0.127635, 0.258873, 0.437539, 0.702845, 0.909297],
           "p95": [0.013711, 0.069404, 0.069131, 0.012863, 0.004914],
           "all": 0.791319}

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def curve(table, **match):
    picked = [r for r in table if all(r[k] == v for k, v in match.items())]
    key = "stage" if "stage" in picked[0] else "layer"
    picked.sort(key=lambda r: int(r[key]))
    return (np.array([int(r[key]) for r in picked]), np.array([float(r["mean"]) for r in picked]),
            np.array([float(r["ci_low"]) for r in picked]), np.array([float(r["ci_high"]) for r in picked]))


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def core7_route() -> None:
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 5.2), sharex=True, constrained_layout=True)
    m1_style = {"position": ("#2878b5", "-", "position"), "content": ("#d94b3d", "--", "content"),
                "random_pair": ("#7f7f7f", ":", "unrelated pair")}
    m6_style = {"image": ("#2878b5", "image"), "question": ("#d94b3d", "question"),
                "options": ("#2a9d55", "options"), "labels": ("#ee8a28", "labels")}
    m2_style = {"attn": ("#2878b5", "attention"), "mlp": ("#ee8a28", "MLP")}
    for row, (key, spec) in enumerate(MODELS.items()):
        last = spec["layers"] - 1
        m1 = rows(spec["conf"] / "m1_residual_confirmation.csv")
        m6 = rows(spec["conf"] / "m6_token_groups_confirmation.csv")
        m2 = rows(spec["conf"] / "m2_components_confirmation.csv")
        ax = axes[row, 0]
        for contrast, (color, ls, label) in m1_style.items():
            x, y, lo, hi = curve(m1, contrast_type=contrast, metric="normalized_recovery")
            ax.plot(100 * x / last, y, color=color, ls=ls, lw=1.4, label=label)
            ax.fill_between(100 * x / last, lo, hi, color=color, alpha=0.15, lw=0)
        ax.axhline(0.5, color="black", lw=0.6, alpha=0.4)
        ax.set_ylabel(f"{spec['name']}\nrecovery")
        ax = axes[row, 1]
        for group, (color, label) in m6_style.items():
            x, y, lo, hi = curve(m6, contrast_type="content", group=group, metric="normalized_recovery")
            ax.plot(100 * x / last, y, color=color, lw=1.4, marker="o", ms=1.8, label=label)
            ax.fill_between(100 * x / last, lo, hi, color=color, alpha=0.15, lw=0)
        ax = axes[row, 2]
        for site, (color, label) in m2_style.items():
            x, y, lo, hi = curve(m2, contrast_type="position", site=site, metric="normalized_recovery")
            ax.plot(100 * x / last, y, color=color, lw=1.4, label=label)
            ax.fill_between(100 * x / last, lo, hi, color=color, alpha=0.15, lw=0)
    for col in range(3):
        lims = [axes[r, col].get_ylim() for r in range(3)]
        lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
        for r in range(3):
            axes[r, col].set_ylim(lo, hi)
            axes[r, col].axhline(0, color="black", lw=0.5, alpha=0.3)
        axes[2, col].set_xlabel("relative depth (% of decoder layers)")
    axes[0, 0].set_title("M1: final-token patch")
    axes[0, 1].set_title("M6: token-group patch (content pairs)")
    axes[0, 2].set_title("M2: component patch (position pairs)")
    for col in range(3):
        axes[0, col].legend(loc="upper left", frameon=False, handlelength=1.8)
    save(fig, "final_core7_route.png")


def head_m3(key):
    if key == "qwen":
        return QWEN_M3
    summary = json.loads(MODELS[key]["heads"].read_text())
    ks = ["1", "2", "4", "8", "16"]
    per = summary["per_k"]
    return {"mean": [per[k]["top_k"]["mean"] for k in ks], "low": [per[k]["top_k"]["ci_low"] for k in ks],
            "high": [per[k]["top_k"]["ci_high"] for k in ks], "p95": [per[k]["random_p95"] for k in ks],
            "all": summary["all_heads_mean_nr"]}


def heads_and_lens() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True)
    k = np.array([1, 2, 4, 8, 16])
    ax = axes[0]
    for key, spec in MODELS.items():
        m3 = head_m3(key)
        mean, low, high = map(np.array, (m3["mean"], m3["low"], m3["high"]))
        ax.errorbar(k, mean, yerr=[mean - low, high - mean], color=spec["color"], marker="o", ms=3,
                    lw=1.4, capsize=2, label=spec["name"])
        ax.plot(k, m3["p95"], color=spec["color"], ls="--", lw=1, marker="s", ms=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(k, [str(v) for v in k])
    ax.set_xlabel("frozen top-$k$ heads")
    ax.set_ylabel("held-out recovery")
    ax.set_title("(a) M3: top-$k$ heads vs random p95 (dashed)")
    ax.legend(frameon=False, loc="upper left")

    specs = list(MODELS.values()) + [PALI]
    panels = [(axes[1], "correct_restricted_prob", "letters", "(b) correct-label prob., A/B/C/D", 0.25),
              (axes[2], "requested_mass_8way", "rare_letters", "(c) requested-alphabet mass, Q/Z/R/X", 0.5)]
    for ax, metric, scheme, title, ref in panels:
        for spec in specs:
            base = DATA / "tuned_lens_full_01" / spec["tl"]
            for lens, ls in (("logit", "--"), ("tuned", "-")):
                x, y, _, _ = curve(rows(base / f"m4_m5_m7_{lens}.csv"), metric=metric, scheme=scheme)
                ax.plot(100 * x / spec["layers"], y, color=spec["color"], ls=ls, lw=1.1,
                        alpha=0.55 if spec is PALI else 1.0)
        ax.axhline(ref, color="black", lw=0.5, alpha=0.4)
        ax.set_xlim(40, 101)
        ax.set_xlabel("relative depth (% of residual stages)")
        ax.set_title(title)
    axes[1].set_ylabel("mean over 1,024 variants")
    handles = [plt.Line2D([], [], color="black", ls="--", label="Logit Lens"),
               plt.Line2D([], [], color="black", ls="-", label="Tuned Lens"),
               plt.Line2D([], [], color=PALI["color"], alpha=0.55, label="PaliGemma2 (prov.)")]
    axes[1].legend(handles=handles, frameon=False, loc="upper left")
    save(fig, "final_heads_lens.png")


def stretch_metric(model, study, metric):
    for r in rows(STRETCH):
        if r["model"] == model and r["study"] == study and r["metric"] == metric:
            value = float(r["value"])
            low = float(r["ci_low"]) if r["ci_low"] else None
            high = float(r["ci_high"]) if r["ci_high"] else None
            return value, low, high
    raise KeyError((model, study, metric))


def point(ax, x, value, low, high, color, marker="o"):
    err = None if low is None else [[value - low], [high - value]]
    ax.errorbar([x], [value], yerr=err, color=color, marker=marker, ms=4, capsize=2.5, lw=1.2)


def stretch() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.45), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.55, 1.0, 0.9, 1.15]})
    qc, ic, lc = MODELS["qwen"]["color"], MODELS["intern"]["color"], MODELS["llava"]["color"]

    # (a) E20. InternVL random-head values are implied by the reported
    # selected-minus-random differences (README: 17.8199 and 17.8281).
    ax = axes[0]
    labels = ["image\nrestore", "top-4\ntransf.", "rand.\ntransf.", "top-4\nclamp", "rand.\nclamp"]
    qwen = [stretch_metric("qwen25vl_7b", "E20", m) for m in (
        "image_restoration_raw_effect", "selected_four_head_transfer", "matched_random_four_head_transfer",
        "selected_four_head_clamp_attenuation", "matched_random_clamp_attenuation")]
    it = stretch_metric("internvl35_8b", "E20", "selected_four_head_transfer")
    ic_clamp = stretch_metric("internvl35_8b", "E20", "selected_four_head_clamp_attenuation")
    intern = [stretch_metric("internvl35_8b", "E20", "image_restoration_raw_effect"), it,
              (it[0] - 17.8199, None, None), ic_clamp, (ic_clamp[0] - 17.8281, None, None)]
    x = np.arange(len(labels))
    for offset, values, color, name in ((-0.18, qwen, qc, "Qwen"), (0.18, intern, ic, "InternVL")):
        heights = [v[0] for v in values]
        errs = np.array([[v[0] - v[1], v[2] - v[0]] if v[1] is not None else [0, 0] for v in values]).T
        ax.bar(x + offset, heights, 0.36, color=color, label=name, yerr=errs, capsize=1.5,
               error_kw={"lw": 0.8})
    ax.set_xticks(x, labels, fontsize=6, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("donor-margin effect (logits)")
    ax.set_title("(a) S1 held-out, 24 pairs")
    ax.text(0.98, 0.97, "LLaVA: not run in time", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color=lc)
    ax.legend(frameon=False, loc="center right")

    # (b) E22 Qwen. Corrected discovery truth-token layers and all-option confirmation.
    ax = axes[1]
    single = [(20, -0.1283, -0.2261, -0.0444), (23, -0.2689, -0.3594, -0.1845), (25, -0.0383, -0.0557, -0.0222)]
    for i, (layer, v, lo, hi) in enumerate(single):
        point(ax, i, v, lo, hi, "#7f7f7f", "s")
    v, lo, hi = stretch_metric("qwen25vl_7b", "E22", "all_option_truth_margin_gain")
    point(ax, 3.4, v, lo, hi, qc)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xticks([0, 1, 2, 3.4], ["L20", "L23", "L25", "L20\nall-opt."])
    ax.set_xlim(-0.6, 4.0)
    ax.set_ylabel("truth-margin change (logits)")
    ax.set_title("(b) S2 Qwen")
    ax.text(0.02, 0.97, "single option,\ndiscovery", transform=ax.transAxes, va="top", fontsize=6, color="#555555")
    ax.text(0.98, 0.42, "35 failures,\n12.9% corrected", transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color=qc)

    # (c) E23 equal-token-control sensitivity.
    ax = axes[2]
    for i, region in enumerate(("question", "option")):
        for offset, model, color in ((-0.14, "internvl35_8b", ic), (0.14, "llava_ov_7b_si", lc)):
            v, lo, hi = stretch_metric(model, "E23", f"{region}_region_specificity")
            point(ax, i + offset, v, lo, hi, color)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xticks([0, 1], ["question", "option"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("specificity (named $-$ matched)")
    ax.set_title("(c) S3, 24 pairs")
    ax.text(0.5, 0.03, "Qwen: rule not met", transform=ax.transAxes, ha="center", va="bottom", fontsize=6, color=qc)
    ax.legend(handles=[plt.Line2D([], [], color=ic, marker="o", ls="", label="InternVL"),
                       plt.Line2D([], [], color=lc, marker="o", ls="", label="LLaVA")],
              frameon=False, loc="upper center", ncol=2, handletextpad=0.1, columnspacing=0.6)
    ax.set_ylim(-0.068, 0.118)

    # (d) E24.
    ax = axes[3]
    metrics = [("top4_minus_random", "top-4\n$-$ random"), ("content_minus_literal_copy_change", "content\n$-$ copy"),
               ("patched_content_minus_literal_preference", "patched\npreference")]
    for i, (metric, _) in enumerate(metrics):
        for offset, model, color in ((-0.14, "qwen25vl_7b", qc), (0.14, "internvl35_8b", ic)):
            v, lo, hi = stretch_metric(model, "E24", metric)
            point(ax, i + offset, v, lo, hi, color)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xticks(range(3), [m[1] for m in metrics], fontsize=6.3)
    ax.set_ylabel("logits")
    ax.set_title("(d) S4, 31 parents")
    ax.text(0.03, 0.03, "LLaVA: running", transform=ax.transAxes, ha="left", va="bottom", fontsize=6, color=lc)
    ax.legend(handles=[plt.Line2D([], [], color=qc, marker="o", ls="", label="Qwen"),
                       plt.Line2D([], [], color=ic, marker="o", ls="", label="InternVL")],
              frameon=False, loc="upper center", ncol=2, handletextpad=0.1, columnspacing=0.6)
    ax.set_ylim(-5.8, 4.3)
    save(fig, "final_stretch.png")


if __name__ == "__main__":
    core7_route()
    heads_and_lens()
    stretch()
