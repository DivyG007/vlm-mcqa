#!/usr/bin/env python3
"""Create compact Qwen/InternVL comparison figures for the mid-semester report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "consolidation" / "report" / "figures"
QWEN = ROOT / "tmp" / "qwen25vl7b_audit" / "qwen25vl_7b_full_core7" / "confirmation"
INTERN = ROOT / "results" / "garuda" / "internvl35_8b_report_prelim_32" / "confirmation"

COLORS = {
    "Qwen2.5-VL-7B": "#2855b8",
    "InternVL3.5-8B": "#d65f35",
    "position": "#2878b5",
    "content": "#d94b3d",
    "random_pair": "#8b8b23",
    "image": "#2878b5",
    "options": "#2a9d55",
    "question": "#d94b3d",
    "labels": "#ee8a28",
    "attn": "#2878b5",
    "mlp": "#ee8a28",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def normalized_layer(row: dict[str, str], final_layer: int) -> float:
    return 100.0 * float(row["layer"]) / final_layer


def behavior() -> None:
    labels = ["Accuracy", "Worst position", "Clean-shuffled gap"]
    qwen = np.array([93.8, 83.3, 64.6])
    intern = np.array([100.0, 100.0, 78.1])
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    q = ax.bar(x - width / 2, qwen, width, label="Qwen2.5-VL-7B",
               color=COLORS["Qwen2.5-VL-7B"])
    i = ax.bar(x + width / 2, intern, width, label="InternVL3.5-8B",
               color=COLORS["InternVL3.5-8B"])
    ax.bar_label(q, fmt="%.1f%%", padding=2, fontsize=8)
    ax.bar_label(i, fmt="%.1f%%", padding=2, fontsize=8)
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 112)
    ax.set_xticks(x, labels)
    ax.set_title("Matched 24-item smoke comparison")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "compare_behavior.png")


def m1_m6() -> None:
    models = [
        ("Qwen2.5-VL-7B", QWEN, 27, "full confirmation, 64 pairs"),
        ("InternVL3.5-8B", INTERN, 35, "preliminary confirmation, 32 pairs"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.1), sharex="col",
                             constrained_layout=True)

    for col, (model, root, final_layer, sample) in enumerate(models):
        m1 = rows(root / "tables" / "m1_residual_confirmation.csv")
        ax = axes[0, col]
        for contrast in ["position", "content", "random_pair"]:
            subset = [r for r in m1 if r["metric"] == "normalized_recovery"
                      and r["contrast_type"] == contrast]
            ax.plot([normalized_layer(r, final_layer) for r in subset],
                    [float(r["mean"]) for r in subset], marker="o", ms=2.5,
                    lw=1.8, color=COLORS[contrast],
                    label=contrast.replace("_", " "))
        ax.axhline(0.5, color="#333333", ls="--", lw=1)
        ax.set_title(f"{model}\nM1 final-token transfer ({sample})", fontsize=10)
        ax.set_ylim(-0.08, 1.08)
        ax.grid(alpha=0.25)
        if col == 0:
            ax.set_ylabel("Normalized recovery")
            ax.legend(frameon=False, fontsize=8, loc="upper left")

        m6 = rows(root / "tables" / "m6_token_groups_confirmation.csv")
        ax = axes[1, col]
        for group in ["image", "options", "question", "labels"]:
            subset = [r for r in m6 if r["metric"] == "normalized_recovery"
                      and r["contrast_type"] == "content" and r["group"] == group]
            ax.plot([normalized_layer(r, final_layer) for r in subset],
                    [float(r["mean"]) for r in subset], marker="o", ms=2.5,
                    lw=1.8, color=COLORS[group], label=group)
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_title("M6 content transfer by token group", fontsize=10)
        ax.set_xlabel("Normalized model depth (%)")
        ax.set_ylim(-0.22, 1.28)
        ax.grid(alpha=0.25)
        if col == 0:
            ax.set_ylabel("Normalized recovery")
            ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    save(fig, "compare_m1_m6.png")


def m2() -> None:
    models = [
        ("Qwen2.5-VL-7B", QWEN, 27, "64 pairs"),
        ("InternVL3.5-8B", INTERN, 35, "32 pairs, preliminary"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.45), constrained_layout=True,
                             sharey=True)
    for ax, (model, root, final_layer, sample) in zip(axes, models):
        data = rows(root / "tables" / "m2_components_confirmation.csv")
        for site in ["attn", "mlp"]:
            subset = [r for r in data if r["metric"] == "normalized_recovery"
                      and r["contrast_type"] == "position" and r["site"] == site]
            ax.plot([normalized_layer(r, final_layer) for r in subset],
                    [float(r["mean"]) for r in subset], marker="o", ms=2.7,
                    lw=1.9, color=COLORS[site],
                    label="Attention" if site == "attn" else "MLP")
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_title(f"{model} ({sample})", fontsize=10)
        ax.set_xlabel("Normalized model depth (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Position-transfer recovery")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("M2: attention carries the larger discriminative update", fontsize=12)
    save(fig, "compare_m2.png")


def m3() -> None:
    k = np.array([1, 2, 4, 8, 16])
    qwen = np.array([0.115560, 0.244082, 0.417044, 0.683652, 0.894855])
    qwen_p95 = np.array([0.013711, 0.069404, 0.069131, 0.012863, 0.004914])
    intern_summary = json.load((INTERN / "heads_confirmation_summary.json").open())
    intern = np.array([intern_summary["per_k"][str(v)]["top_k"]["mean"] for v in k])
    intern_p95 = np.array([intern_summary["per_k"][str(v)]["random_p95"] for v in k])

    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    ax.plot(k, qwen, marker="o", lw=2.1, color=COLORS["Qwen2.5-VL-7B"],
            label="Qwen top-k (64 pairs)")
    ax.plot(k, intern, marker="o", lw=2.1, color=COLORS["InternVL3.5-8B"],
            label="InternVL top-k (32 pairs, preliminary)")
    ax.plot(k, qwen_p95, ls="--", lw=1.2, color=COLORS["Qwen2.5-VL-7B"],
            alpha=0.75, label="Qwen random p95")
    ax.plot(k, intern_p95, ls="--", lw=1.2, color=COLORS["InternVL3.5-8B"],
            alpha=0.75, label="InternVL random p95")
    ax.set_xscale("log", base=2)
    ax.set_xticks(k, [str(v) for v in k])
    ax.set_xlabel("Frozen top-k heads patched")
    ax.set_ylabel("Normalized recovery")
    ax.set_ylim(-0.1, 1.0)
    ax.set_title("M3: cumulative head-set recovery on held-out pairs")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    save(fig, "compare_m3.png")


def lens_milestones() -> None:
    labels = ["A-D label\nreadable", "Requested rare\nalphabet wins", "Content word\nreadable"]
    qwen = np.array([23 / 28, 25 / 28, 27 / 28]) * 100
    intern = np.array([25 / 36, 30 / 36, 32 / 36]) * 100
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    ax.hlines(y, qwen, intern, color="#b9bec8", lw=3, zorder=1)
    ax.scatter(qwen, y, s=70, color=COLORS["Qwen2.5-VL-7B"],
               label="Qwen2.5-VL-7B", zorder=2)
    ax.scatter(intern, y, s=70, color=COLORS["InternVL3.5-8B"],
               label="InternVL3.5-8B", zorder=2)
    for idx, (q, i) in enumerate(zip(qwen, intern)):
        ax.text(q + 0.7, idx - 0.13, f"{q:.0f}%", fontsize=8,
                color=COLORS["Qwen2.5-VL-7B"])
        ax.text(i + 0.7, idx + 0.24, f"{i:.0f}%", fontsize=8,
                color=COLORS["InternVL3.5-8B"])
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(60, 102)
    ax.set_xlabel("Depth at first stable readout (% of language stack)")
    ax.set_title("M4/M5/M7: shared ordering of late readout milestones")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    save(fig, "compare_lens_milestones.png")


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    behavior()
    m1_m6()
    m2()
    m3()
    lens_milestones()


if __name__ == "__main__":
    main()
