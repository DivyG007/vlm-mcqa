#!/usr/bin/env python3
"""Make a compact, paper-readable summary of the Qwen-3B A/B/C/D pilot."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POSITION_TO_LABEL = {1: "A", 2: "B", 3: "C", 4: "D"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows_by_stage: dict[int, list[dict[str, str]]] = defaultdict(list)
    with args.input.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_by_stage[int(row["stage_index"])].append(row)

    # Stage 0 is the final prompt-token embedding, before any language layer.
    # Its large token-identity baseline obscures the late computation, so the
    # paper plot starts after that baseline and explains the omission.
    stages = np.array(sorted(stage for stage in rows_by_stage if stage > 0))
    correct_logits: list[float] = []
    incorrect_logits: list[float] = []
    common_logits: list[float] = []
    margins: list[float] = []
    probabilities: list[float] = []

    for stage in stages:
        rows = rows_by_stage[int(stage)]
        correct = []
        incorrect = []
        all_logits = []
        per_position_margin: dict[int, float] = {}
        per_position_probability: dict[int, float] = {}
        for row in rows:
            position = int(row["correct_position"])
            value = float(row["mean_logit"])
            all_logits.append(value)
            if row["label"] == POSITION_TO_LABEL[position]:
                correct.append(value)
            else:
                incorrect.append(value)
            per_position_margin[position] = float(row["mean_correct_margin"])
            per_position_probability[position] = float(row["mean_correct_probability"])
        correct_logits.append(float(np.mean(correct)))
        incorrect_logits.append(float(np.mean(incorrect)))
        common_logits.append(float(np.mean(all_logits)))
        margins.append(float(np.mean(list(per_position_margin.values()))))
        probabilities.append(float(np.mean(list(per_position_probability.values()))))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.55), constrained_layout=True)

    ax = axes[0]
    ax.plot(stages, correct_logits, color="#2457d6", linewidth=2.0,
            label="correct label")
    ax.plot(stages, incorrect_logits, color="#d74a3f", linewidth=1.9,
            label="mean wrong label")
    ax.plot(stages, common_logits, color="#666666", linewidth=1.3,
            linestyle="--", label="mean of all four")
    ax.axvline(28, color="#111111", linewidth=0.9, linestyle=":")
    ax.set_ylabel("logit")
    ax.set_xlim(1, 36)
    ax.set_title("(a) Generic label rise + correct-label separation", fontsize=8.5)
    ax.legend(fontsize=6.8, frameon=True, loc="upper left")
    ax.tick_params(labelsize=7)
    ax.text(35.8, correct_logits[-1] + 0.2, f"{correct_logits[-1]:.1f}",
            ha="right", va="bottom", color="#2457d6", fontsize=6.8)
    ax.text(35.8, incorrect_logits[-1] - 0.3, f"{incorrect_logits[-1]:.1f}",
            ha="right", va="top", color="#a72d27", fontsize=6.8)

    ax = axes[1]
    probability_axis = ax.twinx()
    margin_line = ax.plot(stages, margins, color="#198754", linewidth=2.0,
                          label="correct $-$ best wrong")
    probability_line = probability_axis.plot(
        stages,
        probabilities,
        color="#b56a00",
        linewidth=1.8,
        label="restricted 4-way probability",
    )
    ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    probability_axis.axhline(0.25, color="#b56a00", linewidth=0.8, linestyle=":")
    ax.axvline(28, color="#111111", linewidth=0.9, linestyle=":")
    ax.set_xlabel("residual-stream stage")
    ax.set_ylabel("logit margin", color="#198754")
    probability_axis.set_ylabel("restricted probability", color="#b56a00")
    probability_axis.set_ylim(0, 1.04)
    ax.set_xlim(1, 36)
    ax.set_title("(b) Relative decision becomes readable at stage 28", fontsize=8.5)
    ax.tick_params(axis="both", labelsize=7)
    probability_axis.tick_params(axis="y", labelsize=7)
    lines = margin_line + probability_line
    ax.legend(lines, [line.get_label() for line in lines], fontsize=6.6,
              frameon=True, loc="upper left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    main()
