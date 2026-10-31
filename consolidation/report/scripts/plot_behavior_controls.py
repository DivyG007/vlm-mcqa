#!/usr/bin/env python3
"""Plot compact behavioural controls from the verified Qwen summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.input.read_text())
    schemes = summary["accuracy_by_scheme"]
    labels = ["A/B/C/D", "1/2/3/4", "Q/Z/R/X"]
    scheme_values = [schemes["letters"], schemes["numbers"], schemes["rare_letters"]]
    image_labels = ["Clean", "Blank", "Shuffled"]
    image_values = [schemes["letters"], summary["blank_accuracy"], summary["shuffled_accuracy"]]

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), sharey=True)
    colors = ["#2f6fbb", "#e28e2c", "#54a24b"]

    for ax, names, values, title in [
        (axes[0], labels, scheme_values, "Output alphabet"),
        (axes[1], image_labels, image_values, "Image dependence"),
    ]:
        x = np.arange(len(names))
        bars = ax.bar(x, np.asarray(values) * 100, color=colors, width=0.65)
        ax.set_xticks(x, names)
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 100 + 2.2,
                f"{value * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_ylabel("accuracy")
    fig.tight_layout(w_pad=1.7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")


if __name__ == "__main__":
    main()
