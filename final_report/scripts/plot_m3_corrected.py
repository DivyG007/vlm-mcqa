#!/usr/bin/env python3
"""Render the authoritative pair-balanced Qwen Method-3 result."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    k = np.array([1, 2, 4, 8, 16])
    mean = np.array([0.115560, 0.244082, 0.417044, 0.683652, 0.894855])
    low = np.array([0.104577, 0.230427, 0.397349, 0.664095, 0.879359])
    high = np.array([0.127635, 0.258873, 0.437539, 0.702845, 0.909297])
    random_p95 = np.array([0.013711, 0.069404, 0.069131, 0.012863, 0.004914])
    all_heads = 0.791319

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    ax.errorbar(
        k,
        mean,
        yerr=np.vstack([mean - low, high - mean]),
        marker="o",
        linewidth=2.2,
        capsize=3,
        color="#2357d9",
        label="Frozen top-k heads (95% pair-clustered CI)",
    )
    ax.plot(k, random_p95, marker="s", linestyle="--", color="#d94343",
            label="Matched random-set 95th percentile")
    ax.axhline(all_heads, color="#2d8a57", linestyle=":", linewidth=2,
               label="All-head reference")
    ax.set_xscale("log", base=2)
    ax.set_xticks(k, [str(value) for value in k])
    ax.set_xlabel("Number of frozen heads patched")
    ax.set_ylabel("Normalized recovery")
    ax.set_ylim(-0.02, 0.98)
    ax.set_title("Qwen2.5-VL-7B: corrected pair-balanced head localization")
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    ax.text(4, 0.445, "4 heads: 0.417 recovery\n(52.7% of all-head effect)",
            fontsize=8, color="#1d3f8f")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

