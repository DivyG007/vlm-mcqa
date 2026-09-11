"""Create auditable figures and CSV tables for ABCD causal interventions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _safe(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _read_json(args.summary)
    rows = summary["aggregate_rows"]
    _write_csv(args.output_dir / "aggregate_interventions.csv", rows)

    residual = [row for row in rows if row["experiment"] == "single_layer_residual_patch"]
    if residual:
        figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for contrast_type in sorted({row["contrast_type"] for row in residual}):
            selected = sorted(
                [row for row in residual if row["contrast_type"] == contrast_type],
                key=lambda row: int(row["layer_index"]),
            )
            stages = [int(row["layer_index"]) + 1 for row in selected]
            recoveries = _safe([row["mean_normalized_recovery"] for row in selected])
            lows = _safe([row["bootstrap_95_low"] for row in selected])
            highs = _safe([row["bootstrap_95_high"] for row in selected])
            axes[0].plot(stages, recoveries, marker="o", markersize=3, label=contrast_type)
            axes[0].fill_between(stages, lows, highs, alpha=0.16)
            axes[1].plot(
                stages,
                _safe([row["target_flip_rate"] for row in selected]),
                marker="o",
                markersize=3,
                label=contrast_type,
            )
        axes[0].axhline(0, color="0.45", linestyle="--", linewidth=1)
        axes[0].axhline(1, color="0.45", linestyle=":", linewidth=1)
        axes[0].set(title="Single-layer residual patching", xlabel="Post-layer stage", ylabel="Normalized recovery")
        axes[1].set(title="Does the target answer actually flip?", xlabel="Post-layer stage", ylabel="Target flip rate", ylim=(-0.03, 1.03))
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend()
        figure.suptitle("ABCD causal residual atlas", fontsize=15, fontweight="bold")
        figure.savefig(args.output_dir / "single_layer_residual_atlas.png", dpi=180)
        plt.close(figure)

    windows = [row for row in rows if row["experiment"] == "residual_window_patch"]
    if windows:
        contrast_types = sorted({row["contrast_type"] for row in windows})
        figure, axes = plt.subplots(
            len(contrast_types), 1, figsize=(13, 4.5 * len(contrast_types)), squeeze=False, constrained_layout=True
        )
        for axis, contrast_type in zip(axes[:, 0], contrast_types, strict=True):
            selected = [row for row in windows if row["contrast_type"] == contrast_type]
            for width in sorted({int(row["window_end"]) - int(row["window_start"]) + 1 for row in selected}):
                width_rows = sorted(
                    [row for row in selected if int(row["window_end"]) - int(row["window_start"]) + 1 == width],
                    key=lambda row: int(row["window_start"]),
                )
                centres = [
                    (int(row["window_start"]) + int(row["window_end"])) / 2 + 1
                    for row in width_rows
                ]
                axis.plot(
                    centres,
                    _safe([row["mean_normalized_recovery"] for row in width_rows]),
                    marker="o",
                    label=f"width {width}",
                )
            axis.axhline(0, color="0.45", linestyle="--", linewidth=1)
            axis.axhline(1, color="0.45", linestyle=":", linewidth=1)
            axis.set(
                title=f"{contrast_type}: sliding residual windows",
                xlabel="Window centre (post-layer stage)",
                ylabel="Normalized recovery",
            )
            axis.grid(alpha=0.2)
            axis.legend(ncol=3)
        figure.savefig(args.output_dir / "residual_window_atlas.png", dpi=180)
        plt.close(figure)

    token_rows = [row for row in rows if row["experiment"] == "token_group_patch"]
    if token_rows:
        contrast_types = sorted({row["contrast_type"] for row in token_rows})
        layers = sorted({int(row["layer_index"]) for row in token_rows})
        groups = sorted({str(row["token_group"]) for row in token_rows})
        figure, axes = plt.subplots(
            1,
            len(contrast_types),
            figsize=(max(11, 6.5 * len(contrast_types)), max(5, 0.48 * len(groups))),
            squeeze=False,
            constrained_layout=True,
        )
        image = None
        for axis, contrast_type in zip(axes[0], contrast_types, strict=True):
            lookup = {
                (str(row["token_group"]), int(row["layer_index"])): float(
                    row["mean_normalized_recovery"]
                )
                for row in token_rows
                if row["contrast_type"] == contrast_type
            }
            matrix = [[lookup.get((group, layer), float("nan")) for layer in layers] for group in groups]
            image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
            axis.set_title(f"{contrast_type}: token-group patch")
            axis.set_xlabel("Post-layer stage")
            axis.set_xticks(range(len(layers)), [layer + 1 for layer in layers])
            axis.set_yticks(range(len(groups)), groups)
        if image is not None:
            figure.colorbar(image, ax=axes.ravel().tolist(), label="Mean normalized recovery", shrink=0.82)
        figure.suptitle("Where answer information can be causally transferred", fontsize=15, fontweight="bold")
        figure.savefig(args.output_dir / "token_group_patch_heatmap.png", dpi=180)
        plt.close(figure)

    decomposition_rows = [row for row in rows if row["experiment"] == "component_patch"]
    if decomposition_rows:
        contrast_types = sorted({row["contrast_type"] for row in decomposition_rows})
        figure, axes = plt.subplots(
            1, len(contrast_types), figsize=(6.5 * len(contrast_types), 5), squeeze=False, constrained_layout=True
        )
        for axis, contrast_type in zip(axes[0], contrast_types, strict=True):
            selected = [row for row in decomposition_rows if row["contrast_type"] == contrast_type]
            for component in sorted({row["component"] for row in selected}):
                component_rows = sorted(
                    [row for row in selected if row["component"] == component],
                    key=lambda row: int(row["layer_index"]),
                )
                axis.plot(
                    [int(row["layer_index"]) + 1 for row in component_rows],
                    [float(row["mean_normalized_recovery"]) for row in component_rows],
                    marker="o",
                    label=component,
                )
            axis.axhline(0, color="0.45", linestyle="--", linewidth=1)
            axis.set(title=contrast_type, xlabel="Post-layer stage", ylabel="Mean normalized recovery")
            axis.grid(alpha=0.2)
            axis.legend()
        figure.suptitle("Attention versus MLP causal contribution", fontsize=15, fontweight="bold")
        figure.savefig(args.output_dir / "component_patch_lines.png", dpi=180)
        plt.close(figure)

    ablation_rows = [row for row in rows if row["experiment"] == "residual_ablation"]
    if ablation_rows:
        contrast_types = sorted({row["contrast_type"] for row in ablation_rows})
        figure, axes = plt.subplots(
            1, len(contrast_types), figsize=(6.5 * len(contrast_types), 5), squeeze=False, constrained_layout=True
        )
        for axis, contrast_type in zip(axes[0], contrast_types, strict=True):
            selected = [row for row in ablation_rows if row["contrast_type"] == contrast_type]
            for ablation_type in sorted({row["ablation_type"] for row in selected}):
                type_rows = sorted(
                    [row for row in selected if row["ablation_type"] == ablation_type],
                    key=lambda row: int(row["layer_index"]),
                )
                axis.plot(
                    [int(row["layer_index"]) + 1 for row in type_rows],
                    [float(row["mean_normalized_recovery"]) for row in type_rows],
                    marker="o",
                    label=ablation_type,
                )
            axis.axhline(0, color="0.45", linestyle="--", linewidth=1)
            axis.set(title=contrast_type, xlabel="Post-layer stage", ylabel="Mean normalized recovery")
            axis.grid(alpha=0.2)
            axis.legend()
        figure.suptitle("Final-token residual intervention controls", fontsize=15, fontweight="bold")
        figure.savefig(args.output_dir / "residual_ablation_lines.png", dpi=180)
        plt.close(figure)

    component_rows = [
        row
        for row in rows
        if row["experiment"]
        in {
            "head_patch",
            "edge_ablation",
            "token_path_patch",
            "binding_vector_addition",
        }
    ]
    if component_rows:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in component_rows:
            grouped[row["experiment"]].append(row)
        for experiment, experiment_rows in grouped.items():
            labels = []
            values = []
            errors_low = []
            errors_high = []
            for row in sorted(experiment_rows, key=lambda value: str(value)):
                labels.append(
                    "/".join(
                        str(value)
                        for value in (
                            row.get("contrast_type"),
                            row.get("component"),
                            row.get("token_group"),
                            row.get("layer_index"),
                            row.get("head_index"),
                            row.get("edge_route"),
                            row.get("ablation_type"),
                            row.get("scale"),
                            row.get("path_sender_layer"),
                            row.get("path_receiver_layer"),
                        )
                        if value is not None
                    )
                )
                mean = float(row["mean_normalized_recovery"])
                values.append(mean)
                errors_low.append(mean - float(row["bootstrap_95_low"]))
                errors_high.append(float(row["bootstrap_95_high"]) - mean)
            figure, axis = plt.subplots(figsize=(max(10, len(labels) * 0.32), 5.5), constrained_layout=True)
            axis.bar(range(len(labels)), values, yerr=[errors_low, errors_high], capsize=2)
            axis.axhline(0, color="0.45", linestyle="--", linewidth=1)
            axis.set(title=experiment.replace("_", " ").title(), ylabel="Mean normalized recovery")
            axis.set_xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=7)
            axis.grid(axis="y", alpha=0.2)
            figure.savefig(args.output_dir / f"{experiment}.png", dpi=180)
            plt.close(figure)

    calibration = [row for row in rows if str(row["experiment"]).startswith("calibration")]
    status = {
        "calibration_rows": len(calibration),
        "calibration_all_passed": all(row.get("passed", True) for row in calibration),
        "figure_files": sorted(path.name for path in args.output_dir.glob("*.png")),
    }
    (args.output_dir / "plot_summary.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
