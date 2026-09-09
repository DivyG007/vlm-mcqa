"""Create accuracy and layerwise logit visualizations for Experiment 00."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


COLORS = {"A": "#2563eb", "B": "#dc2626", "C": "#16a34a", "D": "#d97706"}
SCHEME_ORDER = ("letters", "numbers", "rare_letters", "punctuation")
POSITION_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#d97706")
SCHEME_COLORS = {
    "letters": "#2563eb",
    "numbers": "#7c3aed",
    "rare_letters": "#16a34a",
    "punctuation": "#d97706",
}
SCHEME_TITLES = {
    "letters": "A/B/C/D",
    "numbers": "1/2/3/4",
    "rare_letters": "Q/Z/R/X",
    "punctuation": "!/@/#/$",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#1f2937",
            "text.color": "#111827",
            "font.size": 10,
        }
    )


def _accuracy_plots(predictions: list[dict[str, Any]], output_dir: Path, plt: Any) -> None:
    position_rows = [
        row
        for row in predictions
        if row["variant_family"] == "answer_position" and row["label_scheme"] == "letters"
    ]
    by_position: dict[int, list[bool]] = defaultdict(list)
    confusion = [[0 for _ in range(4)] for _ in range(4)]
    for row in position_rows:
        by_position[row["correct_index"]].append(row["is_correct"])
        confusion[row["correct_index"]][row["predicted_index"]] += 1

    positions = list(range(4))
    accuracies = [mean(by_position[position]) for position in positions]
    with (output_dir / "accuracy_by_position.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["correct_position", "count", "accuracy"])
        for position, accuracy in zip(positions, accuracies, strict=True):
            writer.writerow([position + 1, len(by_position[position]), accuracy])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(
        [position + 1 for position in positions],
        accuracies,
        color=[COLORS[label] for label in "ABCD"],
        width=0.68,
    )
    ax.axhline(0.25, color="#6b7280", linestyle="--", linewidth=1.2, label="Random chance")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Correct answer position")
    ax.set_ylabel("Accuracy")
    ax.set_title("Qwen2.5-VL-3B sensitivity to answer order")
    ax.set_xticks([1, 2, 3, 4], ["1 (A)", "2 (B)", "3 (C)", "4 (D)"])
    ax.legend(frameon=False, loc="lower right")
    for bar, accuracy in zip(bars, accuracies, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.025,
            f"{accuracy:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_by_position.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_title("Answer-position confusion matrix")
    ax.set_xlabel("Predicted position")
    ax.set_ylabel("Correct position")
    ax.set_xticks(range(4), list("ABCD"))
    ax.set_yticks(range(4), list("ABCD"))
    for row in range(4):
        for col in range(4):
            ax.text(col, row, confusion[row][col], ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Number of predictions")
    fig.tight_layout()
    fig.savefig(output_dir / "position_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    grouped: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in predictions:
        grouped[(row["variant_family"], row["label_scheme"])].append(row["is_correct"])
    labels = []
    values = []
    counts = []
    schemes = [scheme for scheme in SCHEME_ORDER if any(key[1] == scheme for key in grouped)]
    for family in ("cross_image", "answer_position"):
        for scheme in schemes:
            values_for_group = grouped[(family, scheme)]
            labels.append(f"{family.replace('_', ' ')}\n{scheme.replace('_', ' ')}")
            values.append(mean(values_for_group))
            counts.append(len(values_for_group))
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    bars = ax.bar(
        range(len(labels)),
        values,
        color=["#334155"] * len(schemes) + ["#7c3aed"] * len(schemes),
    )
    ax.axhline(0.25, color="#6b7280", linestyle="--", linewidth=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy across prompt and label interventions")
    ax.set_xticks(range(len(labels)), labels)
    for bar, value, count in zip(bars, values, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.1%}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_by_condition.png", bbox_inches="tight")
    plt.close(fig)

    scheme_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        scheme_rows[row["label_scheme"]].append(row)
    accuracy_values = [mean(row["is_correct"] for row in scheme_rows[s]) for s in schemes]
    compliance_values = [
        mean(row["global_top_token_is_valid_label"] for row in scheme_rows[s])
        for s in schemes
    ]
    mass_values = [
        mean(row["valid_label_probability_mass"] for row in scheme_rows[s]) for s in schemes
    ]
    with (output_dir / "vocabulary_compliance_by_scheme.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "label_scheme",
                "restricted_accuracy",
                "global_top_token_valid_rate",
                "mean_valid_label_probability_mass",
            ]
        )
        for values_for_scheme in zip(
            schemes, accuracy_values, compliance_values, mass_values, strict=True
        ):
            writer.writerow(values_for_scheme)

    x = list(range(len(schemes)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar([value - width for value in x], accuracy_values, width, label="Restricted accuracy")
    ax.bar(x, compliance_values, width, label="Global top token is valid")
    ax.bar(
        [value + width for value in x],
        mass_values,
        width,
        label="Full-vocabulary mass on four labels",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction")
    ax.set_title("Restricted MCQ score versus full-vocabulary compliance")
    ax.set_xticks(x, [scheme.replace("_", " ") for scheme in schemes])
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / "vocabulary_compliance_by_scheme.png", bbox_inches="tight")
    plt.close(fig)


def _lens_scheme(row: dict[str, Any]) -> str:
    if "label_scheme" in row:
        return row["label_scheme"]
    labels = set(row["label_logits"])
    known = {
        "letters": set("ABCD"),
        "numbers": set("1234"),
        "rare_letters": set("QZRX"),
        "punctuation": set("!@#$"),
    }
    for scheme, expected in known.items():
        if labels == expected:
            return scheme
    raise ValueError(f"Cannot identify label scheme from labels: {sorted(labels)}")


def _layerwise_plots(lens_rows: list[dict[str, Any]], output_dir: Path, plt: Any) -> None:
    rows_by_scheme: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in lens_rows:
        rows_by_scheme[_lens_scheme(row)].append(row)
    schemes = [scheme for scheme in SCHEME_ORDER if scheme in rows_by_scheme]
    if not schemes:
        raise ValueError("No layerwise rows were provided")

    scheme_summaries: dict[str, dict[str, Any]] = {}
    dual_scheme_summaries: dict[str, dict[str, Any]] = {}
    with (output_dir / "layerwise_means.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "label_scheme",
                "correct_position",
                "stage_index",
                "stage_name",
                "label",
                "mean_logit",
                "mean_correct_margin",
                "mean_correct_probability",
            ]
        )

        for scheme in schemes:
            scheme_rows = rows_by_scheme[scheme]
            first = scheme_rows[0]
            labels = list(first.get("labels", first["label_logits"].keys()))
            grouped_logits: dict[tuple[int, int, str], list[float]] = defaultdict(list)
            grouped_margin: dict[tuple[int, int], list[float]] = defaultdict(list)
            grouped_probability: dict[tuple[int, int], list[float]] = defaultdict(list)
            stage_names: dict[int, str] = {}
            stage_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in scheme_rows:
                position = row["correct_index"]
                stage = row["stage_index"]
                stage_names[stage] = row["stage_name"]
                stage_rows[stage].append(row)
                for label, value in row["label_logits"].items():
                    grouped_logits[(position, stage, label)].append(value)
                grouped_margin[(position, stage)].append(row["correct_logit_margin"])
                grouped_probability[(position, stage)].append(
                    row["correct_label_probability"]
                )

            stages = sorted(stage_names)
            scheme_summaries[scheme] = {
                "labels": labels,
                "stages": stages,
                "stage_rows": stage_rows,
            }
            for position in range(4):
                for stage in stages:
                    for label in labels:
                        writer.writerow(
                            [
                                scheme,
                                position + 1,
                                stage,
                                stage_names[stage],
                                label,
                                mean(grouped_logits[(position, stage, label)]),
                                mean(grouped_margin[(position, stage)]),
                                mean(grouped_probability[(position, stage)]),
                            ]
                        )

            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
            for position, ax in enumerate(axes.flat):
                for label_index, label in enumerate(labels):
                    values = [
                        mean(grouped_logits[(position, stage, label)])
                        for stage in stages
                    ]
                    ax.plot(
                        stages,
                        values,
                        label=label,
                        color=POSITION_COLORS[label_index],
                        linewidth=2,
                    )
                ax.set_title(f"Correct position {position + 1} ({labels[position]})")
                ax.axhline(0, color="#9ca3af", linewidth=0.8)
                ax.set_ylabel("Mean label logit")
            for ax in axes[-1]:
                ax.set_xlabel("Residual-stream stage")
            handles, legend_labels = axes[0, 0].get_legend_handles_labels()
            fig.suptitle(
                f"How {SCHEME_TITLES[scheme]} logits develop through Qwen2.5-VL-3B",
                y=0.995,
                fontweight="bold",
            )
            fig.legend(
                handles,
                legend_labels,
                title="Output label",
                frameon=False,
                ncol=4,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.965),
            )
            fig.tight_layout(rect=(0, 0, 1, 0.91))
            fig.savefig(
                output_dir / f"layerwise_label_logits_by_position_{scheme}.png",
                bbox_inches="tight",
            )
            plt.close(fig)

            fig, (margin_ax, probability_ax) = plt.subplots(
                1, 2, figsize=(12, 4.8), sharex=True
            )
            for position, label in enumerate(labels):
                margins = [
                    mean(grouped_margin[(position, stage)]) for stage in stages
                ]
                probabilities = [
                    mean(grouped_probability[(position, stage)]) for stage in stages
                ]
                line_label = f"Position {position + 1} ({label})"
                margin_ax.plot(
                    stages,
                    margins,
                    label=line_label,
                    color=POSITION_COLORS[position],
                    linewidth=2,
                )
                probability_ax.plot(
                    stages,
                    probabilities,
                    label=line_label,
                    color=POSITION_COLORS[position],
                    linewidth=2,
                )
            margin_ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1)
            probability_ax.axhline(
                0.25, color="#6b7280", linestyle="--", linewidth=1
            )
            margin_ax.set_title("Correct-label logit margin")
            probability_ax.set_title(
                f"Correct-label probability among {SCHEME_TITLES[scheme]}"
            )
            margin_ax.set_ylabel("Correct - best incorrect logit")
            probability_ax.set_ylabel("Probability")
            probability_ax.set_ylim(0, 1.02)
            for ax in (margin_ax, probability_ax):
                ax.set_xlabel("Residual-stream stage")
                ax.legend(frameon=False)
            fig.suptitle(
                f"Correct-answer progression: {SCHEME_TITLES[scheme]}",
                y=1.01,
                fontweight="bold",
            )
            fig.tight_layout()
            fig.savefig(
                output_dir / f"layerwise_correct_answer_progression_{scheme}.png",
                bbox_inches="tight",
            )
            plt.close(fig)

            if scheme != "letters" and "probe_label_logits" in scheme_rows[0]:
                default_labels = list("ABCD")
                probe_labels = list(scheme_rows[0]["probe_labels"])
                requested_labels = [
                    label for label in probe_labels if label not in default_labels
                ]
                grouped_probe: dict[tuple[int, int, str], list[float]] = defaultdict(list)
                for row in scheme_rows:
                    for label, value in row["probe_label_logits"].items():
                        grouped_probe[(row["correct_index"], row["stage_index"], label)].append(
                            value
                        )

                fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
                for position, ax in enumerate(axes.flat):
                    for label_index, label in enumerate(default_labels):
                        ax.plot(
                            stages,
                            [mean(grouped_probe[(position, stage, label)]) for stage in stages],
                            color=POSITION_COLORS[label_index],
                            linestyle="--",
                            linewidth=1.6,
                            alpha=0.72,
                            label=f"default {label}",
                        )
                    for label_index, label in enumerate(requested_labels):
                        ax.plot(
                            stages,
                            [mean(grouped_probe[(position, stage, label)]) for stage in stages],
                            color=POSITION_COLORS[label_index],
                            linewidth=2.2,
                            label=f"requested {label}",
                        )
                    ax.axhline(0, color="#9ca3af", linewidth=0.8)
                    ax.set_title(
                        f"Correct position {position + 1} ({requested_labels[position]})"
                    )
                    ax.set_ylabel("Mean projected logit")
                for ax in axes[-1]:
                    ax.set_xlabel("Residual-stream stage")
                handles, legend_labels = axes[0, 0].get_legend_handles_labels()
                fig.suptitle(
                    f"Default A/B/C/D versus requested {SCHEME_TITLES[scheme]}",
                    y=0.995,
                    fontweight="bold",
                )
                fig.legend(
                    handles,
                    legend_labels,
                    frameon=False,
                    ncol=4,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.955),
                )
                fig.tight_layout(rect=(0, 0, 1, 0.88))
                fig.savefig(
                    output_dir / f"layerwise_default_vs_requested_{scheme}.png",
                    bbox_inches="tight",
                )
                plt.close(fig)

                dual_scheme_summaries[scheme] = {
                    "stages": stages,
                    "stage_rows": stage_rows,
                }

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True)
    for scheme in schemes:
        stages = scheme_summaries[scheme]["stages"]
        stage_rows = scheme_summaries[scheme]["stage_rows"]
        common_logits = [
            mean(mean(row["label_logits"].values()) for row in stage_rows[stage])
            for stage in stages
        ]
        margins = [
            mean(row["correct_logit_margin"] for row in stage_rows[stage])
            for stage in stages
        ]
        probabilities = [
            mean(row["correct_label_probability"] for row in stage_rows[stage])
            for stage in stages
        ]
        display = SCHEME_TITLES[scheme]
        color = SCHEME_COLORS[scheme]
        axes[0].plot(stages, common_logits, label=display, color=color, linewidth=2)
        axes[1].plot(stages, margins, label=display, color=color, linewidth=2)
        axes[2].plot(stages, probabilities, label=display, color=color, linewidth=2)
    axes[0].axhline(0, color="#6b7280", linestyle="--", linewidth=1)
    axes[1].axhline(0, color="#6b7280", linestyle="--", linewidth=1)
    axes[2].axhline(0.25, color="#6b7280", linestyle="--", linewidth=1)
    axes[0].set_title("Common candidate-label logit")
    axes[1].set_title("Correct minus best-incorrect logit")
    axes[2].set_title("Correct-label four-way probability")
    axes[0].set_ylabel("Mean logit across four labels")
    axes[1].set_ylabel("Mean margin")
    axes[2].set_ylabel("Mean probability")
    axes[2].set_ylim(0, 1.02)
    for ax in axes:
        ax.set_xlabel("Residual-stream stage")
        ax.legend(frameon=False)
    fig.suptitle(
        "Layerwise comparison across output-label schemes",
        y=1.02,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "layerwise_scheme_comparison.png", bbox_inches="tight"
    )
    plt.close(fig)

    if dual_scheme_summaries:
        with (output_dir / "layerwise_default_to_requested_switch.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "label_scheme",
                    "stage_index",
                    "mean_default_alphabet_probability_mass_8way",
                    "mean_requested_alphabet_probability_mass_8way",
                    "mean_requested_correct_minus_position_default_logit",
                ]
            )
            for scheme, values in dual_scheme_summaries.items():
                for stage in values["stages"]:
                    rows = values["stage_rows"][stage]
                    writer.writerow(
                        [
                            scheme,
                            stage,
                            mean(
                                row["default_alphabet_probability_mass_8way"]
                                for row in rows
                            ),
                            mean(
                                row["requested_alphabet_probability_mass_8way"]
                                for row in rows
                            ),
                            mean(
                                row["requested_correct_minus_position_default_logit"]
                                for row in rows
                            ),
                        ]
                    )

        fig, axes = plt.subplots(
            2,
            len(dual_scheme_summaries),
            figsize=(5.2 * len(dual_scheme_summaries), 8.0),
            sharex="col",
            squeeze=False,
        )
        for column, (scheme, values) in enumerate(dual_scheme_summaries.items()):
            stages = values["stages"]
            stage_rows = values["stage_rows"]
            default_mass = [
                mean(
                    row["default_alphabet_probability_mass_8way"]
                    for row in stage_rows[stage]
                )
                for stage in stages
            ]
            requested_mass = [
                mean(
                    row["requested_alphabet_probability_mass_8way"]
                    for row in stage_rows[stage]
                )
                for stage in stages
            ]
            deltas = [
                mean(
                    row["requested_correct_minus_position_default_logit"]
                    for row in stage_rows[stage]
                )
                for stage in stages
            ]
            axes[0, column].plot(
                stages, default_mass, label="Default A/B/C/D", color="#64748b", linewidth=2
            )
            axes[0, column].plot(
                stages,
                requested_mass,
                label=f"Requested {SCHEME_TITLES[scheme]}",
                color=SCHEME_COLORS[scheme],
                linewidth=2.3,
            )
            axes[0, column].axhline(0.5, color="#9ca3af", linestyle="--", linewidth=1)
            axes[0, column].set_ylim(0, 1)
            axes[0, column].set_title(SCHEME_TITLES[scheme])
            axes[0, column].set_ylabel("8-way probability mass")
            axes[0, column].legend(frameon=False)
            axes[1, column].plot(
                stages, deltas, color=SCHEME_COLORS[scheme], linewidth=2.3
            )
            axes[1, column].axhline(0, color="#6b7280", linestyle="--", linewidth=1)
            axes[1, column].set_ylabel("Requested correct − position-matched A/B/C/D logit")
            axes[1, column].set_xlabel("Residual-stream stage")
        fig.suptitle(
            "Does the model prefer default MCQ letters before switching alphabets?",
            y=1.01,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / "layerwise_default_to_requested_switch.png",
            bbox_inches="tight",
        )
        plt.close(fig)

    letter_rows = rows_by_scheme.get("letters", [])
    if letter_rows and "semantic_content_logits" in letter_rows[0]:
        stage_names: dict[int, str] = {}
        by_stage: dict[int, list[dict[str, Any]]] = defaultdict(list)
        by_position_stage: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in letter_rows:
            stage = row["stage_index"]
            stage_names[stage] = row["stage_name"]
            by_stage[stage].append(row)
            by_position_stage[(row["correct_index"], stage)].append(row)
        stages = sorted(stage_names)

        with (output_dir / "layerwise_content_symbol_means.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "stage_index",
                    "stage_name",
                    "mean_correct_content_margin",
                    "mean_correct_symbol_margin",
                    "mean_correct_content_probability_4way",
                    "mean_correct_symbol_probability_4way",
                ]
            )
            for stage in stages:
                rows = by_stage[stage]
                writer.writerow(
                    [
                        stage,
                        stage_names[stage],
                        mean(row["correct_content_logit_margin"] for row in rows),
                        mean(row["correct_logit_margin"] for row in rows),
                        mean(row["correct_content_probability"] for row in rows),
                        mean(row["correct_label_probability"] for row in rows),
                    ]
                )

        content_margins = [
            mean(row["correct_content_logit_margin"] for row in by_stage[stage])
            for stage in stages
        ]
        symbol_margins = [
            mean(row["correct_logit_margin"] for row in by_stage[stage])
            for stage in stages
        ]
        content_probabilities = [
            mean(row["correct_content_probability"] for row in by_stage[stage])
            for stage in stages
        ]
        symbol_probabilities = [
            mean(row["correct_label_probability"] for row in by_stage[stage])
            for stage in stages
        ]
        fig, (margin_ax, probability_ax) = plt.subplots(1, 2, figsize=(12.5, 4.8))
        margin_ax.plot(
            stages, content_margins, label="Answer content (color)", color="#16a34a", linewidth=2.3
        )
        margin_ax.plot(
            stages, symbol_margins, label="Answer symbol (A/B/C/D)", color="#2563eb", linewidth=2.3
        )
        margin_ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1)
        margin_ax.set_title("Correct minus best-incorrect logit")
        margin_ax.set_ylabel("Mean four-way margin")
        probability_ax.plot(
            stages,
            content_probabilities,
            label="Answer content (color)",
            color="#16a34a",
            linewidth=2.3,
        )
        probability_ax.plot(
            stages,
            symbol_probabilities,
            label="Answer symbol (A/B/C/D)",
            color="#2563eb",
            linewidth=2.3,
        )
        probability_ax.axhline(0.25, color="#6b7280", linestyle="--", linewidth=1)
        probability_ax.set_ylim(0, 1.02)
        probability_ax.set_title("Restricted four-way correct probability")
        probability_ax.set_ylabel("Mean probability")
        for ax in (margin_ax, probability_ax):
            ax.set_xlabel("Residual-stream stage")
            ax.legend(frameon=False)
        fig.suptitle(
            "When answer content becomes answer-symbol identity",
            y=1.01,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(output_dir / "layerwise_content_vs_symbol_letters.png", bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        for position, ax in enumerate(axes.flat):
            ax.plot(
                stages,
                [
                    mean(
                        row["correct_content_probability"]
                        for row in by_position_stage[(position, stage)]
                    )
                    for stage in stages
                ],
                label="Correct content",
                color="#16a34a",
                linewidth=2.2,
            )
            ax.plot(
                stages,
                [
                    mean(
                        row["correct_label_probability"]
                        for row in by_position_stage[(position, stage)]
                    )
                    for stage in stages
                ],
                label="Correct symbol",
                color="#2563eb",
                linewidth=2.2,
            )
            ax.axhline(0.25, color="#6b7280", linestyle="--", linewidth=1)
            ax.set_title(f"Correct position {position + 1} ({'ABCD'[position]})")
            ax.set_ylabel("Restricted four-way probability")
            ax.set_ylim(0, 1.02)
        for ax in axes[-1]:
            ax.set_xlabel("Residual-stream stage")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(
            "Content and symbol emergence by answer position",
            y=0.995,
            fontweight="bold",
        )
        fig.legend(
            handles,
            labels,
            frameon=False,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        fig.savefig(
            output_dir / "layerwise_content_symbol_by_position_letters.png",
            bbox_inches="tight",
        )
        plt.close(fig)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--layerwise", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    predictions = _read_jsonl(args.predictions)
    lens_rows = _read_jsonl(args.layerwise)
    _accuracy_plots(predictions, args.output_dir, plt)
    _layerwise_plots(lens_rows, args.output_dir, plt)
    print(f"[plots] wrote visualizations to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
