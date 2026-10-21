"""Explicit future table/SVG export, with exclusive writes and finite values."""
from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping


def write_table(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_bars(path: Path, values: list[tuple[str, float | None]], *, title: str, unit: str) -> None:
    """Simple standalone scientific export. Missing values remain labelled missing.

    No image is rendered/imported while authoring this function. Visual QA of its
    actual output is a deferred execution task, not a claimed completed check.
    """
    finite = [v for _, v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    high = max([abs(v) for v in finite] + [1.0])
    height = 95 + 48 * len(values)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" viewBox="0 0 960 {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="22" y="28" font-size="18">{html.escape(title)}</text>',
             f'<text x="22" y="52" font-size="12">{html.escape(unit)}; missing evidence is not zero</text>']
    for index, (label, value) in enumerate(values):
        y = 74 + 48 * index
        parts.append(f'<text x="22" y="{y+17}" font-size="13">{html.escape(label)}</text>')
        valid = isinstance(value, (int, float)) and math.isfinite(value)
        if valid:
            width = 330 * abs(value) / high
            x = 610 if value >= 0 else 610 - width
            color = "#237a57" if value >= 0 else "#b24444"
            parts.append(f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="24" fill="{color}"/>')
        parts.append(f'<text x="350" y="{y+17}" font-size="13">{value:.5g}</text>' if valid else
                     f'<text x="350" y="{y+17}" font-size="13">missing</text>')
    parts.append('</svg>')
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(parts) + "\n")


def write_analysis_artifacts(directory: Path, report: Mapping[str, Any]) -> list[str]:
    """Append exports only inside a just-created, exclusively owned report root."""
    paths = []
    if "comparison" in report:
        rows = [{"model": row["model"], "letters_accuracy": row.get("forced_letters_accuracy"),
                 "image_control_gap": row.get("clean_minus_shuffled"),
                 "top4_mean_nr": row["top4"].get("mean_nr"),
                 "top4_selected_layer_fraction": row["top4"].get("fraction_of_selected_layer_all_heads"),
                 "head_source": row["top4"].get("source")} for row in report["comparison"]]
        write_table(directory / "comparison.csv", rows,
                    ["model", "letters_accuracy", "image_control_gap", "top4_mean_nr", "top4_selected_layer_fraction", "head_source"])
        paths.append("comparison.csv")
        for key in ("letters_accuracy", "top4_selected_layer_fraction"):
            filename = key + ".svg"
            write_bars(directory / filename, [(r["model"], r[key]) for r in rows],
                       title=key.replace("_", " "), unit="fraction")
            paths.append(filename)
    if report.get("rows") and "final_correct" in report["rows"][0]:
        fields = ["variant_id", "stage", "scene_id", "analysis_phase", "final_correct", "content_proxy_correct",
                  "confidence_gap", "truth_label_margin", "unresolved", "descriptive_flags"]
        write_table(directory / "descriptive_cases.csv", report["rows"], fields)
        paths.append("descriptive_cases.csv")
    if report.get("phases"):
        rows = []
        for phase, block in report["phases"].items():
            for name, values in block.get("metrics", {}).items():
                rows.append({"phase": phase, "metric": name, "mean": values["mean"],
                             "ci95_lower": values["ci95"][0] if values["ci95"] else None,
                             "ci95_upper": values["ci95"][1] if values["ci95"] else None,
                             "n_scenes": values["n_clusters"], "n_rows": values["n_rows"]})
        write_table(directory / "scene_metrics.csv", rows,
                    ["phase", "metric", "mean", "ci95_lower", "ci95_upper", "n_scenes", "n_rows"])
        paths.append("scene_metrics.csv")
    with (directory / "interpretation.md").open("x", encoding="utf-8") as stream:
        stream.write("# Analysis scope\n\n")
        for limitation in report.get("required_interpretation", report.get("limitations", [])):
            stream.write(f"- {limitation}\n")
        stream.write("\nNo causal diagnosis or pristine holdout is established by descriptive exports.\n")
    return [*paths, "interpretation.md"]


def write_study_artifacts(directory: Path, raw: list[Mapping[str, Any]],
                          summary: Mapping[str, Any]) -> list[str]:
    """Export actual study rows and scalar cluster summaries without recomputing tests."""
    flattened = []
    for envelope in raw:
        children = envelope.get("rows")
        if isinstance(children, list):
            inherited = {key: envelope[key] for key in ("contrast_id", "pair_id", "role", "split") if key in envelope}
            flattened.extend({**inherited, **row} for row in children)
        else:
            flattened.append(dict(envelope))
    fields = sorted({key for row in flattened for key in row})
    paths = []
    if fields:
        exported = [{key: json.dumps(value, sort_keys=True, allow_nan=False)
                     if isinstance(value, (list, dict)) else value for key, value in row.items()}
                    for row in flattened]
        write_table(directory / "measurements.csv", exported, fields)
        paths.append("measurements.csv")
    estimates = []
    def collect(value, path):
        if isinstance(value, Mapping):
            point = value.get("mean", value.get("mean_margin_change"))
            if isinstance(point, (int, float)) and math.isfinite(point):
                interval = value.get("ci95")
                estimates.append({"endpoint": "/".join(path), "mean": point,
                                  "ci95_lower": interval[0] if isinstance(interval, list) and len(interval) == 2 else None,
                                  "ci95_upper": interval[1] if isinstance(interval, list) and len(interval) == 2 else None,
                                  "n_units": value.get("n_units", value.get("n_clusters", value.get("n_pairs")))})
            for key, child in value.items():
                collect(child, [*path, str(key)])
    collect(summary, [])
    if estimates:
        write_table(directory / "effect_estimates.csv", estimates,
                    ["endpoint", "mean", "ci95_lower", "ci95_upper", "n_units"])
        write_bars(directory / "effect_estimates.svg", [(r["endpoint"], r["mean"]) for r in estimates],
                   title="Study estimates (uncertainty in accompanying CSV)", unit="see endpoint definitions")
        paths.extend(["effect_estimates.csv", "effect_estimates.svg"])
    return paths
