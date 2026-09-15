"""Model selection (Step 4/5) and cross-model comparison (Step 6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .aggregate import write_csv

LINEAGE = {"qwen2": "qwen2", "qwen2.5": "qwen2", "qwen3": "qwen3", "gemma": "gemma", "gemma2": "gemma"}
METHODS = {
    "M1 residual patching": "patching_{split}.jsonl",
    "M2 attn/MLP projection": "component_projection_behavior.jsonl",
    "M2 attn/MLP causal": "components_{split}.jsonl",
    "M3 heads": "heads_{split}_summary.json",
    "M4 logit lens": "lens_behavior.jsonl",
    "M5 content vs symbol": "patching_{split}.jsonl",
    "M6 position vs content": "patching_{split}.jsonl",
    "M7 arbitrary symbols": "lens_behavior.jsonl",
}
SHARED_METHODS = {"M2 attn/MLP projection", "M4 logit lens", "M7 arbitrary symbols"}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def screening_row(run_dir: Path) -> dict[str, Any]:
    run = _load(run_dir / "run.json")
    beh = _load(run_dir / "behavior_screening_summary.json")
    cal = _load(run_dir / "calibration.json")
    yield_ = beh.get("pair_yield", {})
    gates = cal.get("gates", {})
    return {
        "model_key": run.get("model_key"), "family": run.get("family"), "hf_id": run.get("hf_id"),
        "params_b": run.get("params_b"), "measured_params_b": (run.get("adapter", {}).get("parameter_count") or 0) / 1e9,
        "lm_backbone": run.get("lm_backbone"),
        "worst_position_accuracy": beh.get("worst_position_accuracy"),
        "letters_accuracy": beh.get("letters_accuracy"),
        "strict_item_consistency": beh.get("strict_item_consistency"),
        "shuffled_accuracy": beh.get("shuffled_accuracy"), "blank_accuracy": beh.get("blank_accuracy"),
        "clean_minus_shuffled": beh.get("clean_minus_shuffled"),
        "worst_scheme_position_accuracy": beh.get("worst_scheme_position_accuracy"),
        "robustness_cells": beh.get("robustness_cells"),
        "global_top_is_label_letters": (beh.get("global_top_is_label_rate") or {}).get("letters"),
        "label_mass_letters": (beh.get("mean_label_mass_full") or {}).get("letters"),
        "compliance_rate": beh.get("compliance_rate"),
        "usable_discovery_pairs": yield_.get("discovery", {}).get("usable_pairs", 0),
        "usable_confirmation_pairs": yield_.get("confirmation", {}).get("usable_pairs", 0),
        "calibration_passed": cal.get("all_passed", False),
        "calibration_worst_ratio": max((g["value"] / g["limit"] for g in gates.values() if g.get("limit")), default=None),
        "elapsed_seconds": beh.get("elapsed_seconds"), "peak_gpu_gib": beh.get("peak_gpu_gib"),
        "run_dir": str(run_dir),
    }


def _eligibility(row: Mapping[str, Any], t: Mapping[str, Any]) -> list[str]:
    failures = []
    if (row["worst_position_accuracy"] or 0) <= t["min_worst_position_accuracy"]:
        failures.append("worst_position_accuracy")
    if (row["clean_minus_shuffled"] or 0) < t["min_shuffled_gap"]:
        failures.append("image_dependence")
    if row["usable_discovery_pairs"] < t["min_usable_discovery_pairs"]:
        failures.append("discovery_pairs")
    if row["usable_confirmation_pairs"] < t["min_usable_confirmation_pairs"]:
        failures.append("confirmation_pairs")
    if not row["calibration_passed"]:
        failures.append("calibration")
    if (row["params_b"] or 99) >= t["max_params_b"]:
        failures.append("params")
    return failures


def _quality_key(row):
    """Rank by letters worst-position first; use proposal robustness only on a tie."""
    return (-(row["worst_position_accuracy"] or 0),
            -int(row["proposal_robustness_pass"]),
            -(row["worst_scheme_position_accuracy"] or 0),
            -row["usable_discovery_pairs"],
            -(row["clean_minus_shuffled"] or 0),
            row["calibration_worst_ratio"] if row["calibration_worst_ratio"] is not None else 9,
            row["elapsed_seconds"] if row["elapsed_seconds"] is not None else 1e9,
            row["model_key"])


def select(screen_dirs: Sequence[Path], thresholds: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    rows = [screening_row(Path(d)) for d in screen_dirs if (Path(d) / "run.json").exists()]
    for row in rows:
        row["failed_gates"] = _eligibility(row, thresholds)
        row["eligible"] = not row["failed_gates"]
        row["proposal_robustness_pass"] = (
            row["worst_scheme_position_accuracy"] is not None
            and row["worst_scheme_position_accuracy"] >= thresholds.get("min_proposal_robustness_accuracy", 0.45)
        )
    winners = {}
    for family in sorted({r["family"] for r in rows}):
        members = [r for r in rows if r["family"] == family]
        eligible = [r for r in members if r["eligible"]]
        if eligible:
            smallest = min(r["params_b"] for r in eligible)
            tied = [r for r in eligible if r["params_b"] - smallest < 0.3]
            choice, provisional = sorted(tied, key=_quality_key)[0], False
        else:
            below = [r for r in members if (r["params_b"] or 99) < thresholds["max_params_b"]] or members
            choice, provisional = sorted(below, key=_quality_key)[0], True
        winners[family] = {"model_key": choice["model_key"], "provisional": provisional,
                           "reason": "smallest eligible checkpoint" if not provisional else
                           "no eligible checkpoint; best sub-10B worst-position accuracy (reconsider dataset difficulty)"}
    ranked = sorted((r for r in rows if r["model_key"] in {w["model_key"] for w in winners.values()}),
                    key=lambda r: (not r["eligible"], _quality_key(r)))
    chosen = []
    if ranked:
        first = ranked[0]
        chosen.append(first["model_key"])
        rest = ranked[1:]
        if rest:
            best = rest[0]
            primary_tied = [r for r in rest if r["eligible"] == best["eligible"]
                            and r["worst_position_accuracy"] == best["worst_position_accuracy"]]
            if len(primary_tied) > 1:
                # Exact primary ties are settled by robustness before diversity.
                robust_tied = [r for r in primary_tied
                               if r["proposal_robustness_pass"] == best["proposal_robustness_pass"]
                               and r["worst_scheme_position_accuracy"] == best["worst_scheme_position_accuracy"]]
                diverse = [r for r in robust_tied
                           if LINEAGE.get(r["lm_backbone"]) != LINEAGE.get(first["lm_backbone"])]
                second = (diverse or [best])[0]
            else:
                # Preserve the pre-existing 3-point diversity preference when
                # there is no exact primary-score tie to break.
                comparable = [r for r in rest if r["eligible"] == best["eligible"]
                              and best["worst_position_accuracy"] - r["worst_position_accuracy"]
                              <= thresholds.get("comparable_accuracy_window", 0.03)]
                diverse = [r for r in comparable
                           if LINEAGE.get(r["lm_backbone"]) != LINEAGE.get(first["lm_backbone"])]
                second = (diverse or [best])[0]
            chosen.append(second["model_key"])
    decision = {"thresholds": dict(thresholds), "family_winners": winners, "winner_ranking":
                [r["model_key"] for r in ranked], "core7_models": chosen, "table": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selection.json").write_text(json.dumps(decision, indent=2, sort_keys=True, default=str))
    write_csv(out_dir / "screening_table.csv", [{k: v for k, v in r.items() if k != "failed_gates"} |
                                                {"failed_gates": ";".join(r["failed_gates"])} for r in rows])
    (out_dir / "screening_table.md").write_text(_markdown(rows, winners, chosen))
    return decision


def _markdown(rows, winners, chosen) -> str:
    fmt = lambda v: "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    cols = ["model_key", "family", "params_b", "worst_position_accuracy", "letters_accuracy",
            "worst_scheme_position_accuracy", "proposal_robustness_pass", "clean_minus_shuffled", "usable_discovery_pairs",
            "usable_confirmation_pairs", "calibration_passed", "eligible"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(rows, key=lambda r: (r["family"], r["params_b"] or 0)):
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    lines += ["", "Family winners: " + ", ".join(f"{f}={w['model_key']}{' (provisional)' if w['provisional'] else ''}"
                                             for f, w in winners.items()),
              "", "Full Core-7 models: " + ", ".join(chosen), ""]
    return "\n".join(lines)


def _by_model(run_dirs: Sequence[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for run_dir in map(Path, run_dirs):
        groups.setdefault(_load(run_dir / "run.json").get("model_key") or run_dir.name, []).append(run_dir)
    return groups


def completion_matrix(run_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    """Which outputs exist; behavior-split analyses are marked shared."""
    out = []
    for model, dirs in _by_model(run_dirs).items():
        row: dict[str, Any] = {"model_key": model, "run_dir": ";".join(map(str, dirs))}
        for method, pattern in METHODS.items():
            if method in SHARED_METHODS:
                row[method] = {"shared": any((d / pattern).exists() for d in dirs)}
            else:
                row[method] = {split: any((d / pattern.format(split=split)).exists() for d in dirs)
                               for split in ("discovery", "confirmation")}
        out.append(row)
    return out


def compare(run_dirs: Sequence[Path], out_dir: Path) -> dict[str, Any]:
    """Cross-model normalized-depth comparison of frozen and confirmed transitions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for model, dirs in _by_model(run_dirs).items():
        n = next((_load(d / "run.json").get("num_layers") for d in dirs), None) or 1
        entry: dict[str, Any] = {"model_key": model, "num_layers": n, "run_dirs": [str(d) for d in dirs]}
        for run_dir in dirs:
            for split in ("discovery", "confirmation"):
                agg = _load(run_dir / f"aggregate_{split}.json")
                if not agg:
                    continue
                for key in ("position_transfer_layer", "content_transfer_layer", "attn_peak_layer",
                            "mlp_peak_layer", "cp_content_peak_layer"):
                    value = agg.get(key)
                    entry[f"{split}:{key}"] = value
                    entry[f"{split}:{key}:depth"] = None if value is None else (value + 1) / n
                readable = agg.get("letters_readable_stage")
                if readable is not None:
                    entry["letters_readable_depth"] = readable / n
                entry[f"{split}:head_verdict"] = ((agg.get("heads") or {}).get("verdict") or [None])[0]
            confirm = _load(run_dir / "confirmation_report.json")
            if confirm:
                entry["confirmation_checks"] = {k: v.get("replicated") for k, v in confirm.get("checks", {}).items()
                                                if isinstance(v, dict) and "replicated" in v}
        models.append(entry)
    result = {"models": models, "completion": completion_matrix(run_dirs),
              "note": "AAA text-LM comparison is a literature comparison, not a matched experiment."}
    (out_dir / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result
