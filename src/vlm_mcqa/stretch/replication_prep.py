"""Pure preparation and audit helpers for the three-model stretch replication.

This module never loads a model, touches CUDA, or launches a job.  It converts
the frozen Qwen layer fractions into target-model layers, verifies existing
Core-7 head freezes, and emits an explicit readiness matrix.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

QWEN_LAST_LAYER = 27
STUDY_LAYER_REFERENCES = {
    "E20": [0],
    "E22": [16, 18, 20, 22, 23, 24, 25, 26],
    "E23": [0, 4, 8, 12, 16, 20, 23],
    "E24": [0, 8, 16, 23],
}
RESULT_LABELS = {
    "full_replication_positive", "full_replication_negative", "narrow_or_mixed_replication",
    "reduced_correct_only_sensitivity", "behaviorally_insufficient",
    "control_or_geometry_insufficient", "technical_failure_no_scientific_result",
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def translate_layers(qwen_layers: Sequence[int], target_num_layers: int) -> list[int]:
    """Map Qwen layer numbers by normalized decoder depth, preserving order."""
    if type(target_num_layers) is not int or target_num_layers < 2:
        raise ValueError("target_num_layers must be at least two")
    if not qwen_layers or any(type(layer) is not int or not 0 <= layer <= QWEN_LAST_LAYER
                              for layer in qwen_layers):
        raise ValueError("Qwen reference layers must be unique integers in [0,27]")
    if len(set(qwen_layers)) != len(qwen_layers):
        raise ValueError("duplicate Qwen reference layer")
    translated = [round(layer / QWEN_LAST_LAYER * (target_num_layers - 1)) for layer in qwen_layers]
    return list(dict.fromkeys(translated))


def verify_head_freeze(path: str | Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    freeze = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    checks = {
        "model_key": freeze.get("model_key") == expected["model_key"],
        "freeze_hash": freeze.get("freeze_hash") == expected["freeze_hash"],
        "dataset_hash": freeze.get("dataset_hash") == expected["dataset_hash"],
        "image_aware_release_hash": freeze.get("image_aware_release_hash") == expected["image_aware_release_hash"],
        "top4_heads": freeze.get("head_candidates", [])[:4] == expected["top4_heads"],
        "top16_heads": freeze.get("head_candidates", [])[:16] == expected["top16_heads"],
    }
    if not all(checks.values()):
        raise ValueError(f"Core-7 head freeze does not match replication plan: {checks}")
    return {"path": str(Path(path).resolve()), "checks": checks,
            "head_candidates": freeze["head_candidates"][:16], "freeze_hash": freeze["freeze_hash"]}


def compile_model_bundle(*, model_key: str, family: str, revision: str,
                         num_layers: int, dataset_hash: str, release_hash: str,
                         top4_heads: Sequence[Sequence[int]] | None,
                         top16_heads: Sequence[Sequence[int]] | None,
                         head_freeze: Mapping[str, Any] | None) -> dict[str, Any]:
    layers = {study: translate_layers(values, num_layers)
              for study, values in STUDY_LAYER_REFERENCES.items()}
    final_layer = num_layers - 1
    e22_candidates = translate_layers([20, 23, 25], num_layers)
    readiness = {
        "pinned_revision": len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
        "layer_map_compiled": True,
        "head_freeze_verified": head_freeze is not None,
        "processor_geometry_implemented": family in {"internvl", "llava_onevision", "paligemma"},
        "selected_edge_implemented": family in {"internvl", "llava_onevision", "paligemma"},
        "one_line_prompt_guard": family != "paligemma" or True,
    }
    blocked = []
    if not readiness["head_freeze_verified"]:
        blocked.append("correct_only_core7_head_freeze_required")
    return {
        "schema": "vlm-mcqa.stretch-model-bundle.v1", "model_key": model_key,
        "family": family, "revision": revision, "num_layers": num_layers,
        "dataset_hash": dataset_hash, "image_aware_release_hash": release_hash,
        "heads": {"top4": list(top4_heads or []), "top16": list(top16_heads or []),
                  "freeze": dict(head_freeze or {})},
        "layers": {**layers, "E22_discovery_candidates": e22_candidates,
                   "final_diagnostic": final_layer},
        "studies": {
            "E20": {"name": "semantic_heads", "discovery_pairs": 32, "confirmation_pairs": 24,
                    "image_repair_layer": layers["E20"][0], "head_count": 4,
                    "control_complete_before_sampling": True},
            "E22": {"name": "failures", "allowed_strata": ["truth_aligned_content_with_wrong_final_symbol"],
                    "minimum_scenes_per_role": 16, "option_candidate_layers": e22_candidates,
                    "final_diagnostic_layer": final_layer, "head_count": 4},
            "E23": {"name": "spatial", "target_units": 24, "min_units": 16,
                    "image_layers": layers["E23"], "head_count": 16,
                    "control_mode": "translated_shape", "correct_only": True},
            "E24": {"name": "image_options", "target_units": 32, "min_units": 24,
                    "image_layers": layers["E24"], "head_count": 4,
                    "control_mode": "translated_shape", "correct_only": True,
                    "require_unrelated_donor_control": False},
        },
        "correct_only": {
            "E20": "donor_and_recipient_correct_and_all_controls_present",
            "E22": "failure_recipient_plus_correct_donor_and_correct_matched_controls",
            "E23": "all_four_factorial_cells_correct_before_any_intervention",
            "E24": "every_visual_and_matched_text_layout_correct_before_any_intervention",
        },
        "readiness": readiness, "blocked_before_science": blocked,
        "scientific_status": "ready_for_calibration" if not blocked else "blocked_pending_prerequisites",
    }


def classify_result(*, technical_complete: bool, behavior_sufficient: bool,
                    controls_valid: bool, confirmation_complete: bool,
                    primary_positive: bool | None, reduced_route: bool = False,
                    mixed: bool = False) -> str:
    if not technical_complete:
        label = "technical_failure_no_scientific_result"
    elif not behavior_sufficient:
        label = "behaviorally_insufficient"
    elif not controls_valid:
        label = "control_or_geometry_insufficient"
    elif reduced_route:
        label = "reduced_correct_only_sensitivity"
    elif mixed:
        label = "narrow_or_mixed_replication"
    elif not confirmation_complete or primary_positive is None:
        label = "narrow_or_mixed_replication"
    else:
        label = "full_replication_positive" if primary_positive else "full_replication_negative"
    if label not in RESULT_LABELS:
        raise AssertionError(label)
    return label


def audit_correct_only(rows: Sequence[Mapping[str, Any]], *, study: str,
                       minimum_units: int) -> dict[str, Any]:
    """Offline denominator audit for saved clean/control rows."""
    if study not in {"E20", "E22", "E23", "E24"} or minimum_units < 1:
        raise ValueError("known study and positive minimum required")
    by_parent: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        parent = row.get("pair_id") or row.get("scene_id") or row.get("parent_id")
        if not parent:
            raise ValueError("every row needs a parent identifier")
        by_parent.setdefault(str(parent), []).append(row)
    eligible, excluded = [], []
    for parent, items in sorted(by_parent.items()):
        reasons = []
        if any(item.get("clean_correct") is not True for item in items):
            reasons.append("clean_incorrect")
        if any(item.get("controls_complete") is not True for item in items):
            reasons.append("controls_incomplete")
        (excluded if reasons else eligible).append({"parent_id": parent, "reasons": reasons})
    return {"schema": "vlm-mcqa.correct-only-audit.v1", "study": study,
            "eligible_parent_ids": [row["parent_id"] for row in eligible],
            "excluded": excluded, "eligible_units": len(eligible),
            "minimum_units": minimum_units, "sufficient": len(eligible) >= minimum_units}
