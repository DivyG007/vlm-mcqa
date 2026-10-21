"""Future explicitly gated semantic-image campaign; no model-loading CLI.

Authored through static inspection only. No imports, tests or runtime evaluation
were performed during implementation. A caller must establish a current calibrated
Session and the separate user authorization; this module chooses no compute host.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import digest, validate_bidirectional, validate_output_path
from .runtime import evaluate_contrast, validate_runtime_context


def validate_sample_plan(plan: Mapping[str, Any], contrasts: Sequence[Mapping[str, Any]],
                         recipe: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Bind an unchanged, full-target plan to its actual directed contrasts.

    An insufficient or shortfall sample is not silently promoted to the frozen
    target. A different target requires a separate prospective protocol decision.
    """
    if plan.get("schema") != "stretch.fixed-text-sample.v1":
        raise ValueError("unsupported sample plan schema")
    canonical = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual_hash = digest(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode())
    if plan.get("plan_sha256") != actual_hash:
        raise ValueError("sample plan hash mismatch")
    if plan.get("status") != "ready":
        raise ValueError("sample target not reached; no automatic shortfall fallback")
    for key in ("model_key", "dataset_hash"):
        if plan.get("baseline_identity", {}).get(key) != recipe.get(key):
            raise ValueError(f"sample baseline {key} differs from recipe")
    ids = plan.get("contrast_ids", [])
    pairs = plan.get("pair_ids", [])
    if not ids or len(ids) != len(set(ids)) or not pairs or len(pairs) != len(set(pairs)):
        raise ValueError("sample requires unique directions and pair identities")
    for key in ("target_pairs", "min_pairs", "layouts_per_pair"):
        if type(plan.get(key)) is not int or plan[key] < 1:
            raise ValueError(f"explicit positive {key} required")
    if not plan["min_pairs"] <= len(pairs) == plan["target_pairs"]:
        raise ValueError("ready status does not match full requested pair target")
    by_id = {}
    for row in contrasts:
        if row["contrast_id"] in by_id:
            raise ValueError("duplicate contrast manifest direction")
        by_id[row["contrast_id"]] = row
    if any(identity not in by_id for identity in ids):
        raise ValueError("selected direction missing from contrast manifest")
    selected = [dict(by_id[identity]) for identity in ids]
    if any(row["split"] != plan.get("split") for row in selected):
        raise ValueError("sample/contrast split mismatch")
    if {row["pair_id"] for row in selected} != set(pairs):
        raise ValueError("sample pair coverage differs from selected directions")
    layouts = plan.get("selected_layouts", {})
    if set(layouts) != set(pairs):
        raise ValueError("selected layout scope differs from selected pairs")
    for pair in pairs:
        expected_layouts = layouts[pair]
        if len(expected_layouts) != plan["layouts_per_pair"] or len(set(expected_layouts)) != len(expected_layouts):
            raise ValueError("sample layout target not reached")
        rows = [row for row in selected if row["pair_id"] == pair]
        if len(rows) != 2 * plan["layouts_per_pair"]:
            raise ValueError("unexpected number of directions per pair")
        if {row.get("layout_id") for row in rows} != set(expected_layouts):
            raise ValueError("selected directions do not match planned layouts")
        if any(sum(row.get("layout_id") == layout for row in rows) != 2 for layout in expected_layouts):
            raise ValueError("each sampled layout requires exactly two directions")
    validate_bidirectional(selected)
    if plan.get("control_complete") is not True:
        raise ValueError("semantic sample was not prospectively control-complete")
    return selected


def run_semantic_campaign(session: Any, contrasts: Sequence[Mapping[str, Any]],
                          recipe: Mapping[str, Any], sample_plan: Mapping[str, Any], *,
                          calibration: Mapping[str, Any], identity: Mapping[str, Any],
                          output_dir: str | Path, protected_inputs: Sequence[str | Path],
                          bootstrap_seed: int, bootstrap_samples: int,
                          execute: bool = False) -> dict[str, Any]:
    """Evaluate a frozen selected sample into fresh durable files, when authorized.

    This is not a server launcher or resume mechanism. Failed/partial directories
    remain preserved; a retry needs a fresh identity and directory. Clean baseline
    exclusion is recorded rather than replaced with a favorable extra pair.
    Source/checkpoint/calibration association retains the explicit caller-binding
    limitations documented by the runtime gate.
    """
    # Explicit denial precedes output creation, source inspection or model calls.
    validate_runtime_context(session, recipe, calibration, identity, execute=execute)
    if not protected_inputs:
        raise ValueError("explicit immutable input paths are required")
    if identity.get("sample_plan_sha256") != sample_plan.get("plan_sha256"):
        raise ValueError("execution identity not bound to sample plan")
    selected = validate_sample_plan(sample_plan, contrasts, recipe)
    # Validate the prospective statistical contract before any model evaluation,
    # not after a costly run with an incompatible random-control recipe.
    from .effects import validate_effect_recipe

    validate_effect_recipe(recipe)
    if type(bootstrap_seed) is not int or type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError("explicit integer bootstrap seed and at least 100 draws required")
    protected = [Path(session.data_dir), Path(session.out_dir)]
    for source in protected_inputs:
        source = Path(source)
        if not source.exists():
            raise ValueError(f"immutable input missing: {source}")
        protected.append(source if source.is_dir() else source.parent)
    output = validate_output_path(output_dir, protected)
    scope = [{"contrast_id": row["contrast_id"], "pair_id": row["pair_id"]} for row in selected]
    header = {"schema": "stretch-semantic-campaign-v1", "status": "started",
              "recipe": dict(recipe), "sample_plan": dict(sample_plan), "identity": dict(identity),
              "analysis_scope": scope, "bootstrap_seed": bootstrap_seed,
              "bootstrap_samples": bootstrap_samples,
              "validation_caveat": "composed real-model calibration must be established separately"}
    # Reject nonserializable metadata before creating a campaign attempt.
    encoded_header = json.dumps(header, indent=2, sort_keys=True, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "attempt.json").open("x", encoding="utf-8") as stream:
        stream.write(encoded_header + "\n")
    results = []
    try:
        with (output / "directions.jsonl").open("x", encoding="utf-8") as stream:
            for contrast in selected:
                result = evaluate_contrast(
                    session, contrast, recipe, calibration=calibration, identity=identity,
                    execute=execute, contrast_manifest=selected)
                stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                results.append(result)
        # This helper analyzes only these new explicit results, not historical
        # confirmation data or the separately sampled donor-control cohorts.
        from .effects import summarize_semantic_effects

        summary = summarize_semantic_effects(
            results, recipe, expected_directions=scope,
            expected_split=sample_plan["split"],
            bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples)
        with (output / "effects.json").open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        completion = {"status": "attempted_all_selected_directions",
                      "execution_complete": len(results) == len(selected),
                      "evaluated_directions": sum(row["status"] == "evaluated" for row in results),
                      "selected_directions": len(selected), "selected_pairs": len(sample_plan["pair_ids"]),
                      "recipe_sha256": recipe["recipe_sha256"],
                      "sample_plan_sha256": sample_plan["plan_sha256"],
                      "scientific_confirmation": "requires_protocol_control_and_artifact_audit"}
        with (output / "completion.json").open("x", encoding="utf-8") as stream:
            json.dump(completion, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return {"completion": completion, "summary": summary, "output_dir": str(output)}
    except BaseException as error:
        # Preserve attempt header and every flushed raw direction. No overwrite,
        # cleanup, hidden retry, status-history rewrite or cloud publication.
        with (output / "failure.json").open("x", encoding="utf-8") as stream:
            json.dump({"status": "failed", "error_type": type(error).__name__,
                       "error": str(error), "finished_directions": len(results),
                       "recipe_sha256": recipe["recipe_sha256"],
                       "sample_plan_sha256": sample_plan["plan_sha256"]},
                      stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        raise
