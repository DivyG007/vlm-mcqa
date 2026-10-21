"""E20 complete semantic-head study with deterministic matched control cohorts."""
from __future__ import annotations

from collections import Counter
import copy
import math
from typing import Any, Mapping
from pathlib import Path

from .campaign import validate_sample_plan
from .extension_calibration import (StudyWriter, calibration_report, core_guard, document, grouped_summary,
                                    measured_check, primitive_checks, protocol, require_calibration,
                                    require_execution, sha)
from .runtime import evaluate_contrast, evaluate_head_control


def bind_calibration_contrast(config: Mapping[str, Any], plan: Mapping[str, Any]) -> dict:
    """Bind technical composition checks to a model-eligible prepared direction.

    The protocol-level calibration direction is only a technical replay case.
    Replication models can answer that frozen Qwen example incorrectly, so the
    prepared correct-only plan supplies the first deterministic eligible
    direction instead. This does not select a scientific layer, head, or effect.
    """
    eligible = plan.get("contrast_ids")
    if not isinstance(eligible, list) or not eligible or not all(isinstance(x, str) for x in eligible):
        raise ValueError("ready semantic preparation lacks eligible calibration direction")
    prepared = dict(config)
    prepared["requested_calibration_contrast_id"] = config.get("calibration_contrast_id")
    prepared["calibration_contrast_id"] = eligible[0]
    return prepared


def coverage_is_adequate(coverage: Mapping[str, Any], *, selected_count: int, min_pairs: int) -> bool:
    """Apply the active phase's declared minimum, including bounded pilots."""
    return (coverage.get("unmatched_control_groups") == 0
            and coverage.get("semantic_pairs", 0) >= min_pairs
            and coverage.get("semantic_evaluated") == selected_count
            and coverage.get("controls_evaluated") == coverage.get("controls_requested"))


def prepare_study(session: Any, config: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[dict, dict]:
    """Generate baseline rows and an eligible sample without hand-written Python.

    Candidate order/sample policy is configured before baseline inference. A
    shortfall is a preserved scientific insufficiency, never a lower target.
    """
    require_execution(context)
    core_guard(session, context)
    if config.get("sample_plan") is not None:
        plan = document(config["sample_plan"])
        return dict(config), {"schema": "stretch.semantic-preparation.v1", "status": plan["status"],
                              "sample_plan_sha256": plan["plan_sha256"], "baseline_source": "supplied"}
    from .sampling import select_fixed_text_sample
    from .data import validate_output_path
    import json
    recipe_source = document(config["recipe"])
    recipe = recipe_source.get("recipe", recipe_source)
    from .effects import validate_effect_recipe
    validate_effect_recipe(recipe)
    body = _body(config, context, recipe)
    raw = document(config["contrasts"])
    contrasts = raw["contrasts"] if isinstance(raw, dict) else raw
    candidates = [c for c in contrasts if c["split"] == config["phase"]]
    from .data import validate_fixed_text_contrast
    # Audit actual endpoint metadata for the entire selected phase before any
    # baseline inference; the contrast's declared split alone is insufficient.
    for contrast in candidates:
        validate_fixed_text_contrast(contrast, session.variants)
    endpoints = sorted({c[key] for c in candidates for key in ("donor_variant_id", "recipient_variant_id")})
    sampling = dict(body["sampling"][config["phase"]])
    if config.get("sampling") is not None and config["sampling"] != sampling:
        raise ValueError("preparation sampling differs from frozen phase policy")
    protected = [Path(session.data_dir), Path(session.out_dir)]
    for value in config.get("protected_inputs", []):
        path = Path(value)
        protected.append(path if path.is_dir() else path.parent)
    for key in ("recipe", "contrasts", "protocol", "frozen_protocol"):
        value = config.get(key)
        if isinstance(value, (str, Path)):
            protected.append(Path(value).resolve().parent)
    output = validate_output_path(context["preparation_dir"], protected)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        with (output / "baseline.jsonl").open("x", encoding="utf-8") as stream:
            for vid in endpoints:
                variant = session.variants[vid]
                readout = session.clean(vid).readout
                row = {**readout, "condition": "clean", "variant_id": vid, "item_id": variant["item_id"],
                       "position": variant["correct_index"], "scheme": variant["label_scheme"],
                       "split": variant["split"], "scene_id": variant["scene_id"], "labels": variant["labels"],
                       "correct_symbol": variant["correct_symbol"], "model_key": session.adapter.spec.key,
                       "dataset_hash": session.manifest["dataset_hash"]}
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
        import hashlib
        baseline_hash = hashlib.sha256((output / "baseline.jsonl").read_bytes()).hexdigest()
        plan = select_control_complete_sample(session, contrasts, rows, split=config["phase"],
            target_pairs=sampling["target_pairs"], min_pairs=sampling["min_pairs"],
            layouts_per_pair=sampling["layouts_per_pair"], seed=sampling["seed"],
            excluded_pair_ids=sampling["excluded_pair_ids"], baseline_identity={
                "model_key": session.adapter.spec.key, "dataset_hash": session.manifest["dataset_hash"],
                "prediction_source_sha256": baseline_hash}, control_policy=body["controls"])
        with (output / "sample_plan.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n")
        prepared = {**dict(config), "sample_plan": str(output / "sample_plan.json")}
        if plan["status"] == "ready" and config["phase"] != "confirmation":
            prepared = bind_calibration_contrast(prepared, plan)
        report = {"schema": "stretch.semantic-preparation.v1", "status": plan["status"],
                  "coverage": plan["coverage"], "sample_plan_sha256": plan["plan_sha256"],
                  "sample_plan_path": prepared["sample_plan"], "baseline_sha256": baseline_hash,
                  "calibration_contrast_id": prepared.get("calibration_contrast_id"),
                  "requested_calibration_contrast_id": prepared.get("requested_calibration_contrast_id"),
                  "output_dir": str(output), "historical_exposure": "caller_exclusion_audit_required"}
        with (output / "preparation.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
        return prepared, report
    except BaseException as error:
        with (output / "failure.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"error": str(error), "completed_baselines": len(rows)}, sort_keys=True) + "\n")
        raise


def _inputs(config: Mapping[str, Any]) -> tuple[dict, list[dict], dict]:
    recipe = document(config["recipe"])
    recipe = recipe.get("recipe", recipe)
    from .effects import validate_effect_recipe
    validate_effect_recipe(recipe)
    contrasts = document(config["contrasts"])
    if isinstance(contrasts, dict):
        contrasts = contrasts["contrasts"]
    plan = document(config["sample_plan"])
    from .effects import validate_effect_recipe
    validate_effect_recipe(recipe)
    return recipe, contrasts, plan


def _identity(context: Mapping[str, Any], recipe: Mapping[str, Any], phase: str) -> dict:
    identity = {**context["identity"], **{k: recipe[k] for k in
                ("model_key", "dataset_hash", "freeze_hash", "head_source_sha256", "recipe_sha256")}}
    identity["calibration_sha256"] = sha(context["core_calibration"])
    identity["calibration_binding"] = {k: identity[k] for k in
                                      ("source_identity", "checkpoint_identity", "image_aware_release_hash")}
    identity["eligibility_decision"] = "approved_for_this_stretch_protocol"
    identity["confirmation_protocol_frozen"] = phase == "confirmation"
    return identity


def _body(config: Mapping[str, Any], context: Mapping[str, Any], recipe: Mapping[str, Any]) -> dict:
    body = protocol(config, context, "semantic_heads")
    if body.get("recipe_sha256") != recipe["recipe_sha256"] or body.get("eligibility_decision") != "approved":
        raise ValueError("protocol must bind exact recipe and explicit scientific eligibility decision")
    controls = body.get("controls", {})
    if (type(controls.get("seed")) is not int or type(controls.get("per_recipient")) is not int
            or controls["per_recipient"] < 1 or type(controls.get("max_donor_reuse")) is not int
            or controls["max_donor_reuse"] < 1 or type(controls.get("match_family")) is not bool):
        raise ValueError("explicit bounded control matching policy required")
    for key in ("min_pairs", "bootstrap_seed", "bootstrap_samples"):
        if type(body.get(key)) is not int or body[key] < (100 if key == "bootstrap_samples" else 0):
            raise ValueError(f"explicit valid {key} required")
    if body["min_pairs"] < 1:
        raise ValueError("positive min_pairs required")
    edges = controls.get("margin_bin_edges", [])
    if (not isinstance(edges, list) or any(type(x) not in (int, float) or not math.isfinite(x) for x in edges)
            or sorted(set(edges)) != edges):
        raise ValueError("margin_bin_edges must be sorted distinct finite baseline thresholds")
    for phase in ("discovery", "confirmation"):
        sampling = body["sampling"][phase]
        if (any(type(sampling[k]) is not int or sampling[k] < 1 for k in ("target_pairs", "min_pairs", "layouts_per_pair"))
                or sampling["min_pairs"] > sampling["target_pairs"] or type(sampling["seed"]) not in (int, str)
                or not isinstance(sampling["excluded_pair_ids"], list)):
            raise ValueError("complete explicit discovery/confirmation sampling policies required")
    return body


def _match_controls(session: Any, selected: list[dict], policy: Mapping[str, Any],
                    used: Counter | None = None) -> tuple[list[dict], list[dict], Counter]:
    """Match transitions and recipient exactly; outcomes only establish correctness.

    Deterministic candidate order is fixed before querying correctness. Donors
    are drawn only from the current phase. Reuse exclusions remain visible.
    """
    chosen, exclusions = [], []
    used = Counter() if used is None else Counter(used)
    for semantic in selected:
        recipient = session.variants[semantic["recipient_variant_id"]]
        target = semantic["donor_symbol"]
        for kind, count in (("position", 1), ("random_pair", policy["per_recipient"])):
            pool = []
            for candidate in session.variants.values():
                if (candidate["split"] != recipient["split"] or candidate["labels"] != recipient["labels"]
                        or candidate["label_scheme"] != recipient["label_scheme"]
                        or candidate["variant_kind"] != "position" or candidate["correct_symbol"] != target):
                    continue
                if kind == "position":
                    if candidate["item_id"] != recipient["item_id"]:
                        continue
                elif (not candidate.get("counterfactual_pair_id")
                      or candidate["counterfactual_pair_id"] == recipient.get("counterfactual_pair_id")
                      or candidate["image"] == recipient["image"]
                      or (policy["match_family"] and candidate["question_family"] != recipient["question_family"])):
                    continue
                pool.append(candidate)
            pool.sort(key=lambda v: sha([policy["seed"], semantic["contrast_id"], kind, v["variant_id"]]))
            retained = 0
            for donor in pool:
                if retained == count:
                    break
                if kind == "random_pair" and used[donor["variant_id"]] >= policy["max_donor_reuse"]:
                    continue
                if not (session.clean(donor["variant_id"]).correct and session.clean(recipient["variant_id"]).correct):
                    continue
                edges = policy.get("margin_bin_edges", [])
                if edges:
                    source_margin = session.clean(semantic["donor_variant_id"]).readout.get("margin")
                    donor_margin = session.clean(donor["variant_id"]).readout.get("margin")
                    if (type(source_margin) not in (int, float) or type(donor_margin) not in (int, float)
                            or not math.isfinite(source_margin) or not math.isfinite(donor_margin)
                            or sum(source_margin >= edge for edge in edges) != sum(donor_margin >= edge for edge in edges)):
                        continue
                row = {"contrast_id": f"E20-control:{semantic['contrast_id']}:{kind}:{retained}",
                       "contrast_type": kind, "split": recipient["split"],
                       "pair_id": recipient["counterfactual_pair_id"],
                       "donor_variant_id": donor["variant_id"], "recipient_variant_id": recipient["variant_id"],
                       "donor_symbol": donor["correct_symbol"], "recipient_symbol": recipient["correct_symbol"],
                       "donor_content": donor["semantic_answer"], "recipient_content": recipient["semantic_answer"],
                       "label_scheme": recipient["label_scheme"], "matched_semantic_contrast_id": semantic["contrast_id"]}
                chosen.append(row)
                used[donor["variant_id"]] += kind == "random_pair"
                retained += 1
            if retained != count:
                exclusions.append({"semantic_contrast_id": semantic["contrast_id"], "control_type": kind,
                                   "requested": count, "retained": retained, "reason": "no_eligible_match_under_frozen_policy"})
    return chosen, exclusions, used


def match_controls(session: Any, selected: list[dict], policy: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    """Compatibility wrapper for one already-frozen semantic cohort."""
    chosen, exclusions, _ = _match_controls(session, selected, policy)
    return chosen, exclusions


def select_control_complete_sample(session: Any, contrasts: list[dict], baseline_rows: list[dict], *,
                                   split: str, target_pairs: int, layouts_per_pair: int,
                                   min_pairs: int, seed: str | int, excluded_pair_ids: list[str],
                                   baseline_identity: Mapping[str, Any],
                                   control_policy: Mapping[str, Any]) -> dict:
    """Select semantic pairs only after all direction-level controls are frozen.

    Pair ordering and layout ordering are outcome-independent hashes. Clean
    correctness and margin bins are used only as prospective eligibility gates.
    Donor reuse is allocated greedily in that frozen pair order. A rejected pair
    cannot consume donors, and execution receives the exact hashed control rows.
    """
    from .sampling import select_fixed_text_sample, _hash
    phase_pairs = {row["pair_id"] for row in contrasts if row.get("split") == split}
    if not phase_pairs:
        raise ValueError("no semantic pairs in requested phase")
    preliminary = select_fixed_text_sample(
        contrasts, session.variants, baseline_rows, split=split,
        target_pairs=len(phase_pairs), layouts_per_pair=layouts_per_pair,
        min_pairs=1, seed=seed, excluded_pair_ids=excluded_pair_ids,
        baseline_identity=baseline_identity)
    used = Counter()
    complete_pairs, controls_by_pair, control_exclusions = [], {}, {}
    for pair in preliminary["pair_ids"]:
        layouts = preliminary["eligible_layouts_by_pair"][pair][:layouts_per_pair]
        semantic = [row for row in contrasts
                    if row.get("split") == split and row["pair_id"] == pair
                    and row.get("layout_id") in layouts]
        semantic.sort(key=lambda row: row["contrast_id"])
        expected = 2 * layouts_per_pair
        if len(semantic) != expected:
            raise ValueError("eligible semantic pair lost a bidirectional layout")
        controls, exclusions, tentative = _match_controls(session, semantic, control_policy, used)
        if exclusions:
            control_exclusions[pair] = exclusions
            continue
        expected_controls = expected * (1 + control_policy["per_recipient"])
        if len(controls) != expected_controls:
            raise ValueError("control matcher returned incomplete cohort without exclusions")
        used = tentative
        complete_pairs.append(pair)
        controls_by_pair[pair] = controls
    selected_pairs = complete_pairs[:target_pairs]
    selected_layouts = {pair: preliminary["eligible_layouts_by_pair"][pair][:layouts_per_pair]
                        for pair in selected_pairs}
    contrast_ids = [row["contrast_id"] for pair in selected_pairs
                    for layout in selected_layouts[pair]
                    for row in sorted((r for r in contrasts if r["pair_id"] == pair
                                       and r.get("layout_id") == layout), key=lambda r: r["contrast_id"])]
    control_rows = [row for pair in selected_pairs for row in controls_by_pair[pair]]
    control_hash = _hash(control_rows)
    status = ("insufficient" if len(selected_pairs) < min_pairs else
              "target_shortfall" if len(selected_pairs) < target_pairs else "ready")
    plan = copy.deepcopy(preliminary)
    plan.update({"status": status, "target_pairs": target_pairs, "min_pairs": min_pairs,
                 "pair_ids": selected_pairs, "contrast_ids": contrast_ids,
                 "selected_layouts": selected_layouts,
                 "control_complete": True,
                 "control_policy_sha256": _hash(dict(control_policy)),
                 "matched_controls": control_rows,
                 "matched_controls_sha256": control_hash})
    reasons = plan["exclusions"]["pair_reasons"]
    for pair in control_exclusions:
        reasons[pair] = "control_incomplete"
    plan["exclusions"]["control_reasons"] = control_exclusions
    plan["exclusions"]["unselected_control_complete_pairs_after_cap"] = complete_pairs[target_pairs:]
    plan["coverage"].update({
        "baseline_eligible_pairs": len(preliminary["pair_ids"]),
        "control_complete_pairs": len(complete_pairs),
        "control_incomplete_pairs": len(control_exclusions),
        "selected_pairs": len(selected_pairs),
        "selected_directions": len(contrast_ids),
        "selected_controls": len(control_rows),
        "target_shortfall": max(0, target_pairs - len(selected_pairs)),
        "control_exclusion_groups": sum(len(rows) for rows in control_exclusions.values()),
        "unrelated_donor_reuse": dict(used),
    })
    plan["eligibility"] = "clean_correct_endpoints_and_complete_frozen_direction_controls"
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = _hash(plan)
    return plan


def validate_frozen_controls(plan: Mapping[str, Any], selected: list[dict],
                             policy: Mapping[str, Any]) -> list[dict]:
    """Validate the exact preparation-time controls without rematching donors."""
    from .sampling import _hash
    rows = plan.get("matched_controls")
    if (plan.get("control_complete") is not True or not isinstance(rows, list)
            or plan.get("control_policy_sha256") != _hash(dict(policy))
            or plan.get("matched_controls_sha256") != _hash(rows)):
        raise ValueError("sample plan lacks immutable control-complete binding")
    expected = {(semantic["contrast_id"], kind)
                for semantic in selected for kind in ("position", "random_pair")}
    counts = Counter((row.get("matched_semantic_contrast_id"), row.get("contrast_type")) for row in rows)
    if set(counts) != expected:
        raise ValueError("frozen controls do not cover every selected semantic direction")
    for semantic_id, kind in expected:
        wanted = 1 if kind == "position" else policy["per_recipient"]
        if counts[semantic_id, kind] != wanted:
            raise ValueError("frozen control multiplicity differs from protocol")
    if len({row.get("contrast_id") for row in rows}) != len(rows):
        raise ValueError("duplicate frozen control identities")
    random_reuse = Counter(row["donor_variant_id"] for row in rows if row["contrast_type"] == "random_pair")
    if random_reuse and max(random_reuse.values()) > policy["max_donor_reuse"]:
        raise ValueError("frozen unrelated donor exceeds reuse cap")
    return [dict(row) for row in rows]


def calibrate_study(session: Any, config: Mapping[str, Any], context: Mapping[str, Any]) -> dict:
    require_execution(context)
    core_guard(session, context)
    recipe, contrasts, _ = _inputs(config)
    body = _body(config, context, recipe)
    from ..core7.hooks import Site
    sites = [Site(recipe["image_site"], recipe["image_layer"])]
    sites += [Site("head", layer) for layer in recipe["head_layers"]]
    examples = list(config["calibration_variant_ids"])
    checks = primitive_checks(session, examples, sites, config["tolerance"])
    selected = next((c for c in contrasts if c["contrast_id"] == config["calibration_contrast_id"]), None)
    if selected is None or selected["split"] == "confirmation":
        raise ValueError("explicit non-confirmation composed calibration direction required")
    identity = _identity(context, recipe, "discovery")
    reverse = next((c for c in contrasts if c["donor_variant_id"] == selected["recipient_variant_id"]
                    and c["recipient_variant_id"] == selected["donor_variant_id"]), None)
    if reverse is None:
        raise ValueError("calibration direction lacks reverse")
    from .trajectory import _BatchSession
    first = evaluate_contrast(_BatchSession(session, 1), selected, recipe, calibration=context["core_calibration"],
                              identity=identity, execute=True, contrast_manifest=[selected, reverse])
    second = evaluate_contrast(_BatchSession(session, 2), selected, recipe, calibration=context["core_calibration"],
                               identity=identity, execute=True, contrast_manifest=[selected, reverse])
    if first["status"] != "evaluated" or second["status"] != "evaluated":
        raise ValueError("composed calibration requires eligible explicit pair")
    a, b = {r["name"]: r for r in first["rows"]}, {r["name"]: r for r in second["rows"]}
    checks["actual_composition_batch_single"] = measured_check(max(abs(a[n]["patched_diff"] - b[n]["patched_diff"]) for n in a), config["tolerance"])
    checks["actual_image_head_self_shams"] = measured_check(max(abs(r["raw_margin_change"]) for r in first["rows"]
        if r["operation"] in ("self_image", "self_head")), config["tolerance"])
    return calibration_report(session, context, "semantic_heads", body, checks, examples)


def run_study(session: Any, config: Mapping[str, Any], context: Mapping[str, Any]) -> dict:
    require_execution(context)
    recipe, contrasts, plan = _inputs(config)
    body = _body(config, context, recipe)
    require_calibration(session, context, "semantic_heads", body)
    if plan.get("split") != config["phase"]:
        raise ValueError("sample phase mismatch")
    planned = body["sampling"][config["phase"]]
    if any(plan.get(k) != planned[k] for k in ("target_pairs", "min_pairs", "layouts_per_pair", "seed")):
        raise ValueError("sample targets/order differ from frozen protocol")
    if set(plan.get("exclusions", {}).get("caller_pair_ids", [])) != set(planned["excluded_pair_ids"]):
        raise ValueError("sample exclusion reserve differs from frozen protocol")
    if plan.get("status") != "ready":
        writer = StudyWriter(session, config, context, "semantic_heads", body)
        writer.write("sample_plan.json", plan)
        return writer.finish(plan.get("coverage", {}), "insufficient_baseline_eligible_pairs",
                             {"sample_status": plan.get("status")}, status="insufficient")
    selected = validate_sample_plan(plan, contrasts, recipe)
    writer = StudyWriter(session, config, context, "semantic_heads", body)
    identity = _identity(context, recipe, config["phase"])
    semantic_results, control_results = [], []
    try:
        controls = validate_frozen_controls(plan, selected, body["controls"])
        unmatched = []
        writer.write("matched_controls.json", controls)
        for row in unmatched:
            writer.append(row, excluded=True)
        for contrast in selected:
            result = evaluate_contrast(session, contrast, recipe, calibration=context["core_calibration"],
                                        identity=identity, execute=True, contrast_manifest=selected)
            semantic_results.append(result)
            writer.append(result, excluded=result["status"] != "evaluated")
        for contrast in controls:
            result = evaluate_head_control(session, contrast, recipe, calibration=context["core_calibration"],
                                           identity=identity, execute=True)
            result["matched_semantic_contrast_id"] = contrast["matched_semantic_contrast_id"]
            control_results.append(result)
            writer.append(result, excluded=result["status"] != "evaluated")
        from .effects import summarize_semantic_effects, descriptive_control_summary
        effects = summarize_semantic_effects(semantic_results, recipe,
            expected_directions=[{"contrast_id": c["contrast_id"], "pair_id": c["pair_id"]} for c in selected],
            expected_split=config["phase"], bootstrap_seed=body["bootstrap_seed"], bootstrap_samples=body["bootstrap_samples"])
        control_summary = descriptive_control_summary(control_results)
        donor_counts = Counter(c["donor_variant_id"] for c in controls if c["contrast_type"] == "random_pair")
        coverage = {"semantic_requested": len(selected), "semantic_evaluated": sum(r["status"] == "evaluated" for r in semantic_results),
                    "controls_requested": len(selected) * (1 + body["controls"]["per_recipient"]),
                    "controls_evaluated": sum(r["status"] == "evaluated" for r in control_results),
                    "unmatched_control_groups": len(unmatched), "semantic_pairs": len(plan["pair_ids"]),
                    "unrelated_donor_reuse": dict(donor_counts)}
        adequate = coverage_is_adequate(coverage, selected_count=len(selected),
                                        min_pairs=planned["min_pairs"])
        summary = {"semantic_effects": effects, "controls": control_summary,
                   "baseline_head_only_diagnostic": "self_head_shared_sham/top4 is recorded separately from clean; both are measured",
                   "donor_dependence": "recipient-pair intervals do not remove reused donor dependence"}
        return writer.finish(coverage, "requires_bounded_effect_and_control_interpretation" if adequate else "insufficient_coverage",
                             summary, status="complete" if adequate else "insufficient",
                             freeze_body=body if config["phase"] == "discovery" and adequate else None)
    except BaseException as error:
        writer.failure(error)
        raise
