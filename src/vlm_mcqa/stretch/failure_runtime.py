"""E22 failure-compatible rescue/remapping studies; no correct/correct recovery."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .extension_calibration import (StudyWriter, calibration_report, core_guard, document, forward,
    grouped_summary, measured_check, primitive_checks, protocol, require_calibration, require_execution, sha)
from .protocol import load_frozen_heads

DONOR_CLASSES = {"same_image_remap", "semantic_match", "unrelated_same_label", "wrong_donor"}
DISCOVERY_SPLITS = {"discovery", "behavior", "screening", "calibration"}


def _prospective_rules(body: Mapping[str, Any]) -> dict:
    """Use the complete already-frozen E19 predicates, never guessed flags."""
    from .failure_protocol import _validate_rules
    source = document(body["strata_rules_path"])
    if not isinstance(source, dict):
        raise ValueError("frozen failure rules or E19 failure-design report required")
    rules = source.get("rules", source)
    _validate_rules(rules)
    if (rules["rules_sha256"] != body["strata_rules_sha256"]
            or rules["strata_flags"] != body["allowed_strata"]
            or rules["min_scenes"] != body["min_stratum_scenes"]):
        raise ValueError("prospective failure predicates/counts differ from frozen scientific protocol")
    tolerance = body.get("tie_tolerance")
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("explicit finite nonnegative failure tie_tolerance required")
    return rules


def select_phase_cases(report: Mapping[str, Any], dataset: Mapping[str, Any], rules: Mapping[str, Any],
                       *, phase: str) -> tuple[list[dict], dict]:
    """Apply E19 rules to ONE prospective phase without its internal hash-resplit.

    Selection uses only clean behavior/content diagnostics and frozen metadata,
    confidence and reuse predicates. Causal intervention outcomes are unavailable.
    """
    from .failure_protocol import _validate_rules, _hash
    from .statistics import cluster_estimate
    _validate_rules(rules)
    if phase not in {"discovery", "confirmation"} or report.get("status") != "descriptive_not_causal":
        raise ValueError("one explicit phase and audited descriptive baseline report required")
    items = {item["item_id"]: item for item in dataset["items"]}
    rows = [dict(row) for row in report["rows"] if row["stage"] == rules["stage"]]
    if len({row["variant_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate prospective failure-stage observations")
    if any(row.get("split") not in ({"confirmation"} if phase == "confirmation" else DISCOVERY_SPLITS)
           for row in rows):
        raise ValueError("prospective descriptive observations cross scientific phase")
    for row in rows:
        item = items.get(row["item_id"], {})
        for key in ("question_family", "reasoning_depth"):
            if row.get(key) is None:
                row[key] = item.get(key)
        if row.get("reasoning_depth") is None:
            row["reasoning_depth"] = item.get("program_depth")
    correct = [row for row in rows if row.get("final_correct") is True and not row.get("unresolved")]
    reliability = cluster_estimate(correct, value="content_proxy_correct", cluster="scene_id",
                                  seed=rules["seed"], draws=rules["bootstrap_draws"])
    reliable = (reliability["n_clusters"] >= rules["min_scenes"] and reliability["ci95"] is not None
                and reliability["ci95"][0] >= rules["reliability_min"])
    cases, strata, reuse = [], {}, Counter()
    for flag in rules["strata_flags"]:
        failures = [row for row in rows if row.get("final_correct") is False
                    and flag in row.get("descriptive_flags", []) and not row.get("unresolved")]
        matches, unmatched = [], []
        for failure in sorted(failures, key=lambda row: _hash([rules["seed"], row["variant_id"]])):
            margin = failure.get("confidence_gap")
            bucket = sum(margin >= edge for edge in rules["margin_bins"]) if margin is not None else None
            candidates = []
            for candidate in correct:
                if (candidate["scene_id"] == failure["scene_id"]
                        or reuse[candidate["scene_id"]] >= rules["max_control_reuse"]
                        or candidate.get("split") != failure.get("split")):
                    continue
                if any(failure.get(key) is None or failure.get(key) != candidate.get(key)
                       for key in rules["match_fields"]):
                    continue
                cm = candidate.get("confidence_gap")
                if bucket is None or cm is None or sum(cm >= edge for edge in rules["margin_bins"]) != bucket:
                    continue
                candidates.append(candidate)
            if not candidates:
                unmatched.append(failure["variant_id"])
                continue
            control = min(candidates, key=lambda row: _hash([rules["seed"], failure["variant_id"], row["variant_id"]]))
            reuse[control["scene_id"]] += 1
            matches.append({"failure_variant_id": failure["variant_id"], "failure_scene_id": failure["scene_id"],
                            "control_variant_id": control["variant_id"], "control_scene_id": control["scene_id"]})
            failure_case_id = f"{phase}:{flag}:failure:{failure['variant_id']}"
            cases.extend([{"case_id": failure_case_id, "recipient_variant_id": failure["variant_id"],
                           "stratum": flag, "role": "failure", "source_phase": phase},
                          {"case_id": f"{phase}:{flag}:matched:{failure['variant_id']}:{control['variant_id']}",
                           "recipient_variant_id": control["variant_id"], "stratum": flag,
                           "role": "matched_correct", "matched_failure_variant_id": failure["variant_id"],
                           "matched_failure_case_id": failure_case_id, "source_phase": phase}])
        failure_scenes = len({match["failure_scene_id"] for match in matches})
        control_scenes = len({match["control_scene_id"] for match in matches})
        enough = min(failure_scenes, control_scenes) >= rules["min_scenes"]
        strata[flag] = {"reliability_gate_passed": reliable, "sample_gate_passed": enough,
                        "eligible_failure_variants": [row["variant_id"] for row in failures],
                        "matched_correct_controls": matches, "unmatched_failure_variants": unmatched,
                        "matched_failure_scenes": failure_scenes, "matched_control_scenes": control_scenes,
                        "status": "ready" if reliable and enough else "insufficient"}
    return cases, {"status": "ready" if reliable and all(s["status"] == "ready" for s in strata.values()) else "insufficient",
                   "phase": phase, "content_proxy_reliability": reliability, "strata": strata,
                   "control_scene_reuse": dict(reuse), "candidate_stage_rows": len(rows),
                   "unresolved_rows": sum(row.get("unresolved", True) for row in rows),
                   "selection": "frozen clean-baseline rules; same-phase scene-independent matched controls; no causal outcomes"}


def prepare_study(session: Any, config: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[dict, dict]:
    """Prepare fresh phase-local wrong/correct cases after scientific rules freeze.

    Preflight inputs contain candidate IDs, never guessed failure/correct roles.
    All candidates and the single frozen lens stage are saved before matching.
    """
    require_execution(context)
    core_guard(session, context)
    body, _ = _body(session, config, context)
    if config.get("candidate_scope_path") is None:
        if config["phase"] == "confirmation":
            raise ValueError("fresh failure confirmation requires prospective candidate_scope_path preparation")
        return dict(config), {"study": "failures", "status": "declared_inputs",
                              "provenance": "externally_prepared_discovery_cases; exploratory_only"}
    rules = _prospective_rules(body)
    if rules["stage"] > session.adapter.num_layers:
        raise ValueError("frozen failure lens stage exceeds current model")
    scope = document(config["candidate_scope_path"])
    ids = scope.get("variant_ids")
    if (scope.get("schema") != "stretch.failure-candidate-scope.v1" or scope.get("phase") != config["phase"]
            or not isinstance(ids, list) or not ids or any(not isinstance(v, str) or not v for v in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("phase-bound nonempty unique prospective candidate IDs required")
    allowed = {"confirmation"} if config["phase"] == "confirmation" else set(body["discovery_allowed_splits"])
    for vid in ids:
        if vid not in session.variants or session.variants[vid]["split"] not in allowed:
            raise ValueError("prospective failure candidate outside frozen phase before inference")
    scenes = {session.variants[vid]["scene_id"] for vid in ids}
    pairs = {session.variants[vid].get("counterfactual_pair_id") for vid in ids} - {None}
    for variant in session.variants.values():
        if ((variant["scene_id"] in scenes or variant.get("counterfactual_pair_id") in pairs)
                and variant["split"] not in allowed):
            raise ValueError("prospective failure scene/pair crosses frozen phase boundary before inference")
    from .data import validate_output_path
    protected = [Path(session.data_dir), Path(session.out_dir), Path(config["candidate_scope_path"]).resolve().parent,
                 Path(body["strata_rules_path"]).resolve().parent]
    for value in config.get("protected_inputs", []):
        path = Path(value)
        protected.append(path if path.is_dir() else path.parent)
    output = validate_output_path(context["preparation_dir"], protected)
    output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    ordered_ids = sorted(ids, key=lambda vid: (sha([rules["seed"], vid]), vid))
    binding = {"identity_sha256": sha(context["identity"]), "rules_sha256": rules["rules_sha256"],
               "scope_sha256": sha(scope), "candidate_scope_file_sha256": hashlib.sha256(Path(config["candidate_scope_path"]).read_bytes()).hexdigest(),
               "rules_file_sha256": hashlib.sha256(Path(body["strata_rules_path"]).read_bytes()).hexdigest()}
    write("attempt.json", {"schema": "stretch.failure-preparation-attempt.v1", "phase": config["phase"],
                           "binding": binding, "identity": context["identity"], "ordered_variant_ids": ordered_ids,
                           "rules": rules, "tie_tolerance": body["tie_tolerance"]})
    behavior, lens = [], []
    try:
        with (output / "baseline.jsonl").open("x", encoding="utf-8") as behavior_out, \
             (output / "lens.jsonl").open("x", encoding="utf-8") as lens_out:
            for vid in ordered_ids:
                variant, clean = session.variants[vid], session.clean(vid)
                prepared = session.prepared(vid)
                metadata = {"variant_id": vid, "item_id": variant["item_id"], "scheme": variant["label_scheme"],
                            "position": variant["correct_index"], "split": variant["split"]}
                b = {**clean.readout, **metadata, "condition": "clean"}
                site = "resid_pre:0" if rules["stage"] == 0 else f"resid_post:{rules['stage'] - 1}"
                content_ids = [prepared.content_ids.get(value) for value in variant["option_contents"]]
                content_ok = None not in content_ids and len(set(content_ids)) == 4
                l = {**metadata, "stage": rules["stage"], "content_logits": None,
                     "content_first_token_ids": content_ids, "distinct_content_first_tokens": content_ok}
                if content_ok:
                    logits = session.adapter.handle.readout(clean.sites[site]).float().cpu()
                    l["content_logits"] = [float(logits[token]) for token in content_ids]
                behavior_out.write(json.dumps(b, sort_keys=True, allow_nan=False) + "\n")
                behavior_out.flush()
                lens_out.write(json.dumps(l, sort_keys=True, allow_nan=False) + "\n")
                lens_out.flush()
                behavior.append(b)
                lens.append(l)
        from .failure_analysis import analyze_failures
        dataset = {"items": list(session.items.values()), "variants": list(session.variants.values())}
        descriptive = analyze_failures(dataset, behavior, lens, stages=[rules["stage"]],
                                       expected_variant_ids=ordered_ids, tie_tolerance=body["tie_tolerance"])
        write("descriptive_baselines.json", descriptive)
        cases, selection = select_phase_cases(descriptive, dataset, rules, phase=config["phase"])
        validate_case_scope(session, cases, body, config["phase"])
        write("selection.json", selection)
        files = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                 for name in ("attempt.json", "baseline.jsonl", "lens.jsonl", "descriptive_baselines.json", "selection.json")}
        manifest = {"schema": "stretch.failure-cases.v1", "phase": config["phase"], "rules_sha256": rules["rules_sha256"],
                    "cases": cases, "generation": {"mode": "prospective_frozen_rules", **binding, "artifacts": files},
                    "claim_limit": "baseline-conditioned sample; heldout causal outcomes, not unobserved baseline behavior"}
        manifest["cases_sha256"] = sha(manifest)
        write("cases.json", manifest)
        report = {"schema": "stretch.failure-preparation.v1", "study": "failures", "status": selection["status"],
                  "phase": config["phase"], "coverage": selection, "binding": binding,
                  "cases_sha256": manifest["cases_sha256"], "cases_path": str(output / "cases.json"),
                  "artifacts": files, "scientific_confirmation": False}
        write("preparation.json", report)
        updated = {**dict(config), "cases": str(output / "cases.json"),
                   "_prepared_failure_cases_sha256": manifest["cases_sha256"]}
        return updated, report
    except BaseException as error:
        write("failure.json", {"error_type": type(error).__name__, "error": str(error),
                               "baseline_rows": len(behavior), "lens_rows": len(lens)})
        raise


def validate_case_scope(session: Any, units: list[dict], body: Mapping[str, Any], phase: str) -> None:
    """Validate the complete case scope before any preparation/baseline inference."""
    if phase not in ("discovery", "confirmation"):
        raise ValueError("explicit failure-study phase required")
    allowed = body.get("discovery_allowed_splits", ["discovery"])
    if (not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed)
            or not set(allowed).issubset(DISCOVERY_SPLITS)):
        raise ValueError("discovery split policy must explicitly exclude held-out confirmation")
    for unit in units:
        if unit["stratum"] not in body["allowed_strata"] or unit["role"] not in ("failure", "matched_correct"):
            raise ValueError("unit outside frozen strata/roles")
        recipient = session.variants[unit["recipient_variant_id"]]
        if recipient["split"] not in ({"confirmation"} if phase == "confirmation" else set(allowed)):
            raise ValueError("failure recipient outside frozen phase/split scope")
        donor_id = unit.get("donor_variant_id")
        if donor_id is not None and session.variants[donor_id]["split"] != recipient["split"]:
            raise ValueError("explicit failure donor crosses split")


def _body(session: Any, config: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[dict, dict]:
    body = protocol(config, context, "failures")
    if "discovery_allowed_splits" not in body:
        raise ValueError("explicit non-held-out discovery split policy required")
    validate_case_scope(session, [], body, config["phase"])
    if (not body.get("allowed_strata") or len(body["allowed_strata"]) > 2
            or len(set(body["allowed_strata"])) != len(body["allowed_strata"])):
        raise ValueError("freeze at most two explicit descriptive strata")
    if not set(body["donor_classes"]).issubset(DONOR_CLASSES) or not body["donor_classes"]:
        raise ValueError("explicit supported donor classes required")
    if not isinstance(body.get("strata_rules_sha256"), str) or len(body["strata_rules_sha256"]) != 64:
        raise ValueError("explicit frozen reliability/strata rule identity required")
    if body.get("strata_rules_path") is not None:
        _prospective_rules(body)
    elif config["phase"] == "confirmation" or config.get("candidate_scope_path") is not None:
        raise ValueError("prospective failure preparation requires complete frozen strata_rules_path")
    frozen = load_frozen_heads(body["head_source"], model_key=session.adapter.spec.key,
                               dataset_hash=session.manifest["dataset_hash"])
    if body["head_source_sha256"] != frozen["source_sha256"] or body["head_k"] not in (4, 16):
        raise ValueError("frozen head source/top-k mismatch")
    for key in ("random_sets", "min_stratum_scenes", "max_donor_reuse", "bootstrap_samples"):
        if type(body[key]) is not int or body[key] < (100 if key == "bootstrap_samples" else 1):
            raise ValueError(f"explicit positive {key} required")
    for key in ("seed", "bootstrap_seed"):
        if type(body[key]) is not int:
            raise ValueError(f"explicit integer {key} required")
    for group in ("image", "options", "final"):
        site = body["sites"][group]
        if site["site"] not in ("resid_pre", "resid_post") or not 0 <= site["layer"] < session.adapter.num_layers:
            raise ValueError("invalid failure intervention boundary")
    if body["sites"]["final"] != {"site": "resid_post", "layer": session.adapter.num_layers - 1}:
        raise ValueError("failure final repair must remain the last-layer diagnostic reference")
    image_enabled = body.get("image_repair_enabled")
    evidence = body.get("e20_restoration_evidence_sha256")
    if type(image_enabled) is not bool:
        raise ValueError("explicit image_repair_enabled decision required")
    if image_enabled:
        if (body["sites"]["image"] != {"site": "resid_post", "layer": 0}
                or not isinstance(evidence, str) or not re.fullmatch(r"[0-9a-f]{64}", evidence)):
            raise ValueError("E22 image repair requires valid layer-0 E20 restoration evidence identity")
    elif evidence is not None:
        raise ValueError("disabled image repair cannot claim E20 restoration evidence")
    candidates = body.get("option_candidate_layers", [])
    if (not isinstance(candidates, list) or len(set(candidates)) != len(candidates)
            or any(type(layer) is not int or not 0 <= layer < session.adapter.num_layers for layer in candidates)):
        raise ValueError("option_candidate_layers must be unique valid decoder layers")
    if config["phase"] == "discovery" and not candidates:
        raise ValueError("failure discovery requires prespecified option candidate layers")
    tolerance = body.get("layer_tie_tolerance")
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("finite nonnegative layer_tie_tolerance required")
    if config["phase"] == "confirmation" and (candidates
            or body.get("selected_option_layer") != body["sites"]["options"]["layer"]):
        raise ValueError("confirmation must use exactly the discovery-selected option layer")
    if body.get("option_scope") not in ("truth", "all"):
        raise ValueError("explicit truth or all option_scope required")
    primary = body.get("primary_conditions_by_donor_class", {})
    allowed_conditions = {"image_repair", "options_repair", "head_repair", "final_repair"}
    if set(primary) != set(body["donor_classes"]) or any(not names or not set(names).issubset(allowed_conditions)
            or len(names) != len(set(names)) for names in primary.values()):
        raise ValueError("freeze nonempty primary repair endpoints for every donor class")
    return body, frozen


def candidate_units(config: Mapping[str, Any], body: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    """Adapt saved internal strata or an explicit new-case manifest without guessing."""
    exclusions = []
    if config.get("cases") is not None:
        supplied = document(config["cases"])
        if supplied.get("schema") != "stretch.failure-cases.v1" or supplied.get("phase") != config["phase"]:
            raise ValueError("phase-bound explicit failure-cases manifest required")
        if supplied.get("rules_sha256") != body["strata_rules_sha256"]:
            raise ValueError("new-case manifest does not bind frozen strata rules")
        generated = supplied.get("generation", {}).get("mode") == "prospective_frozen_rules"
        if generated:
            if (supplied.get("cases_sha256") != sha({k: v for k, v in supplied.items() if k != "cases_sha256"})
                    or config.get("_prepared_failure_cases_sha256") != supplied["cases_sha256"]):
                raise ValueError("prospective cases changed or were not prepared in this campaign")
        elif config["phase"] == "confirmation":
            raise ValueError("externally asserted failure roles cannot supply fresh confirmation")
        units = supplied["cases"]
        if len({u["case_id"] for u in units}) != len(units):
            raise ValueError("duplicate failure case ID")
        return units, exclusions
    report = document(config["strata_manifest"])
    if report.get("schema") != "stretch.failure-strata.v1" or config["phase"] == "confirmation":
        raise ValueError("existing internal-validation strata are not fresh confirmation cases")
    if (report.get("manifest_sha256") != sha({k: v for k, v in report.items() if k != "manifest_sha256"})
            or report.get("rules", {}).get("rules_sha256") != body["strata_rules_sha256"]):
        raise ValueError("changed failure strata report/rules")
    phase = config["strata_phase"]
    if phase not in ("discovery", "internal_validation"):
        raise ValueError("explicit internal strata phase required")
    source = report[phase]["strata"] if phase in report else report["phases"][phase]["strata"]
    units = []
    for stratum in body["allowed_strata"]:
        entry = source[stratum]
        if (entry.get("status") != "candidate_for_causal_design" or entry.get("reliability_gate_passed") is not True
                or entry.get("sample_gate_passed") is not True or entry.get("matched_control_gate_passed") is not True):
            exclusions.append({"stratum": stratum, "reason": "saved_readout_reliability_or_sample_gate_failed"})
            continue
        for vid in entry["variant_ids"]:
            units.append({"case_id": f"{stratum}:failure:{vid}", "recipient_variant_id": vid,
                          "stratum": stratum, "role": "failure", "source_phase": phase})
        for match in entry.get("matched_correct_controls", []):
            units.append({"case_id": f"{stratum}:matched:{match['failure_variant_id']}:{match['control_variant_id']}",
                          "recipient_variant_id": match["control_variant_id"], "stratum": stratum,
                          "role": "matched_correct", "matched_failure_variant_id": match["failure_variant_id"],
                          "matched_failure_case_id": f"{stratum}:failure:{match['failure_variant_id']}",
                          "source_phase": phase})
    return units, exclusions


def validate_prepared_cases(config: Mapping[str, Any], context: Mapping[str, Any], body: Mapping[str, Any]) -> None:
    """Bind generated roles to this phase's actual source/model/rules/raw bytes."""
    if config.get("_prepared_failure_cases_sha256") is None:
        if config["phase"] == "confirmation":
            raise ValueError("confirmation cases require current prospective preparation")
        return
    path = Path(config["cases"])
    supplied = document(path)
    generation = supplied.get("generation", {})
    if (generation.get("mode") != "prospective_frozen_rules"
            or generation.get("identity_sha256") != sha(context["identity"])
            or generation.get("rules_sha256") != body["strata_rules_sha256"]
            or generation.get("scope_sha256") != sha(document(config["candidate_scope_path"]))):
        raise ValueError("prospective failure cases differ from current model/source/data/phase/rules")
    for key, source in (("candidate_scope_file_sha256", config["candidate_scope_path"]),
                        ("rules_file_sha256", body["strata_rules_path"])):
        if generation.get(key) != hashlib.sha256(Path(source).read_bytes()).hexdigest():
            raise ValueError("prospective failure scope/rules bytes changed")
    files = generation.get("artifacts", {})
    expected = {"attempt.json", "baseline.jsonl", "lens.jsonl", "descriptive_baselines.json", "selection.json"}
    if set(files) != expected:
        raise ValueError("prospective failure baseline artifact scope incomplete")
    for name, digest in files.items():
        if hashlib.sha256((path.parent / name).read_bytes()).hexdigest() != digest:
            raise ValueError("prospective failure baseline artifact changed")
    selection = document(path.parent / "selection.json")
    strata = selection.get("strata", {})
    if (selection.get("phase") != config["phase"] or selection.get("status") != "ready"
            or set(strata) != set(body["allowed_strata"])
            or any(entry.get("status") != "ready" or entry.get("reliability_gate_passed") is not True
                   or entry.get("sample_gate_passed") is not True for entry in strata.values())):
        raise ValueError("insufficient prospective failure strata cannot execute causal interventions")


def match_failure_donors(session: Any, units: list[dict], body: Mapping[str, Any], phase: str) -> list[dict]:
    """Return one attempt per requested class, including no-valid-donor records."""
    validate_case_scope(session, units, body, phase)
    attempts, reuse = [], Counter()
    for unit in units:
        if unit["stratum"] not in body["allowed_strata"] or unit["role"] not in ("failure", "matched_correct"):
            raise ValueError("unit outside frozen strata/roles")
        recipient = session.variants[unit["recipient_variant_id"]]
        actual = session.clean(recipient["variant_id"])
        expected_correct = unit["role"] == "matched_correct"
        if actual.correct != expected_correct:
            attempts.append({**unit, "status": "baseline_stratum_changed", "baseline_correct": actual.correct})
            continue
        explicit = "donor_variant_id" in unit
        if explicit and unit.get("donor_class") not in body["donor_classes"]:
            raise ValueError("explicit donor_class outside frozen classes")
        classes = [unit["donor_class"]] if explicit else body["donor_classes"]
        for donor_class in classes:
            if explicit and unit["donor_variant_id"] is None:
                attempts.append({**unit, "status": "no_valid_donor", "reason": "explicit_no_donor"})
                continue
            pool = []
            for donor in session.variants.values():
                if donor["split"] != recipient["split"] or donor["variant_id"] == recipient["variant_id"]:
                    continue
                same_item = donor["item_id"] == recipient["item_id"]
                if donor_class == "same_image_remap":
                    eligible = same_item and donor["image"] == recipient["image"] and (
                        donor["correct_symbol"] != recipient["correct_symbol"] or donor["labels"] != recipient["labels"])
                elif donor_class == "semantic_match":
                    eligible = (not same_item and donor["image"] != recipient["image"]
                        and donor["semantic_answer"] == recipient["semantic_answer"]
                        and donor["question_family"] == recipient["question_family"]
                        and donor["labels"] == recipient["labels"])
                else:
                    eligible = (donor.get("counterfactual_pair_id") != recipient.get("counterfactual_pair_id")
                        and donor["scene_id"] != recipient["scene_id"] and donor["correct_symbol"] == recipient["correct_symbol"]
                        and donor["labels"] == recipient["labels"])
                if eligible and reuse[donor["variant_id"]] < body["max_donor_reuse"]:
                    primary = body.get("primary_conditions_by_donor_class", {}).get(donor_class, [])
                    needed = ([recipient["semantic_answer"]] if body.get("option_scope", "truth") == "truth"
                              else recipient["option_contents"])
                    if "options_repair" in primary and not set(needed).issubset(donor.get("option_contents", [])):
                        continue
                    if "image_repair" in primary and donor.get("prompt") != recipient.get("prompt"):
                        continue
                    pool.append(donor)
            pool.sort(key=lambda d: sha([body["seed"], unit["case_id"], donor_class, d["variant_id"]]))
            primary = body.get("primary_conditions_by_donor_class", {}).get(donor_class, [])
            aligned = []
            for candidate in pool:
                try:
                    if "image_repair" in primary:
                        a, b = session.prepared(recipient["variant_id"]), session.prepared(candidate["variant_id"])
                        if a.input_ids != b.input_ids or a.groups["image"] != b.groups["image"]:
                            continue
                    if "options_repair" in primary:
                        a, b = session.prepared(recipient["variant_id"]), session.prepared(candidate["variant_id"])
                        target = [recipient["semantic_answer"]] if body["option_scope"] == "truth" else recipient["option_contents"]
                        apos, bpos = _option_alignment(session, a, recipient, target), _option_alignment(session, b, candidate, target)
                        if [a.input_ids[p] for p in apos] != [b.input_ids[p] for p in bpos]:
                            continue
                    aligned.append(candidate)
                except (KeyError, ValueError):
                    continue
            pool = aligned
            if explicit:
                donor = next((d for d in pool if d["variant_id"] == unit["donor_variant_id"]), None)
                if donor is None or session.clean(donor["variant_id"]).correct != (donor_class != "wrong_donor"):
                    raise ValueError("explicit donor violates frozen class/correctness/reuse; no substitution")
            else:
                donor = next((d for d in pool if session.clean(d["variant_id"]).correct == (donor_class != "wrong_donor")), None)
            row = {**unit, "case_id": unit["case_id"] if explicit else unit["case_id"] + ":" + donor_class,
                   "source_case_id": unit["case_id"], "donor_class": donor_class}
            if donor is None:
                attempts.append({**row, "donor_variant_id": None, "status": "no_valid_donor"})
            else:
                reuse[donor["variant_id"]] += 1
                attempts.append({**row, "donor_variant_id": donor["variant_id"], "status": "matched"})
    return attempts


def _option_alignment(session: Any, prepared: Any, variant: Mapping[str, Any], target_contents: list[str]) -> list[int]:
    """Find content spans inside their declared option lines, not arbitrary substrings."""
    from ..core7.adapters import locate
    tokenizer = session.adapter.tokenizer
    cursor = max(prepared.groups["image"], default=-1) + 1
    spans = {}
    for label, content in zip(variant["labels"], variant["option_contents"], strict=True):
        line = locate(tokenizer, prepared.input_ids, f"{label}. {content}", cursor)
        if not line:
            raise ValueError("unresolved option line")
        local = locate(tokenizer, [prepared.input_ids[p] for p in line], content, 0)
        if not local:
            raise ValueError("unresolved option content")
        spans[content] = [line[p] for p in local]
        cursor = line[-1] + 1
    return [p for content in target_contents for p in spans[content]]


def evaluate_failure_case(session: Any, case: Mapping[str, Any], body: Mapping[str, Any], frozen: Mapping[str, Any],
                          context: Mapping[str, Any], *, self_donor: bool = False, batch_size: int = 1) -> dict:
    core_guard(session, context)
    from ..core7.hooks import Site, Intervention
    from ..core7.heads import matched_random
    from ..core7.scoring import symbol_readout
    import random
    recipient = session.variants[case["recipient_variant_id"]]
    donor = recipient if self_donor else session.variants[case["donor_variant_id"]]
    if recipient["split"] != donor["split"]:
        raise ValueError("failure donor crosses split")
    prep, donor_prep = session.prepared(recipient["variant_id"]), session.prepared(donor["variant_id"])
    clean, donor_clean = session.clean(recipient["variant_id"]), session.clean(donor["variant_id"])
    labels = session.label_ids(prep, recipient["labels"])
    truth = recipient["correct_index"]
    if recipient["option_contents"][truth] != recipient["semantic_answer"]:
        raise ValueError("recipient truth mapping invalid")
    baseline = symbol_readout(clean.logits, labels, truth)
    specs = [("clean", [], "recipient baseline")]
    skipped = []
    groups = ("image", "options", "final") if body["image_repair_enabled"] else ("options", "final")
    for group in groups:
        boundary = body["sites"][group]
        site = Site(boundary["site"], boundary["layer"])
        try:
            if group == "options":
                target = [recipient["semantic_answer"]] if body["option_scope"] == "truth" else recipient["option_contents"]
                recipient_pos = _option_alignment(session, prep, recipient, target)
                donor_pos = _option_alignment(session, donor_prep, donor, target)
                if [prep.input_ids[p] for p in recipient_pos] != [donor_prep.input_ids[p] for p in donor_pos]:
                    raise ValueError("option content token IDs mismatch")
            else:
                recipient_pos, donor_pos = prep.groups[group], donor_prep.groups[group]
            if not recipient_pos or len(recipient_pos) != len(donor_pos):
                raise ValueError("different/empty token groups")
            if group == "image" and (prep.input_ids != donor_prep.input_ids or recipient_pos != donor_pos):
                raise ValueError("image transfer requires exact nonvisual prompt/token alignment")
            _, donor_capture = forward(session, donor_prep, capture=[site], capture_positions=donor_pos)
            _, self_capture = forward(session, prep, capture=[site], capture_positions=recipient_pos)
            specs.extend([(group + "_repair", [Intervention(site, recipient_pos, donor_capture[site.key][0])],
                           "donor " + group + " with recipient truth readout"),
                          (group + "_self", [Intervention(site, recipient_pos, self_capture[site.key][0])], "self " + group)])
            if group == "options":
                for layer in body.get("option_candidate_layers", []):
                    candidate_site = Site(boundary["site"], layer)
                    _, candidate_donor = forward(session, donor_prep, capture=[candidate_site], capture_positions=donor_pos)
                    _, candidate_self = forward(session, prep, capture=[candidate_site], capture_positions=recipient_pos)
                    specs.extend([
                        (f"options_candidate_L{layer}",
                         [Intervention(candidate_site, recipient_pos, candidate_donor[candidate_site.key][0])],
                         "discovery candidate option repair"),
                        (f"options_candidate_self_L{layer}",
                         [Intervention(candidate_site, recipient_pos, candidate_self[candidate_site.key][0])],
                         "discovery candidate option self identity"),
                    ])
        except (KeyError, ValueError) as error:
            if "non-finite" in str(error):
                raise
            skipped.append({"condition": group + "_repair", "reason": str(error)})
    heads = [tuple(h) for h in frozen["head_candidates"][:body["head_k"]]]

    def head_items(selected, source):
        per_layer = defaultdict(list)
        for layer, head in selected:
            per_layer[layer].append(head)
        return [Intervention(Site("head", layer), [prep.final_position], source.sites[f"head:{layer}"], heads=hs)
                for layer, hs in sorted(per_layer.items())]

    specs.extend([("head_repair", head_items(heads, donor_clean), "frozen donor heads; remapping readout"),
                  ("head_self", head_items(heads, clean), "baseline heads only")])
    rng = random.Random(body["seed"])
    for draw in range(body["random_sets"]):
        chosen = matched_random(heads, frozen["head_layers"], session.adapter.handle.num_heads, rng)
        specs.append((f"random_head/{draw}", head_items(chosen, donor_clean), "matched random donor heads"))
    from .trajectory import _BatchSession
    rows = []
    for chunk in _BatchSession(session, batch_size).batches(specs):
        import copy
        interventions = []
        for i, (_, items, _) in enumerate(chunk):
            for item in items:
                placed = copy.copy(item)
                placed.rows = [i]
                interventions.append(placed)
        logits, _ = forward(session, prep, batch=len(chunk), interventions=interventions)
        for (name, _, lineage), result in zip(chunk, logits):
            readout = symbol_readout(result, labels, truth)
            predicted = readout["predicted_index"]
            predicted_content = recipient["option_contents"][predicted]
            donor_truth_symbol = donor["correct_symbol"]
            donor_symbol_id = donor_prep.label_ids.get(donor_truth_symbol)
            donor_emitted_index = donor_clean.readout["predicted_index"]
            donor_emitted_symbol = donor["labels"][donor_emitted_index]
            donor_emitted_content = donor["option_contents"][donor_emitted_index]
            donor_emitted_id = donor_prep.label_ids.get(donor_emitted_symbol)
            rows.append({"case_id": case["case_id"], "condition": name, "stratum": case["stratum"], "role": case["role"],
                "matched_failure_case_id": case.get("matched_failure_case_id"),
                "donor_class": case["donor_class"], "recipient_variant_id": recipient["variant_id"],
                "donor_variant_id": donor["variant_id"], "sample_unit": recipient["scene_id"],
                "donor_scene_id": donor["scene_id"], "recipient_pair_id": recipient.get("counterfactual_pair_id"),
                "split": recipient["split"], "lineage": lineage, "baseline_correct": clean.correct,
                "donor_correct": donor_clean.correct, "target_content": recipient["semantic_answer"],
                "predicted_content": predicted_content, "recipient_truth_symbol": recipient["correct_symbol"],
                "donor_truth_symbol": donor_truth_symbol, "donor_truth_content": donor["semantic_answer"],
                "donor_restricted_emitted_symbol": donor_emitted_symbol,
                "donor_restricted_emitted_content": donor_emitted_content,
                "raw_margin_change": readout["margin"] - baseline["margin"], "correct_label_margin": readout["margin"],
                "corrected": (not clean.correct) and readout["correct"], "harmed": clean.correct and not readout["correct"],
                "semantic_remapping_success": predicted_content == donor["semantic_answer"],
                "donor_truth_remap_success": predicted_content == donor["semantic_answer"],
                "donor_emitted_content_remap_success": (predicted_content == donor_emitted_content)
                    if donor_emitted_content in recipient["option_contents"] else None,
                "donor_symbol_copy": recipient["labels"][predicted] == donor_truth_symbol,
                "donor_truth_symbol_copy": recipient["labels"][predicted] == donor_truth_symbol,
                "donor_emitted_symbol_copy": recipient["labels"][predicted] == donor_emitted_symbol,
                "full_vocab_donor_symbol_copy": donor_symbol_id is not None and int(result.argmax()) == donor_symbol_id,
                "full_vocab_donor_emitted_symbol_copy": donor_emitted_id is not None and int(result.argmax()) == donor_emitted_id,
                "full_vocab_donor_argmax_copy": int(result.argmax()) == int(donor_clean.logits.argmax()),
                "readout": readout, "normalized_recovery": None})
    return {"case_id": case["case_id"], "status": "evaluated", "rows": rows, "condition_exclusions": skipped}


def calibrate_study(session: Any, config: Mapping[str, Any], context: Mapping[str, Any]) -> dict:
    require_execution(context)
    core_guard(session, context)
    body, frozen = _body(session, config, context)
    from ..core7.hooks import Site
    site_groups = ("image", "options", "final") if body["image_repair_enabled"] else ("options", "final")
    sites = [Site(body["sites"][group]["site"], body["sites"][group]["layer"]) for group in site_groups]
    supplied = document(config["calibration_case"])
    if any(session.variants[supplied[key]]["split"] == "confirmation"
           for key in ("recipient_variant_id", "donor_variant_id")):
        raise ValueError("failure calibration cannot inspect confirmation")
    if session.variants[supplied["recipient_variant_id"]]["split"] != session.variants[supplied["donor_variant_id"]]["split"]:
        raise ValueError("failure calibration donor crosses split")
    if any(session.variants[vid]["split"] == "confirmation" for vid in config["calibration_variant_ids"]):
        raise ValueError("failure primitive calibration cannot inspect confirmation")
    if (supplied.get("role") != "failure" or supplied.get("stratum") not in body["allowed_strata"]
            or supplied.get("donor_class") not in body["donor_classes"]):
        raise ValueError("failure calibration requires a declared frozen-stratum failure case")
    if session.clean(supplied["recipient_variant_id"]).correct:
        raise ValueError("wrong-answer calibration requires an actually incorrect recipient")
    checks = primitive_checks(session, config["calibration_variant_ids"], sites, config["tolerance"])
    sham = evaluate_failure_case(session, supplied, body, frozen, context, self_donor=True, batch_size=1)
    first = evaluate_failure_case(session, supplied, body, frozen, context, batch_size=1)
    batch = evaluate_failure_case(session, supplied, body, frozen, context, batch_size=2)
    checks["wrong_answer_self_identity"] = measured_check(max(abs(r["raw_margin_change"]) for r in sham["rows"]), config["tolerance"])
    paired = {r["condition"]: r for r in batch["rows"]}
    checks["actual_failure_batch_single"] = measured_check(max(abs(r["correct_label_margin"] - paired[r["condition"]]["correct_label_margin"]) for r in first["rows"]), config["tolerance"])
    return calibration_report(session, context, "failures", body, checks, config["calibration_variant_ids"])


def primary_denominators(attempts: list[dict], rows: list[dict], variants: Mapping[str, Any], body: Mapping[str, Any]) -> list[dict]:
    """Explicit no-observed-success denominator, distinct from causal effect data."""
    evaluated = {(r["case_id"], r["condition"]): r for r in rows}
    output = []
    for attempt in attempts:
        classes = [attempt["donor_class"]] if attempt.get("donor_class") else body["donor_classes"]
        for donor_class in classes:
            for condition in body["primary_conditions_by_donor_class"][donor_class]:
                row = evaluated.get((attempt["case_id"], condition))
                output.append({"case_id": attempt["case_id"], "stratum": attempt["stratum"],
                    "role": attempt["role"], "donor_class": donor_class, "repair_condition": condition,
                    "condition": "/".join((attempt["stratum"], attempt["role"], donor_class, condition)),
                    "sample_unit": variants[attempt["recipient_variant_id"]]["scene_id"],
                    "evaluable": row is not None, "correction_success": bool(row and row["corrected"]),
                    "harm": bool(row and row["harmed"]),
                    "missing_reason": None if row else attempt.get("status", "condition_excluded"),
                    "raw_margin_change": float(bool(row and row["corrected"]))})
    return output


def run_study(session: Any, config: Mapping[str, Any], context: Mapping[str, Any]) -> dict:
    require_execution(context)
    body, frozen = _body(session, config, context)
    require_calibration(session, context, "failures", body)
    validate_prepared_cases(config, context, body)
    units, exclusions = candidate_units(config, body)
    validate_case_scope(session, units, body, config["phase"])
    writer = StudyWriter(session, config, context, "failures", body)
    try:
        baselines = []
        for unit in units:
            variant = session.variants[unit["recipient_variant_id"]]
            baselines.append({**unit, "sample_unit": variant["scene_id"],
                              "readout": session.clean(variant["variant_id"]).readout})
        writer.write("recipient_baselines.json", baselines)
        attempts = match_failure_donors(session, units, body, config["phase"])
        writer.write("donor_matches.json", attempts)
        for exclusion in exclusions:
            writer.append(exclusion, excluded=True)
        for case in attempts:
            if case["status"] != "matched":
                writer.append(case, excluded=True)
                continue
            result = evaluate_failure_case(session, case, body, frozen, context, batch_size=config["batch_size"])
            for row in result["rows"]:
                writer.append(row)
            for exclusion in result["condition_exclusions"]:
                writer.append({"case_id": case["case_id"], **exclusion}, excluded=True)
        rates = [{**r, "condition": "/".join((r["stratum"], r["role"], r["donor_class"], r["condition"]))} for r in writer.rows]
        summary = grouped_summary(rates, seed=body["bootstrap_seed"], samples=body["bootstrap_samples"])
        denominator_rows = primary_denominators(attempts, writer.rows, session.variants, body)
        writer.write("primary_attempt_denominators.json", denominator_rows)
        correction = grouped_summary(denominator_rows, seed=body["bootstrap_seed"], samples=body["bootstrap_samples"])
        evaluable_correction = grouped_summary([r for r in denominator_rows if r["evaluable"]],
            seed=body["bootstrap_seed"], samples=body["bootstrap_samples"])
        endpoint_coverage = {}
        adequate = bool(denominator_rows)
        for stratum in body["allowed_strata"]:
            for donor_class, conditions in body["primary_conditions_by_donor_class"].items():
                for condition in conditions:
                    key = "/".join((stratum, donor_class, condition))
                    counts = {}
                    for role in ("failure", "matched_correct"):
                        scope = [r for r in denominator_rows if r["stratum"] == stratum and r["donor_class"] == donor_class
                                 and r["repair_condition"] == condition and r["role"] == role]
                        counts[role] = {"requested_scenes": len({r["sample_unit"] for r in scope}),
                                        "evaluable_scenes": len({r["sample_unit"] for r in scope if r["evaluable"]}),
                                        "requested_cases": len(scope), "evaluable_cases": sum(r["evaluable"] for r in scope)}
                        adequate = adequate and counts[role]["evaluable_scenes"] >= body["min_stratum_scenes"]
                    endpoint_coverage[key] = counts
        coverage = {"recipient_units_requested": len(units), "donor_attempts": len(attempts), "raw_rows": len(writer.rows),
                    "exclusions": len(writer.exclusions), "primary_endpoints": endpoint_coverage,
                    "donor_reuse": dict(Counter(c.get("donor_variant_id") for c in attempts if c.get("donor_variant_id")))}
        selection = None
        freeze_body = None
        if config["phase"] == "discovery" and adequate:
            from .discovery_selection import select_failure_option_layer
            selection = select_failure_option_layer(writer.rows,
                candidate_layers=body["option_candidate_layers"], seed=body["seed"],
                draws=body["bootstrap_samples"], minimum_units=body["min_stratum_scenes"],
                tie_tolerance=body["layer_tie_tolerance"], identity_tolerance=config["tolerance"])
            adequate = selection["status"] == "ready"
            if adequate:
                freeze_body = dict(body)
                freeze_body["sites"] = {key: dict(value) for key, value in body["sites"].items()}
                freeze_body["sites"]["options"]["layer"] = selection["selected_layer"]
                freeze_body["selected_option_layer"] = selection["selected_layer"]
                freeze_body["option_candidate_layers"] = []
        return writer.finish(coverage, "rescue_and_remapping_evidence; not_ground_truth_failure_diagnosis" if adequate else "insufficient_reliable_failure_strata",
            {"margin_effects_evaluable_only": summary, "correction_rates_all_requested_attempts": correction,
             "correction_rates_evaluable_only": evaluable_correction,
             "option_layer_selection": selection,
             "denominator_warning": "no donor/repair is counted as no observed correction, not zero causal effect; both denominators reported",
             "warning": "first-token content proxies and donor steering do not establish grounding/binding labels"},
            status="complete" if adequate else "insufficient", freeze_body=freeze_body)
    except BaseException as error:
        writer.failure(error)
        raise
