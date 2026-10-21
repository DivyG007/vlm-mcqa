"""Frozen descriptive error-stratum rules and independent-scene matching.

Existing behavior records support internal validation, not a pristine causal
holdout. This module never upgrades vocabulary proxies into causal diagnoses.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Mapping

from .statistics import cluster_estimate


FLAGS = {"truth_aligned_content_with_wrong_final_symbol",
         "wrong_content_proxy_agrees_with_final_choice",
         "global_argmax_outside_requested_labels",
         "semantic_choice_varies_across_observed_mappings"}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def freeze_failure_rules(protocol: Mapping[str, Any]) -> dict[str, Any]:
    required = {"stage", "strata_flags", "min_scenes", "reliability_min", "seed",
                "bootstrap_draws", "margin_bins", "match_fields", "max_control_reuse"}
    if not required.issubset(protocol):
        raise ValueError(f"failure rules missing {sorted(required - protocol.keys())}")
    for key in ("stage", "seed", "min_scenes", "bootstrap_draws", "max_control_reuse"):
        if type(protocol[key]) is not int:
            raise ValueError(f"integer {key} required")
    if protocol["stage"] < 0 or min(protocol["min_scenes"], protocol["max_control_reuse"]) < 1 or protocol["bootstrap_draws"] < 100:
        raise ValueError("invalid failure-rule counts")
    flags = protocol["strata_flags"]
    if not isinstance(flags, list) or not 1 <= len(flags) <= 2 or len(set(flags)) != len(flags) or not set(flags) <= FLAGS:
        raise ValueError("one or two distinct prespecified descriptive strata required")
    threshold = protocol["reliability_min"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("reliability_min must be in [0,1]")
    bins = protocol["margin_bins"]
    if not isinstance(bins, list) or not bins or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bins) or sorted(set(bins)) != bins:
        raise ValueError("ordered distinct finite margin bins required")
    if not protocol["match_fields"] or not set(protocol["match_fields"]) <= {"question_family", "reasoning_depth", "scheme"}:
        raise ValueError("explicit supported matching fields required")
    frozen = {"schema": "stretch.failure-rules.v1", **dict(protocol),
              "claim_scope": "descriptive_internal_validation_on_existing_records"}
    frozen["rules_sha256"] = _hash(frozen)
    return frozen


def _validate_rules(rules: Mapping[str, Any]) -> None:
    body = {k: v for k, v in rules.items() if k != "rules_sha256"}
    if rules.get("rules_sha256") != _hash(body):
        raise ValueError("failure rules changed after freeze")
    rebuilt = freeze_failure_rules({k: v for k, v in body.items() if k not in {"schema", "claim_scope"}})
    if rebuilt != dict(rules):
        raise ValueError("invalid frozen failure rule schema")


def evaluate_failure_rules(report: Mapping[str, Any], dataset: Mapping[str, Any],
                           rules: Mapping[str, Any]) -> dict[str, Any]:
    _validate_rules(rules)
    if report.get("status") != "descriptive_not_causal":
        raise ValueError("audited descriptive failure report required")
    items = {item["item_id"]: item for item in dataset["items"]}
    if len(items) != len(dataset["items"]):
        raise ValueError("duplicate item identities")
    variants = {variant["variant_id"]: variant for variant in dataset["variants"]}
    if len(variants) != len(dataset["variants"]):
        raise ValueError("duplicate variant identities")
    rows, seen = [], set()
    for original in report["rows"]:
        if original["stage"] != rules["stage"]:
            continue
        row = dict(original)
        vid = row["variant_id"]
        if vid in seen:
            raise ValueError("duplicate prospective stage row")
        seen.add(vid)
        v = variants.get(vid, {})
        item = items.get(v.get("item_id"), {})
        for key in ("question_family", "reasoning_depth"):
            row[key] = row.get(key) if row.get(key) is not None else item.get(key)
        if row.get("reasoning_depth") is None:
            row["reasoning_depth"] = item.get("program_depth")
        scene = row.get("scene_id")
        # Stable scene partition is independent of correctness/flags/margins.
        row["analysis_phase"] = ("discovery" if int(_hash([rules["seed"], scene]), 16) % 2 == 0
                                 else "internal_validation") if scene else "unresolved"
        rows.append(row)
    output = {}
    for phase in ("discovery", "internal_validation"):
        selected = [row for row in rows if row["analysis_phase"] == phase]
        correct = [r for r in selected if r.get("final_correct") is True and not r.get("unresolved")]
        reliability = cluster_estimate(correct, value="content_proxy_correct", cluster="scene_id",
                                       seed=rules["seed"], draws=rules["bootstrap_draws"])
        reliable = (reliability["n_clusters"] >= rules["min_scenes"]
                    and reliability["ci95"] is not None
                    and reliability["ci95"][0] >= rules["reliability_min"])
        summaries, control_use = {}, Counter()
        for flag in rules["strata_flags"]:
            failures = [r for r in selected if r.get("final_correct") is False
                        and flag in r.get("descriptive_flags", []) and not r.get("unresolved")]
            eligible_scenes = sorted({r["scene_id"] for r in failures if r.get("scene_id")})
            matches, unmatched = [], []
            for failure in sorted(failures, key=lambda r: _hash([rules["seed"], r["variant_id"]])):
                margin = failure.get("confidence_gap")
                bucket = sum(margin >= edge for edge in rules["margin_bins"]) if margin is not None else None
                candidates = []
                for candidate in correct:
                    if candidate["scene_id"] == failure["scene_id"] or control_use[candidate["scene_id"]] >= rules["max_control_reuse"]:
                        continue
                    if any(failure.get(key) is None or failure.get(key) != candidate.get(key) for key in rules["match_fields"]):
                        continue
                    cm = candidate.get("confidence_gap")
                    if bucket is None or cm is None or sum(cm >= edge for edge in rules["margin_bins"]) != bucket:
                        continue
                    candidates.append(candidate)
                if candidates:
                    chosen = min(candidates, key=lambda r: _hash([rules["seed"], failure["variant_id"], r["variant_id"]]))
                    control_use[chosen["scene_id"]] += 1
                    matches.append({"failure_variant_id": failure["variant_id"], "failure_scene_id": failure["scene_id"],
                                    "control_variant_id": chosen["variant_id"], "control_scene_id": chosen["scene_id"]})
                else:
                    unmatched.append(failure["variant_id"])
            enough = len(eligible_scenes) >= rules["min_scenes"]
            matched_scenes = len({r["failure_scene_id"] for r in matches})
            summaries[flag] = {"status": "candidate_for_causal_design" if enough and reliable and matched_scenes >= rules["min_scenes"] else "descriptive_only",
                               "reliability_gate_passed": reliable, "sample_gate_passed": enough,
                               "matched_control_gate_passed": matched_scenes >= rules["min_scenes"],
                               "variant_ids": [r["variant_id"] for r in failures],
                               "scene_ids": eligible_scenes, "matched_correct_controls": matches,
                               "unmatched_failure_variants": unmatched,
                               "n_scene_matched": matched_scenes,
                               "causal_confirmation_eligibility": "not_established_by_existing_record_partition"}
        output[phase] = {"content_proxy_reliability": reliability, "strata": summaries,
                         "known_scenes": len({r["scene_id"] for r in selected if r.get("scene_id")}),
                         "control_scene_reuse": dict(control_use),
                         "metrics": {metric: cluster_estimate(selected, value=metric, cluster="scene_id",
                                                                  seed=rules["seed"], draws=rules["bootstrap_draws"])
                                     for metric in ("final_correct", "content_proxy_correct", "unresolved")}}
    result = {"schema": "stretch.failure-strata.v1", "status": "descriptive_internal_validation",
              "rules": dict(rules), "phases": output,
              "input_report_sha256": _hash(report), "rows": rows,
              "confidence_matching": "predicted_top_label_minus_runner_up; same nonnegative scale for correct and wrong cases",
              "coverage": {"selected_stage_rows": len(rows), "unresolved_rows": sum(r.get("unresolved", True) for r in rows)},
              "limitations": ["Existing records were previously inspected; the internal split is not a new causal holdout.",
                              "Descriptive flags overlap and do not identify grounding or binding causes.",
                              "Correct controls are matched by prespecified metadata and confidence bins, not causal effects."]}
    result["manifest_sha256"] = _hash(result)
    return result
