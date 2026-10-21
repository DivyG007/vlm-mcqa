"""Outcome-independent candidate ordering and clean-baseline eligibility only.

Pure source implementation; no model, filesystem, CLI or network operations.
The resulting plan is neither a protocol freeze nor a confirmation result.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from .data import validate_bidirectional, validate_fixed_text_contrast


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _order(values: Iterable[str], seed: str | int, namespace: str) -> list[str]:
    return sorted(values, key=lambda value: (_hash([seed, namespace, value]), value))


def _baseline_state(row: Mapping[str, Any], variant: Mapping[str, Any], identity: Mapping[str, Any]) -> str:
    if (row.get("item_id") != variant["item_id"]
            or type(row.get("position")) is not int or row["position"] != variant["correct_index"]
            or row.get("scheme") != variant["label_scheme"]):
        return "metadata_mismatch"
    for field, expected in (("split", variant["split"]), ("labels", variant["labels"]),
                            ("scene_id", variant["scene_id"]),
                            ("correct_symbol", variant["correct_symbol"]),
                            ("model_key", identity["model_key"]),
                            ("dataset_hash", identity["dataset_hash"])):
        if field in row and row[field] != expected:
            return "metadata_mismatch"
    logits = row.get("label_logits")
    if (not isinstance(logits, (list, tuple)) or len(logits) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits)):
        return "unknown"
    winners = [i for i, value in enumerate(logits) if value == max(logits)]
    if len(winners) != 1 or type(row.get("correct")) is not bool or type(row.get("predicted_index")) is not int:
        return "unknown"
    correct = winners[0] == variant["correct_index"]
    if row["predicted_index"] != winners[0] or row["correct"] != correct:
        return "metadata_mismatch"
    return "correct" if correct else "incorrect"


def select_fixed_text_sample(contrasts: Iterable[Mapping[str, Any]],
                             variants: Mapping[str, Mapping[str, Any]],
                             baseline_rows: Iterable[Mapping[str, Any]], *,
                             split: str, target_pairs: int, layouts_per_pair: int,
                             min_pairs: int, seed: str | int,
                             excluded_pair_ids: Iterable[str],
                             baseline_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Select exact-size bidirectional layouts in prespecified hashed order.

    Reject malformed contrast structure; exclude missing/duplicate/unknown or
    inconsistent baseline endpoints. ``correct`` alone never establishes
    eligibility: the restricted four-label logits must corroborate it.
    Baseline/source identity is caller-attested, not freshly hash-verified.
    """
    if split not in {"discovery", "confirmation"}:
        raise ValueError("explicit discovery or confirmation split required")
    if any(type(v) is not int or v <= 0 for v in (target_pairs, layouts_per_pair, min_pairs)):
        raise ValueError("positive integer sample counts required")
    if min_pairs > target_pairs:
        raise ValueError("minimum cannot exceed target")
    if type(seed) not in (str, int) or seed == "":
        raise ValueError("explicit string or integer seed required")
    identity = dict(baseline_identity)
    for key in ("model_key", "dataset_hash", "prediction_source_sha256"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise ValueError(f"explicit baseline identity missing {key}")
    sha = identity["prediction_source_sha256"]
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("prediction_source_sha256 must be full lowercase SHA256")
    excluded = set(excluded_pair_ids)
    if any(not isinstance(p, str) or not p for p in excluded):
        raise ValueError("excluded pair ids must be nonempty strings")
    splits: dict[tuple[str, str], str] = {}
    for key, variant in variants.items():
        if key != variant.get("variant_id"):
            raise ValueError("variant index/key mismatch")
        if variant.get("split") not in {"calibration", "screening", "behavior", "discovery", "confirmation"}:
            raise ValueError("missing or unknown variant split")
        for field in ("model_key", "dataset_hash"):
            if field in variant and variant[field] != identity[field]:
                raise ValueError("variant/baseline identity mismatch")
        for field in ("scene_id", "counterfactual_pair_id"):
            value = variant.get(field)
            if field == "scene_id" and (not isinstance(value, str) or not value):
                raise ValueError("scene id required for split audit")
            if value:
                unit = field, value
                if unit in splits and splits[unit] != variant["split"]:
                    raise ValueError("scene or semantic pair crosses splits")
                splits[unit] = variant["split"]
    rows = list(contrasts)
    validate_bidirectional(rows)
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        validate_fixed_text_contrast(row, variants)
        donor, recipient = (variants[row[k]] for k in ("donor_variant_id", "recipient_variant_id"))
        if row.get("layout_id") != row["prompt_sha256"]:
            raise ValueError("layout identity must be exact prompt hash")
        for side, v in (("donor", donor), ("recipient", recipient)):
            if row.get(side + "_scene_id") != v["scene_id"]:
                raise ValueError("contrast scene metadata mismatch")
            if type(v["correct_index"]) is not int:
                raise ValueError("correct index must be integer, not bool")
        if row["split"] == split:
            grouped[row["pair_id"]][row["layout_id"]].append(row)
    for layouts in grouped.values():
        for layout_rows in layouts.values():
            edges = {(r["donor_variant_id"], r["recipient_variant_id"]) for r in layout_rows}
            if len(layout_rows) != 2 or len(edges) != 2 or len(set().union(*(set(e) for e in edges))) != 2:
                raise ValueError("each layout requires exactly one bidirectional endpoint pair")
    # These complete candidate orders are defined before inspecting predictions.
    pair_order = _order(grouped, seed, "pairs")
    layout_orders = {p: _order(grouped[p], seed, "layouts:" + p) for p in pair_order}
    baselines: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    input_counts = Counter()
    for row in baseline_rows:
        input_counts["rows"] += 1
        if row.get("condition") != "clean":
            input_counts["nonclean_rows_ignored"] += 1
            continue
        variant_id = row.get("variant_id")
        if not isinstance(variant_id, str) or not variant_id:
            input_counts["invalid_variant_rows"] += 1
            continue
        baselines[variant_id].append(row)
    endpoint_ids = {r[k] for p in grouped.values() for rs in p.values() for r in rs
                    for k in ("donor_variant_id", "recipient_variant_id")}
    endpoint_states = {}
    for vid in sorted(endpoint_ids):
        matches = baselines.get(vid, [])
        endpoint_states[vid] = ("missing" if not matches else "duplicate" if len(matches) != 1
                                else _baseline_state(matches[0], variants[vid], identity))
    counts = Counter(endpoint_states.values())
    layout_exclusions: dict[str, dict[str, list[str]]] = {}
    eligible_by_pair = {}
    pair_reasons = {}
    for pair in pair_order:
        eligible = []
        layout_exclusions[pair] = {}
        if pair in excluded:
            pair_reasons[pair] = "caller_excluded"
        for layout in layout_orders[pair]:
            endpoints = {r[k] for r in grouped[pair][layout]
                         for k in ("donor_variant_id", "recipient_variant_id")}
            reasons = sorted({endpoint_states[v] for v in endpoints if endpoint_states[v] != "correct"})
            if reasons:
                layout_exclusions[pair][layout] = reasons
            else:
                eligible.append(layout)
        if len(eligible) < layouts_per_pair and pair not in pair_reasons:
            pair_reasons[pair] = "layout_shortfall"
        eligible_by_pair[pair] = eligible
    eligible_pairs = [p for p in pair_order if p not in pair_reasons]
    selected = eligible_pairs[:target_pairs]
    selected_layouts = {p: eligible_by_pair[p][:layouts_per_pair] for p in selected}
    contrast_ids = [r["contrast_id"] for p in selected for layout in selected_layouts[p]
                    for r in sorted(grouped[p][layout], key=lambda r: r["contrast_id"])]
    status = ("insufficient" if len(selected) < min_pairs else
              "target_shortfall" if len(selected) < target_pairs else "ready")
    plan = {"schema": "stretch.fixed-text-sample.v1", "split": split, "status": status,
            "target_pairs": target_pairs, "min_pairs": min_pairs,
            "layouts_per_pair": layouts_per_pair, "seed": seed,
            "pair_ids": selected, "contrast_ids": contrast_ids, "selected_layouts": selected_layouts,
            "eligible_layouts_by_pair": eligible_by_pair,
            "candidate_pair_order": pair_order, "candidate_layout_order": layout_orders,
            "baseline_identity": identity, "baseline_identity_verification": "caller_attested_not_freshly_verified",
            "historical_exposure": "not_established_by_excluded_ids_caller_audit_required",
            "protocol_freeze": "separate_required", "confirmation_claim": False,
            "eligibility": "clean_restricted_four_label_correct_both_endpoints",
            "coverage": {"candidate_pairs": len(pair_order), "eligible_pairs": len(eligible_pairs),
                         "selected_pairs": len(selected), "selected_directions": len(contrast_ids),
                         "endpoint_states": {k: counts[k] for k in ("correct", "incorrect", "missing", "unknown", "duplicate", "metadata_mismatch")},
                         "baseline_input": dict(input_counts),
                         "layout_shortfall_pairs": sum(v == "layout_shortfall" for v in pair_reasons.values()),
                         "target_shortfall": max(0, target_pairs - len(selected))},
            "exclusions": {"caller_pair_ids": sorted(excluded), "pair_reasons": pair_reasons,
                           "layout_reasons": layout_exclusions,
                           "unselected_eligible_pairs_after_cap": eligible_pairs[target_pairs:]}}
    plan["plan_sha256"] = _hash(plan)
    return plan
