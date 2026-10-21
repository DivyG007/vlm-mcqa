"""Descriptive clean-behavior/lens joins. No causal grounding/binding diagnoses."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable


def unique_winner(values: Any, *, tolerance: float = 0.0) -> int | None:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tie tolerance must be finite and nonnegative")
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        return None
    winners = [i for i, value in enumerate(values) if max(values) - value <= tolerance]
    return winners[0] if len(winners) == 1 else None


def _valid_mapping(v: dict | None) -> bool:
    return bool(v and isinstance(v.get("option_contents"), list)
                and len(v["option_contents"]) == 4
                and all(isinstance(c, str) for c in v["option_contents"])
                and len(set(v["option_contents"])) == 4
                and type(v.get("correct_index")) is int and 0 <= v["correct_index"] < 4
                and v["option_contents"][v["correct_index"]] == v.get("semantic_answer")
                and isinstance(v.get("labels"), list) and len(v["labels"]) == 4
                and all(isinstance(c, str) for c in v["labels"]) and len(set(v["labels"])) == 4
                and v.get("correct_symbol") == v["labels"][v["correct_index"]]
                and all(v.get(k) is not None for k in ("item_id", "scene_id", "label_scheme", "split")))


def _metadata_matches(record: dict, variant: dict) -> bool:
    return (record.get("item_id") == variant["item_id"]
            and record.get("position") == variant["correct_index"]
            and record.get("scheme") == variant["label_scheme"]
            and ("split" not in record or record["split"] == variant["split"]))


def _index_unique(rows: Iterable[dict[str, Any]], key) -> tuple[dict, dict]:
    indexed, duplicates, invalid, total = {}, Counter(), 0, 0
    for row in rows:
        total += 1
        identity = key(row)
        if identity is None:
            invalid += 1
            continue
        if identity in indexed or identity in duplicates:
            duplicates[identity] += 1
            indexed.pop(identity, None)
        else:
            indexed[identity] = row
    return indexed, {"duplicate_keys_excluded": {str(k): n + 1 for k, n in duplicates.items()},
                     "invalid_key_rows": invalid, "input_rows": total,
                     "unique_resolved_keys": len(indexed)}


def _cluster_rates(rows: list[dict], field: str, metrics: list[str]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            groups[str(row[field])].append(row)
    result = {}
    for metric in metrics:
        values = []
        n = 0
        for group in groups.values():
            eligible = [r[metric] for r in group if isinstance(r.get(metric), bool)]
            if eligible:
                values.append(sum(eligible) / len(eligible))
                n += len(eligible)
        result[metric] = {"equal_cluster_mean": sum(values) / len(values) if values else None,
                          "n_clusters": len(values), "n_rows": n}
    return {"unit": field, "metrics": result, "interval": "not_estimated",
            "rows_without_cluster": sum(r.get(field) is None for r in rows)}


def analyze_failures(dataset: dict[str, Any], behavior_rows: Iterable[dict[str, Any]],
                     lens_rows: Iterable[dict[str, Any]], *, stages: list[int],
                     expected_variant_ids: list[str], tie_tolerance: float = 0.0) -> dict[str, Any]:
    """Require prospective stage and variant scope; never select on error outcomes.

    Outcomes are reconstructed from clean BEHAVIOR label logits, independently
    of lens.correct. Ambiguous/duplicate/missing records remain unresolved.
    """
    if not stages or len(stages) != len(set(stages)) or any(type(s) is not int or s < 0 for s in stages):
        raise ValueError("stages must be distinct nonnegative integers")
    if not expected_variant_ids or len(expected_variant_ids) != len(set(expected_variant_ids)):
        raise ValueError("explicit distinct expected_variant_ids required")
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    variants, variant_audit = _index_unique(dataset["variants"], lambda r: r.get("variant_id"))
    expected = set(expected_variant_ids)
    clean, conditions = [], Counter()
    for row in behavior_rows:
        conditions[str(row.get("condition"))] += 1
        if row.get("condition") == "clean":
            clean.append(row)
    behavior, behavior_audit = _index_unique(clean, lambda r: r.get("variant_id"))
    lens, lens_audit = _index_unique(lens_rows, lambda r: (r["variant_id"], r["stage"])
                                   if r.get("variant_id") and type(r.get("stage")) is int else None)
    rows = []
    discrepancies = Counter()
    semantic_predictions = defaultdict(set)
    for variant_id in sorted(expected):
        v = variants.get(variant_id)
        b = behavior.get(variant_id)
        if v is None:
            continue
        winner = unique_winner(b.get("label_logits"), tolerance=tie_tolerance) if b else None
        if winner is not None and _valid_mapping(v) and b and _metadata_matches(b, v):
            semantic_predictions[v["item_id"]].add(v["option_contents"][winner])
    for variant_id in sorted(expected):
        v = variants.get(variant_id)
        b = behavior.get(variant_id)
        for stage in stages:
            l = lens.get((variant_id, stage))
            reasons = []
            winner = unique_winner(b.get("label_logits"), tolerance=tie_tolerance) if b else None
            content = unique_winner(l.get("content_logits"), tolerance=tie_tolerance) if l else None
            valid_mapping = _valid_mapping(v)
            if not valid_mapping:
                reasons.append("missing_or_invalid_dataset_mapping")
            if b is None:
                reasons.append("missing_or_duplicate_clean_behavior")
            elif winner is None:
                reasons.append("ambiguous_or_invalid_behavior_logits")
            if l is None:
                reasons.append("missing_or_duplicate_lens_stage")
            elif content is None:
                reasons.append("missing_ambiguous_or_invalid_content_logits")
            behavior_mismatch = lens_mismatch = False
            if valid_mapping:
                behavior_mismatch = bool(b and not _metadata_matches(b, v))
                lens_mismatch = bool(l and not _metadata_matches(l, v))
                if behavior_mismatch or lens_mismatch:
                    reasons.append("record_dataset_metadata_mismatch")
            final_correct = winner == v["correct_index"] if valid_mapping and b and winner is not None and not behavior_mismatch else None
            content_correct = content == v["correct_index"] if valid_mapping and content is not None and not lens_mismatch else None
            if b and final_correct is not None and b.get("correct") != final_correct:
                discrepancies["behavior_correct_disagreement"] += 1
                reasons.append("saved_behavior_correct_disagreement")
                final_correct = None
            if b and winner is not None and b.get("predicted_index") != winner:
                discrepancies["behavior_predicted_index_disagreement"] += 1
                reasons.append("saved_behavior_prediction_disagreement")
                final_correct = None
            flags = []
            if final_correct is False and content_correct is True:
                flags.append("truth_aligned_content_with_wrong_final_symbol")
            if final_correct is False and content_correct is not None and winner == content:
                flags.append("wrong_content_proxy_agrees_with_final_choice")
            if b and b.get("global_top_is_label") is False:
                flags.append("global_argmax_outside_requested_labels")
            if valid_mapping and len(semantic_predictions[v["item_id"]]) > 1:
                flags.append("semantic_choice_varies_across_observed_mappings")
            if final_correct is False and not flags:
                reasons.append("wrong_answer_without_specific_descriptive_flag")
            rows.append({"variant_id": variant_id, "stage": stage,
                         "item_id": v.get("item_id") if v else None,
                         "scene_id": v.get("scene_id") if v else None,
                         "pair_id": v.get("counterfactual_pair_id") if v else None,
                         "split": v.get("split") if v else None,
                         "scheme": v.get("label_scheme") if v else None,
                         "question_family": v.get("question_family") if v else None,
                         "reasoning_depth": v.get("reasoning_depth") if v else None,
                         "confidence_gap": (max(b["label_logits"]) - sorted(b["label_logits"])[-2])
                         if final_correct is not None else None,
                         "truth_label_margin": (b["label_logits"][v["correct_index"]] -
                                                max(x for i, x in enumerate(b["label_logits"]) if i != v["correct_index"]))
                         if final_correct is not None else None,
                         "final_correct": final_correct, "content_proxy_correct": content_correct,
                         "final_semantic_choice": v["option_contents"][winner] if final_correct is not None else None,
                         "content_proxy_choice": v["option_contents"][content] if content_correct is not None else None,
                         "label_mass_full": b.get("label_mass_full") if b else None,
                         "global_top_is_label": b.get("global_top_is_label") if b else None,
                         "descriptive_flags": flags, "unresolved_reasons": reasons,
                         "unresolved": bool(reasons)})
    metrics = ["final_correct", "content_proxy_correct", "global_top_is_label", "unresolved"]
    summaries = {}
    for stage in stages:
        selected = [r for r in rows if r["stage"] == stage]
        reliability = [r for r in selected if r["final_correct"] is True]
        summaries[str(stage)] = {"scene_clustered": _cluster_rates(selected, "scene_id", metrics),
                                 "pair_clustered": _cluster_rates(selected, "pair_id", metrics),
                                 "correct_behavior_content_reliability": _cluster_rates(reliability, "scene_id", ["content_proxy_correct"]),
                                 "flag_counts_overlapping": dict(Counter(f for r in selected for f in r["descriptive_flags"]))}
    return {"schema_version": "stretch.failures.v1", "status": "descriptive_not_causal", "rows": rows,
            "summaries_by_prespecified_stage": summaries,
            "coverage": {"expected_variants": len(expected), "expected_stage_rows": len(expected) * len(stages),
                         "missing_dataset_variants": sorted(expected - variants.keys()),
                         "missing_or_duplicate_clean_behavior": sorted(expected - behavior.keys()),
                         "missing_or_duplicate_lens_keys": [[v, s] for v in sorted(expected) for s in stages if (v, s) not in lens],
                         "clean_behavior_outside_scope": sorted(behavior.keys() - expected),
                         "lens_rows_outside_scope": sum(v not in expected or s not in stages for v, s in lens),
                         "behavior_conditions": dict(conditions), "dataset": variant_audit,
                         "behavior": behavior_audit, "lens": lens_audit,
                         "discrepancies_per_stage_row": dict(discrepancies)},
            "limitations": ["Flags overlap; unresolved cases remain in coverage and denominators.",
                            "Clean forced-choice behavior defines final outcome, not lens.correct or generated accuracy.",
                            "Content logits are first-token vocabulary proxies; saved rows do not retain token IDs.",
                            "Inspect correct-case reliability before using error flags; this tool sets no reliability gate.",
                            "No grounding/binding causal diagnosis or automatic failure-stratum selection.",
                            "Saved scalar logits cannot reconstruct hidden states, attention or intervention effects.",
                            "Missing InternVL raw input is unavailable evidence, not evidence of zero failures.",
                            "Cluster means are descriptive; no independence claim or held-out confirmation is inferred."]}
