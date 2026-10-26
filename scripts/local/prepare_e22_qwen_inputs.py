#!/usr/bin/env python3
"""Prepare frozen Qwen E22 discovery evidence and a blind confirmation scope."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from vlm_mcqa.stretch.extension_calibration import sha
from vlm_mcqa.stretch.failure_protocol import evaluate_failure_rules, freeze_failure_rules


FLAG = "truth_aligned_content_with_wrong_final_symbol"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--e19-design-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discovery-scenes", type=int, default=32)
    parser.add_argument("--confirmation-scenes", type=int, default=128)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    dataset = {
        "items": read_jsonl(args.dataset / "items.jsonl"),
        "variants": read_jsonl(args.dataset / "variants.jsonl"),
    }
    source = json.loads(args.e19_design_report.read_text(encoding="utf-8"))
    source_rules = source["rules"]
    rules = freeze_failure_rules({
        key: source_rules[key]
        for key in ("stage", "min_scenes", "reliability_min", "seed", "bootstrap_draws",
                    "margin_bins", "match_fields", "max_control_reuse")
    } | {"strata_flags": [FLAG]})
    descriptive = {"status": "descriptive_not_causal", "rows": source["rows"]}
    report = evaluate_failure_rules(descriptive, dataset, rules)

    entry = report["phases"]["discovery"]["strata"][FLAG]
    selected, seen = [], set()
    for match in entry["matched_correct_controls"]:
        if match["failure_scene_id"] not in seen:
            selected.append(match)
            seen.add(match["failure_scene_id"])
        if len(selected) == args.discovery_scenes:
            break
    if len(selected) != args.discovery_scenes:
        raise RuntimeError(f"only {len(selected)} control-complete discovery scenes available")
    entry.update({
        "variant_ids": [match["failure_variant_id"] for match in selected],
        "scene_ids": [match["failure_scene_id"] for match in selected],
        "matched_correct_controls": selected,
        "unmatched_failure_variants": [],
        "n_scene_matched": len(selected),
        "sample_gate_passed": True,
        "matched_control_gate_passed": True,
        "status": "candidate_for_causal_design",
    })
    report["manifest_sha256"] = sha({key: value for key, value in report.items() if key != "manifest_sha256"})

    rows = {row["variant_id"]: row for row in report["rows"]}
    variants = {variant["variant_id"]: variant for variant in dataset["variants"]}
    calibration = None
    for match in selected:
        recipient_id = match["failure_variant_id"]
        recipient = variants[recipient_id]
        donors = [variant for variant in dataset["variants"]
                  if variant["split"] == recipient["split"]
                  and variant["item_id"] == recipient["item_id"]
                  and variant["variant_id"] != recipient_id
                  and variant["image"] == recipient["image"]
                  and (variant["correct_symbol"] != recipient["correct_symbol"]
                       or variant["labels"] != recipient["labels"])
                  and rows.get(variant["variant_id"], {}).get("final_correct") is True]
        if donors:
            donor = min(donors, key=lambda value: sha([rules["seed"], recipient_id, value["variant_id"]]))
            calibration = {
                "calibration_variant_ids": [recipient_id, donor["variant_id"]],
                "calibration_case": {
                    "case_id": "e22-calibration",
                    "recipient_variant_id": recipient_id,
                    "donor_variant_id": donor["variant_id"],
                    "stratum": FLAG,
                    "role": "failure",
                    "donor_class": "same_image_remap",
                },
            }
            break
    if calibration is None:
        raise RuntimeError("no observed-correct same-image remap donor for E22 calibration")

    confirmation_items = [item for item in dataset["items"] if item["split"] == "confirmation"]
    scene_representatives = {}
    for item in sorted(confirmation_items, key=lambda value: value["item_id"]):
        scene_representatives.setdefault(item["scene_id"], item)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for item in scene_representatives.values():
        by_family[item["question_family"]].append(item)
    for values in by_family.values():
        values.sort(key=lambda item: (sha([rules["seed"], item["scene_id"]]), item["scene_id"]))
    selected_items = []
    while len(selected_items) < args.confirmation_scenes:
        progressed = False
        for family in sorted(by_family):
            if by_family[family] and len(selected_items) < args.confirmation_scenes:
                selected_items.append(by_family[family].pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected_items) != args.confirmation_scenes:
        raise RuntimeError("confirmation scene target exceeds available scenes")
    selected_scene_ids = {item["scene_id"] for item in selected_items}
    confirmation_ids = sorted(
        (variant["variant_id"] for variant in dataset["variants"]
         if variant["split"] == "confirmation" and variant["scene_id"] in selected_scene_ids),
        key=lambda vid: (sha([rules["seed"], vid]), vid),
    )
    scope = {
        "schema": "stretch.failure-candidate-scope.v1",
        "phase": "confirmation",
        "selection": "outcome-blind deterministic question-family round-robin over confirmation scenes",
        "target_scenes": args.confirmation_scenes,
        "scene_ids": sorted(selected_scene_ids),
        "variant_ids": confirmation_ids,
    }

    write_json(args.output / "rules.json", rules)
    write_json(args.output / "discovery_strata.json", report)
    write_json(args.output / "confirmation_scope.json", scope)
    write_json(args.output / "calibration.json", calibration)
    write_json(args.output / "preparation.json", {
        "schema": "e22.qwen.preparation.v1",
        "source_e19_manifest_sha256": source["manifest_sha256"],
        "rules_sha256": rules["rules_sha256"],
        "discovery_scenes": len(selected),
        "discovery_failure_variants": len(entry["variant_ids"]),
        "discovery_matched_controls": len(entry["matched_correct_controls"]),
        "confirmation_scenes": len(selected_scene_ids),
        "confirmation_variants": len(confirmation_ids),
        "confirmation_selection_uses_outcomes": False,
    })


if __name__ == "__main__":
    main()
