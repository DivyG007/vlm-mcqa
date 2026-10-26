#!/usr/bin/env python3
"""Prepare outcome-blind E22 candidate scopes for a non-Qwen replication."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from vlm_mcqa.stretch.extension_calibration import sha
from vlm_mcqa.stretch.failure_protocol import freeze_failure_rules


FLAG = "truth_aligned_content_with_wrong_final_symbol"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def select_scope(items: list[dict], variants: list[dict], *, phase: str,
                 allowed_splits: set[str], target_scenes: int, seed: int) -> dict:
    representatives: dict[str, dict] = {}
    for item in items:
        if item["split"] in allowed_splits:
            representatives.setdefault(item["scene_id"], item)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for item in representatives.values():
        by_family[item["question_family"]].append(item)
    for values in by_family.values():
        values.sort(key=lambda item: (sha([seed, phase, item["scene_id"]]), item["scene_id"]))
    selected: list[dict] = []
    while len(selected) < target_scenes:
        progressed = False
        for family in sorted(by_family):
            if by_family[family] and len(selected) < target_scenes:
                selected.append(by_family[family].pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected) != target_scenes:
        raise RuntimeError(f"{phase} target {target_scenes} exceeds {len(selected)} available scenes")
    scene_ids = {item["scene_id"] for item in selected}
    variant_ids = sorted(
        (variant["variant_id"] for variant in variants
         if variant["split"] in allowed_splits and variant["scene_id"] in scene_ids),
        key=lambda vid: (sha([seed, phase, vid]), vid),
    )
    return {
        "schema": "stretch.failure-candidate-scope.v1",
        "phase": phase,
        "selection": "outcome-blind deterministic question-family round-robin over scenes",
        "target_scenes": target_scenes,
        "allowed_splits": sorted(allowed_splits),
        "scene_ids": sorted(scene_ids),
        "variant_ids": variant_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--behavior-baseline", type=Path, required=True,
                        help="model-specific clean baseline JSONL used only for technical calibration")
    parser.add_argument("--discovery-scenes", type=int, default=128)
    parser.add_argument("--confirmation-scenes", type=int, default=128)
    args = parser.parse_args()
    if args.num_layers < 2:
        raise SystemExit("num-layers must be at least two")
    args.output.mkdir(parents=True, exist_ok=False)
    items = read_jsonl(args.dataset / "items.jsonl")
    variants = read_jsonl(args.dataset / "variants.jsonl")
    baseline_rows = read_jsonl(args.behavior_baseline)
    rules = freeze_failure_rules({
        "stage": args.num_layers,
        "min_scenes": 16,
        "reliability_min": 0.7,
        "seed": 0,
        "bootstrap_draws": 2000,
        "margin_bins": [1.0, 2.0, 4.0],
        "match_fields": ["question_family", "reasoning_depth", "scheme"],
        "max_control_reuse": 1,
        "strata_flags": [FLAG],
    })
    discovery = select_scope(items, variants, phase="discovery",
                             allowed_splits={"discovery", "behavior"},
                             target_scenes=args.discovery_scenes, seed=rules["seed"])
    confirmation = select_scope(items, variants, phase="confirmation",
                                allowed_splits={"confirmation"},
                                target_scenes=args.confirmation_scenes, seed=rules["seed"])
    indexed = {variant["variant_id"]: variant for variant in variants}
    baseline = {row["variant_id"]: row for row in baseline_rows
                if row.get("model_key") == args.model_key and row.get("condition") == "clean"
                and row.get("variant_id") in indexed}
    calibration = None
    calibration_candidates = sorted(
        (variant_id for variant_id, row in baseline.items()
         if row.get("correct") is False and indexed[variant_id]["split"] in {"discovery", "behavior"}),
        key=lambda vid: (sha([rules["seed"], "calibration", vid]), vid),
    )
    for recipient_id in calibration_candidates:
        recipient = indexed[recipient_id]
        if baseline.get(recipient_id, {}).get("correct") is not False:
            continue
        donors = [variant for variant in variants
                  if variant["split"] == recipient["split"]
                  and variant["item_id"] == recipient["item_id"]
                  and variant["variant_id"] != recipient_id
                  and variant["image"] == recipient["image"]
                  and (variant["correct_symbol"] != recipient["correct_symbol"]
                       or variant["labels"] != recipient["labels"])
                  and baseline.get(variant["variant_id"], {}).get("correct") is True]
        if donors:
            donor = min(donors, key=lambda value: sha([rules["seed"], recipient_id,
                                                       value["variant_id"]]))
            calibration = {
                "calibration_variant_ids": [recipient_id, donor["variant_id"]],
                "calibration_case": {
                    "case_id": "e22-next-three-calibration",
                    "recipient_variant_id": recipient_id,
                    "donor_variant_id": donor["variant_id"],
                    "stratum": FLAG,
                    "role": "failure",
                    "donor_class": "same_image_remap",
                },
            }
            break
    if calibration is None:
        raise RuntimeError("no same-image remap pair available for technical calibration")
    write_json(args.output / "rules.json", rules)
    write_json(args.output / "discovery_scope.json", discovery)
    write_json(args.output / "confirmation_scope.json", confirmation)
    write_json(args.output / "calibration.json", calibration)
    write_json(args.output / "preparation.json", {
        "schema": "e22.next-three.preparation.v1",
        "model_key": args.model_key,
        "num_layers": args.num_layers,
        "rules_sha256": rules["rules_sha256"],
        "selection_uses_outcomes": False,
        "calibration_selection": "model-specific prior clean baseline; technical calibration only",
        "behavior_baseline_sha256": __import__("hashlib").sha256(
            args.behavior_baseline.read_bytes()).hexdigest(),
        "discovery_scenes": len(discovery["scene_ids"]),
        "discovery_variants": len(discovery["variant_ids"]),
        "confirmation_scenes": len(confirmation["scene_ids"]),
        "confirmation_variants": len(confirmation["variant_ids"]),
    })


if __name__ == "__main__":
    main()
