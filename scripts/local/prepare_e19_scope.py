#!/usr/bin/env python3
"""Prepare a prospective E19 scope from release metadata, without model outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_scope(dataset_dir: Path, stage: int) -> dict:
    variants_path = dataset_dir / "variants.jsonl"
    if not variants_path.is_file():
        raise FileNotFoundError(f"missing release variants: {variants_path}")
    selected = []
    with variants_path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "behavior" and row.get("variant_kind") == "position":
                selected.append(row["variant_id"])
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("behavior-position scope must be nonempty and unique")
    return {
        "schema": "stretch.e19-scope.v1",
        "selection": "all release variants with split=behavior and variant_kind=position",
        "selection_uses_model_outcomes": False,
        "stages": [stage],
        "expected_variant_ids": sorted(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage < 0:
        parser.error("stage must be nonnegative")
    if args.output.exists():
        parser.error("output must not exist")
    scope = build_scope(args.dataset, args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
