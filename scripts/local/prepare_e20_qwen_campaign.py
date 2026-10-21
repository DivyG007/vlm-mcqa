#!/usr/bin/env python3
"""Materialize the immutable, no-model-load inputs for Qwen E20 calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vlm_mcqa.campaigns.config import load_config
from vlm_mcqa.campaigns.preparation import prepare_inventory
from vlm_mcqa.campaigns.verification import verify_inputs
from vlm_mcqa.stretch.data import build_fixed_text_contrasts, load_dataset
from vlm_mcqa.stretch.protocol import compile_recipe, load_frozen_heads


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL_KEY = "qwen25vl_7b"
DATASET_HASH = "f5573e25f3aa2bf2"
IMAGE_AWARE_HASH = "eb7b79e56977bbaeed5fc981c5c8e09a4e26eaf8b1702283abe2d4bbfc718cbd"


def write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def explicit_two_gpu_map() -> dict[str, str]:
    """Keep vision/embedding and early layers together; leave hook headroom."""
    placement = {
        "model.visual": "cuda:0",
        "model.language_model.embed_tokens": "cuda:0",
        # Non-persistent RoPE buffers otherwise remain on CPU because this
        # shared module has no checkpoint parameters for Accelerate to place.
        "model.language_model.rotary_emb": "cuda:0",
        "model.language_model.norm": "cuda:1",
        "lm_head": "cuda:1",
    }
    for layer in range(28):
        placement[f"model.language_model.layers.{layer}"] = "cuda:0" if layer <= 12 else "cuda:1"
    return placement


def protocol_document(recipe_hash: str, calibration_pair_id: str) -> dict[str, Any]:
    return {
        "schema": "rachit.e20-qwen-protocol.v1",
        "recipe_sha256": recipe_hash,
        "eligibility_decision": "approved",
        "primary_upstream_site": {"kind": "resid_post", "layer": 0,
                                  "selection": "proposal_fixed_not_outcome_selected"},
        "calibration_pair_ids_excluded_from_science": [calibration_pair_id],
        "min_pairs": 16,
        "bootstrap_seed": 0,
        "bootstrap_samples": 2000,
        "controls": {
            "seed": 0, "per_recipient": 1, "max_donor_reuse": 1,
            "match_family": True, "margin_bin_edges": [0.0, 1.0, 2.0, 4.0, 8.0],
        },
        "sampling": {
            "discovery": {"target_pairs": 32, "min_pairs": 16, "layouts_per_pair": 2,
                          "seed": 0, "excluded_pair_ids": [calibration_pair_id]},
            "confirmation": {"target_pairs": 24, "min_pairs": 16, "layouts_per_pair": 2,
                             "seed": 0, "excluded_pair_ids": []},
        },
        "claim_boundary": "calibration_and_technical_pilots_are_not_scientific_evidence",
    }


def campaign_document(*, source_root: Path, source_manifest: Path, checkpoint_root: Path,
                      checkpoint_manifest: Path, dataset_root: Path, decision_path: Path,
                      decision_sha256: str, protocol_path: Path, recipe_path: Path,
                      contrasts_path: Path, head_source: Path, calibration: dict[str, Any],
                      output_placeholder: Path) -> dict[str, Any]:
    return {
        "schema": "vlm-mcqa-campaign/v1",
        "kind": "stretch",
        "seed": 0,
        "output_dir": str(output_placeholder),
        "source": {"root": str(source_root), "manifest": str(source_manifest)},
        "model": {
            "key": MODEL_KEY,
            "models_config": str(source_root / "configs/core7/qwen25vl_7b_pinned.json"),
            "checkpoint_dir": str(checkpoint_root),
            "checkpoint_manifest": str(checkpoint_manifest),
        },
        "data": {"directory": str(dataset_root), "dataset_hash": DATASET_HASH,
                 "image_aware_release_hash": IMAGE_AWARE_HASH},
        "deployment": {
            "dtype": "float16", "device_map": explicit_two_gpu_map(),
            "offload_policy": "forbid", "batch_size": 1,
            "prepared_cache": 8, "clean_cache": 96,
        },
        "protocol": {"decision_path": str(decision_path), "decision_sha256": decision_sha256},
        "study": {
            "name": "semantic_heads", "phases": ["discovery", "confirmation"],
            "config": {
                "protocol": str(protocol_path), "recipe": str(recipe_path),
                "contrasts": str(contrasts_path),
                "calibration_variant_ids": [calibration["donor_variant_id"],
                                            calibration["recipient_variant_id"]],
                "calibration_contrast_id": calibration["contrast_id"],
                "tolerance": 0.06,
                "protected_inputs": [str(dataset_root), str(head_source)],
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt-base", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    attempt_base = args.attempt_base.resolve()
    head_source = source_root / "consolidation/core7_release_results/qwen25vl_7b_core7/m3_heads.json"
    if len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit):
        raise ValueError("full lowercase source commit required")
    for required in (source_root / "pyproject.toml", checkpoint_root / "stage_completion.json",
                     dataset_root / "dataset_manifest.json", dataset_root / "release_manifest.json",
                     head_source):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_root.exists():
        raise FileExistsError(output_root)
    for immutable in (source_root, checkpoint_root, dataset_root, head_source.parent):
        if output_root == immutable or output_root in immutable.parents or immutable in output_root.parents:
            raise ValueError(f"output overlaps immutable input: {immutable}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    attempt_base.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()

    source_inventory = output_root / "source_inventory.json"
    checkpoint_inventory = output_root / "checkpoint_inventory.json"
    prepare_inventory(source_root, source_inventory, kind="source", commit=args.source_commit)
    prepare_inventory(checkpoint_root, checkpoint_inventory, kind="checkpoint",
                      hf_id=MODEL_ID, revision=REVISION)

    dataset = load_dataset(dataset_root)
    if dataset["manifest"].get("dataset_hash") != DATASET_HASH:
        raise ValueError("unexpected E20 dataset hash")
    if dataset["release_manifest"].get("image_aware_release_hash") != IMAGE_AWARE_HASH:
        raise ValueError("unexpected E20 image-aware release hash")
    contrasts = build_fixed_text_contrasts(dataset, schemes=["letters"],
                                           splits=["discovery", "confirmation"])
    counts = {split: sum(row["split"] == split for row in contrasts)
              for split in ("discovery", "confirmation")}
    if counts != {"discovery": 1280, "confirmation": 1024}:
        raise ValueError(f"unexpected fixed-text contrast coverage: {counts}")
    calibration = next(row for row in contrasts if row["split"] == "discovery")
    reverse = [row for row in contrasts
               if row["donor_variant_id"] == calibration["recipient_variant_id"]
               and row["recipient_variant_id"] == calibration["donor_variant_id"]]
    if len(reverse) != 1:
        raise ValueError("calibration contrast lacks one reverse direction")
    contrasts_path = output_root / "fixed_text_contrasts.json"
    write_new(contrasts_path, {
        "schema": "stretch-fixed-text-contrasts-v1", "provenance": dataset["provenance"],
        "source_manifest": dataset["manifest"], "release_manifest": dataset["release_manifest"],
        "contrasts": contrasts, "runtime_validation": "pending",
        "historical_exposure": "calibration_pair_excluded; full_union_audit_pending",
    })

    frozen = load_frozen_heads(head_source, model_key=MODEL_KEY, dataset_hash=DATASET_HASH)
    recipe = compile_recipe(frozen, num_heads=28, num_layers=28, image_layer=0,
                            image_site="resid_post", seed=0, random_sets=20,
                            expanded_controls=False)
    recipe_path = output_root / "recipe.json"
    write_new(recipe_path, {"recipe": recipe, "head_source": frozen})
    protocol_path = output_root / "protocol.json"
    write_new(protocol_path, protocol_document(recipe["recipe_sha256"], calibration["pair_id"]))
    decision_path = output_root / "decision.json"
    write_new(decision_path, {
        "schema": "rachit.e20-qwen-decision.v1", "approved": True,
        "rationale": "User authorized Qwen-first E20 on the existing verified CLEVR release; calibration is gated and is not scientific evidence.",
        "model_key": MODEL_KEY, "hf_id": MODEL_ID, "revision": REVISION,
        "dataset_hash": DATASET_HASH, "freeze_hash": frozen["freeze_hash"],
        "calibration_pair_id": calibration["pair_id"],
        "calibration_pair_selection": "first deterministic structural discovery contrast; excluded from later science",
    })
    config = campaign_document(
        source_root=source_root, source_manifest=source_inventory,
        checkpoint_root=checkpoint_root, checkpoint_manifest=checkpoint_inventory,
        dataset_root=dataset_root, decision_path=decision_path,
        decision_sha256=file_hash(decision_path), protocol_path=protocol_path,
        recipe_path=recipe_path, contrasts_path=contrasts_path, head_source=head_source,
        calibration=calibration, output_placeholder=attempt_base / "UNMATERIALIZED_JOB_ID",
    )
    config_path = output_root / "campaign_base.json"
    write_new(config_path, config)
    verified = verify_inputs(load_config(config_path))
    write_new(output_root / "verification.json", verified)
    write_new(output_root / "completion.json", {
        "schema": "rachit.e20-qwen-preparation.v1", "status": "complete",
        "source_git_commit": args.source_commit, "model_revision": REVISION,
        "contrast_counts": counts, "recipe_sha256": recipe["recipe_sha256"],
        "condition_count": recipe["condition_count"],
        "calibration_contrast_id": calibration["contrast_id"],
        "calibration_pair_id": calibration["pair_id"],
        "calibration_pair_excluded_from_science": True,
        "device_map": explicit_two_gpu_map(), "verification_config_sha256": verified["config_sha256"],
    })
    print(json.dumps(json.loads((output_root / "completion.json").read_text()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
