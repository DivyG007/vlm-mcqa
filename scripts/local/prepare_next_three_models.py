#!/usr/bin/env python3
"""Compile and verify all CPU-only inputs for the next-three-model stretch run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlm_mcqa.stretch.replication_prep import compile_model_bundle, verify_head_freeze  # noqa: E402
from vlm_mcqa.stretch.protocol import compile_recipe, load_frozen_heads  # noqa: E402


MODEL_RUNTIME = {
    "internvl35_8b": {"family": "internvl", "num_layers": 36, "num_heads": 32,
                       "revision": "741a7d03020411e666c6109218ab71e08151ef86",
                       "freeze": ROOT / "consolidation/core7_release_results/internvl35_8b_core7/freeze.json"},
    "llava_ov_7b_si": {"family": "llava_onevision", "num_layers": 28, "num_heads": 28,
                        "revision": "cae7503dce88887557f73db1882328d21d9a7e21",
                        "freeze": ROOT / "consolidation/core7_release_results/llava_ov_7b_si_core7/freeze.json"},
    "paligemma2_10b_mix": {"family": "paligemma", "num_layers": 42, "num_heads": 16,
                            "revision": "b26d16fb4251090ba4a4aa5af9fca1f8248ed5b6",
                            "freeze": None},
}


def compile_all(plan_path: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    bundles, recipes, e20_protocols = {}, {}, {}
    for model_key, runtime in MODEL_RUNTIME.items():
        declared = plan["models"][model_key]
        freeze = None
        if runtime["freeze"] is not None:
            freeze = verify_head_freeze(runtime["freeze"], {
                "model_key": model_key, "freeze_hash": declared["core7_freeze"],
                "dataset_hash": plan["data"]["dataset_hash"],
                "image_aware_release_hash": plan["data"]["image_aware_release_hash"],
                "top4_heads": declared["top4_heads"], "top16_heads": declared["top16_heads"],
            })
            freeze["path"] = runtime["freeze"].relative_to(ROOT).as_posix()
        bundles[model_key] = compile_model_bundle(
            model_key=model_key, family=runtime["family"], revision=runtime["revision"],
            num_layers=runtime["num_layers"], dataset_hash=plan["data"]["dataset_hash"],
            release_hash=plan["data"]["image_aware_release_hash"],
            top4_heads=declared.get("top4_heads"), top16_heads=declared.get("top16_heads"),
            head_freeze=freeze)
        if runtime["freeze"] is not None:
            loaded = load_frozen_heads(runtime["freeze"], model_key=model_key,
                                       dataset_hash=plan["data"]["dataset_hash"])
            loaded["source_path"] = runtime["freeze"].relative_to(ROOT).as_posix()
            recipes[model_key] = compile_recipe(
                loaded, num_heads=runtime["num_heads"], num_layers=runtime["num_layers"],
                image_layer=0, image_site="resid_post", seed=0, random_sets=20,
                expanded_controls=False)
            bundles[model_key]["e20_recipe_path"] = f"{model_key}.e20_recipe.json"
            recipe = recipes[model_key]
            e20_protocols[model_key] = {
                "schema": "rachit.e20-cross-model-protocol.v1",
                "eligibility_decision": "approved", "recipe_sha256": recipe["recipe_sha256"],
                "claim_boundary": "calibration_and_technical_pilots_are_not_scientific_evidence",
                "primary_upstream_site": {"kind": "resid_post", "layer": 0,
                                          "selection": "relative_depth_frozen_not_outcome_selected"},
                "controls": {"margin_bin_edges": [0.0, 1.0, 2.0, 4.0, 8.0],
                             "match_family": True, "max_donor_reuse": 1,
                             "per_recipient": 1, "seed": 0},
                "sampling": {
                    "discovery": {"target_pairs": 32, "min_pairs": 16, "layouts_per_pair": 2,
                                  "seed": 0, "excluded_pair_ids": ["pair_CLEVR_train_000071"]},
                    "confirmation": {"target_pairs": 24, "min_pairs": 16, "layouts_per_pair": 2,
                                     "seed": 0, "excluded_pair_ids": []}},
                "min_pairs": 16, "bootstrap_seed": 0, "bootstrap_samples": 2000,
                "calibration_pair_ids_excluded_from_science": ["pair_CLEVR_train_000071"],
            }
            bundles[model_key]["e20_protocol_path"] = f"{model_key}.e20_protocol.json"
    try:
        source_plan = plan_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        source_plan = str(plan_path.resolve())
    return {"schema": "vlm-mcqa.next-three-preflight.v1", "source_plan": source_plan,
            "gpu_work_performed": False, "bundles": bundles,
            "recipes": recipes, "e20_protocols": e20_protocols,
            "launch_order": ["internvl35_8b", "llava_ov_7b_si", "paligemma2_10b_mix"],
            "reuse_matrix": {
                "E20": {"reuse": ["fixed_text_contrasts", "CLEVR_images", "sampling_rules"],
                        "recompute": ["clean_correctness", "control_complete_sample_plan", "all_effects"]},
                "E22": {"reuse": ["CLEVR_variants", "stratum_definition", "donor_rules"],
                        "recompute": ["trajectories", "failure_strata", "donors", "selected_option_layer"]},
                "E23": {"reuse": ["task_manifest", "CLEVR_images", "factorial_annotations"],
                        "recompute": ["processor_geometry", "correct_only_cohort", "selected_image_layer"]},
                "E24": {"reuse": ["source_crops", "collages", "task_manifest", "pixel_hashes"],
                        "recompute": ["processor_geometry", "correct_only_cohort", "selected_image_layer"]}},
            "global_gate": "run calibration before every discovery; freeze discovery before confirmation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/stretch/next_three_models_replication_plan_2026-10-27.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs/stretch/next_three")
    parser.add_argument("--check", action="store_true", help="validate and print without writing")
    args = parser.parse_args(argv)
    result = compile_all(args.plan.resolve())
    if not args.check:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for model_key, bundle in result["bundles"].items():
            (args.output_dir / f"{model_key}.protocol.json").write_text(
                json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if model_key in result["recipes"]:
                (args.output_dir / f"{model_key}.e20_recipe.json").write_text(
                    json.dumps(result["recipes"][model_key], indent=2, sort_keys=True) + "\n", encoding="utf-8")
                (args.output_dir / f"{model_key}.e20_protocol.json").write_text(
                    json.dumps(result["e20_protocols"][model_key], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        readiness = {key: value for key, value in result.items() if key not in {"recipes", "e20_protocols"}}
        (args.output_dir / "readiness.json").write_text(
            json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
