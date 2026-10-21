"""Source-only prospective fixtures; these tests have not been executed."""
import copy
import hashlib
import json
import unittest

from vlm_mcqa.stretch.effects import descriptive_control_summary, summarize_semantic_effects, validate_effect_recipe


def recipe_fixture():
    sets = {"top4": [[0, h] for h in range(4)]}
    sets.update({f"random4_{i:02d}": [[0, h] for h in range(4, 8)] for i in range(20)})
    conditions = [{"name": "clean", "operation": "clean"},
                  {"name": "image_restore", "operation": "image_restore"}]
    for name in sets:
        for operation in ("head_transfer", "restore_clamp"):
            conditions.append({"name": f"{operation}/{name}", "operation": operation, "head_set": name})
    recipe = {"sets": sets, "conditions": conditions, "condition_count": len(conditions), "random_sets": 20,
              "model_key": "test_model", "dataset_hash": "test_data", "freeze_hash": "test_freeze", "head_source_sha256": "test_heads"}
    recipe["recipe_sha256"] = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return recipe


def result_fixture(recipe, cid, pair, transfer=4.0, denominator=10.0):
    rows = []
    for condition in recipe["conditions"]:
        name = condition["name"]
        patched = {"clean": 0.0, "image_restore": 5.0}.get(name, 0.0)
        if name == "head_transfer/top4":
            patched = transfer
        elif name == "restore_clamp/top4":
            patched = 1.0
        elif name.startswith("restore_clamp/random"):
            patched = 4.0
        rows.append({**condition, "patched_diff": patched, "recovery_denominator": denominator,
                     "normalized_recovery": patched / denominator if abs(denominator) >= 1e-6 else None,
                     "recipe_sha256": recipe["recipe_sha256"], "contrast_id": cid, "pair_id": pair})
    return {"contrast_id": cid, "pair_id": pair, "status": "evaluated", "rows": rows,
            "role": "semantic", "contrast_type": "semantic_image_fixed_text", "split": "discovery",
            "identity": {key: recipe[key] for key in ("recipe_sha256", "model_key", "dataset_hash", "freeze_hash", "head_source_sha256")}}


def summarize(results, recipe, expected=None, **kwargs):
    return summarize_semantic_effects(results, recipe,
                                      expected_directions=expected or [{"contrast_id": r["contrast_id"], "pair_id": r["pair_id"]} for r in results],
                                      expected_split="discovery", bootstrap_seed=1, bootstrap_samples=100, **kwargs)


class EffectsTests(unittest.TestCase):
    def test_pair_means_not_direction_weights(self):
        recipe = recipe_fixture()
        results = [result_fixture(recipe, "a", "p", 1.0),
                   *[result_fixture(recipe, f"b{i}", "q", 9.0) for i in range(3)]]
        report = summarize(results, recipe)
        metric = report["pair_bootstrap"]["intervals"]["head_transfer_minus_meanrandom"]
        self.assertEqual(metric["mean"], 5.0)
        self.assertEqual(metric["n_pairs"], 2)
        self.assertEqual(report["coverage"]["complete_directions"], 4)

    def test_clamp_uses_shared_restored_reference(self):
        recipe = recipe_fixture()
        report = summarize([result_fixture(recipe, "a", "p"), result_fixture(recipe, "b", "q")], recipe)
        effects = report["direction_effects"][0]
        self.assertEqual(effects["top4_clamp_attenuation"], 4.0)
        self.assertEqual(effects["clamp_attenuation_minus_meanrandom"], 3.0)
        self.assertTrue(report["positive_criterion"]["both_primary_ci95_lower_bounds_above_zero"])
        self.assertEqual(report["positive_criterion"]["scientific_confirmation"], "not_established_by_scalar_summary")

    def test_missing_direction_disables_positive_and_exposes_pair(self):
        recipe = recipe_fixture()
        expected = [{"contrast_id": "a", "pair_id": "p"}, {"contrast_id": "b", "pair_id": "p"},
                    {"contrast_id": "c", "pair_id": "q"}]
        report = summarize([result_fixture(recipe, "a", "p"), result_fixture(recipe, "c", "q")], recipe, expected)
        self.assertEqual(report["coverage"]["expected_directions"], 3)
        self.assertEqual(report["coverage"]["complete_pairs"], 1)
        self.assertEqual(report["coverage"]["pair_audit"][0]["incomplete_direction_ids"], ["b"])
        self.assertIsNone(report["positive_criterion"]["both_primary_ci95_lower_bounds_above_zero"])

    def test_ineligible_direction_is_not_silently_removed(self):
        recipe = recipe_fixture()
        excluded = result_fixture(recipe, "a", "p")
        excluded.update(status="excluded_baseline_eligibility", rows=[])
        report = summarize([excluded, result_fixture(recipe, "b", "q")], recipe)
        self.assertEqual(report["coverage"]["expected_pairs"], 2)
        self.assertEqual(report["coverage"]["direction_audit"][0]["status"], "excluded_baseline_eligibility")
        self.assertIsNone(report["positive_criterion"]["conjunctive_primary_and_reference_pattern"])

    def test_missing_random_condition_is_incomplete_not_control_reweighting(self):
        recipe = recipe_fixture()
        result = result_fixture(recipe, "a", "p")
        result["rows"] = [r for r in result["rows"] if r["name"] != "restore_clamp/random4_19"]
        report = summarize([result], recipe)
        self.assertEqual(report["coverage"]["complete_directions"], 0)
        self.assertEqual(report["coverage"]["direction_audit"][0]["missing_conditions"], ["restore_clamp/random4_19"])

    def test_duplicates_recipe_mismatch_and_nonfinite_rejected(self):
        recipe = recipe_fixture()
        result = result_fixture(recipe, "a", "p")
        with self.assertRaises(ValueError):
            summarize([result, result], recipe, [{"contrast_id": "a", "pair_id": "p"}])
        broken = copy.deepcopy(result)
        broken["rows"].append(broken["rows"][0])
        with self.assertRaises(ValueError):
            summarize([broken], recipe)
        broken = copy.deepcopy(result)
        broken["identity"]["recipe_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            summarize([broken], recipe)
        broken = copy.deepcopy(result)
        broken["rows"][0]["patched_diff"] = float("nan")
        with self.assertRaises(ValueError):
            summarize([broken], recipe)

    def test_small_denominators_change_only_secondary_sensitivity(self):
        recipe = recipe_fixture()
        report = summarize([result_fixture(recipe, "a", "p", denominator=0.1),
                            result_fixture(recipe, "b", "q")], recipe)
        self.assertEqual(report["coverage"]["complete_pairs"], 2)
        self.assertEqual(report["pair_bootstrap"]["intervals"]["head_transfer_minus_meanrandom"]["n_pairs"], 2)
        self.assertEqual(report["normalized_recovery_sensitivity"]["n_pairs_all_directions_stable"], 1)

    def test_controls_cannot_enter_semantic_cohort(self):
        recipe = recipe_fixture()
        control = result_fixture(recipe, "a", "p")
        control.update(role="control", contrast_type="position")
        with self.assertRaises(ValueError):
            summarize([control], recipe)
        control["rows"] = [{"name": "head_transfer/top4", "raw_margin_change": 2.0, "sample_unit": "p"}]
        report = descriptive_control_summary([control])
        self.assertEqual(report["cohorts"][0]["n_recipient_pairs"], 1)
        self.assertEqual(report["cohorts"][0]["contrast_type"], "position")

    def test_split_and_model_identity_cannot_be_mixed(self):
        recipe = recipe_fixture()
        result = result_fixture(recipe, "a", "p")
        result["split"] = "confirmation"
        with self.assertRaises(ValueError):
            summarize([result], recipe)
        result["split"] = "discovery"
        result["identity"]["model_key"] = "different_model"
        with self.assertRaises(ValueError):
            summarize([result], recipe)

    def test_recipe_validation_fails_before_runtime_for_nonprimary_control_count(self):
        recipe = recipe_fixture()
        recipe["random_sets"] = 19
        recipe.pop("recipe_sha256")
        recipe["recipe_sha256"] = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with self.assertRaises(ValueError):
            validate_effect_recipe(recipe)


if __name__ == "__main__":
    unittest.main()
