"""Authored-only causal fixtures; no tests executed during implementation."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vlm_mcqa.stretch import controls, trajectory, failure_runtime
from vlm_mcqa.stretch.extension_calibration import grouped_summary
from vlm_mcqa.stretch.runtime import REQUIRED_GATES


class GateTests(unittest.TestCase):
    def test_all_new_entrypoints_deny_before_any_effect(self):
        for module in (controls, trajectory, failure_runtime):
            for name in ("calibrate_study", "run_study"):
                with self.assertRaises(PermissionError):
                    getattr(module, name)(object(), {}, {"execute": False})
        with self.assertRaises(PermissionError):
            controls.prepare_study(object(), {}, {"execute": False})

    def test_pair_summary_averages_dependent_directions(self):
        rows = [{"condition": "x", "sample_unit": "a", "raw_margin_change": 1.0},
                {"condition": "x", "sample_unit": "a", "raw_margin_change": 3.0},
                {"condition": "x", "sample_unit": "b", "raw_margin_change": 8.0}]
        result = grouped_summary(rows, seed=2, samples=100)
        self.assertEqual(result["x"]["mean"], 5.0)
        self.assertEqual(result["x"]["n_units"], 2)

    def test_supplied_wrong_donor_cannot_be_silently_replaced(self):
        variant = {"variant_id": "r", "item_id": "r", "split": "discovery", "image": "r.png",
                   "scene_id": "r", "counterfactual_pair_id": "p", "correct_symbol": "A",
                   "labels": list("ABCD"), "semantic_answer": "red", "question_family": "colour"}
        donor = {**variant, "variant_id": "d", "item_id": "d", "image": "d.png", "scene_id": "d",
                 "counterfactual_pair_id": "q", "semantic_answer": "blue"}
        session = SimpleNamespace(variants={"r": variant, "d": donor},
                                  clean=lambda vid: SimpleNamespace(correct=vid == "d"))
        body = {"allowed_strata": ["proxy"], "donor_classes": ["same_image_remap"], "seed": 1,
                "max_donor_reuse": 1}
        units = [{"case_id": "u", "stratum": "proxy", "role": "failure", "recipient_variant_id": "r",
                  "donor_class": "same_image_remap", "donor_variant_id": "d"}]
        with self.assertRaises(ValueError):
            failure_runtime.match_failure_donors(session, units, body, "discovery")
        units[0]["donor_variant_id"] = None
        attempts = failure_runtime.match_failure_donors(session, units, body, "discovery")
        self.assertEqual(attempts[0]["status"], "no_valid_donor")

    def test_no_donor_stays_in_requested_denominator(self):
        attempts = [{"case_id": "missing", "stratum": "proxy", "role": "failure", "donor_class": "semantic_match",
                     "recipient_variant_id": "r", "status": "no_valid_donor"}]
        body = {"donor_classes": ["semantic_match"],
                "primary_conditions_by_donor_class": {"semantic_match": ["head_repair"]}}
        rows = failure_runtime.primary_denominators(attempts, [], {"r": {"scene_id": "s"}}, body)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["evaluable"])
        self.assertEqual(rows[0]["raw_margin_change"], 0.0)
        self.assertEqual(rows[0]["missing_reason"], "no_valid_donor")

    def test_failure_scope_rejects_entire_mixed_phase_list_before_baselines(self):
        def forbidden(*args, **kwargs):
            self.fail("held-out scope must be rejected before any baseline or preparation")
        session = SimpleNamespace(variants={"d": {"split": "discovery"}, "c": {"split": "confirmation"}},
                                  clean=forbidden, prepared=forbidden)
        units = [{"case_id": vid, "stratum": "proxy", "role": "failure", "recipient_variant_id": vid}
                 for vid in ("d", "c")]
        with self.assertRaises(ValueError):
            failure_runtime.match_failure_donors(session, units, {"allowed_strata": ["proxy"]}, "discovery")
        with self.assertRaises(ValueError):
            failure_runtime.validate_case_scope(session, units, {"allowed_strata": ["proxy"],
                "discovery_allowed_splits": ["discovery", "confirmation"]}, "discovery")

    def test_semantic_preparation_rejects_false_discovery_before_baselines(self):
        from vlm_mcqa.stretch.data import CONTRAST_TYPE
        variant = {"prompt": "same", "question": "colour?", "labels": list("ABCD"),
                   "option_contents": ["red", "blue", "green", "yellow"], "label_scheme": "letters",
                   "split": "confirmation", "counterfactual_pair_id": "p"}
        session = SimpleNamespace(variants={"d": dict(variant), "r": dict(variant)},
                                  clean=lambda *args: self.fail("baseline must not be evaluated"))
        config = {"phase": "discovery", "recipe": {}, "contrasts": [{
            "contrast_id": "false-discovery", "contrast_type": CONTRAST_TYPE, "pair_id": "p",
            "split": "discovery", "donor_variant_id": "d", "recipient_variant_id": "r"}]}
        with patch.object(controls, "core_guard"), patch.object(controls, "_body", return_value={}), \
             patch("vlm_mcqa.stretch.effects.validate_effect_recipe"):
            with self.assertRaisesRegex(ValueError, "contrast split mismatch"):
                controls.prepare_study(session, config, {"execute": True})


def synthetic_session():
    """Tiny analytic replay: image -> option boundary -> four heads -> final.

    Implements the adapter interface, not a real checkpoint. It distinguishes
    hybrid option captures from copying ordinary donor final-token states.
    """
    import torch
    from vlm_mcqa.core7.session import Clean
    from vlm_mcqa.core7.hooks import Site
    from vlm_mcqa.core7.scoring import symbol_readout

    variants = {}
    for vid, answer in (("d", "red"), ("r", "blue")):
        options, labels = ["red", "blue", "green", "yellow"], list("ABCD")
        variants[vid] = {"variant_id": vid, "item_id": vid, "scene_id": vid, "image": vid + ".png",
            "counterfactual_pair_id": "p", "split": "discovery", "labels": labels, "option_contents": options,
            "semantic_answer": answer, "correct_index": options.index(answer), "correct_symbol": labels[options.index(answer)],
            "question": "Colour?", "question_family": "colour", "label_scheme": "letters", "prompt": "same"}
    prepared = {vid: SimpleNamespace(variant_id=vid, input_ids=list(range(10)), num_tokens=10, final_position=9,
        groups={"image": [0], "options": [1, 2, 3, 4], "question": [5, 6, 7, 8], "final": [9]},
        label_ids={label: i for i, label in enumerate("ABCD")}) for vid in variants}

    class Adapter:
        spec = SimpleNamespace(key="fixture", hf_id="fixture/model", revision="a" * 40)
        handle = SimpleNamespace(num_heads=8)
        num_layers = 4

        def describe(self):
            return {"model_key": "fixture", "class": "analytic_replay"}

        def decoder_forward(self, prep, *, batch=1, interventions=(), capture=(), capture_positions=None):
            state = torch.zeros(batch, 10, 8)
            state[:, 0, 0] = 1 if prep.variant_id == "d" else -1
            captured = {}

            def boundary(tensor, kind, layer):
                value = tensor.clone()
                for item in interventions:
                    if item.site != Site(kind, layer):
                        continue
                    rows = range(batch) if item.rows is None else item.rows
                    supplied = item.values.reshape(len(item.positions), -1)
                    for row in rows:
                        for column, position in enumerate(item.positions):
                            if item.heads is None:
                                value[row, position] = supplied[column]
                            else:
                                for head in item.heads:
                                    value[row, position, head] = supplied[column, head]
                if Site(kind, layer) in capture:
                    captured[f"{kind}:{layer}"] = value[:, capture_positions].clone() if capture_positions is not None else value.clone()
                return value

            for layer in range(4):
                state = boundary(state, "resid_pre", layer)
                heads = torch.zeros_like(state)
                if layer == 1:
                    state[:, 1:5, 0] = state[:, 0:1, 0]
                if layer == 2:
                    heads[:, 9, :4] = state[:, 1:2, 0]
                heads = boundary(heads, "head", layer)
                if layer == 2:
                    state[:, 9, 0] = heads[:, 9].sum(-1)
                state = boundary(state, "resid_post", layer)
            signal = state[:, 9, 0]
            logits = torch.stack((signal, -signal, torch.zeros_like(signal), torch.zeros_like(signal)), dim=-1)
            return logits, captured

    adapter = Adapter()
    cache = {}
    def clean(vid):
        if vid not in cache:
            sites = [Site(kind, layer) for layer in range(4) for kind in ("resid_pre", "resid_post", "head")]
            logits, captured = adapter.decoder_forward(prepared[vid], capture=sites, capture_positions=[9])
            cache[vid] = Clean(logits[0], {key: value[0, 0] for key, value in captured.items()},
                               symbol_readout(logits[0], [0, 1, 2, 3], variants[vid]["correct_index"]))
        return cache[vid]
    session = SimpleNamespace(adapter=adapter, variants=variants, items={v["item_id"]: v for v in variants.values()},
        profile={"require_correct": True}, prepared=lambda vid: prepared[vid], clean=clean,
        label_ids=lambda prep, labels: [prep.label_ids[label] for label in labels],
        manifest={"dataset_hash": "data"}, release_manifest={"image_aware_release_hash": "images"})
    from vlm_mcqa.stretch.data import digest
    contrast = {"contrast_id": "c", "contrast_type": "semantic_image_fixed_text", "pair_id": "p", "split": "discovery",
        "donor_variant_id": "d", "recipient_variant_id": "r", "donor_symbol": "A", "recipient_symbol": "B",
        "donor_content": "red", "recipient_content": "blue", "prompt_sha256": digest(b"same")}
    frozen = {"head_candidates": [[2, h] for h in range(4)], "head_layers": [2]}
    route = {"id": "analytic", "sender": {"site": "resid_post", "layer": 0, "group": "image"},
        "receiver": {"site": "resid_post", "layer": 1, "group": "options"},
        "final": {"site": "resid_post", "layer": 3, "group": "final"}, "head_k": 4, "random_sets": 2, "seed": 3}
    context = {"execute": True, "identity": {"model_key": "fixture", "dataset_hash": "data",
        "image_aware_release_hash": "images", "source_identity": "source", "checkpoint_identity":
        {"hf_id": "fixture/model", "revision": "a" * 40}}, "core_calibration": {"all_passed": True,
        "gates": {name: {"passed": True} for name in REQUIRED_GATES}, "adapter": adapter.describe(), "variants": ["d"]}}
    return session, contrast, route, frozen, context


class AnalyticRouteTests(unittest.TestCase):
    def test_conditioned_path_and_baseline_clamp(self):
        session, contrast, route, frozen, context = synthetic_session()
        result = trajectory.evaluate_route(session, contrast, route, frozen, context)
        self.assertEqual(result["status"], "evaluated")
        rows = {r["condition"]: r for r in result["rows"]}
        self.assertGreater(rows["staged_chain"]["raw_margin_change"], 0)
        self.assertEqual(rows["receiver_clamp"]["raw_margin_change"], 0)
        self.assertEqual(rows["head_clamp_after_image"]["raw_margin_change"], 0)
        self.assertEqual(rows["random_staged/0"]["raw_margin_change"], 0)
        self.assertIn("receiver", rows["staged_chain"]["lineage"])

    def test_every_staged_self_condition_is_identity(self):
        session, contrast, route, frozen, context = synthetic_session()
        result = trajectory.evaluate_route(session, contrast, route, frozen, context, self_sender=True, batch_size=2)
        self.assertTrue(all(abs(r["raw_margin_change"]) < 1e-6 for r in result["rows"]))


if __name__ == "__main__":
    unittest.main()
