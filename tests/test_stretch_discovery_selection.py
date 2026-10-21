import unittest

from vlm_mcqa.stretch.discovery_selection import select_failure_option_layer, select_task_image_layer


class DiscoverySelectionTests(unittest.TestCase):
    def test_e22_selects_best_valid_layer_and_earlier_tie(self):
        rows = []
        for layer, gain in ((20, 1.0), (23, 2.0), (25, 2.005)):
            for unit in ("a", "b", "c"):
                rows.extend([
                    {"condition": f"options_candidate_L{layer}", "case_id": unit,
                     "role": "failure", "raw_margin_change": gain,
                     "semantic_remapping_success": True, "donor_symbol_copy": False,
                     "readout": {"global_top_is_label": True}},
                    {"condition": f"options_candidate_L{layer}", "case_id": "control-" + unit,
                     "matched_failure_case_id": unit, "role": "matched_correct", "raw_margin_change": 0.0,
                     "readout": {"global_top_is_label": True}},
                    {"condition": f"options_candidate_self_L{layer}", "case_id": unit,
                     "role": "failure", "raw_margin_change": 0.0},
                ])
        result = select_failure_option_layer(rows, candidate_layers=[20, 23, 25], seed=0,
            draws=100, minimum_units=3, tie_tolerance=.01, identity_tolerance=.01)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["selected_layer"], 23)
        self.assertEqual(result["tied_layers"], [23, 25])

    def test_e22_rejects_copy_or_nonidentity_candidate(self):
        rows = []
        for unit in ("a", "b"):
            rows.extend([
                {"condition": "options_candidate_L20", "case_id": unit, "role": "failure",
                 "raw_margin_change": 2.0, "semantic_remapping_success": False,
                 "donor_symbol_copy": True, "readout": {"global_top_is_label": True}},
                {"condition": "options_candidate_L20", "case_id": "c" + unit,
                 "matched_failure_case_id": unit, "role": "matched_correct", "raw_margin_change": 0.0,
                 "readout": {"global_top_is_label": True}},
                {"condition": "options_candidate_self_L20", "case_id": unit,
                 "role": "failure", "raw_margin_change": .5},
            ])
        result = select_failure_option_layer(rows, candidate_layers=[20], seed=0, draws=100,
            minimum_units=2, tie_tolerance=.01, identity_tolerance=.01)
        self.assertEqual(result["status"], "insufficient")

    def test_e22_pairs_donor_expanded_failures_to_prospective_controls(self):
        rows = []
        for unit in ("a", "b"):
            rows.extend([
                {"condition": "options_candidate_L20", "case_id": unit + ":semantic_match",
                 "donor_class": "semantic_match", "role": "failure", "raw_margin_change": 1.0,
                 "semantic_remapping_success": True, "donor_symbol_copy": False,
                 "readout": {"global_top_is_label": True}},
                {"condition": "options_candidate_L20", "case_id": "control-" + unit,
                 "donor_class": "semantic_match", "matched_failure_case_id": unit,
                 "role": "matched_correct", "raw_margin_change": 0.0,
                 "readout": {"global_top_is_label": True}},
                {"condition": "options_candidate_self_L20", "case_id": unit + ":semantic_match",
                 "donor_class": "semantic_match", "role": "failure", "raw_margin_change": 0.0},
            ])
        result = select_failure_option_layer(rows, candidate_layers=[20], seed=0, draws=100,
            minimum_units=2, tie_tolerance=.01, identity_tolerance=.01)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["estimates"][20]["n_units"], 2)

    def test_e23_and_e24_cluster_repeated_rows_by_parent(self):
        spatial, visual = [], []
        for pair in ("a", "b"):
            for layer, effect in ((0, 1.0), (8, 2.0)):
                spatial.extend([
                    {"operation": f"zero-region-L{layer}", "region": "question", "pair_id": pair, "margin_change": -effect},
                    {"operation": f"zero-region-L{layer}", "region": "question-matched", "pair_id": pair, "margin_change": 0.0},
                    {"operation": f"self-region-L{layer}", "region": "question", "pair_id": pair, "margin_change": 0.0},
                ])
                visual.extend([
                    {"operation": f"donor-panel-L{layer}", "pair_id": pair,
                     "correct_correct_transfer": True, "content_vs_copy_identifiable": True,
                     "content_vs_literal_logit_change": effect},
                    {"operation": f"self-region-L{layer}", "pair_id": pair, "margin_change": 0.0},
                ])
        for study, rows in (("spatial", spatial), ("image_options", visual)):
            result = select_task_image_layer(study, rows, candidate_layers=[0, 8], seed=0,
                draws=100, minimum_units=2, tie_tolerance=.01, identity_tolerance=.01)
            self.assertEqual(result["selected_layer"], 8)


if __name__ == "__main__":
    unittest.main()
