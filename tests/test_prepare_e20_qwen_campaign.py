import unittest
from pathlib import Path

from vlm_mcqa.campaigns.config import validate_config
from scripts.local.prepare_e20_qwen_campaign import (
    DATASET_HASH,
    IMAGE_AWARE_HASH,
    campaign_document,
    explicit_two_gpu_map,
    protocol_document,
)


class PrepareE20QwenCampaignTests(unittest.TestCase):
    def test_explicit_map_covers_two_gpus_and_all_decoder_layers(self):
        placement = explicit_two_gpu_map()
        self.assertEqual(set(placement.values()), {"cuda:0", "cuda:1"})
        self.assertEqual(placement["model.visual"], "cuda:0")
        self.assertEqual(placement["model.language_model.rotary_emb"], "cuda:0")
        self.assertEqual(placement["model.language_model.layers.12"], "cuda:0")
        self.assertEqual(placement["model.language_model.layers.13"], "cuda:1")
        self.assertEqual(placement["model.language_model.layers.27"], "cuda:1")
        self.assertEqual(sum(key.startswith("model.language_model.layers.") for key in placement), 28)

    def test_protocol_excludes_calibration_pair_from_discovery_only(self):
        protocol = protocol_document("a" * 64, "pair_calibration")
        self.assertEqual(protocol["primary_upstream_site"], {
            "kind": "resid_post", "layer": 0,
            "selection": "proposal_fixed_not_outcome_selected",
        })
        self.assertEqual(protocol["sampling"]["discovery"]["excluded_pair_ids"], ["pair_calibration"])
        self.assertEqual(protocol["sampling"]["confirmation"]["excluded_pair_ids"], [])

    def test_campaign_is_float16_two_gpu_calibration_input(self):
        root = Path("/source")
        calibration = {"contrast_id": "c", "donor_variant_id": "d", "recipient_variant_id": "r"}
        config = campaign_document(
            source_root=root, source_manifest=Path("/prepared/source.json"),
            checkpoint_root=Path("/checkpoint"), checkpoint_manifest=Path("/prepared/checkpoint.json"),
            dataset_root=Path("/dataset"), decision_path=Path("/prepared/decision.json"),
            decision_sha256="b" * 64, protocol_path=Path("/prepared/protocol.json"),
            recipe_path=Path("/prepared/recipe.json"), contrasts_path=Path("/prepared/contrasts.json"),
            head_source=Path("/source/heads.json"), calibration=calibration,
            output_placeholder=Path("/results/placeholder"),
        )
        self.assertEqual(config["deployment"]["dtype"], "float16")
        self.assertEqual(set(config["deployment"]["device_map"].values()), {"cuda:0", "cuda:1"})
        self.assertEqual(config["deployment"]["offload_policy"], "forbid")
        self.assertEqual(config["data"]["dataset_hash"], DATASET_HASH)
        self.assertEqual(config["data"]["image_aware_release_hash"], IMAGE_AWARE_HASH)
        self.assertEqual(config["study"]["phases"], ["discovery", "confirmation"])
        validate_config(config)


if __name__ == "__main__":
    unittest.main()
