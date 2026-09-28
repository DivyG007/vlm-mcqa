"""End-to-end Core-7 checks on tiny random VLMs built locally from configs.

No weights or processors are downloaded: each family's native Transformers
class is instantiated from a tiny config, and a character-level stub tokenizer
plus a family-specific input builder replace the real processor. This checks
hook sites, decoder replay (including multimodal RoPE and Gemma2 sandwich
norms/softcapping), batching, every calibration gate, and that each Core-7
stage runs and writes its artifacts. It says nothing about real-model results.
"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
import warnings
from pathlib import Path

from PIL import Image

from support import HAS_TRANSFORMERS, requires_transformers

if HAS_TRANSFORMERS:
    import torch
    import transformers

VOCAB = 300
IMG, VSTART, VEND = 250, 251, 252
CONTENTS = ["red", "blue", "green", "yellow"]


class CharTokenizer:
    """Printable characters -> ids 3..; ids >= 250 are image/special tokens."""

    unk_token_id = 0
    all_special_ids = [0, 1, 2, IMG, VSTART, VEND]

    def __init__(self) -> None:
        chars = [chr(c) for c in range(32, 127)] + ["\n"]
        self.to_id = {ch: i + 3 for i, ch in enumerate(chars)}
        self.to_ch = {i: ch for ch, i in self.to_id.items()}

    def encode(self, text, add_special_tokens=False):
        return [self.to_id[ch] for ch in text]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self.to_ch.get(int(i), "" if skip_special_tokens else "\x00") for i in ids)

    def convert_tokens_to_ids(self, token):
        return self.unk_token_id


def _text(**extra):
    base = dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    base.update(extra)
    return base


def _pixels(image, shape):
    seed = int(sum(image.resize((4, 4)).convert("L").getdata())) + 1
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def build(family):
    from transformers import (InternVLConfig, InternVLForConditionalGeneration, LlavaOnevisionConfig,
                              LlavaOnevisionForConditionalGeneration, PaliGemmaConfig,
                              PaliGemmaForConditionalGeneration, Qwen2_5_VLConfig,
                              Qwen2_5_VLForConditionalGeneration)

    torch.manual_seed(0)
    if family == "qwen_vl":
        cfg = Qwen2_5_VLConfig(
            text_config=_text(rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=14,
                               spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
                               fullatt_block_indexes=[0], window_size=56),
            image_token_id=IMG, vision_start_token_id=VSTART, vision_end_token_id=VEND, video_token_id=253)
        model = Qwen2_5_VLForConditionalGeneration(cfg)

        def encode(image, ids):
            ids = [VSTART] + [IMG] * 4 + [VEND] + ids
            return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                        pixel_values=_pixels(image, (16, 3 * 2 * 14 * 14)), image_grid_thw=torch.tensor([[1, 4, 4]]))
    elif family == "paligemma":
        cfg = PaliGemmaConfig(
            text_config=dict(model_type="gemma2", **_text(head_dim=16, query_pre_attn_scalar=16, sliding_window=16,
                                                          final_logit_softcapping=30.0, attn_logit_softcapping=50.0)),
            vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                               image_size=28, patch_size=14, projection_dim=64),
            image_token_index=IMG, projection_dim=64)
        cfg.text_config.vocab_size = VOCAB
        model = PaliGemmaForConditionalGeneration(cfg)

        def encode(image, ids):
            ids = [IMG] * 4 + [2] + ids
            return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                        pixel_values=_pixels(image, (1, 3, 28, 28)),
                        token_type_ids=torch.zeros(1, len(ids), dtype=torch.long))
    elif family == "internvl":
        cfg = InternVLConfig(
            text_config=dict(model_type="qwen3", **_text(head_dim=16)),
            vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                               image_size=[28, 28], patch_size=[14, 14]),
            image_token_id=IMG, downsample_ratio=0.5, projector_hidden_act="gelu")
        model = InternVLForConditionalGeneration(cfg)

        def encode(image, ids):
            ids = [VSTART, IMG, VEND] + ids
            return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                        pixel_values=_pixels(image, (1, 3, 28, 28)))
    else:
        cfg = LlavaOnevisionConfig(
            text_config=dict(model_type="qwen2", **_text()),
            vision_config=dict(model_type="siglip_vision_model", hidden_size=32, intermediate_size=64,
                               num_hidden_layers=2, num_attention_heads=2, image_size=28, patch_size=14),
            image_token_index=IMG, vision_feature_layer=-1, image_grid_pinpoints=[[28, 28]],
            vision_aspect_ratio="anyres_max_1")
        model = LlavaOnevisionForConditionalGeneration(cfg)

        def encode(image, ids):
            ids = [IMG] * 10 + ids
            return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                        pixel_values=_pixels(image, (1, 2, 3, 28, 28)), image_sizes=torch.tensor([[28, 28]]))
    return model.eval(), encode


def make_adapter(family):
    from vlm_mcqa.core7.adapters import ModelSpec, VLMAdapter

    model, build_inputs = build(family)
    tokenizer = CharTokenizer()

    class Processor:
        pass

    processor = Processor()
    processor.tokenizer = tokenizer
    spec = ModelSpec(key=f"tiny_{family}", family=family, hf_id="local/tiny", params_b=0.001,
                     lm_backbone={"qwen_vl": "qwen2.5", "paligemma": "gemma2", "internvl": "qwen3",
                                  "llava_onevision": "qwen2"}[family], dtype="float32", device_map="cpu")

    class TinyAdapter(VLMAdapter):
        def encode(self, image, prompt):
            return build_inputs(image, tokenizer.encode(prompt + "\n")), prompt

    return TinyAdapter(model, processor, spec)


def make_dataset(root: Path) -> None:
    from vlm_mcqa.clevr_mcq4.build import expand

    rng = random.Random(0)
    (root / "images").mkdir(parents=True)
    items = []
    for i in range(6):
        split = "discovery" if i < 3 else "confirmation"
        options = list(CONTENTS)
        rng.shuffle(options)
        for member, answer in (("x", options[0]), ("y", options[1])):
            image = f"images/p{i}_{member}.png"
            Image.new("RGB", (28, 28), (rng.randrange(256), rng.randrange(256), rng.randrange(256))).save(root / image)
            items.append({"item_id": f"p{i}_{member}", "split": split, "scene_id": f"s{i}",
                          "counterfactual_pair_id": f"pair{i}", "pair_member": member, "image": image,
                          "question": "Which color?", "option_contents": options, "semantic_answer": answer,
                          "question_family": "count_compare", "program_depth": 3})
    for i, split in enumerate(["screening"] * 4 + ["behavior"] * 3):
        image = f"images/s{i}.png"
        Image.new("RGB", (28, 28), (i * 30, 100, 200 - i * 20)).save(root / image)
        items.append({"item_id": f"single{i}", "split": split, "scene_id": f"single{i}",
                      "counterfactual_pair_id": None, "pair_member": None, "image": image,
                      "question": "What color is it?", "option_contents": list(CONTENTS),
                      "semantic_answer": CONTENTS[i % 4], "question_family": "count", "program_depth": 2})
    expand(root, items, seed=0)


PROFILE = {"screen_items": 4, "robust_items": 2, "compliance_items": 2, "behavior_items": 2, "pairs": 3,
           "contrast_limit_per_type": 2, "layer_stride": 1, "group_layer_stride": 2, "batch_size": 3,
           "ablation_recipients": 2, "head_layers": 2, "head_screen_contrasts": 2, "head_candidates": 4,
           "topk": [1, 2], "random_sets": 2, "schemes": ["letters", "numbers", "rare_letters"],
           "token_groups": ["image", "question", "options", "labels"], "require_correct": False}


@requires_transformers
class TinyModelCore7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warnings.filterwarnings("ignore")
        transformers.logging.set_verbosity_error()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name) / "data"
        make_dataset(cls.data)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _session(self, family, name):
        from vlm_mcqa.core7.session import Session, provenance

        adapter = make_adapter(family)
        session = Session(adapter, self.data, Path(self.tmp.name) / name, PROFILE, seed=0)
        session.write_json("run.json", provenance({
            "model_key": adapter.spec.key, "family": family, "params_b": 0.001,
            "lm_backbone": adapter.spec.lm_backbone, "num_layers": adapter.num_layers,
            "adapter": adapter.describe()}))
        return session

    def test_hooks_and_calibration_pass_for_all_families(self) -> None:
        from vlm_mcqa.core7.calibration import run_calibration

        for family in ("qwen_vl", "paligemma", "internvl", "llava_onevision"):
            with self.subTest(family=family):
                session = self._session(family, f"cal_{family}")
                report = run_calibration(session, num_variants=2)
                failed = {k: g for k, g in report["gates"].items() if not g["passed"]}
                self.assertFalse(failed, f"{family}: {failed}")
                if family == "paligemma":
                    self.assertTrue(report["adapter"]["sandwich_norm"])
                self.assertLess(report["informational"]["direct_projection_additivity_max_error"], 1e-3)

    def test_final_residual_patch_matches_donor_and_restores(self) -> None:
        from vlm_mcqa.core7.hooks import Intervention, Site

        session = self._session("qwen_vl", "patch_qwen")
        variants = [v for v in session.variants.values() if v["split"] == "discovery"][:2]
        a, b = (session.prepared(v["variant_id"]) for v in variants)
        donor = session.clean(variants[1]["variant_id"])
        last = session.adapter.num_layers - 1
        patched, _ = session.adapter.decoder_forward(a, interventions=[
            Intervention(Site("resid_post", last), [a.final_position], donor.sites[f"resid_post:{last}"])])
        # Patching the final residual reproduces the donor's final-token logits exactly.
        self.assertLess(float((patched[0] - donor.logits).abs().max()), 1e-4)
        clean, _ = session.adapter.decoder_forward(a)
        self.assertLess(float((clean[0] - session.clean(variants[0]["variant_id"]).logits).abs().max()), 1e-5)

    def test_all_core7_stages_write_artifacts(self) -> None:
        from vlm_mcqa.core7.aggregate import aggregate_run, confirm, freeze
        from vlm_mcqa.core7.behavior import run_behavior
        from vlm_mcqa.core7.calibration import run_calibration
        from vlm_mcqa.core7.components import run_components
        from vlm_mcqa.core7.heads import run_heads
        from vlm_mcqa.core7.lens import run_lens
        from vlm_mcqa.core7.patching import run_patching
        from vlm_mcqa.core7.plots import plot_run, plot_screening
        from vlm_mcqa.core7.report import select

        for family in ("paligemma", "internvl"):
            with self.subTest(family=family):
                session = self._session(family, f"run_{family}")
                out = session.out_dir
                calibration = run_calibration(session, num_variants=1)
                self.assertTrue(all(g["value"] <= g["limit"] for g in calibration["gates"].values()
                                    if g["passed"]))
                self.assertTrue(all(g["checks"] for g in calibration["gates"].values()))
                beh = run_behavior(session, split="screening", n_items=4, robust_items=2, compliance_items=2,
                                   pair_splits=("discovery", "confirmation"))
                self.assertIn("worst_position_accuracy", beh)
                self.assertIn("discovery", beh["pair_yield"])
                lens = run_lens(session, split="behavior", n_items=2)
                self.assertEqual(lens["stage_rows"], lens["variants"] * (session.adapter.num_layers + 1))
                patch = run_patching(session, split="discovery")
                self.assertGreater(patch["m1_rows"], 0)
                self.assertGreater(patch["group_patching"]["rows"], 0)
                run_components(session, split="discovery")
                heads = run_heads(session, split="discovery", freeze=None)
                self.assertIn(heads["verdict"][0], ("sparse", "distributed"))
                self.assertEqual(heads["cases"]["limited_case_selection"], "round_robin_across_pairs")
                self.assertEqual(heads["cases"]["selected_pairs_by_type"]["position"], 2)
                self.assertEqual(heads["per_k"]["1"]["top_k"]["n_clusters"], 2)
                self.assertEqual(heads["per_k"]["1"]["top_k"]["sampling_unit"], "pair")
                agg = aggregate_run(out, "discovery")
                self.assertIn("attn_curve", agg)
                frozen = freeze(out, out / "freeze.json")
                self.assertEqual(frozen["head_candidates"], heads["candidates"])
                run_patching(session, split="confirmation")
                run_components(session, split="confirmation")
                run_heads(session, split="confirmation", freeze=frozen)
                report = confirm(out, out / "freeze.json")
                self.assertIn("position_transfer_layer", report["checks"])
                figures = plot_run(out)
                self.assertIn("m1_residual_discovery.png", figures)
                self.assertIn("m3_topk_confirmation.png", figures)
                rows = [json.loads(l) for l in (out / "patching_discovery.jsonl").read_text().splitlines()]
                kinds = {r["contrast_type"] for r in rows}
                self.assertTrue({"position", "content", "content_and_position", "identity"} <= kinds, kinds)
                decision = select([out], {"min_worst_position_accuracy": 0.9,
                                          "min_proposal_robustness_accuracy": 0.45, "min_shuffled_gap": 0.3,
                                          "min_usable_discovery_pairs": 64, "min_usable_confirmation_pairs": 48,
                                          "max_params_b": 10.0, "comparable_accuracy_window": 0.03}, out / "sel")
                self.assertTrue(decision["family_winners"][family]["provisional"])
                self.assertIn("screening_robustness_cells.png", plot_screening(decision, out / "sel"))
                from vlm_mcqa.core7.plots import plot_compare
                from vlm_mcqa.core7.report import compare
                comparison = compare([out], out / "cmp")
                self.assertEqual(len(comparison["models"]), 1)
                self.assertIn("shared", comparison["completion"][0]["M4 logit lens"])
                self.assertNotIn("confirmation", comparison["completion"][0]["M4 logit lens"])
                plot_compare(comparison, out / "cmp")


if __name__ == "__main__":
    unittest.main()
