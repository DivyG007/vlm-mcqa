"""Pure-data recipe compilation. No model loading or experiment execution.

The compiler preserves historical head order and does not select sites. Image
layers and execution identity must be explicitly supplied by a future protocol.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import re
from typing import Any, Mapping

from .data import digest

QWEN_CORRECTED_M3_FREEZE = "4f991401f5cb7239"
QWEN_CORRECTED_DATASET = "f5573e25f3aa2bf2"


def load_frozen_heads(path: str | Path, *, model_key: str, dataset_hash: str) -> dict[str, Any]:
    """Accept actual M3 freeze or corrected compact Qwen M3, never localization only.

    Compact Qwen source lacks model/data fields; its known release schema/hash
    permits only the exact corrected Qwen model/data binding, retained explicitly
    rather than pretending those fields existed in the source.
    """
    payload = Path(path).read_bytes()
    frozen = json.loads(payload)
    freeze_hash = frozen.get("freeze_hash")
    if not freeze_hash:
        raise ValueError("head source requires freeze_hash")
    if model_key == "qwen25vl_7b" and freeze_hash != QWEN_CORRECTED_M3_FREEZE:
        raise ValueError("Qwen requires corrected pair-balanced M3 freeze")
    compact = "frozen_candidates" in frozen and "head_candidates" not in frozen
    if compact:
        if (model_key != "qwen25vl_7b" or dataset_hash != QWEN_CORRECTED_DATASET
                or freeze_hash != QWEN_CORRECTED_M3_FREEZE
                or frozen.get("contrast_type") != "position" or frozen.get("label_scheme") != "letters"
                or frozen.get("semantic_pairs_per_split") != 64
                or frozen.get("directed_contrasts_per_split") != 96
                or not str(frozen.get("method", "")).startswith("M3 pair-balanced")):
            raise ValueError("metadata-free compact source is only the corrected Qwen M3 release schema")
    elif "model_key" not in frozen or "dataset_hash" not in frozen:
        raise ValueError("actual head freeze requires explicit model_key and dataset_hash")
    if frozen.get("model_key", model_key) != model_key or frozen.get("dataset_hash", dataset_hash) != dataset_hash:
        raise ValueError("head source model/data mismatch")
    if "head_candidates" in frozen:
        candidates = frozen["head_candidates"]
    elif "frozen_candidates" in frozen:
        candidates = []
        for name in frozen["frozen_candidates"]:
            match = re.fullmatch(r"L(\d+)H(\d+)", name)
            if match is None:
                raise ValueError(f"invalid frozen head {name}")
            candidates.append([int(match[1]), int(match[2])])
    else:
        raise ValueError("localization-only freeze is not a frozen M3 head source")
    if len(candidates) < 16 or any(len(h) != 2 or any(type(x) is not int or x < 0 for x in h) for h in candidates):
        raise ValueError("requires at least sixteen valid frozen candidates")
    if len({tuple(h) for h in candidates}) != len(candidates):
        raise ValueError("duplicate frozen heads")
    layers = frozen.get("head_layers")
    if not layers or any(type(x) is not int or x < 0 for x in layers) or len(layers) != len(set(layers)):
        raise ValueError("requires distinct frozen head_layers")
    if not {h[0] for h in candidates}.issubset(layers):
        raise ValueError("candidate outside frozen reference layers")
    return {"model_key": model_key, "dataset_hash": dataset_hash, "freeze_hash": freeze_hash,
            "head_candidates": candidates, "head_layers": layers,
            "source_path": str(Path(path).resolve()), "source_sha256": digest(payload),
            "model_data_binding": "known_corrected_qwen_compact_release" if compact else "source"}


def compile_recipe(frozen: Mapping[str, Any], *, num_heads: int, num_layers: int,
                   image_layer: int, image_site: str, seed: int, random_sets: int = 20,
                   expanded_controls: bool = False) -> dict[str, Any]:
    """Prespecified top4/top16 + exact-size per-layer matched random controls.

    Random sets exclude their corresponding top-k, matching existing M3.
    All-head and attention references span ALL frozen head_layers. Image
    restoration precedes every clamped head; no layer is selected from effects.
    """
    if type(num_heads) is not int or num_heads <= 0 or type(num_layers) is not int or num_layers <= 0:
        raise ValueError("explicit positive model dimensions required")
    layers = list(frozen["head_layers"])
    candidates = [list(h) for h in frozen["head_candidates"]]
    if any(l >= num_layers for l in layers) or any(h >= num_heads for _, h in candidates):
        raise ValueError("frozen heads outside model dimensions")
    if image_site not in ("resid_pre", "resid_post"):
        raise ValueError("explicit image_site resid_pre or resid_post required")
    latest_image_layer = min(layers) - (image_site == "resid_post")
    if type(image_layer) is not int or not 0 <= image_layer <= latest_image_layer:
        raise ValueError("image restoration boundary must precede every frozen reference head")
    if type(random_sets) is not int or random_sets < 1:
        raise ValueError("random_sets must be positive")
    rng = random.Random(seed)
    sets: dict[str, list[list[int]]] = {}
    for k in (4, 16):
        chosen = candidates[:k]
        if len(chosen) != k:
            raise ValueError("incomplete frozen top16")
        sets[f"top{k}"] = chosen
        if k == 16 and not expanded_controls:
            continue
        for draw in range(random_sets):
            sampled = []
            for layer, count in sorted(Counter(l for l, _ in chosen).items()):
                pool = [h for h in range(num_heads) if [layer, h] not in chosen]
                if len(pool) < count:
                    raise ValueError(f"cannot match {count} random heads in layer {layer}")
                sampled.extend([layer, h] for h in rng.sample(pool, count))
            sets[f"random{k}_{draw:02d}"] = sampled
    sets["all_heads"] = [[l, h] for l in layers for h in range(num_heads)]
    conditions = [{"name": "clean", "operation": "clean"},
                  {"name": "image_restore", "operation": "image_restore"},
                  {"name": "self_image", "operation": "self_image"},
                  {"name": "final_residual", "operation": "final_residual", "layer": num_layers - 1}]
    for name in sets:
        conditions.extend([
            {"name": f"head_transfer/{name}", "operation": "head_transfer", "head_set": name},
            {"name": f"restore_clamp/{name}", "operation": "restore_clamp", "head_set": name},
        ])
        if expanded_controls:
            conditions.append({"name": f"self_head/{name}", "operation": "self_head", "head_set": name})
    if not expanded_controls:
        conditions.append({"name": "self_head_shared_sham/top4", "operation": "self_head", "head_set": "top4"})
    conditions.append({"name": "attention_reference", "operation": "attention_reference", "layers": layers})
    recipe = {"schema": "stretch-semantic-image-v1", "model_key": frozen["model_key"],
              "dataset_hash": frozen["dataset_hash"], "freeze_hash": frozen["freeze_hash"],
              "head_source_sha256": frozen["source_sha256"], "num_heads": num_heads, "num_layers": num_layers,
              "image_layer": image_layer, "image_site": image_site,
              "head_layers": layers, "seed": seed, "random_sets": random_sets,
              "expanded_controls": expanded_controls, "condition_count": len(conditions),
              "sets": sets, "conditions": conditions, "sample_unit": "semantic_pair",
              "readout": "donor_label_minus_recipient_label", "runtime_validation": "pending",
              "primary_head_set": "top4", "secondary_head_set": "top16",
              "sampling_and_confirmation_freeze": "pending_separate_protocol"}
    recipe["recipe_sha256"] = digest(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode())
    return recipe
