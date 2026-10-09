"""Fitting corpus: held-out CLEVR scenes, a seeded scene-level train/val split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..clevr_mcq4.variants import read_jsonl

EVALUATION_SPLITS = ("behavior", "discovery", "confirmation")


def _scene_key(seed: int, scene_id: str) -> str:
    return hashlib.sha256(f"{seed}:{scene_id}".encode()).hexdigest()


def load_corpus(data_dir: Path, config: Mapping[str, Any], *, pilot: bool = False) -> dict[str, Any]:
    """Select corpus variants and split scenes into train/val.

    The whole scene (all alphabets and positions) stays on one side. With
    ``pilot`` only ``pilot_train_scenes``/``pilot_val_scenes`` from the front of
    each frozen ordering are used; a pilot is exploratory and never replaces
    the full fit.
    """
    data_dir = Path(data_dir)
    dataset = json.loads((data_dir / "dataset_manifest.json").read_text())
    release = json.loads((data_dir / "release_manifest.json").read_text()) \
        if (data_dir / "release_manifest.json").exists() else {}
    for key, actual in (("dataset_hash", dataset.get("dataset_hash")),
                        ("image_aware_release_hash", release.get("image_aware_release_hash"))):
        expected = config.get(key)
        if expected and actual != expected:
            raise ValueError(f"{key} mismatch: config {expected} vs data {actual}")
    items = {i["item_id"]: i for i in read_jsonl(data_dir / "items.jsonl")}
    splits = set(config["corpus_splits"])
    if splits & set(EVALUATION_SPLITS):
        raise ValueError("corpus may not include Core-7 evaluation splits")
    corpus_items = {k: v for k, v in items.items() if v["split"] in splits}
    held_out_scenes = {v["scene_id"] for v in items.values() if v["split"] in EVALUATION_SPLITS}
    held_out_images = {v["image"] for v in items.values() if v["split"] in EVALUATION_SPLITS}
    scenes = sorted({v["scene_id"] for v in corpus_items.values()})
    leak = (set(scenes) & held_out_scenes) | ({v["image"] for v in corpus_items.values()} & held_out_images)
    if leak:
        raise ValueError(f"corpus overlaps evaluation scenes/images: {sorted(leak)[:5]}")
    seed = int(config["seed"])
    ordered = sorted(scenes, key=lambda s: _scene_key(seed, s))
    n_val = int(config["val_scenes"])
    val_scenes, train_scenes = ordered[:n_val], ordered[n_val:]
    if pilot:
        train_scenes = train_scenes[:int(config["pilot_train_scenes"])]
        val_scenes = val_scenes[:int(config["pilot_val_scenes"])]
    schemes = set(config["schemes"])
    by_scene: dict[str, list[dict[str, Any]]] = {s: [] for s in scenes}
    for variant in read_jsonl(data_dir / "variants.jsonl"):
        item = corpus_items.get(variant["item_id"])
        if item and variant["variant_kind"] == "position" and variant["label_scheme"] in schemes:
            by_scene[item["scene_id"]].append(variant)
    def pick(chosen):
        out = [v for s in chosen for v in sorted(by_scene[s], key=lambda v: v["variant_id"])]
        # Seeded shuffle so every cache shard mixes scenes; order is reproducible.
        return sorted(out, key=lambda v: _scene_key(seed, v["variant_id"]))
    train, val = pick(train_scenes), pick(val_scenes)
    manifest = {
        "corpus_splits": sorted(splits), "seed": seed, "pilot": pilot,
        "dataset_hash": dataset.get("dataset_hash"),
        "image_aware_release_hash": release.get("image_aware_release_hash"),
        "train_scenes": train_scenes, "val_scenes": val_scenes,
        "train_variants": len(train), "val_variants": len(val),
        "schemes": sorted(schemes),
    }
    manifest["corpus_sha256"] = hashlib.sha256(json.dumps(
        {**manifest, "train_ids": [v["variant_id"] for v in train], "val_ids": [v["variant_id"] for v in val]},
        sort_keys=True).encode()).hexdigest()
    return {"train": train, "val": val, "items": items, "manifest": manifest}
