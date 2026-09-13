"""Dataset access, cached preparations, clean site caches and run provenance."""

from __future__ import annotations

import json
import os
import platform
import random
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from ..clevr_mcq4.variants import read_jsonl
from .adapters import Prepared, VLMAdapter
from .hooks import Site, all_sites
from .scoring import symbol_readout

DEFAULT_PROFILES = Path(__file__).resolve().parents[3] / "configs/core7/profiles.json"
CLEAN_SITE_KINDS = ("resid_pre", "resid_post", "attn", "mlp", "head")


def load_profiles(path: Path | None = None) -> dict[str, Any]:
    return json.loads(Path(path or DEFAULT_PROFILES).read_text())


@dataclass
class Clean:
    """Clean decoder replay of one variant: final-token logits and site values."""

    logits: torch.Tensor           # [V] float32 CPU (replay path)
    sites: dict[str, torch.Tensor]  # site key -> [width] float32 CPU at the final token
    readout: dict[str, Any]         # symbol_readout over the variant's own labels

    @property
    def correct(self) -> bool:
        return bool(self.readout["correct"])


class LRU(OrderedDict):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.capacity:
            self.popitem(last=False)


class Session:
    def __init__(self, adapter: VLMAdapter, data_dir: Path, out_dir: Path, profile: Mapping[str, Any],
                 *, seed: int = 0, prepared_cache: int = 24, clean_cache: int = 512):
        self.adapter = adapter
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.profile = dict(profile)
        self.seed = seed
        self.rng = random.Random(seed)
        self.items = {i["item_id"]: i for i in read_jsonl(self.data_dir / "items.jsonl")}
        self.variants = {v["variant_id"]: v for v in read_jsonl(self.data_dir / "variants.jsonl")}
        contrasts_path = self.data_dir / "contrasts.jsonl"
        self.contrasts = read_jsonl(contrasts_path) if contrasts_path.exists() else []
        manifest_path = self.data_dir / "dataset_manifest.json"
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        release_path = self.data_dir / "release_manifest.json"
        self.release_manifest = json.loads(release_path.read_text()) if release_path.exists() else {}
        self._prepared = LRU(prepared_cache)
        self._clean = LRU(clean_cache)
        self._images: LRU = LRU(64)
        self.sites = all_sites(adapter.num_layers, CLEAN_SITE_KINDS)

    # ------------------------------------------------------------------ data

    def image(self, relative: str):
        from PIL import Image

        if relative not in self._images:
            with Image.open(self.data_dir / relative) as loaded:
                self._images.put(relative, loaded.convert("RGB"))
        return self._images[relative]

    def items_in(self, split: str, limit: int | None = None) -> list[dict[str, Any]]:
        items = sorted((i for i in self.items.values() if i["split"] == split), key=lambda i: i["item_id"])
        return items[:limit] if limit else items

    def variants_of(self, item_id: str, *, schemes: Sequence[str] | None = None,
                    kind: str = "position") -> list[dict[str, Any]]:
        out = [v for v in self._by_item()[item_id] if v["variant_kind"] == kind
               and (schemes is None or v["label_scheme"] in schemes)]
        return sorted(out, key=lambda v: (v["label_scheme"], v["correct_index"]))

    def _by_item(self) -> dict[str, list[dict[str, Any]]]:
        if not hasattr(self, "_by_item_cache"):
            table = defaultdict(list)
            for v in self.variants.values():
                table[v["item_id"]].append(v)
            self._by_item_cache = table
        return self._by_item_cache

    def pair_ids(self, split: str) -> list[str]:
        return sorted({i["counterfactual_pair_id"] for i in self.items.values()
                       if i["split"] == split and i.get("counterfactual_pair_id")})

    # ------------------------------------------------------------------ model calls

    def prepared(self, variant_id: str, *, image_override: str | None = None) -> Prepared:
        key = (variant_id, image_override)
        if key not in self._prepared:
            variant = self.variants[variant_id]
            image = self.image(image_override or variant["image"])
            self._prepared.put(key, self.adapter.prepare(variant, image))
        return self._prepared[key]

    def label_ids(self, prepared: Prepared, labels: Sequence[str]) -> list[int]:
        ids = [prepared.label_ids.get(label) for label in labels]
        if any(i is None for i in ids):
            raise ValueError(f"{prepared.variant_id}: labels {labels} are not single-token continuations")
        return ids  # type: ignore[return-value]

    def clean(self, variant_id: str) -> Clean:
        if variant_id not in self._clean:
            prepared = self.prepared(variant_id)
            variant = self.variants[variant_id]
            logits, captured = self.adapter.decoder_forward(
                prepared, capture=self.sites, capture_positions=[prepared.final_position])
            logits = logits[0].cpu()
            readout = symbol_readout(logits, self.label_ids(prepared, variant["labels"]), variant["correct_index"])
            sites = {k: v[0, 0] for k, v in captured.items()}
            self._clean.put(variant_id, Clean(logits, sites, readout))
        return self._clean[variant_id]

    def site_value(self, variant_id: str, site: Site) -> torch.Tensor:
        return self.clean(variant_id).sites[site.key]

    # ------------------------------------------------------------------ output

    def jsonl(self, name: str):
        return (self.out_dir / name).open("w", encoding="utf-8")

    def write_json(self, name: str, value: Any) -> None:
        (self.out_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")

    def batches(self, values: Sequence[Any]) -> Iterator[list[Any]]:
        size = int(self.profile.get("batch_size", 1))
        for start in range(0, len(values), size):
            yield list(values[start:start + size])

    def layers(self, stride_key: str = "layer_stride") -> list[int]:
        stride = int(self.profile.get(stride_key, 1))
        n = self.adapter.num_layers
        layers = list(range(0, n, stride))
        if layers[-1] != n - 1:
            layers.append(n - 1)
        return layers


def write_jsonl_row(stream, row: Mapping[str, Any]) -> None:
    stream.write(json.dumps(row, sort_keys=True, default=float) + "\n")


def provenance(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[3]
    try:
        from .source_snapshot import source_state

        commit, dirty = source_state(repo)
    except (OSError, ValueError):
        commit, dirty = os.environ.get("SOURCE_GIT_COMMIT", "unknown"), None
    import transformers

    info = {
        "source_git_commit": commit,
        "source_tree_dirty": dirty,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "started_unix": time.time(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    info.update(extra or {})
    return info
