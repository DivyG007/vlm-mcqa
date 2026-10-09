"""Capture residual stages at the frozen fit positions into resumable shards.

One full VLM forward per prompt, with Core-7 hooks capturing ``resid_pre:0``
and every ``resid_post`` at the fit positions plus the final position. The
final position is not stored for fitting; it only checks per prompt that the
final-stage lens reproduces the model logits.

Layout: ``<cache>/<split>/shard_XXXXX.safetensors`` with one ``[T, d]`` bf16
tensor per stage (``stage_00`` .. ``stage_{L}``) and a sibling ``.json`` with
per-variant offsets. Shards are written atomically and skipped on resume, so a
pre-empted spot instance continues where it stopped.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from ..core7.hooks import Site, hooked
from .positions import POSITION_RULE, fit_positions
from .translator import file_sha256


def stage_sites(num_layers: int) -> list[Site]:
    return [Site("resid_pre", 0)] + [Site("resid_post", layer) for layer in range(num_layers)]


def shard_paths(directory: Path, index: int) -> tuple[Path, Path]:
    base = Path(directory) / f"shard_{index:05d}"
    return base.with_suffix(".safetensors"), base.with_suffix(".json")


def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


@torch.inference_mode()
def capture_variant(adapter: Any, variant: Mapping[str, Any], image: Any) -> dict[str, Any]:
    """Return ``stages`` ``[L+1, P, d]`` (fp32 CPU) at fit positions and the final-stage gate error."""
    inputs, _ = adapter.encode(image, variant["prompt"])
    ids = inputs["input_ids"][0].tolist()
    groups = adapter.token_groups(ids, variant)
    positions = fit_positions(ids, groups, adapter.tokenizer, num_labels=len(variant["labels"]))
    final = len(ids) - 1
    sites = stage_sites(adapter.num_layers)
    with hooked(adapter.handle, capture=sites, capture_positions=[*positions, final]) as captured:
        output = adapter.model(**inputs, use_cache=False, return_dict=True)
    stages = torch.stack([captured[site.key][0] for site in sites])  # [L+1, P+1, d]
    final_lens = adapter.handle.readout(stages[-1, -1:]).float().cpu()[0]
    model_logits = output.logits[0, -1].float().cpu()
    gate = float((final_lens - model_logits).abs().max())
    return {"stages": stages[:, :-1], "positions": positions, "num_tokens": len(ids), "final_gate": gate}


def build_cache(adapter: Any, load_image: Callable[[str], Any], variants: Sequence[Mapping[str, Any]],
                out_dir: Path, *, shard_size: int, tracker: Any = None,
                log: Callable[[str], None] = print) -> dict[str, Any]:
    """Cache one split; returns per-shard records (existing complete shards are reused)."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_layers = adapter.num_layers
    shards, totals = [], {"variants": 0, "skipped": 0, "tokens": 0, "max_final_gate": 0.0}
    started = time.time()
    for index, start in enumerate(range(0, len(variants), shard_size)):
        chunk = variants[start:start + shard_size]
        tensor_path, meta_path = shard_paths(out_dir, index)
        if tensor_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("variant_ids") == [v["variant_id"] for v in chunk]:
                shards.append(meta)
                _accumulate(totals, meta)
                continue
            raise ValueError(f"existing {meta_path} was built from different variants; use a fresh cache dir")
        stage_rows: list[list[torch.Tensor]] = [[] for _ in range(num_layers + 1)]
        records, skipped, offset = [], [], 0
        shard_started = time.time()
        for variant in chunk:
            try:
                got = capture_variant(adapter, variant, load_image(variant["image"]))
            except ValueError as error:  # labels not locatable: record, never silently drop
                skipped.append({"variant_id": variant["variant_id"], "reason": str(error)})
                continue
            count = len(got["positions"])
            for s in range(num_layers + 1):
                stage_rows[s].append(got["stages"][s].to(torch.bfloat16))
            records.append({"variant_id": variant["variant_id"], "scene_id": variant["scene_id"],
                            "offset": offset, "count": count, "positions": got["positions"],
                            "num_tokens": got["num_tokens"], "final_gate": got["final_gate"]})
            offset += count
        if not records:
            raise RuntimeError(f"shard {index}: every variant was skipped: {skipped[:3]}")
        tensors = {f"stage_{s:02d}": torch.cat(stage_rows[s]).contiguous() for s in range(num_layers + 1)}
        tmp = tensor_path.with_name(tensor_path.name + ".tmp")
        save_file(tensors, str(tmp))
        os.replace(tmp, tensor_path)
        meta = {"shard": index, "variant_ids": [v["variant_id"] for v in chunk], "records": records,
                "skipped": skipped, "tokens": offset, "num_stages": num_layers + 1,
                "hidden_size": int(tensors["stage_00"].shape[1]), "position_rule": POSITION_RULE,
                "sha256": file_sha256(tensor_path), "seconds": time.time() - shard_started}
        _atomic_json(meta_path, meta)
        shards.append(meta)
        _accumulate(totals, meta)
        elapsed = time.time() - started
        log(f"[cache:{out_dir.name}] shard {index} tokens={offset} variants={len(records)} "
            f"skipped={len(skipped)} {meta['seconds']:.1f}s (total {elapsed / 60:.1f} min)")
        if tracker is not None:
            tracker.log({f"cache/{out_dir.name}/shard": index, f"cache/{out_dir.name}/tokens": totals["tokens"],
                         f"cache/{out_dir.name}/seconds_per_variant": meta["seconds"] / max(len(records), 1),
                         f"cache/{out_dir.name}/max_final_gate": totals["max_final_gate"],
                         "cache/peak_gpu_gib": _peak_gib()})
    return {"shards": [{k: m[k] for k in ("shard", "tokens", "sha256")} for m in shards], **totals}


def _accumulate(totals: dict[str, Any], meta: Mapping[str, Any]) -> None:
    totals["variants"] += len(meta["records"])
    totals["skipped"] += len(meta["skipped"])
    totals["tokens"] += meta["tokens"]
    totals["max_final_gate"] = max(totals["max_final_gate"],
                                   max((r["final_gate"] for r in meta["records"]), default=0.0))


def _peak_gib() -> float:
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def load_stage(split_dir: Path, stage: int) -> torch.Tensor:
    """Concatenate one stage over all shards of a split (``[T, d]`` bf16, CPU)."""
    from safetensors import safe_open

    split_dir = Path(split_dir)
    parts = []
    index = 0
    while True:
        tensor_path, meta_path = shard_paths(split_dir, index)
        if not tensor_path.exists():
            break
        if not meta_path.exists():
            raise RuntimeError(f"incomplete shard {tensor_path}")
        with safe_open(str(tensor_path), framework="pt") as handle:
            parts.append(handle.get_tensor(f"stage_{stage:02d}"))
        index += 1
    if not parts:
        raise FileNotFoundError(f"no cache shards in {split_dir}")
    return torch.cat(parts)
