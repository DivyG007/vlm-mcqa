"""Validated processor geometry and bounded selected-attention operations.

Geometry is reconstructed per model family and must exactly match the visual
tokens emitted by the real processor.  A mismatch stops the run before any
scientific intervention rather than inventing a flattened patch layout.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import threading
from typing import Any, Mapping, Sequence

from .task_prep import Rect, TokenCell, ViewTransform, map_region

_ATTENTION_LOCK = threading.Lock()
SUPPORTED_GEOMETRY_FAMILIES = {"qwen_vl", "paligemma", "internvl", "llava_onevision"}


def require_supported_geometry(adapter: Any) -> None:
    family = adapter.spec.family
    if family not in SUPPORTED_GEOMETRY_FAMILIES or not hasattr(adapter.processor, "image_processor"):
        raise ValueError(f"unsupported or unavailable processor geometry: {family}")
    if family == "paligemma":
        if adapter.spec.attn_implementation != "eager":
            raise ValueError("PaliGemma/Gemma2 requires eager attention to preserve softcapping")
    elif adapter.spec.attn_implementation != "sdpa":
        raise ValueError(f"{family} selected-edge studies require explicit SDPA attention")


# Backward-compatible import name.  New callers use require_supported_geometry.
require_qwen_geometry = require_supported_geometry


def _visual(adapter: Any, inputs: Mapping[str, Any]) -> list[int]:
    positions = [i for i, token in enumerate(inputs["input_ids"][0].tolist())
                 if token in adapter.image_token_ids]
    if not positions:
        raise ValueError("processor output has no identifiable visual tokens")
    return positions


def _cells(positions: Sequence[int], view: ViewTransform, height: int, width: int) -> list[TokenCell]:
    if height < 1 or width < 1 or len(positions) != height * width:
        raise ValueError("visual positions do not form the declared grid")
    dx = (view.destination.x1 - view.destination.x0) / width
    dy = (view.destination.y1 - view.destination.y0) / height
    return [TokenCell(int(positions[r * width + c]), view.view_id,
                      Rect(view.destination.x0 + c * dx, view.destination.y0 + r * dy,
                           view.destination.x0 + (c + 1) * dx, view.destination.y0 + (r + 1) * dy))
            for r in range(height) for c in range(width)]


def _qwen(adapter: Any, inputs: Mapping[str, Any], image: Any):
    processor = adapter.processor.image_processor
    if getattr(processor, "do_center_crop", False) or getattr(processor, "do_pad", False):
        raise ValueError("Qwen crop/pad transforms need a separately validated bridge")
    raw = inputs.get("image_grid_thw")
    if raw is None or tuple(raw.shape) != (1, 3):
        raise ValueError("one explicit Qwen still-image grid required")
    temporal, height, width = [int(v) for v in raw[0].tolist()]
    merge = int(adapter.model.config.vision_config.spatial_merge_size)
    patch = int(adapter.model.config.vision_config.patch_size)
    if temporal != 1 or merge < 1 or patch < 1 or height % merge or width % merge:
        raise ValueError("unsupported temporal/merged Qwen grid")
    positions = _visual(adapter, inputs)
    gh, gw = height // merge, width // merge
    source = Rect(0, 0, image.width, image.height)
    view = ViewTransform("qwen-still", source, Rect(0, 0, width * patch, height * patch))
    return [view], _cells(positions, view, gh, gw), {
        "grid_thw": [temporal, height, width], "spatial_merge_size": merge,
        "patch_size": patch, "view_token_counts": [len(positions)]}


def _paligemma(adapter: Any, inputs: Mapping[str, Any], image: Any):
    positions = _visual(adapter, inputs)
    side = math.isqrt(len(positions))
    if side * side != len(positions):
        raise ValueError("PaliGemma visual-token count is not a square fixed-resolution grid")
    declared = getattr(adapter.processor, "image_seq_length", None)
    if declared is not None and int(declared) != len(positions):
        raise ValueError("PaliGemma image_seq_length differs from emitted visual tokens")
    source = Rect(0, 0, image.width, image.height)
    view = ViewTransform("paligemma-fixed", source, Rect(0, 0, side, side))
    return [view], _cells(positions, view, side, side), {
        "fixed_grid": [side, side], "view_token_counts": [len(positions)],
        "prompt_contract": "one_line_before_processor_appended_terminator"}


def _internvl_ratios(minimum: int, maximum: int) -> list[tuple[int, int]]:
    # Match GotOcr2ImageProcessor's stable width-major enumeration exactly.
    values = [(columns, rows) for columns in range(1, maximum + 1)
              for rows in range(1, maximum + 1)
              if minimum <= columns * rows <= maximum]
    return sorted(values, key=lambda pair: pair[0] * pair[1])


def _internvl_ratio(width: int, height: int, ratios: Sequence[tuple[int, int]], tile: int):
    aspect, area = width / height, width * height
    best, delta = None, float("inf")
    for columns, rows in ratios:
        current = abs(aspect - columns / rows)
        if current < delta or current == delta and area > .5 * tile * tile * columns * rows:
            best, delta = (columns, rows), current
    if best is None:
        raise ValueError("InternVL has no feasible tile ratio")
    return best


def _internvl(adapter: Any, inputs: Mapping[str, Any], image: Any):
    processor, image_processor = adapter.processor, adapter.processor.image_processor
    positions = _visual(adapter, inputs)
    per_view = int(getattr(processor, "image_seq_length", 256))
    side = math.isqrt(per_view)
    if side * side != per_view or len(positions) % per_view:
        raise ValueError("InternVL tokens are incompatible with the declared per-view grid")
    emitted = len(positions) // per_view
    pixels = inputs.get("pixel_values")
    if pixels is not None and int(pixels.shape[0]) != emitted:
        raise ValueError("InternVL pixel-view count differs from visual-token chunks")
    size = getattr(image_processor, "size", {})
    tile = int(size.get("height", size.get("shortest_edge", 448)) if isinstance(size, Mapping) else size)
    minimum = int(getattr(image_processor, "min_patches", getattr(image_processor, "min_num_patches", 1)))
    maximum = int(getattr(image_processor, "max_patches", getattr(image_processor, "max_num_patches", 12)))
    columns, rows = _internvl_ratio(image.width, image.height, _internvl_ratios(minimum, maximum), tile)
    thumbnail = bool(getattr(image_processor, "use_thumbnail", True) and columns * rows > 1)
    if emitted != columns * rows + int(thumbnail):
        raise ValueError("InternVL reconstructed view count differs from processor output")
    views, cells, offset = [], [], 0
    for row in range(rows):
        for column in range(columns):
            source = Rect(column * image.width / columns, row * image.height / rows,
                          (column + 1) * image.width / columns, (row + 1) * image.height / rows)
            view = ViewTransform(f"internvl-tile-{row}-{column}", source, Rect(0, 0, tile, tile))
            views.append(view)
            cells.extend(_cells(positions[offset:offset + per_view], view, side, side))
            offset += per_view
    if thumbnail:
        view = ViewTransform("internvl-thumbnail", Rect(0, 0, image.width, image.height), Rect(0, 0, tile, tile))
        views.append(view)
        cells.extend(_cells(positions[offset:offset + per_view], view, side, side))
        offset += per_view
    if offset != len(positions):
        raise ValueError("InternVL view partition did not consume all tokens")
    return views, cells, {"tile_grid": [rows, columns], "thumbnail": thumbnail,
                           "per_view_grid": [side, side], "view_token_counts": [per_view] * emitted}


def _best_resolution(original: tuple[int, int], candidates: Sequence[Sequence[int]]):
    oh, ow = original
    best, effective_best, waste_best = None, -1, float("inf")
    for candidate in candidates:
        height, width = map(int, candidate)
        scale = min(width / ow, height / oh)
        effective = min(int(ow * scale) * int(oh * scale), ow * oh)
        waste = width * height - effective
        if effective > effective_best or effective == effective_best and waste < waste_best:
            best, effective_best, waste_best = (height, width), effective, waste
    if best is None:
        raise ValueError("LLaVA OneVision has no image_grid_pinpoints")
    return best


def _unpad(original: tuple[int, int], height: int, width: int):
    oh, ow = original
    if ow / oh > width / height:
        new_height = int(round(oh * (width / ow), 7))
        top = (height - new_height) // 2
        return top, height - top, 0, width
    new_width = int(round(ow * (height / oh), 7))
    left = (width - new_width) // 2
    return 0, height, left, width - left


def _llava(adapter: Any, inputs: Mapping[str, Any], image: Any):
    positions = _visual(adapter, inputs)
    config, vision = adapter.model.config, adapter.model.config.vision_config
    side = int(vision.image_size) // int(vision.patch_size)
    base_count = side * side
    if len(positions) < base_count:
        raise ValueError("LLaVA emitted fewer tokens than its base grid")
    source = Rect(0, 0, image.width, image.height)
    base = ViewTransform("llava-base", source, Rect(0, 0, side, side))
    views, cells = [base], _cells(positions[:base_count], base, side, side)
    remaining = positions[base_count:]
    pinpoints = getattr(config, "image_grid_pinpoints", None)
    if not pinpoints:
        raise ValueError("LLaVA image_grid_pinpoints unavailable")
    best_h, best_w = _best_resolution((image.height, image.width), pinpoints)
    patch_image = int(vision.image_size)
    full_h, full_w = best_h // patch_image * side, best_w // patch_image * side
    top, bottom, left, right = _unpad((image.height, image.width), full_h, full_w)
    uh, uw = bottom - top, right - left
    downsampled = False
    aspect_mode = str(getattr(config, "vision_aspect_ratio", "anyres_max_9"))
    if aspect_mode.startswith("anyres_max_"):
        maximum = int(aspect_mode.removeprefix("anyres_max_"))
        ratio = math.sqrt(uh * uw / (maximum * side**2))
        if ratio > 1.1:
            uh, uw = int(uh // ratio), int(uw // ratio)
            downsampled = True
    if len(remaining) != uh * (uw + 1):
        # ``anyres_max_N`` may interpolate the unpadded feature map before one
        # newline vector is appended per row. Infer only dimensions whose exact
        # packed count and aspect agree with the pre-downsampled map.
        candidates = [(height, len(remaining) // height - 1)
                      for height in range(1, uh + 1) if len(remaining) % height == 0
                      and 0 < len(remaining) // height - 1 <= uw]
        if not candidates:
            raise ValueError("LLaVA packed anyres/newline count differs from every valid grid")
        ranked = sorted(candidates, key=lambda shape: (abs(shape[1] / shape[0] - uw / uh),
                                                       -(shape[0] * shape[1]), shape))
        if len(ranked) > 1 and abs(ranked[0][1] / ranked[0][0] - uw / uh) == abs(ranked[1][1] / ranked[1][0] - uw / uh):
            raise ValueError("LLaVA downsampled anyres dimensions are ambiguous")
        uh, uw = ranked[0]
        downsampled = True
    destination = Rect(0, 0, uw, uh) if downsampled else Rect(left, top, right, bottom)
    anyres = ViewTransform("llava-anyres-unpadded", source, destination)
    views.append(anyres)
    spatial, newlines = [], []
    for row in range(uh):
        start = row * (uw + 1)
        spatial.extend(remaining[start:start + uw])
        newlines.append(remaining[start + uw])
    cells.extend(_cells(spatial, anyres, uh, uw))
    return views, cells, {"base_grid": [side, side], "anyres_canvas": [full_h, full_w],
                           "anyres_unpadded": [uh, uw], "anyres_downsampled": downsampled,
                           "vision_aspect_ratio": aspect_mode,
                           "newline_positions": newlines,
                           "view_token_counts": [base_count, len(spatial)]}


def prepare_with_geometry(adapter: Any, variant: Mapping[str, Any], image: Any, *,
                          regions: Mapping[str, Sequence[float]], rule: str,
                          min_fraction: float | None, max_tokens: int,
                          max_region_tokens: int) -> tuple[Any, dict]:
    """Encode once and map source-image rectangles to validated sequence tokens."""
    require_supported_geometry(adapter)
    if any(type(x) is not int or x <= 0 for x in (max_tokens, max_region_tokens)):
        raise ValueError("explicit positive capture budgets required")
    inputs, rendered = adapter.encode(image, variant["prompt"])
    if inputs["input_ids"].shape[1] > max_tokens:
        raise ValueError("task exceeds explicit token budget")
    builders = {"qwen_vl": _qwen, "paligemma": _paligemma,
                "internvl": _internvl, "llava_onevision": _llava}
    views, cells, details = builders[adapter.spec.family](adapter, inputs, image)
    source, groups = Rect(0, 0, image.width, image.height), {}
    for name, coords in regions.items():
        region = Rect(*coords)
        if not source.contains(region):
            raise ValueError("region outside original image")
        selected = list(map_region(region, views, cells, rule=rule, min_fraction=min_fraction))
        if not selected or len(selected) > max_region_tokens:
            raise ValueError(f"empty or over-budget region {name}")
        groups[name] = selected
    prepared = adapter.prepare_inputs(variant, inputs, rendered)
    prepared.groups["text_question"] = list(prepared.groups.get("question", []))
    prepared.groups["text_options"] = list(prepared.groups.get("options", []))
    prepared.groups.update(groups)
    # Retained only in memory for deterministic view-local matched controls.
    # JSON reports contain the compact geometry summary, not dataclass objects.
    prepared.region_token_cells = cells
    prepared.region_views = views
    return prepared, {"schema": "stretch.processor-geometry.v2", "family": adapter.spec.family,
                      "image_size": list(image.size), "groups": groups,
                      "image_positions": _visual(adapter, inputs), "mapping_rule": rule,
                      "min_fraction": min_fraction, **details}


@contextmanager
def selected_attention(adapter: Any, *, heads: Sequence[Sequence[int]],
                       query_positions: Sequence[int], regions: Mapping[str, Sequence[int]],
                       block_region: str | None = None):
    """Record selected SDPA rows and optionally remove one declared region."""
    require_supported_geometry(adapter)
    if not heads or not query_positions or block_region is not None and block_region not in regions:
        raise ValueError("explicit heads/queries/known blocked region required")
    if len({tuple(h) for h in heads}) != len(heads):
        raise ValueError("duplicate attention heads")
    import torch
    selected: dict[int, list[int]] = {}
    for layer, head in heads:
        if type(layer) is not int or type(head) is not int or not 0 <= layer < adapter.num_layers or not 0 <= head < adapter.handle.num_heads:
            raise ValueError("attention head outside model dimensions")
        selected.setdefault(layer, []).append(head)
    if not _ATTENTION_LOCK.acquire(blocking=False):
        raise RuntimeError("selected attention instrumentation already active")
    if adapter.spec.family == "paligemma":
        try:
            from transformers.models.gemma2 import modeling_gemma2
            original_eager = modeling_gemma2.eager_attention_forward
            by_module = {id(adapter.handle.layers[layer].self_attn): layer for layer in selected}
            records, calls = [], {}

            def eager_intercept(module, query, key, value, attention_mask, dropout=0.0,
                                scaling=None, softcap=None, **kwargs):
                output, weights = original_eager(module, query, key, value, attention_mask,
                                                 dropout=dropout, scaling=scaling,
                                                 softcap=softcap, **kwargs)
                layer = by_module.get(id(module))
                if layer is None:
                    return output, weights
                if dropout != 0 or query.ndim != 4 or query.shape[0] != 1 or weights is None:
                    raise ValueError("only deterministic batch-one Gemma2 eager attention supported")
                calls[layer] = calls.get(layer, 0) + 1
                if calls[layer] > 1:
                    raise ValueError("selected Gemma2 layer executed more than once")
                qheads, kheads = weights.shape[1], value.shape[1]
                if qheads % kheads:
                    raise ValueError("invalid Gemma2 grouped-query head alignment")
                repeated_value = value.repeat_interleave(qheads // kheads, dim=1)
                output = output.clone() if block_region is not None else output
                for head in selected[layer]:
                    for pos in query_positions:
                        if type(pos) is not int or not 0 <= pos < weights.shape[-2]:
                            raise ValueError("attention query outside sequence")
                        probabilities = weights[0, head, pos].float()
                        expected = probabilities.to(value.dtype) @ repeated_value[0, head]
                        error = float((expected.float() - output[0, pos, head].float()).abs().max().item())
                        if not math.isfinite(error):
                            raise ValueError("nonfinite Gemma2 attention reconstruction")
                        if block_region is not None:
                            blocked = list(regions[block_region])
                            if not blocked or any(type(i) is not int or not 0 <= i < probabilities.numel() for i in blocked):
                                raise ValueError("invalid blocked positions")
                            probabilities = probabilities.clone()
                            probabilities[blocked] = 0
                            total = probabilities.sum()
                            if not torch.isfinite(total) or total <= 0:
                                raise ValueError("edge intervention removed all accessible keys")
                            probabilities /= total
                            output[0, pos, head] = (probabilities.to(value.dtype) @ repeated_value[0, head]).to(output.dtype)
                        masses = {}
                        for name, positions in regions.items():
                            if not positions or any(type(i) is not int or not 0 <= i < probabilities.numel() for i in positions):
                                raise ValueError("invalid attention region")
                            masses[name] = {"mass": float(probabilities[list(positions)].sum().item()),
                                            "tokens": len(positions)}
                        records.append({"layer": layer, "head": head, "query_position": pos,
                                        "reconstruction_max_error": error, "region_attention": masses,
                                        "blocked_region": block_region})
                return output, weights

            modeling_gemma2.eager_attention_forward = eager_intercept
            try:
                yield records
                if set(calls) != set(selected):
                    raise ValueError("model did not execute eager attention at every selected layer")
            finally:
                modeling_gemma2.eager_attention_forward = original_eager
        finally:
            _ATTENTION_LOCK.release()
        return
    import torch.nn.functional as functional
    original = functional.scaled_dot_product_attention
    active = [None]
    records, hooks, calls = [], [], {}

    def intercept(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                  scale=None, enable_gqa=False):
        layer = active[0]
        kwargs = {"attn_mask": attn_mask, "dropout_p": dropout_p, "is_causal": is_causal}
        if scale is not None:
            kwargs["scale"] = scale
        if enable_gqa:
            kwargs["enable_gqa"] = True
        output = original(query, key, value, **kwargs)
        if layer not in selected:
            return output
        if dropout_p != 0 or query.ndim != 4 or query.shape[0] != 1 or key.shape[0] != 1:
            raise ValueError("only deterministic batch-one SDPA supported")
        calls[layer] = calls.get(layer, 0) + 1
        if calls[layer] > 1:
            raise ValueError("selected layer executed more than once")
        qheads, kheads = query.shape[1], key.shape[1]
        if qheads % kheads or value.shape[1] != kheads:
            raise ValueError("invalid grouped-query head alignment")
        output = output.clone() if block_region is not None else output
        for head in selected[layer]:
            kh = head // (qheads // kheads)
            for pos in query_positions:
                if type(pos) is not int or not 0 <= pos < query.shape[-2]:
                    raise ValueError("attention query outside sequence")
                scores = query[0, head, pos].float() @ key[0, kh].float().T
                scores *= scale if scale is not None else 1 / math.sqrt(query.shape[-1])
                if is_causal:
                    scores[torch.arange(scores.numel(), device=scores.device) > pos] = -torch.inf
                if attn_mask is not None:
                    mask = attn_mask
                    while mask.ndim < 4:
                        mask = mask.unsqueeze(0)
                    row = mask[0, head if mask.shape[1] != 1 else 0, pos if mask.shape[-2] != 1 else 0]
                    scores = scores.masked_fill(~row, -torch.inf) if row.dtype == torch.bool else scores + row.float()
                probabilities = scores.softmax(-1)
                expected = probabilities.to(value.dtype) @ value[0, kh]
                error = float((expected.float() - output[0, head, pos].float()).abs().max().item())
                if not math.isfinite(error):
                    raise ValueError("nonfinite selected-attention reconstruction")
                if block_region is not None:
                    blocked = list(regions[block_region])
                    if not blocked or any(type(i) is not int or not 0 <= i < scores.numel() for i in blocked):
                        raise ValueError("invalid blocked positions")
                    scores[blocked] = -torch.inf
                    probabilities = scores.softmax(-1)
                    changed = probabilities.to(value.dtype) @ value[0, kh]
                    if not torch.isfinite(changed).all():
                        raise ValueError("edge intervention removed all accessible keys")
                    output[0, head, pos] = changed.to(output.dtype)
                masses = {}
                for name, positions in regions.items():
                    if not positions or any(type(i) is not int or not 0 <= i < scores.numel() for i in positions):
                        raise ValueError("invalid attention region")
                    masses[name] = {"mass": float(probabilities[list(positions)].sum().item()),
                                    "tokens": len(positions)}
                records.append({"layer": layer, "head": head, "query_position": pos,
                                "reconstruction_max_error": error, "region_attention": masses,
                                "blocked_region": block_region})
        return output

    try:
        for index in selected:
            module = adapter.handle.layers[index].self_attn
            hooks.append(module.register_forward_pre_hook(lambda _m, _a, i=index: active.__setitem__(0, i)))
            hooks.append(module.register_forward_hook(lambda _m, _a, out: active.__setitem__(0, None)))
        functional.scaled_dot_product_attention = intercept
        yield records
        if set(calls) != set(selected):
            raise ValueError("model did not execute interceptable SDPA at every selected layer")
    finally:
        functional.scaled_dot_product_attention = original
        for hook in hooks:
            hook.remove()
        _ATTENTION_LOCK.release()
