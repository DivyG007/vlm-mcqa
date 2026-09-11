"""Run causal interventions for ABCD visual multiple-choice answering.

The runner separates causal contrasts, hook boundaries, and metrics. It first
prepares image-conditioned language-model embeddings once per example, then
runs interventions only through the Qwen language tower. A calibration step
checks that this fast path reproduces the full VLM forward pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from vlm_mcqa.eval.qwen_shapes import _image_token_id, _resolve_attr, _single_token_id
from vlm_mcqa.interventions.core import (
    HookSite,
    add_to_outputs,
    bbox_to_grid_indices,
    block_attention_edges,
    bootstrap_interval,
    capture_inputs,
    capture_outputs,
    complement_indices,
    evenly_spaced_control_region,
    find_subsequence,
    normalized_recovery,
    path_patch_outputs,
    patch_head_inputs,
    patch_outputs,
)


ABCD = ("A", "B", "C", "D")
SEMANTIC_CONTENTS = ("red", "blue", "green", "yellow")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(dict(value), sort_keys=True) + "\n")
    stream.flush()


@dataclass(frozen=True)
class Contrast:
    contrast_id: str
    contrast_type: str
    source: dict[str, Any]
    target: dict[str, Any]


@dataclass
class PreparedExample:
    record: dict[str, Any]
    full_inputs: dict[str, Any]
    inputs_embeds: Any
    position_ids: Any
    attention_mask: Any
    input_ids: Any
    token_groups: dict[str, list[int]]
    image_grid: tuple[int, int]


def _letter_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record["label_scheme"] == "letters"]


def _format_prompt(question: str, options: Sequence[str], labels: Sequence[str]) -> str:
    lines = [question]
    lines.extend(f"{label}. {option}" for label, option in zip(labels, options, strict=True))
    lines.append("Respond with only the option label.")
    lines.append("Answer:")
    return "\n".join(lines)


def _content_fixed_symbol_target(
    source: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    """Keep the correct symbol fixed while changing the visual answer content."""
    source_content = source["correct_content"]
    target_content = target["correct_content"]
    if source_content == target_content:
        raise ValueError("content contrast must change the semantic answer")
    options = list(source["option_contents"])
    source_index = int(source["correct_index"])
    target_original_index = options.index(target_content)
    options[source_index], options[target_original_index] = (
        options[target_original_index],
        options[source_index],
    )
    derived = dict(target)
    derived.update(
        {
            "sample_id": f"{target['sample_id']}_content_fixed_symbol_{source_index}",
            "variant_family": "content_fixed_symbol",
            "variant_key": f"content_at_{source_index}",
            "option_contents": options,
            "correct_index": source_index,
            "correct_label": source["correct_label"],
            "prompt": _format_prompt(target["question"], options, target["labels"]),
        }
    )
    return derived


def _pair_ids(records: Sequence[dict[str, Any]]) -> list[str]:
    return sorted({str(record["pair_id"]) for record in records})


def select_pair_ids(
    records: Sequence[dict[str, Any]], *, split: str, num_pairs: int
) -> list[str]:
    pair_ids = _pair_ids(records)
    if len(pair_ids) < 64:
        midpoint = len(pair_ids) // 2
    else:
        midpoint = 32
    if split == "smoke":
        selected = pair_ids[:num_pairs]
    elif split == "discovery":
        selected = pair_ids[:midpoint][:num_pairs]
    elif split == "confirmation":
        selected = pair_ids[midpoint:][:num_pairs]
    elif split == "all":
        selected = pair_ids[:num_pairs]
    else:
        raise ValueError(f"unknown split: {split}")
    if not selected:
        raise ValueError(f"no pair ids selected for split={split!r}")
    return selected


def build_contrasts(
    records: Sequence[dict[str, Any]],
    *,
    selected_pairs: Sequence[str],
    contrast_types: Sequence[str],
) -> list[Contrast]:
    selected = set(selected_pairs)
    letters = [record for record in _letter_records(records) if record["pair_id"] in selected]
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in letters:
        by_pair[record["pair_id"]].append(record)

    contrasts: list[Contrast] = []
    for pair_id in selected_pairs:
        pair_records = by_pair[pair_id]
        if "visual" in contrast_types:
            cross = {
                record["pair_member"]: record
                for record in pair_records
                if record["variant_family"] == "cross_image"
            }
            if set(cross) != {"a", "b"}:
                raise ValueError(f"pair {pair_id} lacks a complete visual contrast")
            if cross["a"]["correct_label"] == cross["b"]["correct_label"]:
                raise ValueError(f"pair {pair_id} visual contrast does not change the answer")
            contrasts.append(
                Contrast(f"{pair_id}:visual:a_to_b", "visual", cross["a"], cross["b"])
            )

        if "content" in contrast_types:
            cross = {
                record["pair_member"]: record
                for record in pair_records
                if record["variant_family"] == "cross_image"
            }
            if set(cross) != {"a", "b"}:
                raise ValueError(f"pair {pair_id} lacks a complete content contrast")
            content_target = _content_fixed_symbol_target(cross["a"], cross["b"])
            contrasts.append(
                Contrast(
                    f"{pair_id}:content:a_to_b_fixed_symbol",
                    "content",
                    cross["a"],
                    content_target,
                )
            )

        if "position" in contrast_types:
            position = {
                int(record["correct_index"]): record
                for record in pair_records
                if record["variant_family"] == "answer_position"
                and record["pair_member"] == "a"
            }
            if set(position) != {0, 1, 2, 3}:
                raise ValueError(f"pair {pair_id} lacks all four position variants")
            for source_index, target_index in ((0, 1), (2, 3)):
                contrasts.append(
                    Contrast(
                        f"{pair_id}:position:{source_index}_to_{target_index}",
                        "position",
                        position[source_index],
                        position[target_index],
                    )
                )
    return contrasts


def both_directions(contrast: Contrast) -> Iterator[Contrast]:
    yield contrast
    yield Contrast(
        contrast_id=contrast.contrast_id.rsplit(":", 1)[0] + ":reverse",
        contrast_type=contrast.contrast_type,
        source=contrast.target,
        target=contrast.source,
    )


def _contrast_readout(
    contrast: Contrast,
) -> tuple[str, Sequence[str], str, str]:
    if contrast.contrast_type == "content":
        return (
            "semantic_content",
            SEMANTIC_CONTENTS,
            str(contrast.source["correct_content"]),
            str(contrast.target["correct_content"]),
        )
    return (
        "answer_symbol",
        ABCD,
        str(contrast.source["correct_label"]),
        str(contrast.target["correct_label"]),
    )


def _resolve_layers(model: Any) -> tuple[Any, str]:
    return _resolve_attr(
        model,
        (
            "model.language_model.layers",
            "model.model.language_model.layers",
            "language_model.layers",
        ),
    )


def _tokenize_subsequence(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _first_subsequence(input_ids: Sequence[int], candidates: Sequence[Sequence[int]]) -> list[int]:
    for candidate in candidates:
        starts = find_subsequence(input_ids, candidate)
        if starts:
            start = starts[0]
            return list(range(start, start + len(candidate)))
    return []


def _text_token_groups(
    *,
    tokenizer: Any,
    input_ids: Sequence[int],
    record: Mapping[str, Any],
    image_positions: Sequence[int],
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    groups["question"] = _first_subsequence(
        input_ids,
        [
            _tokenize_subsequence(tokenizer, record["question"]),
            _tokenize_subsequence(tokenizer, " " + record["question"]),
            _tokenize_subsequence(tokenizer, "\n" + record["question"]),
        ],
    )
    option_content: list[int] = []
    option_labels: list[int] = []
    for label, content in zip(record["labels"], record["option_contents"], strict=True):
        line_candidates = [
            _tokenize_subsequence(tokenizer, f"{label}. {content}"),
            _tokenize_subsequence(tokenizer, f"\n{label}. {content}"),
        ]
        line_positions = _first_subsequence(input_ids, line_candidates)
        if not line_positions:
            continue
        line_ids = [input_ids[index] for index in line_positions]
        label_ids = _tokenize_subsequence(tokenizer, label)
        content_ids = _tokenize_subsequence(tokenizer, content)
        label_local = _first_subsequence(line_ids, [label_ids])
        content_local = _first_subsequence(line_ids, [content_ids])
        option_labels.extend(line_positions[index] for index in label_local)
        option_content.extend(line_positions[index] for index in content_local)
    groups["option_labels"] = sorted(set(option_labels))
    groups["option_content"] = sorted(set(option_content))

    # Tokenizers may merge the leading whitespace of prompt spans into the
    # first ordinary token.  The exact full-line search above is preferred,
    # but use the known synthetic-prompt structure as an audited fallback.
    # Special vision/chat delimiters are explicitly excluded from the
    # question range.
    if not groups["question"] and groups["option_labels"] and image_positions:
        special_ids = set(getattr(tokenizer, "all_special_ids", ()))
        groups["question"] = [
            index
            for index in range(max(image_positions) + 1, groups["option_labels"][0])
            if input_ids[index] not in special_ids
        ]

    if not groups["option_content"] and len(groups["option_labels"]) == len(
        record["option_contents"]
    ):
        recovered_content: list[int] = []
        labels = groups["option_labels"]
        for option_index, (label_position, content) in enumerate(
            zip(labels, record["option_contents"], strict=True)
        ):
            end = labels[option_index + 1] if option_index + 1 < len(labels) else len(input_ids)
            segment = input_ids[label_position + 1 : end]
            content_local = _first_subsequence(
                segment,
                [
                    _tokenize_subsequence(tokenizer, content),
                    _tokenize_subsequence(tokenizer, " " + content),
                    _tokenize_subsequence(tokenizer, "\n" + content),
                ],
            )
            recovered_content.extend(label_position + 1 + index for index in content_local)
        groups["option_content"] = sorted(set(recovered_content))
    return groups


def _prepare_example(
    *,
    record: dict[str, Any],
    manifest_dir: Path,
    processor: Any,
    model: Any,
    image_token_id: int,
    image_size: int,
) -> PreparedExample:
    import torch
    from PIL import Image

    image_path = manifest_dir / record["image"]
    with Image.open(image_path) as loaded:
        image = loaded.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": record["prompt"]},
            ],
        }
    ]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    input_ids = inputs["input_ids"]
    with torch.inference_mode():
        inputs_embeds = model.model.get_input_embeddings()(input_ids)
        image_outputs = model.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = model.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        position_ids = model.model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.get("attention_mask"),
            past_key_values=None,
            second_per_grid_ts=None,
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

    ids = input_ids[0].tolist()
    seq_len = len(ids)
    image_positions = [index for index, token_id in enumerate(ids) if token_id == image_token_id]
    grid = inputs["image_grid_thw"][0].tolist()
    merge = int(model.config.vision_config.spatial_merge_size)
    grid_height, grid_width = int(grid[1]) // merge, int(grid[2]) // merge
    if len(image_positions) != int(grid[0]) * grid_height * grid_width:
        raise RuntimeError(
            f"visual-token/grid mismatch for {record['sample_id']}: "
            f"{len(image_positions)} versus {grid} merge={merge}"
        )
    if int(grid[0]) != 1:
        raise RuntimeError("the shapes experiment expects one temporal image grid")
    groups = _text_token_groups(
        tokenizer=processor.tokenizer,
        input_ids=ids,
        record=record,
        image_positions=image_positions,
    )
    groups.update(
        {
            "final": [seq_len - 1],
            "image_all": image_positions,
            "text_all": [index for index in range(seq_len) if index not in set(image_positions)],
        }
    )
    query_region = bbox_to_grid_indices(
        bbox=record["query_bbox"],
        image_size=image_size,
        grid_height=grid_height,
        grid_width=grid_width,
        image_token_indices=image_positions,
        expansion=0,
    )
    query_region_expanded = bbox_to_grid_indices(
        bbox=record["query_bbox"],
        image_size=image_size,
        grid_height=grid_height,
        grid_width=grid_width,
        image_token_indices=image_positions,
        expansion=1,
    )
    groups["image_query"] = query_region
    groups["image_query_expanded"] = query_region_expanded
    groups["image_query_complement"] = complement_indices(query_region_expanded, image_positions)
    groups["image_random_matched"] = evenly_spaced_control_region(
        target_count=len(query_region_expanded),
        universe=image_positions,
        excluded=query_region_expanded,
    )
    distractor_shape = next(
        (shape for shape in record["changed_shapes"] if shape != record["query_shape"]),
        None,
    )
    if distractor_shape is not None:
        distractor_bbox = next(
            obj["bbox"] for obj in record["objects"] if obj["shape"] == distractor_shape
        )
        groups["image_changed_distractor"] = bbox_to_grid_indices(
            bbox=distractor_bbox,
            image_size=image_size,
            grid_height=grid_height,
            grid_width=grid_width,
            image_token_indices=image_positions,
            expansion=1,
        )
    return PreparedExample(
        record=record,
        full_inputs=inputs,
        inputs_embeds=inputs_embeds.detach(),
        position_ids=position_ids,
        attention_mask=inputs.get("attention_mask"),
        input_ids=input_ids,
        token_groups=groups,
        image_grid=(grid_height, grid_width),
    )


def _language_forward(model: Any, prepared: PreparedExample) -> Any:
    outputs = model.model.language_model(
        input_ids=None,
        position_ids=prepared.position_ids,
        attention_mask=prepared.attention_mask,
        inputs_embeds=prepared.inputs_embeds,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
    return logits[0, -1].float()


def _score_logits(
    logits: Any,
    *,
    labels: Sequence[str],
    label_token_ids: Mapping[str, int],
    source_label: str,
    target_label: str,
) -> dict[str, Any]:
    import torch

    ids = torch.tensor([label_token_ids[label] for label in labels], device=logits.device)
    scores = logits.index_select(0, ids)
    partition = torch.logsumexp(logits, dim=0)
    predicted_index = int(scores.argmax().item())
    source_index = labels.index(source_label)
    target_index = labels.index(target_label)
    incorrect_target = torch.cat([scores[:target_index], scores[target_index + 1 :]])
    return {
        "label_logits": {label: float(value) for label, value in zip(labels, scores.tolist(), strict=True)},
        "predicted_label": labels[predicted_index],
        "target_minus_source": float(scores[target_index] - scores[source_index]),
        "target_margin": float(scores[target_index] - incorrect_target.max()),
        "source_margin": float(
            scores[source_index]
            - torch.cat([scores[:source_index], scores[source_index + 1 :]]).max()
        ),
        "valid_label_probability_mass": float(
            torch.exp(torch.logsumexp(scores, dim=0) - partition)
        ),
        "global_top_token_id": int(logits.argmax().item()),
        "global_top_token_is_valid": int(logits.argmax().item()) in ids.tolist(),
    }


def _restricted_prediction(
    logits: Any, *, tokens: Sequence[str], token_ids: Mapping[str, int]
) -> str:
    import torch

    ids = torch.tensor([token_ids[token] for token in tokens], device=logits.device)
    return tokens[int(logits.index_select(0, ids).argmax().item())]


def _residual_sites(layers: Sequence[Any]) -> list[HookSite]:
    return [
        HookSite(f"resid_post_layer_{index}", layer, "decoder_layer_output_post_mlp_residual")
        for index, layer in enumerate(layers)
    ]


def _component_sites(layers: Sequence[Any], component: str) -> list[HookSite]:
    if component == "attention":
        return [
            HookSite(f"attention_out_layer_{index}", layer.self_attn, "attention_output_pre_residual")
            for index, layer in enumerate(layers)
        ]
    if component == "mlp":
        return [
            HookSite(f"mlp_out_layer_{index}", layer.mlp, "mlp_output_pre_residual")
            for index, layer in enumerate(layers)
        ]
    raise ValueError(f"unknown component: {component}")


def _head_sites(layers: Sequence[Any]) -> list[HookSite]:
    return [
        HookSite(f"head_pre_o_proj_layer_{index}", layer.self_attn.o_proj, "attention_pre_o_proj")
        for index, layer in enumerate(layers)
    ]


def _run_with_output_patch(
    *,
    model: Any,
    prepared: PreparedExample,
    sites: Sequence[HookSite],
    donor_cache: Mapping[str, Any],
    recipient_groups: Mapping[str, Sequence[int]],
    donor_groups: Mapping[str, Sequence[int]],
    token_group: str,
) -> Any:
    patches = {
        site.key: (
            site,
            donor_cache[site.key],
            recipient_groups[token_group],
            donor_groups[token_group],
        )
        for site in sites
    }
    with patch_outputs(patches):
        return _language_forward(model, prepared)


def _base_row(
    contrast: Contrast,
    *,
    source_score: Mapping[str, Any],
    target_score: Mapping[str, Any],
    patched_score: Mapping[str, Any],
) -> dict[str, Any]:
    readout_type, readout_tokens, source_readout, target_readout = _contrast_readout(
        contrast
    )
    source_difference = float(source_score["target_minus_source"])
    target_difference = float(target_score["target_minus_source"])
    patched_difference = float(patched_score["target_minus_source"])
    return {
        "contrast_id": contrast.contrast_id,
        "contrast_type": contrast.contrast_type,
        "pair_id": contrast.source["pair_id"],
        "source_sample_id": contrast.source["sample_id"],
        "target_sample_id": contrast.target["sample_id"],
        "source_label": contrast.source["correct_label"],
        "target_label": contrast.target["correct_label"],
        "source_content": contrast.source["correct_content"],
        "target_content": contrast.target["correct_content"],
        "readout_type": readout_type,
        "readout_tokens": list(readout_tokens),
        "source_readout": source_readout,
        "target_readout": target_readout,
        "source_target_difference": source_difference,
        "clean_target_difference": target_difference,
        "patched_target_difference": patched_difference,
        "normalized_recovery": normalized_recovery(
            source_difference, target_difference, patched_difference
        ),
        "patched_predicted_label": patched_score["predicted_label"],
        "target_flip": patched_score["predicted_label"] == target_readout,
        "patched_target_margin": patched_score["target_margin"],
        "patched_source_margin": patched_score["source_margin"],
        "patched_valid_label_mass": patched_score["valid_label_probability_mass"],
        "patched_global_top_valid": patched_score["global_top_token_is_valid"],
        "patched_label_logits": patched_score["label_logits"],
    }


def _window_specs(
    *, num_layers: int, widths: Sequence[int], start_layer: int, end_layer: int
) -> list[tuple[int, int]]:
    end_layer = min(end_layer, num_layers - 1)
    specs: list[tuple[int, int]] = []
    for width in widths:
        for start in range(start_layer, end_layer - width + 2):
            specs.append((start, start + width - 1))
    return specs


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["experiment"],
            row.get("contrast_type"),
            row.get("site_type"),
            row.get("layer_index"),
            row.get("window_start"),
            row.get("window_end"),
            row.get("token_group"),
            row.get("component"),
            row.get("head_index"),
            row.get("edge_route"),
            row.get("scale"),
            row.get("ablation_type"),
            row.get("binding_key"),
            row.get("path_sender_layer"),
            row.get("path_receiver_layer"),
            row.get("path_receiver_group"),
        )
        groups[key].append(row)
    aggregates = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        recoveries = [
            float(row["normalized_recovery"])
            for row in values
            if not math.isnan(float(row.get("normalized_recovery", float("nan"))))
        ]
        low, high = bootstrap_interval(recoveries) if recoveries else (float("nan"), float("nan"))
        aggregates.append(
            {
                "experiment": key[0],
                "contrast_type": key[1],
                "site_type": key[2],
                "layer_index": key[3],
                "window_start": key[4],
                "window_end": key[5],
                "token_group": key[6],
                "component": key[7],
                "head_index": key[8],
                "edge_route": key[9],
                "scale": key[10],
                "ablation_type": key[11],
                "binding_key": key[12],
                "path_sender_layer": key[13],
                "path_receiver_layer": key[14],
                "path_receiver_group": key[15],
                "count": len(values),
                "mean_normalized_recovery": statistics.fmean(recoveries) if recoveries else float("nan"),
                "median_normalized_recovery": statistics.median(recoveries) if recoveries else float("nan"),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "target_flip_rate": statistics.fmean(bool(row.get("target_flip")) for row in values),
                "mean_patched_target_margin": statistics.fmean(
                    float(row.get("patched_target_margin", 0.0)) for row in values
                ),
            }
        )
    return {"aggregate_rows": aggregates, "raw_row_count": len(rows)}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("smoke", "discovery", "confirmation", "all"), default="smoke")
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--contrast-types", default="visual,position")
    parser.add_argument("--modes", default="calibration,residual,window")
    parser.add_argument("--max-pixels", type=int, default=224 * 224)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--window-widths", default="2,4,8")
    parser.add_argument("--window-start-layer", type=int, default=19)
    parser.add_argument("--window-end-layer", type=int, default=35)
    parser.add_argument("--selected-layers", default="24,25,26,27,28,29,30,31")
    parser.add_argument(
        "--token-groups",
        default="final,image_all,image_query_expanded,image_changed_distractor,image_random_matched,question,option_content,option_labels",
    )
    parser.add_argument("--binding-scales", default="0,0.25,0.5,1,1.5")
    parser.add_argument("--binding-vectors", type=Path)
    parser.add_argument("--path-sender-layer", type=int)
    parser.add_argument("--path-receiver-layer", type=int)
    parser.add_argument("--path-sender-group", default="image_query_expanded")
    parser.add_argument("--path-receiver-group", default="final")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for causal interventions")
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    contrast_types = [item.strip() for item in args.contrast_types.split(",") if item.strip()]
    print(f"[load] model={args.model_id} modes={modes}", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=args.max_pixels,
        max_pixels=args.max_pixels,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval().cuda()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers, layers_path = _resolve_layers(model)
    layers = list(layers)
    image_token_id = _image_token_id(model)
    print(
        f"[model] layers={len(layers)} path={layers_path} image_token_id={image_token_id} "
        f"gpu_gib={torch.cuda.memory_allocated() / 2**30:.2f}",
        flush=True,
    )
    label_token_ids = {
        token: _single_token_id(processor.tokenizer, token)[0]
        for token in (*ABCD, *SEMANTIC_CONTENTS)
    }

    all_records = _read_jsonl(args.manifest)
    selected_pairs = select_pair_ids(all_records, split=args.split, num_pairs=args.num_pairs)
    contrasts = build_contrasts(
        all_records, selected_pairs=selected_pairs, contrast_types=contrast_types
    )
    directed = [direction for contrast in contrasts for direction in both_directions(contrast)]
    print(
        f"[data] pairs={len(selected_pairs)} base_contrasts={len(contrasts)} "
        f"directed_contrasts={len(directed)} split={args.split}",
        flush=True,
    )

    prepared_cache: dict[str, PreparedExample] = {}

    def prepared(record: dict[str, Any]) -> PreparedExample:
        sample_id = record["sample_id"]
        if sample_id not in prepared_cache:
            prepared_cache[sample_id] = _prepare_example(
                record=record,
                manifest_dir=args.manifest.parent,
                processor=processor,
                model=model,
                image_token_id=image_token_id,
                image_size=args.image_size,
            )
        return prepared_cache[sample_id]

    residual_sites = _residual_sites(layers)
    selected_layers = _parse_int_list(args.selected_layers)
    if any(index < 0 or index >= len(layers) for index in selected_layers):
        raise ValueError(f"selected layers outside model range: {selected_layers}")
    token_groups = [item.strip() for item in args.token_groups.split(",") if item.strip()]
    binding_scales = [float(item) for item in args.binding_scales.split(",") if item.strip()]
    binding_differences: dict[str, list[Any]] = defaultdict(list)
    loaded_binding_vectors: dict[str, Any] = {}
    if "binding_apply" in modes:
        if args.binding_vectors is None:
            raise ValueError("--binding-vectors is required for binding_apply")
        loaded = torch.load(args.binding_vectors, map_location="cpu", weights_only=True)
        loaded_binding_vectors = dict(loaded["vectors"])
        print(
            f"[binding] loaded_vectors={len(loaded_binding_vectors)} path={args.binding_vectors}",
            flush=True,
        )
    rows: list[dict[str, Any]] = []
    clean_gate_passed = 0
    clean_gate_skipped = 0
    rows_path = args.output_dir / "interventions.jsonl"
    started = time.time()
    completed_interventions = 0

    with rows_path.open("w", encoding="utf-8") as row_stream, torch.inference_mode():
        if "calibration" in modes:
            example = prepared(directed[0].source)
            full_outputs = model(
                **example.full_inputs,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
            fast_logits = _language_forward(model, example)
            full_logits = full_outputs.logits[0, -1].float()
            max_diff = float((fast_logits - full_logits).abs().max())
            label_diff = float(
                max(
                    abs(float(fast_logits[token_id] - full_logits[token_id]))
                    for token_id in label_token_ids.values()
                )
            )
            calibration = {
                "experiment": "calibration_fast_path",
                "sample_id": example.record["sample_id"],
                "max_full_vocabulary_logit_difference": max_diff,
                "max_abcd_logit_difference": label_diff,
                "passed": max_diff <= 1e-3,
            }
            _append_jsonl(row_stream, calibration)
            rows.append(calibration)
            print(f"[calibration] {json.dumps(calibration, sort_keys=True)}", flush=True)
            if not calibration["passed"]:
                raise RuntimeError(f"fast language path failed equivalence: {calibration}")

            # Identity-patch every residual boundary on one example.
            with capture_outputs(residual_sites) as identity_cache:
                identity_clean_logits = _language_forward(model, example)
            for layer_index, site in enumerate(residual_sites):
                identity_logits = _run_with_output_patch(
                    model=model,
                    prepared=example,
                    sites=[site],
                    donor_cache=identity_cache,
                    recipient_groups=example.token_groups,
                    donor_groups=example.token_groups,
                    token_group="final",
                )
                identity_diff = float((identity_logits - identity_clean_logits).abs().max())
                row = {
                    "experiment": "calibration_identity_patch",
                    "site_type": "residual",
                    "layer_index": layer_index,
                    "stage_index": layer_index + 1,
                    "max_full_vocabulary_logit_difference": identity_diff,
                    "passed": identity_diff <= 1e-3,
                }
                _append_jsonl(row_stream, row)
                rows.append(row)
                if not row["passed"]:
                    raise RuntimeError(f"identity patch failed at layer {layer_index}: {row}")
            print("[calibration] all residual identity patches passed", flush=True)

        for contrast_index, contrast in enumerate(directed, start=1):
            source = prepared(contrast.source)
            target = prepared(contrast.target)
            readout_type, readout_tokens, source_readout, target_readout = _contrast_readout(
                contrast
            )
            if source_readout == target_readout:
                raise RuntimeError(
                    f"causal contrast does not change its readout: {contrast.contrast_id}"
                )
            with capture_outputs(residual_sites) as source_cache:
                source_logits = _language_forward(model, source)
            with capture_outputs(residual_sites) as target_cache:
                target_logits = _language_forward(model, target)
            source_score = _score_logits(
                source_logits,
                labels=readout_tokens,
                label_token_ids=label_token_ids,
                source_label=source_readout,
                target_label=target_readout,
            )
            target_score = _score_logits(
                target_logits,
                labels=readout_tokens,
                label_token_ids=label_token_ids,
                source_label=source_readout,
                target_label=target_readout,
            )
            source_behavioral_symbol = _restricted_prediction(
                source_logits, tokens=ABCD, token_ids=label_token_ids
            )
            target_behavioral_symbol = _restricted_prediction(
                target_logits, tokens=ABCD, token_ids=label_token_ids
            )
            behavioral_passed = (
                source_behavioral_symbol == contrast.source["correct_label"]
                and target_behavioral_symbol == contrast.target["correct_label"]
            )
            clean_passed = (
                source_score["predicted_label"] == source_readout
                and target_score["predicted_label"] == target_readout
                and behavioral_passed
            )
            gate_row = {
                "experiment": "clean_readout_gate",
                "contrast_id": contrast.contrast_id,
                "contrast_type": contrast.contrast_type,
                "pair_id": contrast.source["pair_id"],
                "readout_type": readout_type,
                "source_readout": source_readout,
                "target_readout": target_readout,
                "source_predicted_readout": source_score["predicted_label"],
                "target_predicted_readout": target_score["predicted_label"],
                "source_behavioral_symbol": source_behavioral_symbol,
                "target_behavioral_symbol": target_behavioral_symbol,
                "behavioral_passed": behavioral_passed,
                "passed": clean_passed,
            }
            _append_jsonl(row_stream, gate_row)
            rows.append(gate_row)
            if not clean_passed:
                clean_gate_skipped += 1
                if contrast.contrast_type != "content":
                    raise RuntimeError(
                        f"clean answer-symbol prediction failed: {contrast.contrast_id}"
                    )
                continue
            clean_gate_passed += 1

            if "residual" in modes:
                for layer_index, site in enumerate(residual_sites):
                    patched_logits = _run_with_output_patch(
                        model=model,
                        prepared=source,
                        sites=[site],
                        donor_cache=target_cache,
                        recipient_groups=source.token_groups,
                        donor_groups=target.token_groups,
                        token_group="final",
                    )
                    patched_score = _score_logits(
                        patched_logits,
                        labels=readout_tokens,
                        label_token_ids=label_token_ids,
                        source_label=source_readout,
                        target_label=target_readout,
                    )
                    row = {
                        "experiment": "single_layer_residual_patch",
                        "site_type": "residual",
                        "component": "residual_stream",
                        "token_group": "final",
                        "layer_index": layer_index,
                        "stage_index": layer_index + 1,
                        **_base_row(
                            contrast,
                            source_score=source_score,
                            target_score=target_score,
                            patched_score=patched_score,
                        ),
                    }
                    _append_jsonl(row_stream, row)
                    rows.append(row)
                    completed_interventions += 1

            if "window" in modes:
                windows = _window_specs(
                    num_layers=len(layers),
                    widths=_parse_int_list(args.window_widths),
                    start_layer=args.window_start_layer,
                    end_layer=args.window_end_layer,
                )
                for window_start, window_end in windows:
                    selected_sites = residual_sites[window_start : window_end + 1]
                    patched_logits = _run_with_output_patch(
                        model=model,
                        prepared=source,
                        sites=selected_sites,
                        donor_cache=target_cache,
                        recipient_groups=source.token_groups,
                        donor_groups=target.token_groups,
                        token_group="final",
                    )
                    patched_score = _score_logits(
                        patched_logits,
                        labels=readout_tokens,
                        label_token_ids=label_token_ids,
                        source_label=source_readout,
                        target_label=target_readout,
                    )
                    row = {
                        "experiment": "residual_window_patch",
                        "site_type": "residual_window",
                        "component": "residual_stream",
                        "token_group": "final",
                        "window_start": window_start,
                        "window_end": window_end,
                        "window_width": window_end - window_start + 1,
                        "stage_start": window_start + 1,
                        "stage_end": window_end + 1,
                        **_base_row(
                            contrast,
                            source_score=source_score,
                            target_score=target_score,
                            patched_score=patched_score,
                        ),
                    }
                    _append_jsonl(row_stream, row)
                    rows.append(row)
                    completed_interventions += 1

            if "token" in modes:
                for layer_index in selected_layers:
                    site = residual_sites[layer_index]
                    for token_group in token_groups:
                        if token_group not in source.token_groups or token_group not in target.token_groups:
                            raise RuntimeError(
                                f"missing token group {token_group!r} for {contrast.contrast_id}"
                            )
                        if not source.token_groups[token_group] or not target.token_groups[token_group]:
                            raise RuntimeError(
                                f"empty token group {token_group!r} for {contrast.contrast_id}"
                            )
                        if len(source.token_groups[token_group]) != len(target.token_groups[token_group]):
                            raise RuntimeError(
                                f"token group size mismatch {token_group!r}: "
                                f"{len(source.token_groups[token_group])} versus "
                                f"{len(target.token_groups[token_group])}"
                            )
                        patched_logits = _run_with_output_patch(
                            model=model,
                            prepared=source,
                            sites=[site],
                            donor_cache=target_cache,
                            recipient_groups=source.token_groups,
                            donor_groups=target.token_groups,
                            token_group=token_group,
                        )
                        patched_score = _score_logits(
                            patched_logits,
                            labels=readout_tokens,
                            label_token_ids=label_token_ids,
                            source_label=source_readout,
                            target_label=target_readout,
                        )
                        row = {
                            "experiment": "token_group_patch",
                            "site_type": "residual",
                            "component": "residual_stream",
                            "token_group": token_group,
                            "layer_index": layer_index,
                            "stage_index": layer_index + 1,
                            "recipient_token_count": len(source.token_groups[token_group]),
                            **_base_row(
                                contrast,
                                source_score=source_score,
                                target_score=target_score,
                                patched_score=patched_score,
                            ),
                        }
                        _append_jsonl(row_stream, row)
                        rows.append(row)
                        completed_interventions += 1

            if "ablation" in modes:
                for layer_index in selected_layers:
                    site = residual_sites[layer_index]
                    for ablation_type in ("resample_other_answer", "pair_mean", "zero"):
                        donor = target_cache[site.key].clone()
                        source_position = source.token_groups["final"][0]
                        target_position = target.token_groups["final"][0]
                        if ablation_type == "pair_mean":
                            donor[0, target_position] = 0.5 * (
                                target_cache[site.key][0, target_position].float()
                                + source_cache[site.key][0, source_position].float()
                            ).to(donor.dtype)
                        elif ablation_type == "zero":
                            donor[0, target_position].zero_()
                        patched_logits = _run_with_output_patch(
                            model=model,
                            prepared=source,
                            sites=[site],
                            donor_cache={site.key: donor},
                            recipient_groups=source.token_groups,
                            donor_groups=target.token_groups,
                            token_group="final",
                        )
                        patched_score = _score_logits(
                            patched_logits,
                            labels=readout_tokens,
                            label_token_ids=label_token_ids,
                            source_label=source_readout,
                            target_label=target_readout,
                        )
                        row = {
                            "experiment": "residual_ablation",
                            "site_type": "residual",
                            "component": "residual_stream",
                            "token_group": "final",
                            "layer_index": layer_index,
                            "stage_index": layer_index + 1,
                            "ablation_type": ablation_type,
                            "source_margin_drop": float(source_score["source_margin"])
                            - float(patched_score["source_margin"]),
                            **_base_row(
                                contrast,
                                source_score=source_score,
                                target_score=target_score,
                                patched_score=patched_score,
                            ),
                        }
                        _append_jsonl(row_stream, row)
                        rows.append(row)
                        completed_interventions += 1

            if "component" in modes:
                for component in ("attention", "mlp"):
                    all_component_sites = _component_sites(layers, component)
                    component_sites = [all_component_sites[index] for index in selected_layers]
                    with capture_outputs(component_sites) as component_target_cache:
                        _language_forward(model, target)
                    for layer_index, site in zip(selected_layers, component_sites, strict=True):
                        patched_logits = _run_with_output_patch(
                            model=model,
                            prepared=source,
                            sites=[site],
                            donor_cache=component_target_cache,
                            recipient_groups=source.token_groups,
                            donor_groups=target.token_groups,
                            token_group="final",
                        )
                        patched_score = _score_logits(
                            patched_logits,
                            labels=readout_tokens,
                            label_token_ids=label_token_ids,
                            source_label=source_readout,
                            target_label=target_readout,
                        )
                        row = {
                            "experiment": "component_patch",
                            "site_type": component,
                            "component": component,
                            "token_group": "final",
                            "layer_index": layer_index,
                            "stage_index": layer_index + 1,
                            **_base_row(
                                contrast,
                                source_score=source_score,
                                target_score=target_score,
                                patched_score=patched_score,
                            ),
                        }
                        _append_jsonl(row_stream, row)
                        rows.append(row)
                        completed_interventions += 1

            if "head" in modes:
                all_head_sites = _head_sites(layers)
                head_sites = [all_head_sites[index] for index in selected_layers]
                with capture_inputs(head_sites) as head_target_cache:
                    _language_forward(model, target)
                for layer_index, site in zip(selected_layers, head_sites, strict=True):
                    attention = layers[layer_index].self_attn
                    for head_index in range(int(attention.num_heads)):
                        with patch_head_inputs(
                            {
                                site.key: (
                                    site,
                                    head_target_cache[site.key],
                                    [head_index],
                                    source.token_groups["final"],
                                    target.token_groups["final"],
                                    int(attention.head_dim),
                                )
                            }
                        ):
                            patched_logits = _language_forward(model, source)
                        patched_score = _score_logits(
                            patched_logits,
                            labels=readout_tokens,
                            label_token_ids=label_token_ids,
                            source_label=source_readout,
                            target_label=target_readout,
                        )
                        row = {
                            "experiment": "head_patch",
                            "site_type": "attention_head",
                            "component": "attention",
                            "token_group": "final",
                            "layer_index": layer_index,
                            "stage_index": layer_index + 1,
                            "head_index": head_index,
                            **_base_row(
                                contrast,
                                source_score=source_score,
                                target_score=target_score,
                                patched_score=patched_score,
                            ),
                        }
                        _append_jsonl(row_stream, row)
                        rows.append(row)
                        completed_interventions += 1

            if "edge" in modes:
                edge_groups = (
                    "image_all",
                    "image_query_expanded",
                    "image_random_matched",
                    "option_content",
                    "option_labels",
                )
                for layer_index in selected_layers:
                    site = HookSite(
                        f"attention_edges_layer_{layer_index}",
                        layers[layer_index].self_attn,
                        "attention_mask_edges",
                    )
                    for edge_group in edge_groups:
                        if not source.token_groups.get(edge_group):
                            raise RuntimeError(
                                f"empty edge token group {edge_group!r} for {contrast.contrast_id}"
                            )
                        with block_attention_edges(
                            [site],
                            query_positions=source.token_groups["final"],
                            key_positions=source.token_groups[edge_group],
                        ):
                            patched_logits = _language_forward(model, source)
                        patched_score = _score_logits(
                            patched_logits,
                            labels=readout_tokens,
                            label_token_ids=label_token_ids,
                            source_label=source_readout,
                            target_label=target_readout,
                        )
                        row = {
                            "experiment": "edge_ablation",
                            "site_type": "attention_edges",
                            "component": "attention",
                            "token_group": edge_group,
                            "edge_route": f"final_to_{edge_group}",
                            "layer_index": layer_index,
                            "stage_index": layer_index + 1,
                            "source_margin_drop": float(source_score["source_margin"])
                            - float(patched_score["source_margin"]),
                            **_base_row(
                                contrast,
                                source_score=source_score,
                                target_score=target_score,
                                patched_score=patched_score,
                            ),
                        }
                        _append_jsonl(row_stream, row)
                        rows.append(row)
                        completed_interventions += 1

            if "path" in modes:
                if args.path_sender_layer is None or args.path_receiver_layer is None:
                    raise ValueError("path mode requires --path-sender-layer and --path-receiver-layer")
                if args.path_sender_layer >= args.path_receiver_layer:
                    raise ValueError("path sender layer must be before receiver layer")
                sender_site = residual_sites[args.path_sender_layer]
                receiver_site = residual_sites[args.path_receiver_layer]
                with path_patch_outputs(
                    sender_site=sender_site,
                    sender_donor=target_cache[sender_site.key],
                    sender_recipient_positions=source.token_groups[args.path_sender_group],
                    sender_donor_positions=target.token_groups[args.path_sender_group],
                    receiver_site=receiver_site,
                    receiver_clean=source_cache[receiver_site.key],
                    receiver_positions=source.token_groups[args.path_receiver_group],
                ):
                    patched_logits = _language_forward(model, source)
                patched_score = _score_logits(
                    patched_logits,
                    labels=readout_tokens,
                    label_token_ids=label_token_ids,
                    source_label=source_readout,
                    target_label=target_readout,
                )
                row = {
                    "experiment": "token_path_patch",
                    "site_type": "residual_path",
                    "component": "residual_stream",
                    "token_group": args.path_sender_group,
                    "path_sender_layer": args.path_sender_layer,
                    "path_receiver_layer": args.path_receiver_layer,
                    "path_receiver_group": args.path_receiver_group,
                    **_base_row(
                        contrast,
                        source_score=source_score,
                        target_score=target_score,
                        patched_score=patched_score,
                    ),
                }
                _append_jsonl(row_stream, row)
                rows.append(row)
                completed_interventions += 1

            if "binding_fit" in modes:
                for layer_index in selected_layers:
                    site = residual_sites[layer_index]
                    source_vector = source_cache[site.key][0, source.token_groups["final"][0]].float()
                    target_vector = target_cache[site.key][0, target.token_groups["final"][0]].float()
                    binding_key = (
                        f"{contrast.contrast_type}:{contrast.source['correct_label']}"
                        f"->{contrast.target['correct_label']}:layer_{layer_index}:final"
                    )
                    binding_differences[binding_key].append(target_vector - source_vector)

            if "binding_apply" in modes:
                for layer_index in selected_layers:
                    binding_key = (
                        f"{contrast.contrast_type}:{contrast.source['correct_label']}"
                        f"->{contrast.target['correct_label']}:layer_{layer_index}:final"
                    )
                    vector = loaded_binding_vectors.get(binding_key)
                    if vector is None:
                        continue
                    site = residual_sites[layer_index]
                    for scale in binding_scales:
                        with add_to_outputs(
                            {
                                site.key: (
                                    site,
                                    vector,
                                    source.token_groups["final"],
                                    scale,
                                )
                            }
                        ):
                            patched_logits = _language_forward(model, source)
                        patched_score = _score_logits(
                            patched_logits,
                            labels=readout_tokens,
                            label_token_ids=label_token_ids,
                            source_label=source_readout,
                            target_label=target_readout,
                        )
                        row = {
                            "experiment": "binding_vector_addition",
                            "site_type": "residual_direction",
                            "component": "residual_stream",
                            "token_group": "final",
                            "layer_index": layer_index,
                            "stage_index": layer_index + 1,
                            "binding_key": binding_key,
                            "scale": scale,
                            **_base_row(
                                contrast,
                                source_score=source_score,
                                target_score=target_score,
                                patched_score=patched_score,
                            ),
                        }
                        _append_jsonl(row_stream, row)
                        rows.append(row)
                        completed_interventions += 1

            if contrast_index % max(1, args.progress_every) == 0 or contrast_index == len(directed):
                print(
                    f"[progress] contrasts={contrast_index}/{len(directed)} "
                    f"interventions={completed_interventions} elapsed_s={time.time()-started:.1f}",
                    flush=True,
                )

    if "binding_fit" in modes:
        binding_vectors = {
            key: torch.stack(values, dim=0).mean(dim=0)
            for key, values in binding_differences.items()
            if values
        }
        torch.save(
            {
                "vectors": binding_vectors,
                "counts": {key: len(values) for key, values in binding_differences.items()},
                "selected_pair_ids": selected_pairs,
                "source_git_commit": os.environ.get("SOURCE_GIT_COMMIT", "unknown"),
            },
            args.output_dir / "binding_vectors.pt",
        )
        print(f"[binding] fitted_vectors={len(binding_vectors)}", flush=True)

    aggregate = _aggregate_rows(rows)
    summary = {
        **aggregate,
        "model_id": args.model_id,
        "source_git_commit": os.environ.get("SOURCE_GIT_COMMIT", "unknown"),
        "manifest": str(args.manifest),
        "split": args.split,
        "selected_pair_ids": selected_pairs,
        "base_contrast_count": len(contrasts),
        "directed_contrast_count": len(directed),
        "clean_readout_gate_passed": clean_gate_passed,
        "clean_readout_gate_skipped": clean_gate_skipped,
        "modes": modes,
        "contrast_types": contrast_types,
        "num_layers": len(layers),
        "layer_module_path": layers_path,
        "elapsed_seconds": time.time() - started,
        "completed_interventions": completed_interventions,
        "label_token_ids": label_token_ids,
    }
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(
        args.output_dir / "token_maps.json",
        {
            sample_id: {
                "input_token_count": int(value.input_ids.shape[1]),
                "image_grid": list(value.image_grid),
                "token_groups": value.token_groups,
            }
            for sample_id, value in prepared_cache.items()
        },
    )
    print(f"[summary] {json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
