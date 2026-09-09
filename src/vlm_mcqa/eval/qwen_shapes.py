"""Evaluate Qwen2.5-VL on the controlled shapes MCQA dataset.

The evaluator scores the allowed answer labels directly from next-token logits.
For every answer-position variant it also applies a logit lens to the
final-token residual stream after every language layer.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ANSWER_LABELS = ("A", "B", "C", "D")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _single_token_id(tokenizer: Any, label: str) -> tuple[int, dict[str, list[int]]]:
    candidates = {
        "bare": tokenizer.encode(label, add_special_tokens=False),
        "space_prefixed": tokenizer.encode(" " + label, add_special_tokens=False),
    }
    if len(candidates["bare"]) == 1:
        return candidates["bare"][0], candidates
    if len(candidates["space_prefixed"]) == 1:
        return candidates["space_prefixed"][0], candidates
    raise ValueError(f"Answer label {label!r} is not one token: {candidates}")


def _resolve_attr(root: Any, paths: Iterable[str]) -> tuple[Any, str]:
    for path in paths:
        current = root
        try:
            for part in path.split("."):
                current = getattr(current, part)
        except AttributeError:
            continue
        return current, path
    raise AttributeError(f"None of the candidate module paths exist: {list(paths)}")


def _should_collect_lens(record: dict[str, Any]) -> bool:
    return record["variant_family"] == "answer_position"


def _probe_labels(labels: Iterable[str]) -> list[str]:
    """Return default MCQ letters plus the requested labels without duplicates."""
    requested = list(labels)
    return list(dict.fromkeys((*DEFAULT_ANSWER_LABELS, *requested)))


def _image_token_id(model: Any) -> int:
    """Resolve the image placeholder token across Transformers config layouts."""
    candidates = (
        getattr(model.config, "image_token_id", None),
        getattr(getattr(model.config, "text_config", None), "image_token_id", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return int(candidate)
    raise AttributeError("Could not resolve image_token_id from the model configuration")


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[bool]] = defaultdict(list)
    global_compliance: dict[str, list[bool]] = defaultdict(list)
    valid_mass: dict[str, list[float]] = defaultdict(list)
    predictions: Counter[str] = Counter()
    for record in records:
        groups["overall"].append(record["is_correct"])
        groups[f"family:{record['variant_family']}"].append(record["is_correct"])
        groups[f"scheme:{record['label_scheme']}"].append(record["is_correct"])
        groups[
            f"family_scheme:{record['variant_family']}:{record['label_scheme']}"
        ].append(record["is_correct"])
        if record["variant_family"] == "answer_position":
            groups[f"correct_position:{record['correct_index'] + 1}"].append(
                record["is_correct"]
            )
        predictions[record["predicted_label"]] += 1
        scheme = record["label_scheme"]
        global_compliance[scheme].append(record["global_top_token_is_valid_label"])
        valid_mass[scheme].append(record["valid_label_probability_mass"])

    metrics = {
        name: {"count": len(values), "accuracy": sum(values) / len(values)}
        for name, values in sorted(groups.items())
    }
    vocabulary_metrics = {
        scheme: {
            "count": len(global_compliance[scheme]),
            "global_top_token_valid_rate": (
                sum(global_compliance[scheme]) / len(global_compliance[scheme])
            ),
            "mean_valid_label_probability_mass": (
                sum(valid_mass[scheme]) / len(valid_mass[scheme])
            ),
        }
        for scheme in sorted(global_compliance)
    }
    return {
        "metrics": metrics,
        "prediction_counts": dict(sorted(predictions.items())),
        "vocabulary_metrics": vocabulary_metrics,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=224 * 224)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluator")

    print(f"[load] model={args.model_id}", flush=True)
    print(f"[load] HF_HOME={os.environ.get('HF_HOME')}", flush=True)
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
    print(
        f"[load] class={type(model).__name__} "
        f"allocated_gib={torch.cuda.memory_allocated() / 2**30:.2f}",
        flush=True,
    )

    norm, norm_path = _resolve_attr(
        model,
        (
            "model.language_model.norm",
            "model.norm",
            "language_model.model.norm",
            "language_model.norm",
        ),
    )
    lm_head, lm_head_path = _resolve_attr(model, ("lm_head", "language_model.lm_head"))
    print(f"[model] norm={norm_path} lm_head={lm_head_path}", flush=True)
    image_token_id = _image_token_id(model)
    print(f"[model] image_token_id={image_token_id}", flush=True)

    all_records = _read_jsonl(args.manifest)
    if args.limit is not None:
        all_records = all_records[: args.limit]
    print(f"[data] records={len(all_records)} manifest={args.manifest}", flush=True)

    all_labels = sorted({label for record in all_records for label in record["labels"]})
    semantic_tokens = sorted(
        {content for record in all_records for content in record["option_contents"]}
    )
    label_token_ids: dict[str, int] = {}
    tokenization: dict[str, Any] = {}
    for token in [*all_labels, *semantic_tokens]:
        token_id, candidates = _single_token_id(processor.tokenizer, token)
        label_token_ids[token] = token_id
        tokenization[token] = {
            "selected_id": token_id,
            "selected_decoded": processor.tokenizer.decode([token_id]),
            "candidates": candidates,
        }
    _write_json(args.output_dir / "label_tokenization.json", tokenization)
    print(f"[tokens] {json.dumps(tokenization, sort_keys=True)}", flush=True)

    results_path = args.output_dir / "predictions.jsonl"
    lens_path = args.output_dir / "layerwise_label_logits.jsonl"
    prediction_rows: list[dict[str, Any]] = []
    started = time.time()

    with results_path.open("w", encoding="utf-8") as result_stream, lens_path.open(
        "w", encoding="utf-8"
    ) as lens_stream, torch.inference_mode():
        for index, record in enumerate(all_records, start=1):
            image_path = args.manifest.parent / record["image"]
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
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[rendered], images=[image], padding=True, return_tensors="pt"
            )
            inputs = {
                key: value.to(model.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            collect_lens = _should_collect_lens(record)
            outputs = model(
                **inputs,
                output_hidden_states=collect_lens,
                use_cache=False,
                return_dict=True,
            )
            labels = record["labels"]
            ids = torch.tensor(
                [label_token_ids[label] for label in labels], device=outputs.logits.device
            )
            full_logits = outputs.logits[0, -1].float()
            scores = full_logits.index_select(0, ids)
            predicted_index = int(scores.argmax().item())
            label_probabilities = torch.softmax(scores, dim=0)
            full_log_partition = torch.logsumexp(full_logits, dim=0)
            valid_label_probability_mass = torch.exp(
                torch.logsumexp(scores, dim=0) - full_log_partition
            )
            correct_score = scores[record["correct_index"]]
            global_top_token_id = int(full_logits.argmax().item())
            global_top_values, global_top_ids = torch.topk(full_logits, k=5)
            prediction = {
                **record,
                "predicted_index": predicted_index,
                "predicted_label": labels[predicted_index],
                "is_correct": predicted_index == record["correct_index"],
                "label_logits": {
                    label: float(score) for label, score in zip(labels, scores.tolist(), strict=True)
                },
                "restricted_label_probabilities": {
                    label: float(probability)
                    for label, probability in zip(
                        labels, label_probabilities.tolist(), strict=True
                    )
                },
                "valid_label_probability_mass": float(valid_label_probability_mass),
                "correct_label_full_vocabulary_probability": float(
                    torch.exp(correct_score - full_log_partition)
                ),
                "correct_label_global_rank": int((full_logits > correct_score).sum().item() + 1),
                "global_top_token_id": global_top_token_id,
                "global_top_token_text": processor.tokenizer.decode([global_top_token_id]),
                "global_top_token_is_valid_label": global_top_token_id in ids.tolist(),
                "global_top_five": [
                    {
                        "token_id": int(token_id),
                        "text": processor.tokenizer.decode([int(token_id)]),
                        "logit": float(logit),
                    }
                    for token_id, logit in zip(
                        global_top_ids.tolist(), global_top_values.tolist(), strict=True
                    )
                ],
                "correct_logit_margin": float(
                    correct_score
                    - torch.cat(
                        [scores[: record["correct_index"]], scores[record["correct_index"] + 1 :]]
                    ).max()
                ),
                "input_token_count": int(inputs["input_ids"].shape[1]),
                "visual_token_count": int(
                    (inputs["input_ids"] == image_token_id).sum().item()
                ),
            }
            prediction_rows.append(prediction)
            result_stream.write(json.dumps(prediction, sort_keys=True) + "\n")
            result_stream.flush()

            if collect_lens:
                hidden_states = outputs.hidden_states
                if hidden_states is None:
                    raise RuntimeError("Model did not return hidden states")
                probe_labels = _probe_labels(labels)
                probe_ids = torch.tensor(
                    [label_token_ids[label] for label in probe_labels],
                    device=outputs.logits.device,
                )
                probe_head_weight = lm_head.weight.index_select(
                    0, probe_ids.to(lm_head.weight.device)
                )
                semantic_ids = torch.tensor(
                    [label_token_ids[token] for token in semantic_tokens],
                    device=outputs.logits.device,
                )
                semantic_head_weight = lm_head.weight.index_select(
                    0, semantic_ids.to(lm_head.weight.device)
                )
                correct_content_index = semantic_tokens.index(record["correct_content"])
                head_bias = getattr(lm_head, "bias", None)
                for stage_index, hidden in enumerate(hidden_states):
                    vector = hidden[0, -1]
                    # Hugging Face decoder outputs conventionally returns the final
                    # state after final norm, while earlier stages are pre-final-norm.
                    projected = vector if stage_index == len(hidden_states) - 1 else norm(vector)
                    probe_scores = probe_head_weight.float() @ projected.float()
                    if head_bias is not None:
                        probe_scores = probe_scores + head_bias.index_select(
                            0, probe_ids
                        ).float()
                    semantic_scores = semantic_head_weight.float() @ projected.float()
                    if head_bias is not None:
                        semantic_scores = semantic_scores + head_bias.index_select(
                            0, semantic_ids
                        ).float()
                    probe_by_label = dict(
                        zip(probe_labels, probe_scores.tolist(), strict=True)
                    )
                    layer_scores = torch.tensor(
                        [probe_by_label[label] for label in labels],
                        device=probe_scores.device,
                    )
                    correct = layer_scores[record["correct_index"]]
                    wrong = torch.cat(
                        [
                            layer_scores[: record["correct_index"]],
                            layer_scores[record["correct_index"] + 1 :],
                        ]
                    )
                    correct_content_score = semantic_scores[correct_content_index]
                    wrong_content_scores = torch.cat(
                        [
                            semantic_scores[:correct_content_index],
                            semantic_scores[correct_content_index + 1 :],
                        ]
                    )
                    lens_row = {
                        "sample_id": record["sample_id"],
                        "pair_id": record["pair_id"],
                        "pair_member": record["pair_member"],
                        "label_scheme": record["label_scheme"],
                        "labels": labels,
                        "correct_index": record["correct_index"],
                        "correct_label": record["correct_label"],
                        "stage_index": stage_index,
                        "stage_name": (
                            "embedding"
                            if stage_index == 0
                            else "final_norm"
                            if stage_index == len(hidden_states) - 1
                            else f"layer_{stage_index}"
                        ),
                        "label_logits": {
                            label: float(score)
                            for label, score in zip(
                                labels, layer_scores.tolist(), strict=True
                            )
                        },
                        "probe_labels": probe_labels,
                        "probe_label_logits": {
                            label: float(score)
                            for label, score in zip(
                                probe_labels, probe_scores.tolist(), strict=True
                            )
                        },
                        "semantic_content_tokens": semantic_tokens,
                        "semantic_content_logits": {
                            token: float(score)
                            for token, score in zip(
                                semantic_tokens, semantic_scores.tolist(), strict=True
                            )
                        },
                        "correct_content": record["correct_content"],
                        "correct_content_logit_margin": float(
                            correct_content_score - wrong_content_scores.max()
                        ),
                        "correct_content_probability": float(
                            torch.softmax(semantic_scores, dim=0)[correct_content_index]
                        ),
                        "correct_logit_margin": float(correct - wrong.max()),
                        "correct_label_probability": float(
                            torch.softmax(layer_scores, dim=0)[record["correct_index"]]
                        ),
                    }
                    if tuple(labels) != DEFAULT_ANSWER_LABELS:
                        probe_probabilities = torch.softmax(probe_scores, dim=0)
                        default_count = len(DEFAULT_ANSWER_LABELS)
                        requested_correct_logit = probe_by_label[
                            labels[record["correct_index"]]
                        ]
                        position_default_label = DEFAULT_ANSWER_LABELS[
                            record["correct_index"]
                        ]
                        lens_row.update(
                            {
                                "dual_alphabet_probability_scope": (
                                    "softmax restricted to default A/B/C/D plus "
                                    "the four requested labels"
                                ),
                                "default_alphabet_probability_mass_8way": float(
                                    probe_probabilities[:default_count].sum()
                                ),
                                "requested_alphabet_probability_mass_8way": float(
                                    probe_probabilities[default_count:].sum()
                                ),
                                "position_matched_default_label": position_default_label,
                                "requested_correct_minus_position_default_logit": float(
                                    requested_correct_logit
                                    - probe_by_label[position_default_label]
                                ),
                            }
                        )
                    lens_stream.write(json.dumps(lens_row, sort_keys=True) + "\n")
                lens_stream.flush()

            del outputs, inputs, full_logits
            if index % args.progress_every == 0 or index == len(all_records):
                elapsed = time.time() - started
                accuracy = sum(row["is_correct"] for row in prediction_rows) / len(
                    prediction_rows
                )
                print(
                    f"[progress] {index}/{len(all_records)} "
                    f"accuracy={accuracy:.4f} elapsed_s={elapsed:.1f} "
                    f"gpu_gib={torch.cuda.memory_allocated() / 2**30:.2f}",
                    flush=True,
                )

    summary = _aggregate(prediction_rows)
    summary.update(
        {
            "model_id": args.model_id,
            "manifest": str(args.manifest),
            "record_count": len(prediction_rows),
            "elapsed_seconds": time.time() - started,
            "source_git_commit": os.environ.get("SOURCE_GIT_COMMIT", "unknown"),
            "norm_module": norm_path,
            "lm_head_module": lm_head_path,
            "max_pixels": args.max_pixels,
        }
    )
    _write_json(args.output_dir / "summary.json", summary)
    print(f"[summary] {json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
