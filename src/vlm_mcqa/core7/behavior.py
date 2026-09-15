"""Behavioural screening: accuracy, image dependence, compliance, pair yield.

Used by ``screen`` (screening split, all twelve candidates) and by the first
Core-7 step (behaviour split). All accuracies are forced choice among the four
displayed labels from full-VLM next-token logits; full-vocabulary compliance is
reported separately.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Sequence

import torch

from .scoring import group_rates, parse_generated_label, symbol_readout
from .session import Session, write_jsonl_row


def _score(session: Session, variant: dict[str, Any], image_override: str | None, condition: str) -> dict[str, Any]:
    adapter = session.adapter
    image = session.image(image_override) if image_override else session.image(variant["image"])
    if condition == "blank":
        from PIL import Image

        image = Image.new("RGB", image.size, (128, 128, 128))
    prepared = adapter.prepare(variant, image)
    readout = symbol_readout(prepared.clean_logits, session.label_ids(prepared, variant["labels"]),
                             variant["correct_index"], tokenizer=adapter.tokenizer)
    return {
        "variant_id": variant["variant_id"],
        "item_id": variant["item_id"],
        "split": variant["split"],
        "scheme": variant["label_scheme"],
        "position": variant["correct_index"],
        "family": variant["question_family"],
        "depth": variant["program_depth"],
        "pair_id": variant.get("counterfactual_pair_id"),
        "pair_member": variant.get("pair_member"),
        "condition": condition,
        "num_tokens": prepared.num_tokens,
        "num_image_tokens": len(prepared.groups["image"]),
        **readout,
    }


def _shuffled_image(session: Session, item: dict[str, Any], pool: Sequence[dict[str, Any]]) -> str:
    candidates = [o for o in pool if o["item_id"] != item["item_id"]
                  and o["semantic_answer"] != item["semantic_answer"]] or [o for o in pool if o is not item]
    return session.rng.choice(candidates)["image"]


def run_behavior(session: Session, *, split: str, n_items: int | None, robust_items: int,
                 compliance_items: int, controls: bool = True, pair_splits: Sequence[str] = ()) -> dict[str, Any]:
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    schemes = session.profile.get("schemes", ["letters", "numbers", "rare_letters"])
    items = session.items_in(split, n_items)
    rows: list[dict[str, Any]] = []
    with session.jsonl(f"behavior_{split}.jsonl") as stream:
        def emit(row):
            rows.append(row)
            write_jsonl_row(stream, row)

        for index, item in enumerate(items):
            use_schemes = schemes if index < robust_items else ["letters"]
            for variant in session.variants_of(item["item_id"], schemes=use_schemes):
                emit(_score(session, variant, None, "clean"))
            if controls:
                shuffled = _shuffled_image(session, item, items)
                for variant in session.variants_of(item["item_id"], schemes=["letters"]):
                    emit(_score(session, variant, None, "blank"))
                    emit(_score(session, variant, shuffled, "shuffled"))
            if (index + 1) % 25 == 0:
                print(f"[behavior:{split}] {index + 1}/{len(items)} items", flush=True)

        pair_rows = []
        for pair_split in pair_splits:
            for pair_id in session.pair_ids(pair_split):
                for item in (i for i in session.items.values() if i.get("counterfactual_pair_id") == pair_id):
                    for kind in ("position", "distractor_permuted"):
                        for variant in session.variants_of(item["item_id"], schemes=["letters"], kind=kind):
                            row = _score(session, variant, None, "clean")
                            row["variant_kind"] = kind
                            pair_rows.append(row)
                            write_jsonl_row(stream, row)

    compliance = []
    with session.jsonl(f"compliance_{split}.jsonl") as stream:
        for index, item in enumerate(items[:compliance_items]):
            variants = session.variants_of(item["item_id"], schemes=["letters"])
            variant = variants[index % len(variants)]
            text = session.adapter.generate(session.image(variant["image"]), variant["prompt"])
            parsed = parse_generated_label(text, variant["labels"])
            row = {"variant_id": variant["variant_id"], "generated": text, "parsed": parsed,
                   "compliant": parsed is not None, "correct": parsed == variant["correct_symbol"]}
            compliance.append(row)
            write_jsonl_row(stream, row)

    summary = summarize_behavior(rows, pair_rows, compliance, session)
    summary["elapsed_seconds"] = time.time() - started
    if torch.cuda.is_available():
        summary["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
    session.write_json(f"behavior_{split}_summary.json", summary)
    return summary


def pair_yield(pair_rows: Sequence[dict[str, Any]], contrasts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    correct = {r["variant_id"]: r["correct"] for r in pair_rows}
    by_split = defaultdict(lambda: {"pairs": set(), "usable_pairs": set(), "usable_contrasts": defaultdict(int)})
    per_pair_types = defaultdict(set)
    for c in contrasts:
        if c["donor_variant_id"] not in correct or c["recipient_variant_id"] not in correct:
            continue
        entry = by_split[c["split"]]
        entry["pairs"].add(c["pair_id"])
        if correct[c["donor_variant_id"]] and correct[c["recipient_variant_id"]]:
            entry["usable_contrasts"][c["contrast_type"]] += 1
            per_pair_types[(c["split"], c["pair_id"])].add(c["contrast_type"])
    needed = {"position", "content", "content_and_position"}
    for (split, pair_id), types in per_pair_types.items():
        if needed <= types:
            by_split[split]["usable_pairs"].add(pair_id)
    return {split: {"pairs": len(v["pairs"]), "usable_pairs": len(v["usable_pairs"]),
                    "usable_contrasts": dict(v["usable_contrasts"])} for split, v in by_split.items()}


def summarize_behavior(rows, pair_rows, compliance, session: Session) -> dict[str, Any]:
    clean = [r for r in rows if r["condition"] == "clean"]
    letters = [r for r in clean if r["scheme"] == "letters"]
    robust_ids = {r["item_id"] for r in clean if r["scheme"] != "letters"}
    robust = [r for r in clean if r["item_id"] in robust_ids]
    cells = defaultdict(list)
    for r in robust:
        cells[(r["scheme"], r["position"])].append(r["correct"])
    cell_acc = {f"{s}:p{p}": sum(v) / len(v) for (s, p), v in sorted(cells.items())}
    acc = lambda rs: sum(r["correct"] for r in rs) / len(rs) if rs else float("nan")  # noqa: E731
    by_position = group_rates(letters, "position")
    blank = [r for r in rows if r["condition"] == "blank"]
    shuffled = [r for r in rows if r["condition"] == "shuffled"]
    per_item = defaultdict(list)
    for r in letters:
        per_item[r["item_id"]].append(r["correct"])
    return {
        "items": len({r["item_id"] for r in clean}),
        "letters_accuracy": acc(letters),
        "accuracy_by_position": by_position,
        "worst_position_accuracy": min(by_position.values()) if by_position else float("nan"),
        "strict_item_consistency": (sum(all(v) for v in per_item.values()) / len(per_item)) if per_item else float("nan"),
        "accuracy_by_family": group_rates(letters, "family"),
        "accuracy_by_depth": group_rates(letters, "depth"),
        "robustness_cells": cell_acc,
        "worst_scheme_position_accuracy": min(cell_acc.values()) if cell_acc else float("nan"),
        "accuracy_by_scheme": group_rates(robust, "scheme"),
        "global_top_is_label_rate": group_rates(clean, "scheme", "global_top_is_label"),
        "mean_label_mass_full": group_rates(clean, "scheme", "label_mass_full"),
        "blank_accuracy": acc(blank),
        "shuffled_accuracy": acc(shuffled),
        "clean_minus_shuffled": acc(letters) - acc(shuffled) if shuffled else float("nan"),
        "clean_minus_blank": acc(letters) - acc(blank) if blank else float("nan"),
        "compliance_rate": acc([{"correct": c["compliant"]} for c in compliance]),
        "generated_accuracy": acc(compliance),
        "pair_yield": pair_yield(pair_rows, session.contrasts) if pair_rows else {},
        "mean_tokens": sum(r["num_tokens"] for r in clean) / len(clean) if clean else None,
        "mean_image_tokens": sum(r["num_image_tokens"] for r in clean) / len(clean) if clean else None,
    }
