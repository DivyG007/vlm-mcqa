"""Prompt variants, causal contrasts and validation for four-option MCQ items.

The item/variant/contrast schema is shared by CLEVR-MCQ-4 and the IconQA
validation subset.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

PROMPT_TEMPLATE_VERSION = "mcq4-v1"
ALPHABETS: dict[str, tuple[str, str, str, str]] = {
    "letters": ("A", "B", "C", "D"),
    "numbers": ("1", "2", "3", "4"),
    "rare_letters": ("Q", "Z", "R", "X"),
    "punctuation": ("!", "@", "#", "$"),  # optional; not required by the proposal
}
REQUIRED_SCHEMES = ("letters", "numbers", "rare_letters")
CONTRAST_TYPES = ("position", "content", "content_and_position", "identity", "random_pair")


def format_prompt(question: str, options: Sequence[str], labels: Sequence[str]) -> str:
    lines = [question]
    lines.extend(f"{label}. {option}" for label, option in zip(labels, options, strict=True))
    lines.append("Respond with only the option label.")
    lines.append("Answer:")
    return "\n".join(lines)


def rotate_to(options: Sequence[str], answer: str, position: int) -> list[str]:
    """Cyclically rotate ``options`` so ``answer`` lands at ``position``.

    Rotation keeps the distractors' relative order, so each content appears at
    every position exactly once across the four variants.
    """
    shift = (position - list(options).index(answer)) % 4
    return [options[(i - shift) % 4] for i in range(4)]


def variant_id(item_id: str, scheme: str, position: int, kind: str = "position") -> str:
    suffix = "" if kind == "position" else f"|{kind}"
    return f"{item_id}|{scheme}|p{position}{suffix}"


def _variant(item: Mapping[str, Any], scheme: str, position: int, options: list[str], kind: str):
    labels = ALPHABETS[scheme]
    correct = options.index(item["semantic_answer"])
    return {
        "variant_id": variant_id(item["item_id"], scheme, position, kind),
        "item_id": item["item_id"],
        "split": item["split"],
        "scene_id": item["scene_id"],
        "counterfactual_pair_id": item.get("counterfactual_pair_id"),
        "pair_member": item.get("pair_member"),
        "image": item["image"],
        "question": item["question"],
        "question_family": item["question_family"],
        "program_depth": item["program_depth"],
        "label_scheme": scheme,
        "labels": list(labels),
        "option_contents": list(options),
        "semantic_answer": item["semantic_answer"],
        "correct_index": correct,
        "correct_symbol": labels[correct],
        "variant_kind": kind,
        "prompt": format_prompt(item["question"], options, labels),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
    }


def build_variants(
    item: Mapping[str, Any], schemes: Sequence[str] = REQUIRED_SCHEMES, *, permuted_control: bool
) -> list[dict[str, Any]]:
    """All position x alphabet variants, plus distractor-permuted controls."""
    out = []
    for scheme in schemes:
        for position in range(4):
            options = rotate_to(item["option_contents"], item["semantic_answer"], position)
            out.append(_variant(item, scheme, position, options, "position"))
    if permuted_control:
        for position in range(4):
            options = rotate_to(item["option_contents"], item["semantic_answer"], position)
            others = [i for i in range(4) if i != position]
            permuted = list(options)
            # Cycle the three distractors: correct option and symbol stay fixed.
            for src, dst in zip(others, others[1:] + others[:1]):
                permuted[dst] = options[src]
            out.append(_variant(item, "letters", position, permuted, "distractor_permuted"))
    return out


def _contrast(kind, donor, recipient, **extra) -> dict[str, Any]:
    row = {
        "contrast_id": f"{kind}:{donor['variant_id']}->{recipient['variant_id']}",
        "contrast_type": kind,
        "split": recipient["split"],
        "pair_id": recipient.get("counterfactual_pair_id"),
        "donor_variant_id": donor["variant_id"],
        "recipient_variant_id": recipient["variant_id"],
        "donor_symbol": donor["correct_symbol"],
        "recipient_symbol": recipient["correct_symbol"],
        "donor_content": donor["semantic_answer"],
        "recipient_content": recipient["semantic_answer"],
        "label_scheme": recipient["label_scheme"],
    }
    row.update(extra)
    return row


def build_pair_contrasts(
    x_variants: Sequence[Mapping[str, Any]],
    y_variants: Sequence[Mapping[str, Any]],
    rng: random.Random,
    *,
    scheme: str = "letters",
) -> list[dict[str, Any]]:
    """Precompute position, content, content-and-position and identity contrasts."""
    def by_pos(variants, kind="position"):
        return {
            v["correct_index"]: v
            for v in variants
            if v["label_scheme"] == scheme and v["variant_kind"] == kind
        }

    xs, ys = by_pos(x_variants), by_pos(y_variants)
    xperm, yperm = by_pos(x_variants, "distractor_permuted"), by_pos(y_variants, "distractor_permuted")
    if set(xs) != set(range(4)) or set(ys) != set(range(4)):
        raise ValueError("pair lacks four position variants")
    out: list[dict[str, Any]] = []
    for image_variants in (xs, ys):
        for p in range(4):
            donor, recipient = image_variants[p], image_variants[(p + 1) % 4]
            out.append(_contrast("position", donor, recipient))
            out.append(_contrast("position", recipient, donor))
    for p in range(4):
        out.append(_contrast("content", xs[p], ys[p]))
        out.append(_contrast("content", ys[p], xs[p]))
    for donors, recipients in ((xs, ys), (ys, xs)):
        for p in range(4):
            donor = donors[p]
            options = []
            for q, recipient in recipients.items():
                content_pos = recipient["option_contents"].index(donor["semantic_answer"])
                if len({p, q, content_pos}) == 3:
                    options.append((recipient, content_pos))
            recipient, content_pos = rng.choice(options)
            out.append(_contrast(
                "content_and_position", donor, recipient,
                content_symbol=recipient["labels"][content_pos], content_index=content_pos,
            ))
    for perm, base in ((xperm, xs), (yperm, ys)):
        for p in sorted(perm):
            out.append(_contrast("identity", perm[p], base[p]))
            out.append(_contrast("identity", base[p], perm[p]))
    return out


def build_random_pair_contrasts(
    pair_variants: Mapping[str, Sequence[Mapping[str, Any]]], rng: random.Random, *, per_pair: int = 2
) -> list[dict[str, Any]]:
    """Donor from an unrelated pair in the same split, with a different symbol."""
    pair_ids = sorted(pair_variants)
    out = []
    for k, pair_id in enumerate(pair_ids):
        if len(pair_ids) < 2:
            break
        other = pair_ids[(k + 1) % len(pair_ids)]
        recipients = [v for v in pair_variants[pair_id]
                      if v["label_scheme"] == "letters" and v["variant_kind"] == "position"]
        donors = [v for v in pair_variants[other]
                  if v["label_scheme"] == "letters" and v["variant_kind"] == "position"]
        for recipient in rng.sample(recipients, min(per_pair, len(recipients))):
            choices = [d for d in donors if d["correct_symbol"] != recipient["correct_symbol"]]
            out.append(_contrast("random_pair", rng.choice(choices), recipient))
    return out


def assign_splits(
    groups: Sequence[str], sizes: Mapping[str, int], rng: random.Random
) -> dict[str, str]:
    """Assign whole groups (scenes) to splits; groups never cross splits."""
    shuffled = list(groups)
    rng.shuffle(shuffled)
    out, cursor = {}, 0
    for split, size in sizes.items():
        for group in shuffled[cursor:cursor + size]:
            out[group] = split
        cursor += size
    if cursor > len(shuffled):
        raise ValueError(f"need {cursor} groups, have {len(shuffled)}")
    return out


def validate(
    items: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Structural checks; raises on violations and returns summary counts."""
    errors: list[str] = []
    split_of_scene: dict[str, set[str]] = defaultdict(set)
    for item in items:
        split_of_scene[item["scene_id"]].add(item["split"])
        opts = item["option_contents"]
        if len(opts) != 4 or len(set(opts)) != 4:
            errors.append(f"{item['item_id']}: options not four distinct")
        if item["semantic_answer"] not in opts:
            errors.append(f"{item['item_id']}: answer not in options")
    errors += [f"scene {s} in splits {sorted(v)}" for s, v in split_of_scene.items() if len(v) > 1]
    by_item: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for v in variants:
        by_item[v["item_id"]].append(v)
        if v["labels"][v["correct_index"]] != v["correct_symbol"]:
            errors.append(f"{v['variant_id']}: symbol mismatch")
        if v["option_contents"][v["correct_index"]] != v["semantic_answer"]:
            errors.append(f"{v['variant_id']}: answer position mismatch")
    for item in items:
        for scheme in REQUIRED_SCHEMES:
            positions = sorted(
                v["correct_index"] for v in by_item[item["item_id"]]
                if v["label_scheme"] == scheme and v["variant_kind"] == "position"
            )
            if positions != [0, 1, 2, 3]:
                errors.append(f"{item['item_id']}: {scheme} positions {positions}")
    variant_by_id = {v["variant_id"]: v for v in variants}
    for c in contrasts:
        donor, recipient = variant_by_id[c["donor_variant_id"]], variant_by_id[c["recipient_variant_id"]]
        kind = c["contrast_type"]
        same_image = donor["image"] == recipient["image"]
        if kind == "position" and not (same_image and donor["correct_symbol"] != recipient["correct_symbol"]
                                       and donor["semantic_answer"] == recipient["semantic_answer"]):
            errors.append(f"{c['contrast_id']}: bad position contrast")
        if kind == "content" and not (not same_image and donor["correct_symbol"] == recipient["correct_symbol"]
                                      and donor["semantic_answer"] != recipient["semantic_answer"]
                                      and donor["question"] == recipient["question"]):
            errors.append(f"{c['contrast_id']}: bad content contrast")
        if kind == "content_and_position":
            trio = {donor["correct_index"], recipient["correct_index"], c["content_index"]}
            if len(trio) != 3 or same_image:
                errors.append(f"{c['contrast_id']}: bad content-and-position contrast")
        if kind == "identity" and not (same_image and donor["correct_symbol"] == recipient["correct_symbol"]):
            errors.append(f"{c['contrast_id']}: bad identity contrast")
    if errors:
        raise ValueError("dataset validation failed:\n" + "\n".join(errors[:50]))
    return {
        "items_by_split": dict(Counter(i["split"] for i in items)),
        "variants_by_split": dict(Counter(v["split"] for v in variants)),
        "contrasts_by_type": dict(Counter(c["contrast_type"] for c in contrasts)),
        "items_by_family": dict(Counter(i["question_family"] for i in items)),
        "correct_position_counts": dict(Counter(
            v["correct_index"] for v in variants if v["variant_kind"] == "position")),
    }


def read_jsonl(path) -> list[dict[str, Any]]:
    import json
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows: Iterable[Mapping[str, Any]]) -> int:
    import json
    count = 0
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count
