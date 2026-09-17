"""Usable contrast cases and a batched patch-evaluation engine.

A *case* is one directed contrast (donor activations patched into the
recipient's forward) whose donor and recipient are both answered correctly
(the proposal's correct/correct rule). ``evaluate`` turns patched recipient
logits into the readout appropriate for the contrast type:

- position / random_pair: donor symbol versus recipient symbol;
- content: donor content token versus recipient content token (symbol fixed);
- content_and_position: three-way outcome (``position`` = donor symbol,
  ``content`` = label where the donor's content sits in the recipient prompt,
  ``none`` = recipient symbol, ``other``);
- identity: change of the recipient's own margin (should stay near zero).
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .adapters import Prepared
from .hooks import Intervention
from .sampling import round_robin_across_groups
from .scoring import normalized_recovery
from .session import Clean, Session

NEEDED_TYPES = ("position", "content", "content_and_position")


@dataclass
class Case:
    contrast: dict[str, Any]
    donor: dict[str, Any]
    recipient: dict[str, Any]
    donor_clean: Clean
    recipient_clean: Clean
    prepared: Prepared
    label_ids: list[int]           # recipient's four displayed labels
    read_a: int | None             # token id pushed by a successful transfer
    read_b: int | None             # token id of the recipient's clean answer
    donor_diff: float
    recipient_diff: float

    @property
    def kind(self) -> str:
        return self.contrast["contrast_type"]


def _make_case(session: Session, contrast: Mapping[str, Any]) -> Case | None:
    donor = session.variants[contrast["donor_variant_id"]]
    recipient = session.variants[contrast["recipient_variant_id"]]
    donor_clean, recipient_clean = session.clean(donor["variant_id"]), session.clean(recipient["variant_id"])
    if session.profile.get("require_correct", True) and not (donor_clean.correct and recipient_clean.correct):
        return None
    prepared = session.prepared(recipient["variant_id"])
    labels = session.label_ids(prepared, recipient["labels"])
    kind = contrast["contrast_type"]
    if kind == "content":
        a = prepared.content_ids.get(contrast["donor_content"])
        b = prepared.content_ids.get(contrast["recipient_content"])
        if a is None or b is None or a == b:
            return None
    elif kind == "identity":
        a = b = None
    else:
        a = labels[recipient["labels"].index(contrast["donor_symbol"])]
        b = labels[recipient["correct_index"]]
    if a is None:
        donor_diff = recipient_diff = 0.0
    else:
        donor_diff = float(donor_clean.logits[a] - donor_clean.logits[b])
        recipient_diff = float(recipient_clean.logits[a] - recipient_clean.logits[b])
    return Case(dict(contrast), donor, recipient, donor_clean, recipient_clean, prepared, labels,
                a, b, donor_diff, recipient_diff)


def select_cases(session: Session, split: str, types: Sequence[str], *,
                 max_pairs: int | None, limit_per_type: int | None) -> tuple[list[Case], dict[str, Any]]:
    """Correct/correct cases from the first ``max_pairs`` usable pairs.

    A per-type cap is applied *round-robin across semantic pairs*.  Contrast
    files group many directed variants of one pair together, so slicing the
    flattened list would spend a 96-case head-screen budget on only the first
    few pairs.  Round-robin selection maximizes distinct-pair coverage before
    taking a second directed contrast from any pair. Deterministic staggered
    offsets also prevent every pair from contributing the same first
    donor/recipient direction.
    """
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in session.contrasts:
        if c["split"] == split and c["contrast_type"] in types:
            by_pair[c["pair_id"]].append(c)
    cases_by_pair: dict[str, list[Case]] = {}
    skipped = defaultdict(int)
    for pair_id in sorted(by_pair):
        if max_pairs is not None and len(cases_by_pair) >= max_pairs:
            break
        cases = []
        for c in by_pair[pair_id]:
            case = _make_case(session, c)
            if case is None:
                skipped[c["contrast_type"]] += 1
            else:
                cases.append(case)
        present = {case.kind for case in cases}
        if all(t in present for t in NEEDED_TYPES if t in types):
            cases_by_pair[pair_id] = cases
        else:
            skipped["pairs_without_all_types"] += 1
    eligible_pair_ids = sorted(cases_by_pair)
    selected: list[Case] = []
    selected_pairs_by_type: dict[str, list[str]] = {}
    selected_directions_by_type: dict[str, dict[str, int]] = {}
    for kind in types:
        buckets = {
            pair_id: [case for case in cases_by_pair[pair_id] if case.kind == kind]
            for pair_id in sorted(cases_by_pair)
        }
        if limit_per_type is None:
            chosen = [case for pair_id in sorted(buckets) for case in buckets[pair_id]]
        else:
            chosen = round_robin_across_groups(buckets, limit_per_type)
        selected.extend(chosen)
        selected_pairs_by_type[kind] = sorted({case.contrast["pair_id"] for case in chosen})
        directions = defaultdict(int)
        for case in chosen:
            donor_position = getattr(case, "donor", {}).get("correct_index")
            recipient_position = case.recipient.get("correct_index")
            directions[f"p{donor_position}->p{recipient_position}"] += 1
        selected_directions_by_type[kind] = dict(sorted(directions.items()))
    selected_pair_ids = sorted({case.contrast["pair_id"] for case in selected})
    per_pair = defaultdict(int)
    per_scheme = defaultdict(int)
    for case in selected:
        per_pair[case.contrast["pair_id"]] += 1
        scheme = case.recipient.get("label_scheme")
        if scheme is not None:
            per_scheme[scheme] += 1
    info = {"pairs_used": len(selected_pair_ids), "pair_ids": selected_pair_ids,
            "eligible_pairs": len(eligible_pair_ids), "eligible_pair_ids": eligible_pair_ids,
            "cases_by_type": {k: sum(c.kind == k for c in selected) for k in types},
            "cases_by_scheme": dict(sorted(per_scheme.items())),
            "contrasts_per_pair": dict(sorted(per_pair.items())),
            "selected_pairs_by_type": {k: len(v) for k, v in selected_pairs_by_type.items()},
            "selected_pair_ids_by_type": selected_pairs_by_type,
            "selected_directions_by_type": selected_directions_by_type,
            "limited_case_selection": "round_robin_across_pairs" if limit_per_type is not None else "all",
            "skipped": dict(skipped)}
    return selected, info


def evaluate(case: Case, logits: torch.Tensor) -> dict[str, Any]:
    logits = logits.float().cpu()
    scores = logits[torch.tensor(case.label_ids)]
    predicted = int(scores.argmax())
    labels = case.recipient["labels"]
    correct = case.recipient["correct_index"]
    others = torch.cat([scores[:correct], scores[correct + 1:]])
    row: dict[str, Any] = {
        "predicted_symbol": labels[predicted],
        "recipient_margin": float(scores[correct] - others.max()),
        "recipient_margin_change": float(scores[correct] - others.max()) - case.recipient_clean.readout["margin"],
        "symbol_changed": labels[predicted] != case.contrast["recipient_symbol"],
    }
    if case.read_a is not None:
        diff = float(logits[case.read_a] - logits[case.read_b])
        row.update({"patched_diff": diff, "donor_diff": case.donor_diff, "recipient_diff": case.recipient_diff,
                    "normalized_recovery": normalized_recovery(case.recipient_diff, case.donor_diff, diff)})
    if case.kind in ("position", "random_pair"):
        row["flip_to_donor"] = labels[predicted] == case.contrast["donor_symbol"]
    if case.kind == "content_and_position":
        symbol = labels[predicted]
        row["outcome"] = ("position" if symbol == case.contrast["donor_symbol"] else
                          "content" if symbol == case.contrast["content_symbol"] else
                          "none" if symbol == case.contrast["recipient_symbol"] else "other")
        content_id = case.label_ids[case.contrast["content_index"]]
        row["content_minus_recipient"] = float(logits[content_id] - logits[case.read_b])
        row["position_minus_recipient"] = float(logits[case.read_a] - logits[case.read_b])
    return row


def case_meta(case: Case) -> dict[str, Any]:
    c = case.contrast
    return {"contrast_id": c["contrast_id"], "contrast_type": c["contrast_type"], "pair_id": c["pair_id"],
            "donor_variant_id": c["donor_variant_id"], "recipient_variant_id": c["recipient_variant_id"],
            "family": case.recipient["question_family"]}


def run_specs(session: Session, case: Case, specs: Sequence[tuple[dict[str, Any], list[Intervention]]]
              ) -> list[dict[str, Any]]:
    """Evaluate each spec (metadata, interventions) as one batch row of the recipient."""
    rows = []
    for chunk in session.batches(list(specs)):
        interventions = []
        for row_index, (_meta, items) in enumerate(chunk):
            for item in items:
                placed = copy.copy(item)
                placed.rows = [row_index]
                interventions.append(placed)
        logits, _ = session.adapter.decoder_forward(case.prepared, batch=len(chunk), interventions=interventions)
        for (meta, _items), row_logits in zip(chunk, logits):
            rows.append({**case_meta(case), **meta, **evaluate(case, row_logits)})
    return rows
