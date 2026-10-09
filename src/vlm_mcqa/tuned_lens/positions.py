"""Frozen rule for the token positions used to fit translators.

Rule ``text_minus_labels_v1`` (fixed 2026-10-27, before any lens result):
fit on prompt text positions after the last image token, excluding

- the final position (the answer is read out there);
- every option-label token (A/B/C/D, 1-4, Q/Z/R/X);
- the position immediately before each label, whose next-token target *is*
  that label, so label identity is never a fitting target;
- the delimiter token right after a label when it decodes to ``"."``.

Image-token positions are excluded (owner decision): their next-token
predictions are not language predictions.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

POSITION_RULE = "text_minus_labels_v1"


def fit_positions(ids: Sequence[int], groups: Mapping[str, Sequence[int]], tokenizer: Any,
                  *, num_labels: int = 4) -> list[int]:
    """Return sorted fit positions, or raise if the labels could not be located."""
    labels = sorted(groups.get("labels", []))
    if len(labels) != num_labels:
        raise ValueError(f"expected {num_labels} located option labels, found {len(labels)}")
    final = len(ids) - 1
    image = groups.get("image", [])
    start = max(image) + 1 if image else 0
    excluded = set()
    for p in labels:
        excluded.update((p - 1, p))
        if p + 1 < final and tokenizer.decode([ids[p + 1]]).strip() == ".":
            excluded.add(p + 1)
    positions = [p for p in range(start, final) if p not in excluded]
    if not positions:
        raise ValueError("no fit positions left after exclusions")
    return positions
