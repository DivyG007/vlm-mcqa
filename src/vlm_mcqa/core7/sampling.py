"""Deterministic sampling helpers that do not depend on model libraries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

T = TypeVar("T")


def round_robin_across_groups(buckets: Mapping[str, Sequence[T]], limit: int) -> list[T]:
    """Take one value per sorted group before revisiting any group.

    Each group starts at a deterministic, staggered offset.  This avoids
    selecting the same within-pair donor/recipient direction from every pair
    when all buckets share the same ordering.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")
    chosen: list[T] = []
    depth = 0
    keys = sorted(buckets)
    while len(chosen) < limit:
        added = False
        for group_index, key in enumerate(keys):
            if depth < len(buckets[key]):
                index = (group_index + depth) % len(buckets[key])
                chosen.append(buckets[key][index])
                added = True
                if len(chosen) == limit:
                    break
        if not added:
            break
        depth += 1
    return chosen
