"""Method 1 (layer-wise residual patching) and Methods 5/6 (causal factorial).

- Final-token ``resid_post`` patching at every layer for every usable contrast
  type, both directions (stored as separate directed contrasts), with identity
  and random-pair controls.
- Token-group patching (image, question, option-content and option-label
  tokens) for content and content-and-position contrasts: the full residual at
  every position of the group is replaced by the donor's, which tests whether
  answer content enters through visual tokens, the question, or option text.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .contrasts import Case, run_specs, select_cases
from .hooks import Intervention, Site
from .session import Session, write_jsonl_row

RESIDUAL_TYPES = ("position", "content", "content_and_position", "identity", "random_pair")
GROUP_TYPES = ("content", "content_and_position")


def residual_sweep(session: Session, cases: Sequence[Case], layers: Sequence[int], stream) -> int:
    count = 0
    for index, case in enumerate(cases):
        final = case.prepared.final_position
        specs = [({"method": "M1", "site": "resid_post", "layer": layer, "group": "final"},
                  [Intervention(Site("resid_post", layer), [final],
                                case.donor_clean.sites[f"resid_post:{layer}"])])
                 for layer in layers]
        for row in run_specs(session, case, specs):
            write_jsonl_row(stream, row)
            count += 1
        if (index + 1) % 20 == 0:
            print(f"[M1] {index + 1}/{len(cases)} cases", flush=True)
    return count


def group_sweep(session: Session, cases: Sequence[Case], layers: Sequence[int], groups: Sequence[str],
                stream) -> dict[str, int]:
    counts = {"rows": 0, "skipped_size_mismatch": 0}
    for case in cases:
        donor_prepared = session.prepared(case.donor["variant_id"])
        usable = [g for g in groups
                  if donor_prepared.groups.get(g) and len(donor_prepared.groups[g]) == len(case.prepared.groups.get(g, []))]
        counts["skipped_size_mismatch"] += len(groups) - len(usable)
        if not usable:
            continue
        positions = sorted({p for g in usable for p in donor_prepared.groups[g]})
        sites = [Site("resid_post", layer) for layer in layers]
        _, captured = session.adapter.decoder_forward(donor_prepared, capture=sites, capture_positions=positions)
        column = {p: i for i, p in enumerate(positions)}
        specs = []
        for group in usable:
            donor_cols = [column[p] for p in donor_prepared.groups[group]]
            for layer in layers:
                values = captured[f"resid_post:{layer}"][0, donor_cols]
                specs.append(({"method": "M6", "site": "resid_post", "layer": layer, "group": group,
                               "group_size": len(donor_cols)},
                              [Intervention(Site("resid_post", layer), case.prepared.groups[group], values)]))
        for row in run_specs(session, case, specs):
            write_jsonl_row(stream, row)
            counts["rows"] += 1
        del captured
    return counts


def run_patching(session: Session, *, split: str) -> dict[str, Any]:
    profile = session.profile
    cases, info = select_cases(session, split, RESIDUAL_TYPES, max_pairs=profile.get("pairs"),
                               limit_per_type=profile.get("contrast_limit_per_type"))
    layers = session.layers()
    with session.jsonl(f"patching_{split}.jsonl") as stream:
        m1_rows = residual_sweep(session, cases, layers, stream)
        group_cases = [c for c in cases if c.kind in GROUP_TYPES]
        group_counts = group_sweep(session, group_cases, session.layers("group_layer_stride"),
                                   profile.get("token_groups", ["image", "question", "options", "labels"]), stream)
    summary = {"cases": info, "layers": layers, "m1_rows": m1_rows, "group_patching": group_counts}
    session.write_json(f"patching_{split}_summary.json", summary)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
