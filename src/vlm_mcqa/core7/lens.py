"""Observational readouts at every residual stage and component update.

- Method 4: logit lens of every displayed symbol (logits and probabilities).
- Method 7: requested alphabet versus default A/B/C/D at every stage.
- Method 5 (observational): semantic-content first tokens versus symbols.
- Method 2 (projection): direct vocabulary projection of each isolated
  attention/MLP update (frozen final-norm scale), and the incremental lens change
  when that update is added to the residual stream.

Stage ``s`` is the residual entering layer ``s`` (``s = 0`` is the embedding
with image features); stage ``num_layers`` is the final residual before the
final norm, whose lens equals the model's logits (calibration gate).
"""

from __future__ import annotations

from typing import Any

import torch

from .scoring import DEFAULT_LABELS, symbol_readout
from .session import Session, write_jsonl_row


def _label_summary(values: torch.Tensor, correct: int) -> dict[str, float]:
    """Generic versus relative components of a four-label logit (change) vector."""
    others = torch.cat([values[:correct], values[correct + 1:]])
    return {
        "mean_label": float(values.mean()),
        "correct_minus_mean_incorrect": float(values[correct] - others.mean()),
        "correct_minus_best_incorrect": float(values[correct] - others.max()),
    }


def run_lens(session: Session, *, split: str, n_items: int | None) -> dict[str, Any]:
    adapter, handle = session.adapter, session.adapter.handle
    schemes = session.profile.get("schemes", ["letters", "numbers", "rare_letters"])
    n_layers = adapter.num_layers
    counts = {"variants": 0, "stage_rows": 0, "component_rows": 0, "content_skipped": 0}
    with session.jsonl(f"lens_{split}.jsonl") as lens_out, \
            session.jsonl(f"component_projection_{split}.jsonl") as comp_out:
        for index, item in enumerate(session.items_in(split, n_items)):
            for variant in session.variants_of(item["item_id"], schemes=schemes):
                prepared = session.prepared(variant["variant_id"])
                clean = session.clean(variant["variant_id"])
                correct = variant["correct_index"]
                ids = session.label_ids(prepared, variant["labels"])
                default_ids = session.label_ids(prepared, DEFAULT_LABELS)
                content_ids = [prepared.content_ids.get(c) for c in variant["option_contents"]]
                content_ok = None not in content_ids and len(set(content_ids)) == 4
                counts["content_skipped"] += not content_ok

                stages = [clean.sites["resid_pre:0"]] + [clean.sites[f"resid_post:{l}"] for l in range(n_layers)]
                stage_logits = handle.readout(torch.stack(stages)).float()
                for s, logits in enumerate(stage_logits):
                    row = {"variant_id": variant["variant_id"], "item_id": item["item_id"],
                           "scheme": variant["label_scheme"], "position": correct, "stage": s,
                           "family": variant["question_family"],
                           **symbol_readout(logits, ids, correct, tokenizer=adapter.tokenizer)}
                    row["correct_restricted_prob"] = row["restricted_probs"][correct]
                    if variant["label_scheme"] != "letters":
                        eight = logits[torch.tensor([*default_ids, *ids], device=logits.device)]
                        probs = torch.softmax(eight, 0)
                        row.update({
                            "default_logits": [float(v) for v in eight[:4]],
                            "default_mass_8way": float(probs[:4].sum()),
                            "requested_mass_8way": float(probs[4:].sum()),
                            "requested_correct_minus_position_default": float(eight[4 + correct] - eight[correct]),
                        })
                    if content_ok:
                        c = logits[torch.tensor(content_ids, device=logits.device)]
                        others = torch.cat([c[:correct], c[correct + 1:]])
                        row.update({
                            "content_logits": [float(v) for v in c],
                            "content_margin": float(c[correct] - others.max()),
                            "content_restricted_prob": float(torch.softmax(c, 0)[correct]),
                        })
                    write_jsonl_row(lens_out, row)
                    counts["stage_rows"] += 1

                final = clean.sites[f"resid_post:{n_layers - 1}"]
                id_tensor = torch.tensor(ids)
                for layer in range(n_layers):
                    pre = clean.sites[f"resid_pre:{layer}"]
                    attn, mlp = clean.sites[f"attn:{layer}"], clean.sites[f"mlp:{layer}"]
                    mid = pre + attn
                    lens = handle.readout(torch.stack([pre, mid, mid + mlp])).float().cpu()[:, id_tensor]
                    for component, update, before, after in (("attn", attn, lens[0], lens[1]),
                                                             ("mlp", mlp, lens[1], lens[2])):
                        direct = handle.direct_projection(update, final).cpu()[id_tensor]
                        row = {"variant_id": variant["variant_id"], "scheme": variant["label_scheme"],
                               "position": correct, "layer": layer, "component": component,
                               "update_norm": float(update.norm()),
                               **{f"direct_{k}": v for k, v in _label_summary(direct, correct).items()},
                               **{f"incremental_{k}": v
                                  for k, v in _label_summary(after - before, correct).items()}}
                        write_jsonl_row(comp_out, row)
                        counts["component_rows"] += 1
                counts["variants"] += 1
            if (index + 1) % 16 == 0:
                print(f"[lens:{split}] {index + 1} items", flush=True)
    session.write_json(f"lens_{split}_summary.json", counts)
    return counts
