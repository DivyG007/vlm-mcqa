"""Method 2 (causal part): attention versus MLP at every layer.

- Component patching: the final-token attention update or MLP update of layer
  ``L`` is replaced by the donor's, for position, content and
  content-and-position contrasts.
- Resampling ablation: the recipient's clean final-token update is replaced by
  the same update from a different item with a different correct symbol
  (in-distribution; zero ablation is avoided because it produced misleading
  effects in the Qwen pilot). Reports the drop in the recipient's own margin.

The observational projection half of Method 2 lives in ``lens.py``.
"""

from __future__ import annotations

from typing import Any

from .contrasts import run_specs, select_cases
from .hooks import Intervention, Site
from .scoring import symbol_readout
from .session import Session, write_jsonl_row

COMPONENT_TYPES = ("position", "content", "content_and_position")


def run_components(session: Session, *, split: str) -> dict[str, Any]:
    profile = session.profile
    cases, info = select_cases(session, split, COMPONENT_TYPES, max_pairs=profile.get("pairs"),
                               limit_per_type=profile.get("contrast_limit_per_type"))
    layers = session.layers()
    rows = 0
    with session.jsonl(f"components_{split}.jsonl") as stream:
        for index, case in enumerate(cases):
            final = case.prepared.final_position
            specs = [({"method": "M2", "site": kind, "layer": layer, "group": "final"},
                      [Intervention(Site(kind, layer), [final], case.donor_clean.sites[f"{kind}:{layer}"])])
                     for kind in ("attn", "mlp") for layer in layers]
            for row in run_specs(session, case, specs):
                write_jsonl_row(stream, row)
                rows += 1
            if (index + 1) % 20 == 0:
                print(f"[M2] {index + 1}/{len(cases)} cases", flush=True)

    ablation_rows = run_resample_ablation(session, split=split, layers=layers,
                                          pair_ids=info["pair_ids"], kinds=("attn", "mlp"),
                                          out_name=f"component_ablation_{split}.jsonl")
    summary = {"cases": info, "layers": layers, "patch_rows": rows, "ablation_rows": ablation_rows}
    session.write_json(f"components_{split}_summary.json", summary)
    return summary


def ablation_pool(session: Session, split: str, pair_ids) -> list[dict[str, Any]]:
    pairs = set(pair_ids)
    return [v for v in session.variants.values()
            if v["split"] == split and v["label_scheme"] == "letters" and v["variant_kind"] == "position"
            and v.get("counterfactual_pair_id") in pairs]


def resample_donor(session: Session, recipient: dict[str, Any], pool) -> dict[str, Any]:
    choices = [v for v in pool if v["item_id"] != recipient["item_id"]
               and v.get("counterfactual_pair_id") != recipient.get("counterfactual_pair_id")
               and v["correct_symbol"] != recipient["correct_symbol"]]
    return session.rng.choice(choices)


def run_resample_ablation(session: Session, *, split: str, layers, pair_ids, kinds, out_name: str,
                          head_sets: dict[str, list[tuple[int, int]]] | None = None) -> int:
    """Resample-ablate sites (or head sets) on clean, correctly answered recipients."""
    pool = ablation_pool(session, split, pair_ids)
    require = session.profile.get("require_correct", True)
    recipients = [v for v in sorted(pool, key=lambda v: v["variant_id"])
                  if not require or session.clean(v["variant_id"]).correct]
    recipients = recipients[: session.profile.get("ablation_recipients")]
    rows = 0
    with session.jsonl(out_name) as stream:
        for recipient in recipients:
            donor = resample_donor(session, recipient, pool)
            prepared = session.prepared(recipient["variant_id"])
            donor_clean, clean = session.clean(donor["variant_id"]), session.clean(recipient["variant_id"])
            final = prepared.final_position
            specs = []
            if head_sets is None:
                for kind in kinds:
                    for layer in layers:
                        specs.append(({"site": kind, "layer": layer},
                                      [Intervention(Site(kind, layer), [final], donor_clean.sites[f"{kind}:{layer}"])]))
            else:
                for name, heads in head_sets.items():
                    by_layer: dict[int, list[int]] = {}
                    for layer, head in heads:
                        by_layer.setdefault(layer, []).append(head)
                    specs.append(({"site": "head_set", "head_set": name, "size": len(heads)},
                                  [Intervention(Site("head", layer), [final], donor_clean.sites[f"head:{layer}"],
                                                heads=hs) for layer, hs in sorted(by_layer.items())]))
            ids = session.label_ids(prepared, recipient["labels"])
            for chunk in session.batches(specs):
                interventions = []
                for row_index, (_meta, items) in enumerate(chunk):
                    for item in items:
                        item.rows = [row_index]
                        interventions.append(item)
                logits, _ = session.adapter.decoder_forward(prepared, batch=len(chunk), interventions=interventions)
                for (meta, _items), row_logits in zip(chunk, logits):
                    readout = symbol_readout(row_logits.cpu(), ids, recipient["correct_index"])
                    write_jsonl_row(stream, {
                        **meta, "recipient_variant_id": recipient["variant_id"],
                        "donor_variant_id": donor["variant_id"],
                        "clean_margin": clean.readout["margin"], "ablated_margin": readout["margin"],
                        "margin_drop": clean.readout["margin"] - readout["margin"],
                        "still_correct": readout["correct"],
                    })
                    rows += 1
    return rows
