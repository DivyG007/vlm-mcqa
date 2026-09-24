"""Method 3: individual attention-head localization in two stages.

Stage 1 (discovery only): screen every head of the layers localized by Methods
1/2 by exactly patching its pre-``o_proj`` channel slice at the final token
(position contrasts, both directions). The ranking, candidate list and head-set
sizes are frozen in ``freeze.json``.

Stage 2 (validation; confirmatory only on the confirmation split): for the
frozen candidates, individual patching, cumulative top-k joint patching,
matched random head sets (same per-layer counts), the all-heads reference for
those layers, and resampling ablation of top-k versus random sets. No
sparsity is assumed: the verdict may be ``sparse`` or ``distributed``.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from .components import run_resample_ablation
from .contrasts import Case, run_specs, select_cases
from .hooks import Intervention, Site
from .scoring import mean, summarize_clustered
from .session import Session, write_jsonl_row

Head = tuple[int, int]


def _clustered(rows: Sequence[dict[str, Any]], *, seed: int = 0) -> dict[str, Any]:
    return summarize_clustered([r["normalized_recovery"] for r in rows],
                               [r["pair_id"] for r in rows], seed=seed)


def _cluster_mean(rows: Sequence[dict[str, Any]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["pair_id"]].append(row["normalized_recovery"])
    return mean([mean(values) for values in grouped.values()])


def _head_interventions(case: Case, heads: Sequence[Head]) -> list[Intervention]:
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in heads:
        by_layer[layer].append(head)
    final = case.prepared.final_position
    return [Intervention(Site("head", layer), [final], case.donor_clean.sites[f"head:{layer}"], heads=hs)
            for layer, hs in sorted(by_layer.items())]


def layers_from_components(session: Session, split: str, count: int,
                           components_dir: str | Path | None = None) -> list[int]:
    """Layers with the largest mean attention-patch recovery on position contrasts."""
    path = Path(components_dir or session.out_dir) / f"components_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError("run M2 first or pass head layers through freeze.json")
    per_layer = defaultdict(list)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["site"] == "attn" and row["contrast_type"] == "position":
                per_layer[row["layer"]].append(row["normalized_recovery"])
    ranked = sorted(per_layer, key=lambda layer: -mean(per_layer[layer]))
    return sorted(ranked[:count])


def matched_random(heads: Sequence[Head], layers: Sequence[int], num_heads: int, rng: random.Random) -> list[Head]:
    chosen = set(heads)
    counts = defaultdict(int)
    for layer, _ in heads:
        counts[layer] += 1
    out = []
    for layer, n in counts.items():
        pool = [(layer, h) for h in range(num_heads) if (layer, h) not in chosen]
        if len(pool) < n:
            raise ValueError(f"cannot draw {n} matched random heads from layer {layer}; only {len(pool)} remain")
        out.extend(rng.sample(pool, n))
    return out


def run_heads(session: Session, *, split: str, freeze: dict[str, Any] | None,
              components_dir: str | Path | None = None) -> dict[str, Any]:
    profile, adapter = session.profile, session.adapter
    rng = random.Random(session.seed + 3)
    cases, info = select_cases(session, split, ("position",), max_pairs=profile.get("pairs"),
                               limit_per_type=profile.get("head_screen_contrasts"))
    num_heads = adapter.handle.num_heads
    out: dict[str, Any] = {"cases": info, "split": split}

    if freeze is None:
        head_layers = layers_from_components(session, split, int(profile["head_layers"]), components_dir)
        per_head: dict[Head, list[dict[str, Any]]] = defaultdict(list)
        with session.jsonl(f"heads_screen_{split}.jsonl") as stream:
            for index, case in enumerate(cases):
                specs = [({"method": "M3-screen", "layer": l, "head": h}, _head_interventions(case, [(l, h)]))
                         for l in head_layers for h in range(num_heads)]
                for row in run_specs(session, case, specs):
                    write_jsonl_row(stream, row)
                    per_head[(row["layer"], row["head"])].append(row)
                print(f"[M3 screen] {index + 1}/{len(cases)} cases", flush=True)
        ranking = sorted(per_head, key=lambda k: -_cluster_mean(per_head[k]))
        candidates = [list(k) for k in ranking[: int(profile["head_candidates"])]]
        out.update({"stage": "screen+validate(discovery)", "head_layers": head_layers,
                    "ranking": [{"layer": l, "head": h,
                                 "mean_nr": _cluster_mean(per_head[(l, h)]),
                                 "n_pairs": len({r["pair_id"] for r in per_head[(l, h)]})}
                                for l, h in ranking]})
        topk = [k for k in profile["topk"] if k <= len(candidates)]
    else:
        head_layers = freeze["head_layers"]
        candidates = freeze["head_candidates"]
        topk = freeze["head_set_sizes"]
        out["stage"] = "validate(frozen)"
    candidate_heads = [tuple(h) for h in candidates]
    out.update({"candidates": candidates, "topk": topk})

    sets: dict[str, list[Head]] = {}
    for k in topk:
        sets[f"top{k}"] = candidate_heads[:k]
        for r in range(int(profile["random_sets"])):
            sets[f"random{k}_{r}"] = matched_random(candidate_heads[:k], head_layers, num_heads, rng)
    sets["all_heads"] = [(l, h) for l in head_layers for h in range(num_heads)]

    results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with session.jsonl(f"heads_validate_{split}.jsonl") as stream:
        for index, case in enumerate(cases):
            specs = [({"method": "M3-single", "set": f"head_{l}_{h}", "layer": l, "head": h, "size": 1},
                      _head_interventions(case, [(l, h)])) for l, h in candidate_heads]
            specs += [({"method": "M3-set", "set": name, "size": len(heads)}, _head_interventions(case, heads))
                      for name, heads in sets.items()]
            for row in run_specs(session, case, specs):
                write_jsonl_row(stream, row)
                results[row["set"]].append(row)
            print(f"[M3 validate] {index + 1}/{len(cases)} cases", flush=True)

    ablation_sets = {name: heads for name, heads in sets.items()
                     if name.startswith("top") or name == "all_heads" or name.endswith("_0")}
    out["ablation_rows"] = run_resample_ablation(
        session, split=split, layers=head_layers, pair_ids=info["pair_ids"], kinds=(),
        out_name=f"heads_ablation_{split}.jsonl", head_sets=ablation_sets)

    all_heads = _cluster_mean(results["all_heads"])
    per_k = {}
    for k in topk:
        top = _clustered(results[f"top{k}"], seed=k)
        random_means = sorted(_cluster_mean(results[f"random{k}_{r}"])
                              for r in range(int(profile["random_sets"])))
        p95 = random_means[min(len(random_means) - 1, int(math.ceil(0.95 * len(random_means))) - 1)]
        per_k[str(k)] = {"top_k": top, "random_mean": mean(random_means), "random_p95": p95,
                         "fraction_of_all_heads": top["mean"] / all_heads if all_heads else float("nan"),
                         "beats_random_p95": top["mean"] > p95}
    out["per_k"] = per_k
    out["all_heads_mean_nr"] = all_heads
    out["individual"] = {name: _clustered(v) for name, v in results.items() if name.startswith("head_")}
    small = [k for k in topk if k <= 4 and per_k[str(k)]["beats_random_p95"]
             and per_k[str(k)]["fraction_of_all_heads"] >= 0.5]
    out["verdict"] = ("sparse" if small else "distributed",
                      "a set of <=4 heads beats matched random sets and recovers >=50% of the all-heads effect"
                      if small else "no set of <=4 heads meets the sparse criterion")
    session.write_json(f"heads_{split}_summary.json", out)
    return out


def load_freeze(path: str | Path | None) -> dict[str, Any] | None:
    return json.loads(Path(path).read_text()) if path else None
