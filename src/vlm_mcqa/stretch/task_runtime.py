"""Built-in Qwen spatial and single-collage visual-option study evaluators.

Authored without executing imports, tests, processors, models or image assembly.
Runtime calls require an existing explicitly calibrated Session and execution opt-in.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Mapping

from .assets import spatial_from_objects
from .discovery_selection import select_task_image_layer
from .processor_geometry import prepare_with_geometry, require_supported_geometry, selected_attention
from .protocol import load_frozen_heads


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def execution_identity(session: Any, context: Mapping) -> str:
    identity = context.get("identity", {})
    spec = session.adapter.spec
    if (identity.get("model_key") != spec.key or not re.fullmatch(r"[0-9a-fA-F]{40}", spec.revision)
            or identity.get("checkpoint_identity") != {"hf_id": spec.hf_id, "revision": spec.revision}
            or identity.get("eligibility_decision") != "approved_for_this_stretch_protocol"
            or any(not identity.get(k) for k in ("source_identity", "image_aware_release_hash", "protocol_decision_sha256"))):
        raise ValueError("explicit current pinned-model/source/release/protocol identity required")
    return canonical_hash(identity)


def _name(config: Mapping) -> str:
    name = config.get("name", config.get("study"))
    if name not in {"spatial", "image_options"} or "study" in config and config["study"] != name:
        raise ValueError("task study must be spatial or image_options")
    return name


def _guard(session: Any, config: Mapping, context: Mapping) -> tuple[dict, dict]:
    if context.get("execute") is not True:
        raise PermissionError("task runtime requires explicit execution authorization")
    name = _name(config)
    require_supported_geometry(session.adapter)
    execution_identity(session, context)
    if config.get("model_key") != session.adapter.spec.key:
        raise ValueError("task config model differs from supplied Session")
    core = context.get("core_calibration", {})
    if core.get("all_passed") is not True or core.get("adapter") != session.adapter.describe():
        raise ValueError("fresh real Core-7 calibration for this adapter required")
    required = {"cached_matches_full", "capture_is_inert", "identity_patch", "batched_matches_single",
                "replay_deterministic", "residual_decomposition", "lens_final_matches", "symbols_single_token", "token_spans"}
    if any(core.get("gates", {}).get(g, {}).get("passed") is not True for g in required):
        raise ValueError("all nine ordinary calibration gates must pass")
    if config.get("phase") not in {"discovery", "confirmation"} or config.get("split") != config["phase"]:
        raise ValueError("explicit matching phase/split required")
    p = dict(config["protocol"])
    p["task_assets_root"] = str(Path(config["task_assets_root"]).resolve())
    for key in ("target_units", "min_units", "max_tokens", "max_region_tokens", "calibration_tasks", "bootstrap_samples", "random_head_sets", "query_budget_per_group"):
        if type(p.get(key)) is not int or p[key] < 1:
            raise ValueError(f"explicit positive {key} required")
    if p["min_units"] > p["target_units"] or type(p.get("seed")) is not int:
        raise ValueError("invalid explicit minimum/target/seed")
    if p["bootstrap_samples"] < 100:
        raise ValueError("at least 100 bootstrap draws required before task inference")
    for key in ("tolerance", "minimum_accuracy", "minimum_effect", "minimum_image_gap", "layer_tie_tolerance"):
        if type(p.get(key)) not in (int, float) or not math.isfinite(p[key]) or p[key] < 0:
            raise ValueError(f"finite nonnegative {key} required")
    if not 0 <= p["minimum_accuracy"] <= 1:
        raise ValueError("accuracy threshold outside [0,1]")
    if p.get("mapping_rule") not in {"center", "overlap"} or "min_fraction" not in p:
        raise ValueError("explicit geometry rule and fraction (possibly null) required")
    if p.get("control_mode") not in {"token_count", "translated_shape"}:
        raise ValueError("explicit token_count or translated_shape control policy required")
    if name == "image_options" and type(p.get("require_unrelated_donor_control", True)) is not bool:
        raise ValueError("image-option unrelated donor control policy must be boolean")
    query_groups = p.get("attention_query_groups")
    if (not isinstance(query_groups, list) or len(set(query_groups)) != len(query_groups)
            or "final" not in query_groups or any(g not in {"final", "options", "text_question", "visual_options"} for g in query_groups)):
        raise ValueError("explicit supported attention query groups including final required")
    if name == "spatial" and "options" not in query_groups:
        raise ValueError("spatial routing requires actual text-option query positions")
    layers = p.get("image_layers", [])
    if not layers or len(set(layers)) != len(layers) or any(type(l) is not int or not 0 <= l < session.adapter.num_layers for l in layers):
        raise ValueError("explicit unique valid image intervention layers required")
    if config["phase"] == "confirmation" and len(layers) != 1:
        raise ValueError("confirmation requires exactly one discovery-frozen image layer")
    frozen = load_frozen_heads(config["head_freeze_path"], model_key=config["model_key"],
                               dataset_hash=config["head_freeze_dataset_hash"])
    if p.get("head_count") not in (4, 16):
        raise ValueError("head_count must explicitly select original top4 or top16")
    heads = frozen["head_candidates"][:p["head_count"]]
    if any(l >= session.adapter.num_layers or h >= session.adapter.handle.num_heads for l, h in heads):
        raise ValueError("frozen heads incompatible with current adapter")
    p.update({"selected_heads": heads, "head_source_sha256": frozen["source_sha256"],
              "head_freeze_hash": frozen["freeze_hash"], "model_key": config["model_key"], "study": name})
    return p, frozen


def _load_tasks(config: Mapping, p: Mapping) -> tuple[list[dict], str]:
    path = Path(config["tasks_path"])
    payload = path.read_bytes()
    records = json.loads(payload)
    if not isinstance(records, list) or not records:
        raise ValueError("nonempty task list required")
    tasks = []
    for entry in records:
        source_payload = Path(entry["source_metadata_path"]).read_bytes()
        if hashlib.sha256(source_payload).hexdigest() != entry["source"]["source_sha256"]:
            raise ValueError("task source metadata SHA256 mismatch")
        if _name(config) == "spatial":
            if json.loads(source_payload).get("objects") != entry["objects"]:
                raise ValueError("spatial object truth differs from immutable annotated source metadata")
            built = spatial_from_objects(entry["source"], image=entry["image"], objects=entry["objects"],
                                         query_objects=entry["query_objects"], option_sets=entry["option_sets"], labels=entry["labels"])
            objects = {o["id"]: o for o in entry["objects"]}
            for row in built["task"]["conditions"]:
                options = row["option_ids"]
                regions = {"question": objects[row["question_region"]]["bbox"]}
                for oid, obj in objects.items():
                    if obj["color"] in options:
                        regions["option-object-" + oid] = obj["bbox"]
                task = {"task_id": built["task"]["derivative_id"] + ":" + row["condition_id"],
                        "source": entry["source"], "image": entry["image"], "image_sha256": entry["image_sha256"],
                        "labels": row["labels"], "option_semantics": options,
                        "correct_index": options.index(row["correct_option_id"]),
                        "question": row["question"], "prompt": row["question"] + "\n" +
                        "\n".join(f"{l}. {o}" for l, o in zip(row["labels"], options)) + "\nRespond with only the option label.\nAnswer:",
                        "regions": regions, "question_level": row["question_level"],
                        "option_level": row["option_level"], "condition": "spatial"}
                tasks.append(task)
        else:
            # Builder-created visual/matched-text records; semantic data stays private.
            if entry.get("condition") not in {"visual", "matched_text"}:
                raise ValueError("image options require explicit visual and matched_text task records")
            plan = entry["derivative_plan"]
            body = {k: v for k, v in plan.items() if k != "derivative_id"}
            expected_id = "image-options-v1-" + canonical_hash(body)[:20]
            panels = plan["annotations"]["panels"]
            semantics = [row["semantic_id"] for row in panels]
            if (plan.get("derivative_id") != expected_id or entry["task_id"] != expected_id
                    or plan.get("task_semantics") != "exact_visual_matching"
                    or entry["source"] != plan["source"]
                    or entry["labels"] != [row["symbol"] for row in panels]
                    or entry["option_semantics"] != semantics
                    or entry["correct_index"] != semantics.index(plan["annotations"]["target_option"])):
                raise ValueError("image-option task contradicts hashed panel/truth plan")
            assembly = entry["assembly_provenance"]
            if assembly["image_sha256"] != entry["image_sha256"] or assembly["task_derivative_id"] != expected_id:
                raise ValueError("image-option assembly identity mismatch")
            if assembly["mode"] != ("visual" if entry["condition"] == "visual" else "text-query-only"):
                raise ValueError("matched text requires separately assembled query-only image")
            base = Path(assembly["asset_base"]).resolve()
            for path, expected in assembly["asset_sha256"].items():
                asset = Path(path).resolve()
                if base not in asset.parents or hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
                    raise ValueError("image-option source asset changed")
            if hashlib.sha256(Path(assembly["font_path"]).read_bytes()).hexdigest() != assembly["font_sha256"]:
                raise ValueError("image-option assembly font changed")
            if entry["condition"] == "visual":
                ordered_assets = [plan["visible"]["query"]["asset"], *[r["asset"] for r in plan["visible"]["options"]]]
                hashes = [assembly["asset_sha256"][str((base / name).resolve())] for name in ordered_assets]
                if hashes[0] != hashes[entry["correct_index"] + 1] or len(set(hashes[1:])) != 4:
                    raise ValueError("visual-match query/options violate exact source-asset truth")
            tasks.append(dict(entry))
    ids, units, split_audit = set(), defaultdict(list), {}
    for task in tasks:
        source = task["source"]
        key = task["task_id"], task["condition"]
        if key in ids:
            raise ValueError("duplicate task condition")
        ids.add(key)
        for field in ("scene_id", "pair_id"):
            value = source.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError("explicit parent scene/pair required")
            audit_key = field, value
            if audit_key in split_audit and split_audit[audit_key] != source["split"]:
                raise ValueError("task parent crosses split")
            split_audit[audit_key] = source["split"]
        if source["split"] != config["split"]:
            raise ValueError("task source split differs from requested phase")
        labels, semantics = task["labels"], task["option_semantics"]
        correct = task["correct_index"]
        if (len(labels) != 4 or len(set(labels)) != 4 or len(semantics) != 4 or len(set(semantics)) != 4
                or any(not isinstance(x, str) or not x.strip() for x in [*labels, *semantics])
                or type(correct) is not int or not 0 <= correct < 4):
            raise ValueError("invalid option/symbol/truth mapping")
        if _name(config) == "image_options" and task["condition"] == "visual":
            # Exact allowed visible prompt prevents injecting private semantic names.
            if task["prompt"] != "Which option image matches the query image? Respond with only its displayed symbol.":
                raise ValueError("visual-option prompt differs from semantic-free contract")
            expected = {"query", *(f"option-{i}" for i in range(4))}
            if set(task["regions"]) != expected or len(task.get("symbol_regions", {})) != 4:
                raise ValueError("five-panel geometry and four symbol regions required")
        units[source["pair_id"]].append(task)
    order = sorted(units, key=lambda x: (canonical_hash([p["seed"], x]), x))
    # Correct-only replication must screen the complete prepared candidate pool
    # before deterministic sampling; otherwise an early wrong answer can hide a
    # later eligible parent and create an artificial shortfall.
    selected = order if p.get("correct_only", False) else order[:p["target_units"]]
    if _name(config) == "image_options":
        for unit in selected:
            by_task = defaultdict(set)
            for task in units[unit]:
                by_task[task["task_id"]].add(task["condition"])
            if any(conditions != {"visual", "matched_text"} for conditions in by_task.values()):
                raise ValueError("every panel layout needs its matched-text condition")
    return [t for unit in selected for t in sorted(units[unit], key=lambda x: (x["task_id"], x["condition"]))], hashlib.sha256(payload).hexdigest()


def _variant(task: Mapping) -> dict:
    i = task["correct_index"]
    return {"variant_id": task["task_id"] + ":" + task["condition"], "prompt": task["prompt"],
            "question": task["question"], "labels": task["labels"],
            "option_contents": task["option_semantics"], "semantic_answer": task["option_semantics"][i],
            "correct_index": i, "correct_symbol": task["labels"][i]}


def _image(session: Any, task: Mapping, p: Mapping):
    from PIL import Image
    import io
    root = Path(p["task_assets_root"]).resolve()
    path = (root / task["image"]).resolve()
    if root not in path.parents:
        raise ValueError("task image outside immutable dataset root")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != task["image_sha256"]:
        raise ValueError("task image hash mismatch")
    with Image.open(io.BytesIO(payload)) as loaded:
        return loaded.convert("RGB")


def _prepare(session: Any, task: Mapping, p: Mapping, image=None):
    image = _image(session, task, p) if image is None else image
    regions = dict(task["regions"])
    if task["condition"] == "matched_text":
        regions = {"query": regions["query"]}
    else:
        regions.update(task.get("symbol_regions", {}))
    prepared, geometry = prepare_with_geometry(session.adapter, _variant(task), image,
        regions=regions, rule=p["mapping_rule"], min_fraction=p["min_fraction"],
        max_tokens=p["max_tokens"], max_region_tokens=p["max_region_tokens"])
    if "grid_thw" in geometry:
        prepared.region_grid_width = geometry["grid_thw"][2] // geometry["spatial_merge_size"]
    if task["condition"] == "spatial":
        prepared.groups["option"] = sorted({i for name, values in geometry["groups"].items()
                                             if name.startswith("option-object-") for i in values}
                                            - set(prepared.groups["question"]))
        if not prepared.groups["option"]:
            raise ValueError("no option-supported visual tokens outside queried object")
    return prepared, geometry


def _score(session: Any, prepared: Any, task: Mapping, logits: Any) -> dict:
    from ..core7.scoring import symbol_readout
    if logits.ndim == 2:
        logits = logits[0]
    ids = session.label_ids(prepared, task["labels"])
    result = symbol_readout(logits.cpu(), ids, task["correct_index"])
    scores = result["label_logits"]
    result["truth_margin"] = scores[task["correct_index"]] - max(x for i, x in enumerate(scores) if i != task["correct_index"])
    if not all(math.isfinite(x) for x in scores):
        raise ValueError("nonfinite task logits")
    return result


def _controls(prepared: Any, regions: Mapping[str, list[int]], seed: int, *, mode: str) -> dict[str, list[int]]:
    controls = {}
    visual = list(prepared.groups["image"])
    relevant = {i for positions in regions.values() for i in positions}
    available = set(visual) - relevant
    if mode not in {"token_count", "translated_shape"}:
        raise ValueError("unknown control geometry policy")
    for name, positions in regions.items():
        pool = sorted(available, key=lambda i: canonical_hash([seed, name, i]))
        if len(pool) < len(positions):
            raise ValueError("insufficient nonoverlapping token-matched controls")
        if mode == "token_count":
            controls[name] = pool[:len(positions)]
            continue
        cells = getattr(prepared, "region_token_cells", None)
        if not cells:
            raise ValueError("processor geometry omitted view-local cells for translated controls")
        by_view = defaultdict(list)
        for cell in cells:
            by_view[cell.view_id].append(cell)
        target_by_view = defaultdict(list)
        cell_by_token = {cell.token_index: cell for cell in cells}
        if any(token not in cell_by_token for token in positions):
            raise ValueError("target region includes a non-spatial visual token")
        for token in positions:
            target_by_view[cell_by_token[token].view_id].append(token)
        combined = []
        for view_id, target_tokens in sorted(target_by_view.items()):
            view_cells = by_view[view_id]
            xs = sorted({(c.footprint.x0 + c.footprint.x1) / 2 for c in view_cells})
            ys = sorted({(c.footprint.y0 + c.footprint.y1) / 2 for c in view_cells})
            lookup = {c.token_index: (ys.index((c.footprint.y0 + c.footprint.y1) / 2),
                                              xs.index((c.footprint.x0 + c.footprint.x1) / 2))
                      for c in view_cells}
            inverse = {location: token for token, location in lookup.items()}
            shape = [lookup[token] for token in target_tokens]
            origin_r, origin_c = min(r for r, _ in shape), min(c for _, c in shape)
            offsets = [(r - origin_r, c - origin_c) for r, c in shape]
            choices = []
            for row in range(len(ys)):
                for column in range(len(xs)):
                    translated = [(row + r, column + c) for r, c in offsets]
                    if any(location not in inverse for location in translated):
                        continue
                    candidate = sorted(inverse[location] for location in translated)
                    if set(candidate).issubset(available):
                        choices.append(candidate)
            if not choices:
                raise ValueError(f"no translated control in processor view {view_id}")
            combined.extend(min(choices, key=lambda indices: canonical_hash([seed, name, view_id, indices])))
        controls[name] = sorted(combined)
    return controls


def _regions(prepared: Any, task: Mapping) -> dict[str, list[int]]:
    if task["condition"] == "spatial":
        return {name: prepared.groups[name] for name in ("question", "option")}
    if task["condition"] == "matched_text":
        return {"query": prepared.groups["query"]}
    names = ["query", *(f"option-{i}" for i in range(4)), *(f"option-{i}-symbol" for i in range(4))]
    return {name: prepared.groups[name] for name in names}


def _queries(prepared: Any, task: Mapping, p: Mapping) -> tuple[list[int], dict[str, list[int]]]:
    groups = {}
    for name in p["attention_query_groups"]:
        if name == "visual_options":
            positions = sorted({i for k, values in prepared.groups.items() if k.startswith("option-") and not k.endswith("symbol") for i in values})
            if task["condition"] == "matched_text":
                positions = prepared.groups.get("options", [])
        else:
            positions = prepared.groups.get(name, [])
        if not positions:
            raise ValueError(f"requested attention query group {name} is unresolved")
        groups[name] = sorted(positions, key=lambda i: canonical_hash([p["seed"], name, i]))[:p["query_budget_per_group"]]
    return sorted({i for values in groups.values() for i in values}), groups


def calibrate_study(session: Any, config: Mapping, context: Mapping) -> dict:
    p, frozen = _guard(session, config, context)
    _, task_hash = _load_tasks(config, p)
    calibration_config = {**config, "tasks_path": config["calibration_tasks_path"],
                          "phase": "discovery", "split": "discovery"}
    tasks, calibration_task_hash = _load_tasks(calibration_config, p)
    if len(tasks) < p["calibration_tasks"]:
        raise ValueError("insufficient explicitly discovery-only calibration tasks")
    from ..core7.hooks import Site, Intervention
    import torch
    checks, checked = {}, []
    for index, task in enumerate(tasks[:p["calibration_tasks"]]):
        prepared, geometry = _prepare(session, task, p)
        baseline, _ = session.adapter.decoder_forward(prepared)
        regions = _regions(prepared, task)
        queries, query_groups = _queries(prepared, task, p)
        controls = _controls(prepared, regions, p["seed"], mode=p["control_mode"])
        with selected_attention(session.adapter, heads=p["selected_heads"],
                                query_positions=queries, regions=regions) as attention:
            captured_logits, _ = session.adapter.decoder_forward(prepared)
        error = float((captured_logits[0].cpu() - prepared.clean_logits).abs().max().item())
        reconstruction = max(r["reconstruction_max_error"] for r in attention)
        checks[f"capture_inert_{index}"] = {"passed": error <= p["tolerance"], "measured": error, "threshold": p["tolerance"]}
        checks[f"selected_attention_rows_{index}"] = {"passed": reconstruction <= p["tolerance"], "measured": reconstruction, "threshold": p["tolerance"]}
        selected_region = next(iter(regions))
        with selected_attention(session.adapter, heads=p["selected_heads"], query_positions=queries,
                                regions=regions, block_region=selected_region) as blocked:
            blocked_logits, _ = session.adapter.decoder_forward(prepared)
        mass = max(r["region_attention"][selected_region]["mass"] for r in blocked)
        checks[f"edge_block_{index}"] = {"passed": mass <= 1e-7 and bool(torch.isfinite(blocked_logits).all()),
                                         "measured": mass, "threshold": 1e-7}
        positions = regions[selected_region]
        site = Site("resid_post", p["image_layers"][0])
        _, cache = session.adapter.decoder_forward(prepared, capture=[site], capture_positions=positions)
        values = cache[site.key][0]
        identity_logits, _ = session.adapter.decoder_forward(prepared, interventions=[Intervention(site, positions, values)])
        error = float((identity_logits - baseline).abs().max().item())
        checks[f"region_identity_{index}"] = {"passed": error <= p["tolerance"], "measured": error, "threshold": p["tolerance"]}
        checked.append({"task_id": task["task_id"], "condition": task["condition"], "geometry": geometry,
                        "attention_query_groups": query_groups,
                        "control_sizes": {k: len(v) for k, v in controls.items()}})
    return {"schema": "stretch.task-calibration.v1", "study": _name(config), "checks": checks,
            "all_passed": bool(checks) and all(v["passed"] for v in checks.values()),
            "protocol_sha256": canonical_hash(p), "tasks_sha256": task_hash,
            "calibration_tasks_sha256": calibration_task_hash,
            "core_calibration_sha256": canonical_hash(context["core_calibration"]),
            "execution_identity_sha256": execution_identity(session, context),
            "adapter": session.adapter.describe(), "checked_units": checked,
            "head_source_sha256": frozen["source_sha256"]}


def grouped_effects(rows: list[dict], *, seed: int, bootstrap_samples: int) -> dict:
    """Pair-mean effects with percentile bootstrap; repeated layouts stay clustered."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if "margin_change" in row:
            groups[row["operation"] + ":" + row.get("region", "")][row["pair_id"]].append(row["margin_change"])
    out = {}
    rng = random.Random(seed)
    for condition, pairs in sorted(groups.items()):
        means = [sum(values) / len(values) for _, values in sorted(pairs.items())]
        boot = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(bootstrap_samples))
        out[condition] = {"mean_margin_change": sum(means) / len(means), "n_pairs": len(means),
                          "ci95": [boot[int(.025 * (len(boot) - 1))], boot[int(.975 * (len(boot) - 1))]]
                          if len(means) >= 2 else None}
    return out


def routing_analysis(attention_rows: list[dict]) -> dict:
    """Factorial density contrasts, averaged within parent before interpretation."""
    cells = defaultdict(lambda: defaultdict(list))
    for row in attention_rows:
        q, o = row.get("question_level"), row.get("option_level")
        density = row.get("attention_density", {})
        if q not in (0, 1) or o not in (0, 1) or not {"question", "option"}.issubset(density):
            continue
        for query_group in row["query_groups"]:
            cells[(row["pair_id"], query_group)][(q, o)].append(density["question"] - density["option"])
    pairs = []
    for (pair, group), table in sorted(cells.items()):
        if set(table) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            continue
        mean = {key: sum(values) / len(values) for key, values in table.items()}
        pairs.append({"pair_id": pair, "query_group": group,
                      "question_main_effect": ((mean[1, 0] + mean[1, 1]) - (mean[0, 0] + mean[0, 1])) / 2,
                      "option_main_effect": ((mean[0, 1] + mean[1, 1]) - (mean[0, 0] + mean[1, 0])) / 2,
                      "interaction": mean[1, 1] - mean[1, 0] - mean[0, 1] + mean[0, 0]})
    return {"estimand": "question_minus_option_attention_mass_per_token", "per_parent_factorial": pairs,
            "causal_status": "observational; interpret with region/edge interventions"}


def specificity_analysis(rows: list[dict], p: Mapping) -> dict:
    """Paired parent differences: harm at target minus harm at matched controls."""
    table = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["operation"] == "selected-edge-block" and "margin_change" in row:
            table[row["region"]][row["pair_id"]].append(row["margin_change"])
    estimates = {}
    rng = random.Random(p["seed"])
    for region in sorted(k for k in table if not k.endswith("-matched")):
        control = table.get(region + "-matched", {})
        pairs = sorted(set(table[region]) & set(control))
        differences = [sum(control[u]) / len(control[u]) - sum(table[region][u]) / len(table[region][u]) for u in pairs]
        if not differences:
            continue
        draws = sorted(sum(rng.choice(differences) for _ in differences) / len(differences)
                       for _ in range(p["bootstrap_samples"]))
        ci = [draws[int(.025 * (len(draws) - 1))], draws[int(.975 * (len(draws) - 1))]] if len(pairs) >= 2 else None
        estimates[region] = {"mean": sum(differences) / len(differences), "n_pairs": len(pairs), "ci95": ci,
                             "exceeds_prespecified_minimum": ci[0] > p["minimum_effect"] if ci else None}
    return {"estimand": "matched_control_margin_change_minus_target_margin_change",
            "minimum_effect": p["minimum_effect"], "region_tests": estimates,
            "multiplicity": "separate prespecified regions; no omnibus circuit claim"}


def visual_binding_readouts(changed: list[float], baseline: list[float], content: int, literal: int) -> dict:
    """Comparable target margins plus the direct content-versus-copy contrast."""
    def margin(logits: list[float], target: int) -> float:
        return logits[target] - max(x for i, x in enumerate(logits) if i != target)

    return {"donor_content_margin_change": margin(changed, content) - margin(baseline, content),
            "literal_label_margin_change": margin(changed, literal) - margin(baseline, literal),
            "content_vs_literal_logit_change": (changed[content] - changed[literal]) -
                                                (baseline[content] - baseline[literal]),
            "content_vs_literal_logit_difference": changed[content] - changed[literal]}


def binding_analysis(rows: list[dict], p: Mapping) -> dict:
    """Paired visual-option transfer controls, with layouts clustered by problem."""
    from .statistics import cluster_estimate
    tasks = defaultdict(dict)
    for row in rows:
        if (row.get("condition") == "visual" and row.get("region") == "final"
                and row.get("correct_correct_transfer") is True and row.get("content_vs_copy_identifiable") is True):
            if row["operation"] in tasks[row["task_id"]]:
                raise ValueError("duplicate visual-option transfer condition")
            tasks[row["task_id"]][row["operation"]] = row
    contrasts, missing = [], []
    random_names = [f"donor-heads-random-{index}" for index in range(p["random_head_sets"])]
    required = {"donor-frozen-heads", "donor-heads-all-frozen-layers", *random_names}
    for task, conditions in sorted(tasks.items()):
        if not required.issubset(conditions):
            missing.append({"task_id": task, "missing": sorted(required - conditions.keys())})
            continue
        top = conditions["donor-frozen-heads"]
        contrasts.append({"pair_id": top["pair_id"],
                          "top4_minus_random": top["donor_content_margin_change"] -
                              sum(conditions[key]["donor_content_margin_change"] for key in random_names) / len(random_names),
                          "content_minus_literal_copy": top["content_vs_literal_logit_change"],
                          "patched_content_minus_literal": top["content_vs_literal_logit_difference"],
                          "all_layer_head_reference": conditions["donor-heads-all-frozen-layers"]["donor_content_margin_change"]})
    estimates = {metric: cluster_estimate(contrasts, value=metric, cluster="pair_id", seed=p["seed"], draws=p["bootstrap_samples"])
                 for metric in ("top4_minus_random", "content_minus_literal_copy", "patched_content_minus_literal", "all_layer_head_reference")}
    enough = len({row["pair_id"] for row in contrasts}) >= p["min_units"]
    passed = enough and not missing and all(value["ci95"] and value["ci95"][0] >
                 (p["minimum_effect"] if key == "top4_minus_random" else 0)
                 for key, value in estimates.items())
    return {"estimands": estimates, "correct_correct_parent_units": len({r["pair_id"] for r in contrasts}),
            "content_copy_contrast": "direct paired content-minus-literal logits; baseline change and patched preference reported separately",
            "missing_control_tasks": missing, "pattern_criterion_passed": bool(passed),
            "claim_limit": "intervention-defined transfer, not a complete circuit or native multi-image generalization"}


def _behavioral_preflight(session: Any, tasks: list[dict], p: Mapping) -> dict:
    """Measure task eligibility before head/region interventions, preserve controls."""
    from PIL import Image
    rows, unavailable = [], []
    for task in tasks:
        prepared, _ = _prepare(session, task, p)
        clean = _score(session, prepared, task, prepared.clean_logits)
        row = {"task_id": task["task_id"], "condition": task["condition"], "pair_id": task["source"]["pair_id"],
               "clean": clean}
        if task["condition"] == "matched_text":
            rows.append(row)
            continue
        image = _image(session, task, p)
        alternatives = [t for t in tasks if t["condition"] == task["condition"]
                        and t["source"]["pair_id"] != task["source"]["pair_id"]
                        and t["option_semantics"][t["correct_index"]] != task["option_semantics"][task["correct_index"]]]
        if not alternatives:
            unavailable.append(task["task_id"])
            rows.append(row)
            continue
        donor = sorted(alternatives, key=lambda t: canonical_hash([p["seed"], task["task_id"], t["task_id"]]))[0]
        donor_image = _image(session, donor, p)
        blank = image.copy() if task["condition"] == "visual" else Image.new("RGB", image.size, "white")
        shuffled = image.copy() if task["condition"] == "visual" else donor_image.resize(image.size)
        if task["condition"] == "visual":
            box = tuple(int(v) for v in task["regions"]["query"])
            source_box = tuple(int(v) for v in donor["regions"]["query"])
            blank.paste("white", box)
            shuffled.paste(donor_image.crop(source_box).resize((box[2] - box[0], box[3] - box[1])), box[:2])
        for operation, modified in (("blank", blank), ("shuffled", shuffled)):
            control, _ = _prepare(session, task, p, modified)
            row[operation] = _score(session, control, task, control.clean_logits)
        row["control_scope"] = "query_only_options_symbols_preserved" if task["condition"] == "visual" else "whole_image"
        rows.append(row)
    primary = [r for r in rows if r["condition"] != "matched_text"]
    correct_only = p.get("correct_only", False)
    if type(correct_only) is not bool:
        raise ValueError("correct_only must be explicitly boolean")
    excluded = []
    eligible_ids = sorted({r["pair_id"] for r in primary})
    if correct_only:
        by_pair_tasks = defaultdict(list)
        by_pair_rows = defaultdict(list)
        for task in tasks:
            by_pair_tasks[task["source"]["pair_id"]].append(task)
        for row in rows:
            by_pair_rows[row["pair_id"]].append(row)
        eligible_ids = []
        for pair_id in sorted(by_pair_tasks):
            task_rows = by_pair_rows[pair_id]
            reasons = []
            if not task_rows or any(r["clean"].get("correct") is not True for r in task_rows):
                reasons.append("not_every_required_clean_condition_is_correct")
            primary_rows = [r for r in task_rows if r["condition"] != "matched_text"]
            if not primary_rows or any("blank" not in r or "shuffled" not in r for r in primary_rows):
                reasons.append("required_behavior_controls_incomplete")
            if p["study"] == "spatial":
                combos = {(t.get("question_level"), t.get("option_level")) for t in by_pair_tasks[pair_id]}
                if combos != {(0, 0), (0, 1), (1, 0), (1, 1)}:
                    reasons.append("four_cell_spatial_factorial_incomplete")
            else:
                conditions = defaultdict(set)
                for task in by_pair_tasks[pair_id]:
                    conditions[task["task_id"]].add(task["condition"])
                if not conditions or any(value != {"visual", "matched_text"} for value in conditions.values()):
                    reasons.append("visual_and_matched_text_layout_pair_incomplete")
            if reasons:
                excluded.append({"pair_id": pair_id, "reasons": reasons})
            else:
                eligible_ids.append(pair_id)
        eligible_available = len(eligible_ids)
        eligible_ids = sorted(eligible_ids, key=lambda value: (canonical_hash([p["seed"], value]), value))[:p["target_units"]]
        eligible = set(eligible_ids)
        primary = [r for r in primary if r["pair_id"] in eligible]
    else:
        eligible_available = len(eligible_ids)
    clusters = defaultdict(list)
    for row in primary:
        clusters[row["pair_id"]].append(row)
    accuracy = sum(sum(r["clean"]["correct"] for r in rs) / len(rs) for rs in clusters.values()) / len(clusters) if clusters else None
    complete_controls = [rs for rs in clusters.values() if all("shuffled" in r and "blank" in r for r in rs)]
    gap = sum(sum(r["clean"]["correct"] - r["shuffled"]["correct"] for r in rs) / len(rs) for rs in complete_controls) / len(complete_controls) if complete_controls else None
    missing_eligible = [task_id for task_id in unavailable
                        if any(r["task_id"] == task_id and r["pair_id"] in set(eligible_ids) for r in rows)]
    count_ok = len(clusters) >= p["min_units"] if correct_only else len(clusters) == p["target_units"]
    passed = (not missing_eligible and count_ok and accuracy is not None
              and accuracy >= p["minimum_accuracy"] and gap is not None and gap >= p["minimum_image_gap"])
    return {"passed": passed, "accuracy": accuracy, "image_gap": gap, "rows": rows,
            "missing_control_task_ids": unavailable, "missing_control_eligible_task_ids": missing_eligible,
            "correct_only": correct_only, "eligible_pair_ids": eligible_ids,
            "excluded_parents": excluded, "eligible_units_available": eligible_available,
            "eligible_units": len(eligible_ids),
            "target_units": p["target_units"], "minimum_units": p["min_units"],
            "unit": "parent_pair_equal_weight",
            "minimum_accuracy": p["minimum_accuracy"], "minimum_image_gap": p["minimum_image_gap"]}


def run_study(session: Any, config: Mapping, context: Mapping) -> dict:
    p, frozen = _guard(session, config, context)
    tasks, task_hash = _load_tasks(config, p)
    calibration = context.get("extension_calibration", {})
    if (calibration.get("all_passed") is not True or calibration.get("protocol_sha256") != canonical_hash(p)
            or calibration.get("tasks_sha256") != task_hash or calibration.get("adapter") != session.adapter.describe()
            or calibration.get("core_calibration_sha256") != canonical_hash(context["core_calibration"])
            or calibration.get("execution_identity_sha256") != execution_identity(session, context)):
        raise ValueError("matching measured task-extension calibration required")
    unit_ids = sorted({t["source"]["pair_id"] for t in tasks})
    protocol_hash = canonical_hash(p)
    if config["phase"] == "confirmation":
        freeze = json.loads(Path(config["freeze_path"]).read_text(encoding="utf-8"))
        if freeze.get("freeze_sha256") != canonical_hash({k: v for k, v in freeze.items() if k != "freeze_sha256"}):
            raise ValueError("task discovery freeze content hash mismatch")
        if (freeze.get("confirmation_protocol_sha256") != protocol_hash
                or freeze.get("selected_image_layer") != p["image_layers"][0]
                or set(freeze.get("discovery_pair_ids", [])) & set(unit_ids)):
            raise ValueError("confirmation protocol mismatch or discovery pair reuse")
        scene_ids = {t["source"]["scene_id"] for t in tasks}
        if scene_ids & set(freeze.get("discovery_scene_ids", [])):
            raise ValueError("confirmation scene exposed in discovery")
    out = Path(context["output_dir"]).resolve()
    protected = [session.data_dir.resolve(), Path(config["tasks_path"]).resolve().parent,
                 Path(config["calibration_tasks_path"]).resolve().parent, Path(p["task_assets_root"])]
    if out.exists() or any(root == out or root in out.parents or out in root.parents for root in protected):
        raise ValueError("fresh task output outside source dataset required")
    out.mkdir(parents=True, exist_ok=False)
    header = {"schema": "stretch.task-attempt.v1", "study": _name(config), "identity": context["identity"],
              "protocol_sha256": protocol_hash, "tasks_sha256": task_hash,
              "calibration_sha256": canonical_hash(calibration), "phase": config["phase"]}
    (out / "header.json").write_text(json.dumps(header, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    behavior = _behavioral_preflight(session, tasks, p)
    (out / "behavioral_preflight.json").write_text(json.dumps(behavior, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not behavior["passed"]:
        report = {**header, "status": "insufficient", "coverage": {"target_units": p["target_units"], "observed_units": len(unit_ids)},
                  "scientific_verdict": "behaviorally_ineligible_no_mechanistic_operations_executed", "behavior": behavior,
                  "artifacts": {"behavior": str(out / "behavioral_preflight.json")}}
        from .reporting import write_study_artifacts
        report["artifacts"]["exports"] = write_study_artifacts(out, behavior["rows"], report)
        (out / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return report
    if behavior["correct_only"]:
        eligible = set(behavior["eligible_pair_ids"])
        tasks = [task for task in tasks if task["source"]["pair_id"] in eligible]
        unit_ids = sorted(eligible)
    partial_rows = []
    resumed_task_keys = set()
    partial_path = config.get("partial_rows_path")
    if partial_path:
        if config["phase"] != "discovery":
            raise ValueError("partial task-row recovery is discovery-only")
        partial_path = Path(partial_path).resolve()
        partial_header = partial_path.parent / "header.json"
        if not partial_path.is_file() or not partial_header.is_file():
            raise ValueError("partial task recovery requires raw.jsonl and its header.json")
        prior_header = json.loads(partial_header.read_text(encoding="utf-8"))
        for key, expected_value in {
            "schema": "stretch.task-attempt.v1",
            "study": _name(config),
            "protocol_sha256": protocol_hash,
            "tasks_sha256": task_hash,
            "phase": config["phase"],
        }.items():
            if prior_header.get(key) != expected_value:
                raise ValueError(f"partial task recovery header mismatch: {key}")
        current_identity = context["identity"]
        prior_identity = prior_header.get("identity", {})
        for key in ("model_key", "dataset_hash", "image_aware_release_hash",
                    "checkpoint_identity", "split"):
            if prior_identity.get(key) != current_identity.get(key):
                raise ValueError(f"partial task recovery identity mismatch: {key}")
        prior_campaign = json.loads((partial_path.parent.parent / "attempt.json").read_text(encoding="utf-8"))
        prior_decision = prior_campaign.get("verified_inputs", {}).get("protocol_decision")
        current_decision = context.get("protocol_decision")
        if not isinstance(prior_decision, dict) or not isinstance(current_decision, dict):
            raise ValueError("partial task recovery requires both protocol decision documents")
        # The source commit is execution provenance, not a scientific choice.
        # Recovery necessarily runs newer orchestration code, so compare every
        # scientific field while allowing only that provenance value to change.
        scientific_prior = {key: value for key, value in prior_decision.items()
                            if key != "source_git_commit"}
        scientific_current = {key: value for key, value in current_decision.items()
                              if key != "source_git_commit"}
        if scientific_prior != scientific_current:
            raise ValueError("partial task recovery scientific protocol decision changed")
        known_keys = [(task["task_id"], task["condition"]) for task in tasks]
        known_set = set(known_keys)
        observed_order = []
        observed_set = set()
        duplicate_keys = set()
        seen_rows = set()
        with partial_path.open(encoding="utf-8") as prior_stream:
            for line in prior_stream:
                row = json.loads(line)
                task_key = row.get("task_id"), row.get("condition")
                row_key = (*task_key, row.get("operation"), row.get("region", ""))
                if task_key not in known_set:
                    raise ValueError("partial task recovery contains an unknown task")
                if row_key in seen_rows:
                    duplicate_keys.add(row_key)
                seen_rows.add(row_key)
                if task_key not in observed_set:
                    observed_order.append(task_key)
                    observed_set.add(task_key)
                partial_rows.append(row)
        if duplicate_keys or not observed_order:
            raise ValueError("partial task recovery is empty or contains duplicate operations")
        if observed_order != known_keys[:len(observed_order)]:
            raise ValueError("partial task recovery is not a sequential task prefix")
        # The final observed task may have been interrupted at any operation.
        # Retain only earlier whole tasks and recompute the final task from its
        # beginning. This makes recovery independent of signal timing.
        resumed_task_keys = set(observed_order[:-1])
        partial_rows = [row for row in partial_rows
                        if (row["task_id"], row["condition"]) in resumed_task_keys]
    from ..core7.hooks import Site, Intervention
    from PIL import Image
    import torch
    rows, exclusions, attention_rows = list(partial_rows), [], []
    expected = set()
    seen = {(row["task_id"], row["condition"], row["operation"], row.get("region", ""))
            for row in partial_rows}
    with (out / "raw.jsonl").open("x", encoding="utf-8") as stream:
        for row in partial_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        def emit(task, operation, score, *, region="", extra=None):
            key = task["task_id"], task["condition"], operation, region
            if key in seen:
                raise ValueError("duplicate task operation")
            seen.add(key)
            record = {"task_id": task["task_id"], "condition": task["condition"],
                      "pair_id": task["source"]["pair_id"], "scene_id": task["source"]["scene_id"],
                      "split": config["split"], "operation": operation, "region": region, **score, **(extra or {})}
            rows.append(record)
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
        for task in tasks:
            prepared, geometry = _prepare(session, task, p)
            regions = _regions(prepared, task)
            queries, query_groups = _queries(prepared, task, p)
            try:
                controls = _controls(prepared, regions, p["seed"], mode=p["control_mode"])
            except ValueError as error:
                exclusions.append({"task_id": task["task_id"], "condition": task["condition"], "reason": str(error)})
                continue
            task_key = task["task_id"], task["condition"]
            base_operations = ["clean", "blank", "shuffled", "attention_capture", "self-frozen-heads",
                               "zero-head-frozen", "zero-head-all-frozen-layers",
                               *(f"zero-head-random-{i}" for i in range(p["random_head_sets"]))]
            if task["condition"] == "visual":
                base_operations += ["blank-query-only", "shuffled-query-only", "option-content-cycle-labels-fixed"]
                for layer in p["image_layers"]:
                    expected.add((*task_key, f"donor-panel-L{layer}", "query"))
                    # The semantic donor may move to any recipient panel, but the
                    # requested semantic target is fixed by deterministic selection.
                donor_ops = ["donor-frozen-heads", "donor-final"]
                if p.get("require_unrelated_donor_control", True):
                    donor_ops.append("unrelated-frozen-heads")
                for op in donor_ops:
                    expected.add((*task_key, op, "final"))
            for operation in base_operations:
                expected.add((*task_key, operation, ""))
            for region in [*regions, *(name + "-matched" for name in controls)]:
                expected.add((*task_key, "selected-edge-block", region))
                for layer in p["image_layers"]:
                    for op in ("self-region", "zero-region"):
                        expected.add((*task_key, f"{op}-L{layer}", region))
            if task_key in resumed_task_keys:
                # Attention matrices are held in memory until final reporting,
                # so regenerate only this inexpensive observational forward for
                # recovered tasks. All expensive intervention rows remain the
                # exact, hash-bound rows from the interrupted attempt.
                with selected_attention(session.adapter, heads=p["selected_heads"],
                                        query_positions=queries, regions=regions) as attn:
                    logits, _ = session.adapter.decoder_forward(prepared)
                prior_attention = next((row for row in partial_rows
                                        if (row["task_id"], row["condition"], row["operation"],
                                            row.get("region", "")) == (*task_key, "attention_capture", "")), None)
                replay = _score(session, prepared, task, logits)
                if prior_attention is None or replay["label_logits"] != prior_attention["label_logits"]:
                    raise ValueError("partial task recovery attention replay changed")
                for row in attn:
                    attention_rows.append({"task_id": task["task_id"], "condition": task["condition"],
                                           "pair_id": task["source"]["pair_id"],
                                           "question_level": task.get("question_level"),
                                           "option_level": task.get("option_level"),
                                           "query_groups": [name for name, positions in query_groups.items()
                                                            if row["query_position"] in positions],
                                           "attention_density": {name: block["mass"] / block["tokens"]
                                                                 for name, block in row["region_attention"].items()},
                                           **row})
                continue
            baseline, _ = session.adapter.decoder_forward(prepared)
            clean = _score(session, prepared, task, baseline)
            emit(task, "clean", clean, extra={"geometry": geometry})
            # Explicit full-image references and wrong-image behavior controls.
            image = _image(session, task, p)
            alternatives = [t for t in tasks if t["source"]["pair_id"] != task["source"]["pair_id"]
                            and t["condition"] == task["condition"] and t["option_semantics"][t["correct_index"]] != task["option_semantics"][task["correct_index"]]]
            for operation in ("blank", "shuffled"):
                if operation == "shuffled" and not alternatives:
                    exclusions.append({"task_id": task["task_id"], "condition": task["condition"], "reason": "no_matched_shuffled_image"})
                    continue
                changed = Image.new("RGB", image.size, "white") if operation == "blank" else _image(session, sorted(alternatives, key=lambda t: canonical_hash([p["seed"], task["task_id"], t["task_id"]]))[0], p).resize(image.size)
                control, _ = _prepare(session, task, p, changed)
                emit(task, operation, _score(session, control, task, control.clean_logits))
            if task["condition"] == "visual":
                query_box = tuple(int(v) for v in task["regions"]["query"])
                blank_query = image.copy()
                blank_query.paste("white", query_box)
                blank_prepared, _ = _prepare(session, task, p, blank_query)
                emit(task, "blank-query-only", _score(session, blank_prepared, task, blank_prepared.clean_logits))
                if alternatives:
                    donor_task = sorted(alternatives, key=lambda t: canonical_hash([p["seed"], task["task_id"], t["task_id"]]))[0]
                    donor_image = _image(session, donor_task, p)
                    donor_box = tuple(int(v) for v in donor_task["regions"]["query"])
                    shuffled_query = image.copy()
                    shuffled_query.paste(donor_image.crop(donor_box).resize((query_box[2] - query_box[0], query_box[3] - query_box[1])), query_box[:2])
                    query_control, _ = _prepare(session, task, p, shuffled_query)
                    emit(task, "shuffled-query-only", _score(session, query_control, task, query_control.clean_logits),
                         extra={"donor_task_id": donor_task["task_id"], "scope": "query_only_options_symbols_preserved"})
                shifted = image.copy()
                for i in range(4):
                    box = tuple(int(v) for v in task["regions"][f"option-{i}"])
                    source_box = tuple(int(v) for v in task["regions"][f"option-{(i + 1) % 4}"])
                    shifted.paste(image.crop(source_box).resize((box[2] - box[0], box[3] - box[1])), box[:2])
                shifted_task = {**task, "correct_index": (task["correct_index"] - 1) % 4,
                                "option_semantics": task["option_semantics"][1:] + task["option_semantics"][:1]}
                shifted_prepared, _ = _prepare(session, shifted_task, p, shifted)
                emit(task, "option-content-cycle-labels-fixed", _score(session, shifted_prepared, shifted_task, shifted_prepared.clean_logits),
                     extra={"control_correct_index": shifted_task["correct_index"], "control_semantic_order": task["option_semantics"][1:] + task["option_semantics"][:1]})
            with selected_attention(session.adapter, heads=p["selected_heads"], query_positions=queries, regions=regions) as attn:
                logits, _ = session.adapter.decoder_forward(prepared)
            emit(task, "attention_capture", _score(session, prepared, task, logits))
            for row in attn:
                attention_rows.append({"task_id": task["task_id"], "condition": task["condition"],
                                       "pair_id": task["source"]["pair_id"],
                                       "question_level": task.get("question_level"), "option_level": task.get("option_level"),
                                       "query_groups": [name for name, positions in query_groups.items() if row["query_position"] in positions],
                                       "attention_density": {name: block["mass"] / block["tokens"] for name, block in row["region_attention"].items()}, **row})
            all_regions = {**regions, **{name + "-matched": positions for name, positions in controls.items()}}
            for name, positions in all_regions.items():
                for layer in p["image_layers"]:
                    site = Site("resid_post", layer)
                    _, capture = session.adapter.decoder_forward(prepared, capture=[site], capture_positions=positions)
                    values = capture[site.key][0]
                    for op, replacement in (("self-region", values), ("zero-region", torch.zeros_like(values))):
                        result, _ = session.adapter.decoder_forward(prepared, interventions=[Intervention(site, positions, replacement)])
                        score = _score(session, prepared, task, result)
                        emit(task, f"{op}-L{layer}", score, region=name,
                             extra={"margin_change": score["truth_margin"] - clean["truth_margin"], "region_tokens": len(positions)})
                with selected_attention(session.adapter, heads=p["selected_heads"], query_positions=queries,
                                        regions=all_regions, block_region=name):
                    result, _ = session.adapter.decoder_forward(prepared)
                score = _score(session, prepared, task, result)
                emit(task, "selected-edge-block", score, region=name,
                     extra={"margin_change": score["truth_margin"] - clean["truth_margin"], "region_tokens": len(positions)})
            head_sets = {"frozen": p["selected_heads"]}
            head_sets["all-frozen-layers"] = [[l, h] for l in sorted({l for l, _ in p["selected_heads"]})
                                              for h in range(session.adapter.handle.num_heads)]
            for index in range(p["random_head_sets"]):
                chosen = []
                for layer in sorted({h[0] for h in p["selected_heads"]}):
                    actual = {h for l, h in p["selected_heads"] if l == layer}
                    pool = sorted(set(range(session.adapter.handle.num_heads)) - actual,
                                  key=lambda h: canonical_hash([p["seed"], index, layer, h]))
                    if len(pool) < len(actual):
                        raise ValueError("insufficient matched random heads")
                    chosen.extend([[layer, h] for h in pool[:len(actual)]])
                head_sets[f"random-{index}"] = chosen
            for set_name, heads in head_sets.items():
                interventions, self_heads = [], []
                for layer in sorted({h[0] for h in heads}):
                    site = Site("head", layer)
                    _, captures = session.adapter.decoder_forward(prepared, capture=[site], capture_positions=[prepared.final_position])
                    interventions.append(Intervention(site, [prepared.final_position], torch.zeros_like(captures[site.key][0]),
                                                      heads=[h for l, h in heads if l == layer]))
                    if set_name == "frozen":
                        self_heads.append(Intervention(site, [prepared.final_position], captures[site.key][0],
                                                       heads=[h for l, h in heads if l == layer]))
                result, _ = session.adapter.decoder_forward(prepared, interventions=interventions)
                score = _score(session, prepared, task, result)
                emit(task, "zero-head-" + set_name, score,
                     extra={"margin_change": score["truth_margin"] - clean["truth_margin"], "head_set": heads})
                if set_name == "frozen":
                    identity_result, _ = session.adapter.decoder_forward(prepared, interventions=self_heads)
                    identity_score = _score(session, prepared, task, identity_result)
                    emit(task, "self-frozen-heads", identity_score,
                         extra={"margin_change": identity_score["truth_margin"] - clean["truth_margin"]})
            if task["condition"] == "visual":
                # Within-parent semantic-image donors, preserving all four semantic
                # option identities while allowing independent panel/symbol maps.
                donors = [t for t in tasks if t["condition"] == "visual"
                          and t["source"]["pair_id"] == task["source"]["pair_id"]
                          and set(t["option_semantics"]) == set(task["option_semantics"])
                          and t["option_semantics"][t["correct_index"]] != task["option_semantics"][task["correct_index"]]]
                donors = [t for t in donors if set(t["labels"]) == set(task["labels"])
                          and task["option_semantics"].index(t["option_semantics"][t["correct_index"]])
                          != task["labels"].index(t["labels"][t["correct_index"]])]
                donors = sorted(donors, key=lambda t: canonical_hash([p["seed"], task["task_id"], t["task_id"]]))
                if not donors:
                    exclusions.append({"task_id": task["task_id"], "reason": "no_within_parent_donor_distinguishing_content_remap_from_symbol_copy"})
                    continue
                donor = donors[0]
                dp, _ = _prepare(session, donor, p)
                donor_correct = _score(session, dp, donor, dp.clean_logits)["correct"]
                semantic = donor["option_semantics"][donor["correct_index"]]
                content_index = task["option_semantics"].index(semantic)
                donor_symbol = donor["labels"][donor["correct_index"]]
                if donor_symbol not in task["labels"]:
                    raise ValueError("visual donor and recipient need same symbol alphabet")
                copied_index = task["labels"].index(donor_symbol)
                conditions, condition_donor_ids, condition_donor_correct = [], {}, {}
                for layer in p["image_layers"]:
                    site = Site("resid_post", layer)
                    for group in ("query", f"option-{donor['correct_index']}"):
                        recipient_group = "query" if group == "query" else f"option-{content_index}"
                        expected.add((*task_key, f"donor-panel-L{layer}", recipient_group))
                        donor_box, recipient_box = donor["regions"][group], task["regions"][recipient_group]
                        if (donor_box[2] - donor_box[0], donor_box[3] - donor_box[1]) != (recipient_box[2] - recipient_box[0], recipient_box[3] - recipient_box[1]):
                            exclusions.append({"task_id": task["task_id"], "reason": "donor_panel_pixel_geometry_mismatch", "region": group})
                            continue
                        source_positions, recipient_positions = dp.groups[group], prepared.groups[recipient_group]
                        if len(source_positions) != len(recipient_positions):
                            exclusions.append({"task_id": task["task_id"], "reason": "donor_panel_token_count_mismatch", "region": group})
                            continue
                        _, capture = session.adapter.decoder_forward(dp, capture=[site], capture_positions=source_positions)
                        conditions.append((f"donor-panel-L{layer}", recipient_group,
                                           [Intervention(site, recipient_positions, capture[site.key][0])]))
                for set_name, heads in head_sets.items():
                    patches = []
                    for layer in sorted({l for l, _ in heads}):
                        site = Site("head", layer)
                        _, captures = session.adapter.decoder_forward(dp, capture=[site], capture_positions=[dp.final_position])
                        patches.append(Intervention(site, [prepared.final_position], captures[site.key][0],
                                                    heads=[h for l, h in heads if l == layer]))
                    operation = "donor-frozen-heads" if set_name == "frozen" else "donor-heads-" + set_name
                    expected.add((*task_key, operation, "final"))
                    conditions.append((operation, "final", patches))
                last = Site("resid_post", session.adapter.num_layers - 1)
                _, capture = session.adapter.decoder_forward(dp, capture=[last], capture_positions=[dp.final_position])
                conditions.append(("donor-final", "final", [Intervention(last, [prepared.final_position], capture[last.key][0])]))
                unrelated = [t for t in tasks if t["condition"] == "visual"
                             and t["source"]["pair_id"] != task["source"]["pair_id"]
                             and set(t["option_semantics"]) == set(task["option_semantics"])
                             and t["option_semantics"][t["correct_index"]] == semantic
                             and t["labels"][t["correct_index"]] == donor_symbol]
                if not unrelated and p.get("require_unrelated_donor_control", True):
                    exclusions.append({"task_id": task["task_id"], "reason": "no_semantic_matched_unrelated_donor"})
                elif unrelated:
                    random_task = sorted(unrelated, key=lambda t: canonical_hash([p["seed"], task["task_id"], t["task_id"]]))[0]
                    rp, _ = _prepare(session, random_task, p)
                    condition_donor_correct["unrelated-frozen-heads"] = _score(session, rp, random_task, rp.clean_logits)["correct"]
                    random_patches = []
                    for layer in sorted({l for l, _ in p["selected_heads"]}):
                        site = Site("head", layer)
                        _, random_capture = session.adapter.decoder_forward(rp, capture=[site], capture_positions=[rp.final_position])
                        random_patches.append(Intervention(site, [prepared.final_position], random_capture[site.key][0],
                                                           heads=[h for l, h in p["selected_heads"] if l == layer]))
                    conditions.append(("unrelated-frozen-heads", "final", random_patches))
                    condition_donor_ids["unrelated-frozen-heads"] = random_task["task_id"]
                for operation, region, patches in conditions:
                    changed, _ = session.adapter.decoder_forward(prepared, interventions=patches)
                    score = _score(session, prepared, task, changed)
                    emit(task, operation, score, region=region,
                         extra={"donor_task_id": condition_donor_ids.get(operation, donor["task_id"]), "donor_semantic_target": semantic,
                                "donor_content_recipient_symbol": task["labels"][content_index],
                                "donor_literal_symbol": donor_symbol,
                                "content_vs_copy_identifiable": content_index != copied_index,
                                "correct_correct_transfer": bool(clean["correct"] and condition_donor_correct.get(operation, donor_correct)),
                                **visual_binding_readouts(score["label_logits"], clean["label_logits"], content_index, copied_index),
                                "margin_change": score["truth_margin"] - clean["truth_margin"]})
    clean_rows = [r for r in rows if r["operation"] == "clean"]
    used_units = {r["pair_id"] for r in clean_rows}
    accuracy = sum(r["correct"] for r in clean_rows) / len(clean_rows) if clean_rows else None
    missing_conditions = sorted(expected - seen)
    identity_errors = [abs(r["margin_change"]) for r in rows
                       if "margin_change" in r and (r["operation"].startswith("self-region") or r["operation"] == "self-frozen-heads")]
    identity_passed = bool(identity_errors) and max(identity_errors) <= 2 * p["tolerance"]
    complete = len(used_units) == p["target_units"] and not exclusions and not missing_conditions and identity_passed
    transfer_units = {r["pair_id"] for r in rows if r["operation"] == "donor-frozen-heads" and r.get("correct_correct_transfer") is True}
    if _name(config) == "image_options" and len(transfer_units) < p["min_units"]:
        complete = False
    effects = grouped_effects(rows, seed=p["seed"], bootstrap_samples=p["bootstrap_samples"])
    specificity = specificity_analysis(rows, p)
    layer_selection = None
    if config["phase"] == "discovery" and complete:
        layer_selection = select_task_image_layer(
            _name(config), rows, candidate_layers=p["image_layers"], seed=p["seed"],
            draws=p["bootstrap_samples"], minimum_units=p["min_units"],
            tie_tolerance=p["layer_tie_tolerance"], identity_tolerance=2 * p["tolerance"])
        if layer_selection["status"] != "ready":
            complete = False
    verdict = "behaviorally_ineligible" if accuracy is None or accuracy < p["minimum_accuracy"] else "causal_effects_require_specificity_review"
    if not complete:
        verdict = ("insufficient_discovery_layer_selection" if layer_selection is not None
                   and layer_selection["status"] != "ready" else "insufficient_or_incomplete_controls")
    elif any(v["exceeds_prespecified_minimum"] for v in specificity["region_tests"].values()):
        verdict = "selective_region_edge_effects_under_declared_zero_and_block_interventions"
    else:
        verdict = "no_confirmed_selective_region_edge_effect_at_prespecified_minimum"
    report = {**header, "status": "complete" if complete else "insufficient", "coverage": {
                "target_units": p["target_units"], "minimum_units": p["min_units"], "observed_units": len(used_units),
                "tasks": len(tasks), "clean_rows": len(clean_rows), "rows": len(rows), "duplicate_operations": False,
                "exclusions": exclusions, "correct_correct_transfer_units": len(transfer_units)},
                "scientific_verdict": verdict, "accuracy": accuracy,
                "condition_audit": {"expected_operations": len(expected), "observed_operations": len(seen),
                                    "missing": missing_conditions, "self_margin_tolerance": 2 * p["tolerance"],
                                    "self_identity_passed": identity_passed},
                "effects": effects, "attention_interpretation": "observational region masses; edge/zero interventions are separate causal tests",
                "effects_scope": "all-case disruption estimates; binding transfer primary cohort is correct/correct",
                "binding_transfer_primary": {
                    metric: grouped_effects([{**r, "margin_change": r[metric]} for r in rows
                                             if r.get("correct_correct_transfer") is True and metric in r],
                                            seed=p["seed"], bootstrap_samples=p["bootstrap_samples"])
                    for metric in ("donor_content_margin_change", "literal_label_margin_change")},
                "behavior": {key: value for key, value in behavior.items() if key != "rows"},
                "spatial_routing": routing_analysis(attention_rows),
                "specificity": specificity,
                "image_layer_selection": layer_selection,
                "visual_option_binding": binding_analysis(rows, p) if _name(config) == "image_options" else None,
                "artifacts": {"raw": str(out / "raw.jsonl"), "attention": str(out / "attention.json"), "report": str(out / "report.json")}}
    (out / "attention.json").write_text(json.dumps(attention_rows, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if complete and config["phase"] == "discovery":
        selected_layer = layer_selection["selected_layer"]
        confirmation_protocol = dict(p)
        confirmation_protocol["image_layers"] = [selected_layer]
        confirmation_hash = canonical_hash(confirmation_protocol)
        freeze = {"schema": "stretch.task-freeze.v1", "discovery_protocol_sha256": protocol_hash,
                  "confirmation_protocol": confirmation_protocol,
                  "confirmation_protocol_sha256": confirmation_hash,
                  "selected_image_layer": selected_layer, "layer_selection": layer_selection,
                  "discovery_pair_ids": unit_ids, "discovery_scene_ids": sorted({t["source"]["scene_id"] for t in tasks}),
                  "head_source_sha256": frozen["source_sha256"], "discovery_verdict": verdict}
        # Normalize JSON object keys before hashing.  Layer-selection diagnostics
        # can use integer layer keys; after a JSON round trip those become
        # strings and otherwise make an honestly written freeze fail its own
        # confirmation-time integrity check.
        freeze_payload = json.loads(json.dumps(freeze, sort_keys=True, allow_nan=False))
        freeze["freeze_sha256"] = canonical_hash(freeze_payload)
        (out / "freeze.json").write_text(json.dumps(freeze, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        report["artifacts"]["freeze"] = str(out / "freeze.json")
    from .reporting import write_study_artifacts
    report["artifacts"]["exports"] = write_study_artifacts(out, rows, report)
    (out / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
