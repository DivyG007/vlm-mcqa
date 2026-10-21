"""E21 sender-conditioned token-boundary routes, not isolated attention edges."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from .data import validate_fixed_text_contrast
from .data import validate_bidirectional
from .extension_calibration import (StudyWriter, calibration_report, core_guard, difference, document,
    forward, grouped_summary, measured_check, primitive_checks, protocol, require_calibration, require_execution, sha)
from .protocol import load_frozen_heads
from .runtime import _FiniteSession


def site_order(spec: Mapping[str, Any]) -> int:
    offsets = {"resid_pre": 0, "head": 2, "attn": 3, "mlp": 5, "resid_post": 8}
    if spec["site"] not in offsets or type(spec["layer"]) is not int or spec["layer"] < 0:
        raise ValueError("invalid boundary")
    return spec["layer"] * 10 + offsets[spec["site"]]


def validate_route(route: Mapping[str, Any], frozen: Mapping[str, Any], layers: int) -> None:
    if not route.get("id") or any(route[k]["site"] not in ("resid_pre", "resid_post") for k in ("sender", "receiver", "final")):
        raise ValueError("named residual sender/receiver/final boundaries required")
    k = route["head_k"]
    if k not in (4, 16) or len(frozen["head_candidates"]) < k:
        raise ValueError("route requires prespecified frozen top4 or top16")
    heads = frozen["head_candidates"][:k]
    if not (site_order(route["sender"]) < site_order(route["receiver"])
            < min(l * 10 + 2 for l, _ in heads)
            <= max(l * 10 + 2 for l, _ in heads) < site_order(route["final"])):
        raise ValueError("route boundaries must follow sender < receiver < every head < final")
    if (any(route[g]["layer"] >= layers for g in ("sender", "receiver", "final"))
            or route["sender"]["group"] != "image" or route["receiver"]["group"] not in ("options", "question", "labels", "final")
            or route["final"]["group"] != "final"):
        raise ValueError("unsupported group or out-of-range layer")
    if type(route.get("random_sets")) is not int or route["random_sets"] < 1 or type(route.get("seed")) is not int:
        raise ValueError("explicit random-head policy required")


def _inputs(session: Any, config: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[dict, dict, list[dict]]:
    body = protocol(config, context, "route")
    frozen = load_frozen_heads(body["head_source"], model_key=session.adapter.spec.key,
                               dataset_hash=session.manifest["dataset_hash"])
    if body.get("head_source_sha256") != frozen["source_sha256"]:
        raise ValueError("protocol head source bytes mismatch")
    routes = body["routes"]
    if not routes or len({r["id"] for r in routes}) != len(routes):
        raise ValueError("unique explicit route candidates required")
    if len(routes) > body["max_candidates"] or body["selection_rule"] != "max_pair_mean_staged_chain":
        raise ValueError("bounded prespecified candidate budget and selection rule required")
    if (type(body.get("min_pairs")) is not int or body["min_pairs"] < 1
            or type(body.get("bootstrap_seed")) is not int or type(body.get("bootstrap_samples")) is not int
            or body["bootstrap_samples"] < 100):
        raise ValueError("explicit positive route sample/statistical policy required")
    if config["phase"] == "confirmation" and len(routes) != 1:
        raise ValueError("confirmation must evaluate one frozen route")
    for route in routes:
        validate_route(route, frozen, session.adapter.num_layers)
    raw = document(config["contrasts"])
    raw = raw["contrasts"] if isinstance(raw, dict) else raw
    by_id = {r["contrast_id"]: r for r in raw}
    if len(by_id) != len(raw):
        raise ValueError("duplicate route contrast manifest direction")
    if config.get("sample_plan") is not None:
        from .campaign import validate_sample_plan
        plan = document(config["sample_plan"])
        selected = validate_sample_plan(plan, raw, {"model_key": session.adapter.spec.key,
                                                   "dataset_hash": session.manifest["dataset_hash"]})
        if plan.get("split") != config["phase"]:
            raise ValueError("route requires phase-matched eligible sample plan")
        ids = plan["contrast_ids"]
    else:
        ids = config["contrast_ids"]
        selected = [by_id[i] for i in ids]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate route direction IDs")
    validate_bidirectional(selected)
    if not selected:
        raise ValueError("explicit nonempty bidirectional route sample required")
    for contrast in selected:
        validate_fixed_text_contrast(contrast, session.variants)
    if any(c["split"] != config["phase"] for c in selected):
        raise ValueError("route direction phase mismatch")
    return body, frozen, selected


class _BatchSession(_FiniteSession):
    def __init__(self, session: Any, batch_size: int, case: Any = None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("positive batch_size required")
        super().__init__(session, case)
        self.batch_size = batch_size

    def batches(self, values):
        for start in range(0, len(values), self.batch_size):
            yield values[start:start + self.batch_size]


def evaluate_route(session: Any, contrast: Mapping[str, Any], route: Mapping[str, Any], frozen: Mapping[str, Any],
                   context: Mapping[str, Any], *, self_sender: bool = False, batch_size: int = 1) -> dict:
    """Capture under interventions, then transplant into genuinely clean trajectories."""
    core_guard(session, context)
    validate_fixed_text_contrast(contrast, session.variants)
    validate_route(route, frozen, session.adapter.num_layers)
    from ..core7.contrasts import _make_case, run_specs
    from ..core7.hooks import Intervention, Site
    from ..core7.heads import matched_random
    import random
    case = _make_case(session, contrast)
    if case is None:
        return {"contrast_id": contrast["contrast_id"], "status": "excluded_baseline_eligibility", "rows": []}
    if not case.donor_clean.correct or not case.recipient_clean.correct:
        raise ValueError("route transfer requires clean correct/correct")
    prepared = case.prepared
    donor_prepared = prepared if self_sender else session.prepared(case.donor["variant_id"])
    donor_clean = case.recipient_clean if self_sender else case.donor_clean
    if (donor_prepared.input_ids != prepared.input_ids or donor_prepared.groups["image"] != prepared.groups["image"]):
        raise ValueError("fixed-text route token alignment mismatch")
    sender, receiver, final_spec = (Site(route[k]["site"], route[k]["layer"]) for k in ("sender", "receiver", "final"))
    image_pos = prepared.groups["image"]
    receiver_pos = prepared.groups[route["receiver"]["group"]]
    if not image_pos or not receiver_pos:
        return {"contrast_id": contrast["contrast_id"], "status": "excluded_empty_groups", "rows": []}
    pool = [p for p in range(prepared.num_tokens) if p not in set(image_pos + receiver_pos + [prepared.final_position])]
    rng = random.Random(route["seed"])
    if len(pool) < len(receiver_pos):
        return {"contrast_id": contrast["contrast_id"], "status": "excluded_no_size_control", "rows": []}
    control_pos = sorted(rng.sample(pool, len(receiver_pos)))
    _, donor_capture = forward(session, donor_prepared, capture=[sender], capture_positions=image_pos)
    _, clean_capture = forward(session, prepared, capture=[sender, receiver], capture_positions=None)
    restore = Intervention(sender, image_pos, donor_capture[sender.key][0])
    _, hybrid = forward(session, prepared, interventions=[restore], capture=[receiver], capture_positions=None)
    recv_patch = Intervention(receiver, receiver_pos, hybrid[receiver.key][0, receiver_pos])
    complement = [p for p in range(prepared.num_tokens) if p not in set(receiver_pos)]
    reset = Intervention(receiver, complement, clean_capture[receiver.key][0, complement])
    heads = [tuple(h) for h in frozen["head_candidates"][:route["head_k"]]]
    head_sites = [Site("head", layer) for layer in sorted({l for l, _ in heads})]
    _, conditioned_heads = forward(session, prepared, interventions=[recv_patch], capture=head_sites,
                                    capture_positions=[prepared.final_position])

    def head_items(chosen, source):
        groups = defaultdict(list)
        for layer, head in chosen:
            groups[layer].append(head)
        items = []
        for layer, hs in sorted(groups.items()):
            key = f"head:{layer}"
            value = (conditioned_heads[key][0, 0] if source == "conditioned" else
                     donor_clean.sites[key] if source == "donor" else case.recipient_clean.sites[key])
            items.append(Intervention(Site("head", layer), [prepared.final_position], value, heads=hs))
        return items

    staged_heads = head_items(heads, "conditioned")
    _, staged_final = forward(session, prepared, interventions=staged_heads, capture=[final_spec],
                               capture_positions=[prepared.final_position])
    final_patch = Intervention(final_spec, [prepared.final_position], staged_final[final_spec.key][0, 0])
    specs = [
        ("clean", [], "clean"), ("image_total", [restore], "donor-image -> recipient"),
        ("receiver_transfer", [recv_patch], "image-conditioned receiver -> clean"),
        ("receiver_complement_reset", [restore, reset], "image -> receiver boundary; other tokens clean"),
        ("receiver_clamp", [restore, Intervention(receiver, receiver_pos, clean_capture[receiver.key][0, receiver_pos])], "image; receiver baseline"),
        ("size_control_transfer", [Intervention(receiver, control_pos, hybrid[receiver.key][0, control_pos])], "image-conditioned size-control -> clean"),
        ("head_direct", head_items(heads, "donor"), "clean donor heads -> clean"),
        ("staged_chain", staged_heads, "image -> receiver -> frozen heads -> clean"),
        ("head_clamp_after_image", [restore, *head_items(heads, "baseline")], "image; heads baseline"),
        ("head_clamp_after_receiver", [recv_patch, *head_items(heads, "baseline")], "receiver transfer; heads baseline"),
        ("final_staged", [final_patch], "image -> receiver -> heads -> final -> clean"),
        ("final_direct", [Intervention(final_spec, [prepared.final_position], donor_clean.sites[final_spec.key])], "clean donor final -> clean"),
        ("self_receiver", [Intervention(receiver, receiver_pos, clean_capture[receiver.key][0, receiver_pos])], "self receiver"),
        ("self_image", [Intervention(sender, image_pos, clean_capture[sender.key][0, image_pos])], "self image"),
        ("self_heads", head_items(heads, "baseline"), "baseline heads only"),
        ("image_all_retained", [restore, Intervention(receiver, list(range(prepared.num_tokens)), hybrid[receiver.key][0])], "all image-conditioned receiver tokens retained"),
    ]
    for draw in range(route["random_sets"]):
        random_heads = matched_random(heads, frozen["head_layers"], session.adapter.handle.num_heads, rng)
        # Random heads may use the same layers as selected heads, whose complete vectors were captured.
        specs.extend([(f"random_staged/{draw}", head_items(random_heads, "conditioned"), "receiver-conditioned random heads -> clean"),
                      (f"random_clamp/{draw}", [restore, *head_items(random_heads, "baseline")], "image; random heads baseline")])
    checked_session = _BatchSession(session, batch_size, case)
    rows = run_specs(checked_session, case,
        [({"condition": name, "route_id": route["id"], "lineage": lineage,
           "sender": route["sender"], "receiver": route["receiver"], "final": route["final"],
           "receiver_positions": list(receiver_pos), "control_positions": control_pos,
           "self_sender_calibration": self_sender}, items) for name, items, lineage in specs])
    for row, readout in zip(rows, checked_session.adapter.readouts, strict=True):
        row["readout"] = readout
        row.update({"raw_margin_change": row["patched_diff"] - case.recipient_diff,
                    "sample_unit": contrast["pair_id"], "split": contrast["split"],
                    "role": "route", "recipe_sha256": sha(route)})
        if abs(case.donor_diff - case.recipient_diff) < 1e-6:
            row["normalized_recovery"] = None
    return {"contrast_id": contrast["contrast_id"], "pair_id": contrast["pair_id"], "status": "evaluated", "rows": rows}


def calibrate_study(session: Any, config: Mapping[str, Any], context: Mapping[str, Any]) -> dict:
    require_execution(context)
    core_guard(session, context)
    body, frozen, _ = _inputs(session, config, context)
    from ..core7.hooks import Site
    sites = [Site(r[k]["site"], r[k]["layer"]) for r in body["routes"] for k in ("sender", "receiver", "final")]
    sites = list(set(sites))
    checks = primitive_checks(session, config["calibration_variant_ids"], sites, config["tolerance"])
    raw = document(config["contrasts"])
    raw = raw["contrasts"] if isinstance(raw, dict) else raw
    contrast = next(c for c in raw if c["contrast_id"] == config["calibration_contrast_id"])
    if contrast["split"] == "confirmation":
        raise ValueError("route calibration cannot inspect confirmation")
    for route in body["routes"]:
        sham = evaluate_route(session, contrast, route, frozen, context, self_sender=True)
        first = evaluate_route(session, contrast, route, frozen, context, batch_size=1)
        batch = evaluate_route(session, contrast, route, frozen, context, batch_size=2)
        if any(r["status"] != "evaluated" for r in (sham, first, batch)):
            raise ValueError("route calibration case unsuitable")
        checks[route["id"] + "/all_stage_self"] = measured_check(max(abs(r["raw_margin_change"]) for r in sham["rows"]), config["tolerance"])
        by_name = {r["condition"]: r for r in batch["rows"]}
        checks[route["id"] + "/actual_batch_single"] = measured_check(max(abs(r["patched_diff"] - by_name[r["condition"]]["patched_diff"]) for r in first["rows"]), config["tolerance"])
        names = {r["condition"]: r for r in first["rows"]}
        checks[route["id"] + "/all_retained_equivalence"] = measured_check(abs(names["image_total"]["patched_diff"] - names["image_all_retained"]["patched_diff"]), config["tolerance"])
    return calibration_report(session, context, "route", body, checks, config["calibration_variant_ids"])


def run_study(session: Any, config: Mapping[str, Any], context: Mapping[str, Any]) -> dict:
    require_execution(context)
    body, frozen, selected = _inputs(session, config, context)
    require_calibration(session, context, "route", body)
    writer = StudyWriter(session, config, context, "route", body)
    try:
        for contrast in selected:
            for route in body["routes"]:
                result = evaluate_route(session, contrast, route, frozen, context, batch_size=config["batch_size"])
                if result["status"] != "evaluated":
                    writer.append({**result, "route_id": route["id"]}, excluded=True)
                else:
                    for row in result["rows"]:
                        writer.append(row)
        summary = grouped_summary([{**r, "condition": r["route_id"] + "/" + r["condition"]} for r in writer.rows],
                                   seed=body["bootstrap_seed"], samples=body["bootstrap_samples"])
        pairs = len({r["sample_unit"] for r in writer.rows})
        adequate = pairs >= body["min_pairs"] and not writer.exclusions
        frozen_body = None
        if config["phase"] == "discovery" and adequate:
            scores = {r["id"]: summary[r["id"] + "/staged_chain"]["mean"] for r in body["routes"]}
            chosen = sorted(scores, key=lambda rid: (-scores[rid], rid))[0]
            frozen_body = {**body, "routes": [r for r in body["routes"] if r["id"] == chosen]}
        coverage = {"directions_requested": len(selected), "routes": len(body["routes"]),
                    "pairs": pairs, "raw_rows": len(writer.rows), "exclusions": len(writer.exclusions)}
        expected_conditions = {r["id"]: 16 + 2 * r["random_sets"] for r in body["routes"]}
        for contrast in selected:
            for route in body["routes"]:
                scope = [r for r in writer.rows if r["contrast_id"] == contrast["contrast_id"] and r["route_id"] == route["id"]]
                if scope and (len(scope) != expected_conditions[route["id"]]
                              or len({r["condition"] for r in scope}) != len(scope)):
                    raise ValueError("route raw condition scope missing or duplicated")
        coverage["expected_condition_counts"] = expected_conditions
        return writer.finish(coverage, "boundary_route_evidence_requires_controls; not_edge_complete" if adequate else "insufficient",
            summary, status="complete" if adequate else "insufficient", freeze_body=frozen_body)
    except BaseException as error:
        writer.failure(error)
        raise
