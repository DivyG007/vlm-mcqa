"""Explicitly gated runtime building blocks, NOT a model-loading launcher.

Unexecuted implementation. The caller must provide an already loaded Session,
fresh real Core-7 calibration and matching execution provenance. No server,
device, weight download or environment is selected here. Scalar saved baseline
records cannot substitute for newly captured activation trajectories.
"""
from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import Any, Mapping

from .data import digest, validate_fixed_text_contrast, validate_bidirectional

REQUIRED_GATES = {
    "lens_final_matches", "cached_matches_full", "replay_deterministic", "capture_is_inert",
    "identity_patch", "batched_matches_single", "residual_decomposition", "token_spans", "symbols_single_token",
}


def validate_runtime_context(session: Any, recipe: Mapping[str, Any], calibration: Mapping[str, Any],
                             identity: Mapping[str, Any], *, execute: bool = False) -> None:
    """Fail closed before any preparation or model call.

    Calibration binding fields are an explicit caller attestation, not fields
    invented in the historical calibration format. Caller must construct them
    from the current session/source/checkpoint, not copy stale evidence.
    """
    if execute is not True:
        raise PermissionError("runtime requires explicit execute=True after separate execution authorization")
    canonical = {k: v for k, v in recipe.items() if k != "recipe_sha256"}
    if digest(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()) != recipe["recipe_sha256"]:
        raise ValueError("recipe changed after compilation")
    for key in ("model_key", "dataset_hash", "freeze_hash", "head_source_sha256", "recipe_sha256"):
        if not recipe.get(key) or identity.get(key) != recipe[key]:
            raise ValueError(f"execution identity {key} mismatch")
    for key in ("source_identity", "checkpoint_identity", "image_aware_release_hash", "calibration_sha256"):
        if not identity.get(key):
            raise ValueError(f"missing {key}")
    calibration_hash = digest(json.dumps(calibration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if identity["calibration_sha256"] != calibration_hash:
        raise ValueError("calibration payload identity mismatch")
    if session.adapter.spec.key != recipe["model_key"] or session.manifest.get("dataset_hash") != recipe["dataset_hash"]:
        raise ValueError("current Session model/data mismatch")
    if identity["image_aware_release_hash"] != session.release_manifest.get("image_aware_release_hash"):
        raise ValueError("current Session image-aware release mismatch")
    spec = session.adapter.spec
    if not re.fullmatch(r"[0-9a-fA-F]{40}", spec.revision):
        raise ValueError("runtime requires explicitly pinned checkpoint commit revision")
    if identity["checkpoint_identity"] != {"hf_id": spec.hf_id, "revision": spec.revision}:
        raise ValueError("checkpoint identity differs from current adapter specification")
    if (session.adapter.handle.num_heads != recipe["num_heads"]
            or session.adapter.num_layers != recipe["num_layers"]):
        raise ValueError("current model dimensions mismatch")
    gates = calibration.get("gates", {})
    if calibration.get("all_passed") is not True or not REQUIRED_GATES.issubset(gates):
        raise ValueError("all nine real Core-7 calibration gates required")
    if any(gates[name].get("passed") is not True for name in REQUIRED_GATES):
        raise ValueError("failed Core-7 calibration gate")
    if calibration.get("adapter") != session.adapter.describe():
        raise ValueError("calibration does not describe current adapter")
    if not calibration.get("variants"):
        raise ValueError("real calibration variant evidence missing")
    if identity.get("calibration_binding") != {
        key: identity[key] for key in ("source_identity", "checkpoint_identity", "image_aware_release_hash")
    }:
        raise ValueError("calibration must be explicitly bound to current source/checkpoint/release")
    if identity.get("eligibility_decision") != "approved_for_this_stretch_protocol":
        raise ValueError("separate scientific eligibility decision missing")


def _finite(tensor: Any, description: str) -> None:
    import torch
    if not torch.isfinite(tensor).all():
        raise ValueError(f"non-finite {description}")


class _FiniteAdapter:
    """Local proxy: validate patched logits without altering the shared adapter."""
    def __init__(self, adapter: Any, case: Any = None):
        self._adapter = adapter
        self.case = case
        self.readouts = []

    def decoder_forward(self, *args: Any, **kwargs: Any) -> Any:
        logits, captured = self._adapter.decoder_forward(*args, **kwargs)
        _finite(logits, "patched decoder logits")
        for key, value in captured.items():
            _finite(value, key)
        if self.case is not None:
            from ..core7.scoring import symbol_readout
            self.readouts.extend(symbol_readout(row, self.case.label_ids, self.case.recipient["correct_index"])
                                 for row in logits)
        return logits, captured

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)


class _FiniteSession:
    def __init__(self, session: Any, case: Any = None):
        self._session = session
        self.adapter = _FiniteAdapter(session.adapter, case)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)


def _heads(case: Any, selected: list[list[int]], *, source: str) -> list[Any]:
    from ..core7.hooks import Intervention, Site
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in selected:
        by_layer[layer].append(head)
    clean = case.donor_clean if source == "donor" else case.recipient_clean
    return [Intervention(Site("head", layer), [case.prepared.final_position],
                         clean.sites[f"head:{layer}"], heads=heads)
            for layer, heads in sorted(by_layer.items())]


def _image_values(session: Any, case: Any, layer: int, kind: str) -> tuple[Any, Any, list[int]]:
    """Fresh donor/recipient image captures; never read scalar baseline files."""
    from ..core7.hooks import Site
    donor_prepared = session.prepared(case.donor["variant_id"])
    recipient_prepared = case.prepared
    donor_pos, recipient_pos = donor_prepared.groups["image"], recipient_prepared.groups["image"]
    if not donor_pos or donor_pos != recipient_pos or donor_prepared.num_tokens != recipient_prepared.num_tokens:
        raise ValueError("image-token correspondence unvalidated: different token layouts")
    image_set = set(recipient_pos)
    if any(a != b for i, (a, b) in enumerate(zip(donor_prepared.input_ids, recipient_prepared.input_ids))
           if i not in image_set):
        raise ValueError("non-image token IDs differ despite exact source prompt")
    site = Site(kind, layer)
    captures = []
    for prepared in (donor_prepared, recipient_prepared):
        _, captured = _FiniteAdapter(session.adapter).decoder_forward(prepared, capture=[site], capture_positions=recipient_pos)
        value = captured[site.key][0]
        _finite(value, site.key)
        captures.append(value)
    return captures[0], captures[1], list(recipient_pos)


def evaluate_contrast(session: Any, contrast: Mapping[str, Any], recipe: Mapping[str, Any], *,
                      calibration: Mapping[str, Any], identity: Mapping[str, Any], execute: bool = False,
                      contrast_manifest: list[dict[str, Any]],
                      cached_baseline: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate one direction; caller owns split sampling and exclusive output.

    Raw margin change is primary. Core normalized recovery is retained only as
    a secondary diagnostic alongside the denominator. Cases are never selected
    for denominator size. ``cached_baseline`` is provenance-only, not tensors.
    """
    validate_runtime_context(session, recipe, calibration, identity, execute=execute)
    validate_bidirectional(contrast_manifest)
    if not any(row == contrast for row in contrast_manifest):
        raise ValueError("direction absent from supplied bidirectional manifest")
    for row in contrast_manifest:
        validate_fixed_text_contrast(row, session.variants)
    scene_splits = defaultdict(set)
    for item in session.items.values():
        scene_splits[item["scene_id"]].add(item["split"])
    if any(len(splits) != 1 for splits in scene_splits.values()):
        raise ValueError("current dataset scene crosses splits")
    for side in ("donor", "recipient"):
        variant = session.variants[contrast[side + "_variant_id"]]
        item = session.items[variant["item_id"]]
        for key in ("scene_id", "split", "image", "semantic_answer", "counterfactual_pair_id"):
            if item.get(key) != variant.get(key):
                raise ValueError(f"current item/variant {key} mismatch")
        if contrast[side + "_scene_id"] != item["scene_id"]:
            raise ValueError("contrast scene identity mismatch")
    validate_fixed_text_contrast(contrast, session.variants)
    if contrast["split"] == "confirmation" and identity.get("confirmation_protocol_frozen") is not True:
        raise ValueError("confirmation requires prior scientific protocol freeze")
    from ..core7.contrasts import _make_case, run_specs
    from ..core7.hooks import Intervention, Site
    case = _make_case(session, contrast)
    if case is None:
        return {"contrast_id": contrast["contrast_id"], "pair_id": contrast["pair_id"],
                "status": "excluded_baseline_eligibility", "runtime_validation": "not_established",
                "identity": dict(identity), "role": "semantic", "split": contrast["split"],
                "recipe_sha256": recipe["recipe_sha256"], "contrast_type": contrast["contrast_type"]}
    if not case.donor_clean.correct or not case.recipient_clean.correct:
        raise ValueError("clean semantic transfer requires correct/correct; failure study needs separate protocol")
    _finite(case.donor_clean.logits, "donor baseline")
    _finite(case.recipient_clean.logits, "recipient baseline")
    donor_image, self_image, positions = _image_values(session, case, recipe["image_layer"], recipe["image_site"])
    site = Site(recipe["image_site"], recipe["image_layer"])
    restore = Intervention(site, positions, donor_image)
    specs = []
    for condition in recipe["conditions"]:
        operation = condition["operation"]
        selected = recipe["sets"].get(condition.get("head_set"), [])
        if operation == "clean":
            interventions = []
        elif operation == "image_restore":
            interventions = [restore]
        elif operation == "self_image":
            interventions = [Intervention(site, positions, self_image)]
        elif operation == "head_transfer":
            interventions = _heads(case, selected, source="donor")
        elif operation in ("self_head", "sham_clamp"):
            interventions = _heads(case, selected, source="recipient")
        elif operation == "restore_clamp":
            interventions = [restore, *_heads(case, selected, source="recipient")]
        elif operation == "final_residual":
            key = f"resid_post:{condition['layer']}"
            interventions = [Intervention(Site("resid_post", condition["layer"]),
                                          [case.prepared.final_position], case.donor_clean.sites[key])]
        elif operation == "attention_reference":
            interventions = [Intervention(Site("attn", layer), [case.prepared.final_position],
                                          case.donor_clean.sites[f"attn:{layer}"]) for layer in condition["layers"]]
        else:
            raise ValueError(f"unsupported operation {operation}")
        for intervention in interventions:
            _finite(intervention.values, intervention.site.key)
        specs.append(({"method": "stretch-semantic-image", **condition}, interventions))
    checked_session = _FiniteSession(session, case)
    rows = run_specs(checked_session, case, specs)
    for row, readout in zip(rows, checked_session.adapter.readouts, strict=True):
        row["readout"] = readout
        denominator = case.donor_diff - case.recipient_diff
        if abs(denominator) < 1e-6:
            row["normalized_recovery"] = None
        row.update({"raw_margin_change": row["patched_diff"] - case.recipient_diff,
                    "role": "semantic", "split": contrast["split"],
                    "recovery_denominator": denominator,
                    "flip_to_donor": row["predicted_symbol"] == contrast["donor_symbol"],
                    "sample_unit": contrast["pair_id"], "baseline_source": "current_session_clean_replay",
                    "image_values_source": "new_decoder_trajectory_capture",
                    "recipe_sha256": recipe["recipe_sha256"]})
    return {"contrast_id": contrast["contrast_id"], "pair_id": contrast["pair_id"], "status": "evaluated",
            "role": "semantic", "split": contrast["split"], "contrast_type": contrast["contrast_type"],
            "recipe_sha256": recipe["recipe_sha256"],
            "rows": rows, "identity": dict(identity), "cached_baseline_supplied": cached_baseline is not None,
            "cached_baseline_use": "metadata_only; never used as activations",
            "cached_baseline_metadata": dict(cached_baseline or {}),
            "runtime_validation": "requires_separate_composed_intervention_calibration"}


def validate_head_control(contrast: Mapping[str, Any], variants: Mapping[str, Mapping[str, Any]],
                          items: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Pure-data validation for old position/unrelated contrasts, never image swaps.

    Controls are clustered by recipient pair; unrelated donor identity is also
    retained because donor reuse introduces additional dependence. This is not
    a claim that the recipient clusters alone remove all donor dependence.
    """
    kind = contrast["contrast_type"]
    if kind not in ("position", "random_pair"):
        raise ValueError("head control must be position or random_pair")
    donor, recipient = (variants[contrast[side + "_variant_id"]] for side in ("donor", "recipient"))
    if donor["variant_id"] == recipient["variant_id"]:
        raise ValueError("control requires different variants")
    for key in ("split", "labels", "label_scheme"):
        if donor[key] != recipient[key]:
            raise ValueError(f"head control {key} mismatch")
    if contrast["split"] != recipient["split"]:
        raise ValueError("head control split mismatch")
    for side, variant in (("donor", donor), ("recipient", recipient)):
        item = items[variant["item_id"]]
        for key in ("scene_id", "split", "image", "semantic_answer", "counterfactual_pair_id"):
            if item.get(key) != variant.get(key):
                raise ValueError(f"head control item/variant {key} mismatch")
        labels, options, index = variant["labels"], variant["option_contents"], variant["correct_index"]
        if (len(labels) != 4 or len(set(labels)) != 4 or len(options) != 4 or len(set(options)) != 4
                or type(index) is not int or not 0 <= index < 4
                or options[index] != variant["semantic_answer"] or labels[index] != variant["correct_symbol"]):
            raise ValueError("head control option/label/answer mapping invalid")
        if (contrast[side + "_symbol"] != variant["correct_symbol"]
                or contrast[side + "_content"] != variant["semantic_answer"]):
            raise ValueError("head control contrast readout differs from variant")
    if donor["correct_symbol"] == recipient["correct_symbol"]:
        raise ValueError("head control requires changed correct symbol")
    donor_pair, recipient_pair = donor.get("counterfactual_pair_id"), recipient.get("counterfactual_pair_id")
    if not donor_pair or not recipient_pair or contrast["pair_id"] != recipient_pair:
        raise ValueError("head control pair_id must identify recipient semantic pair")
    if kind == "position":
        if (donor["image"] != recipient["image"] or donor["semantic_answer"] != recipient["semantic_answer"]
                or donor_pair != recipient_pair or donor["question"] != recipient["question"]
                or set(donor["option_contents"]) != set(recipient["option_contents"])):
            raise ValueError("position control must keep image, semantic answer, question and option set")
    elif donor_pair == recipient_pair or donor["image"] == recipient["image"]:
        raise ValueError("random_pair requires unrelated semantic pair and image")
    scene_splits = defaultdict(set)
    for item in items.values():
        scene_splits[item["scene_id"]].add(item["split"])
    if any(len(splits) != 1 for splits in scene_splits.values()):
        raise ValueError("control dataset scene crosses splits")
    return {"role": "control", "contrast_type": kind, "sample_unit": recipient_pair, "split": recipient["split"],
            "sample_unit_definition": "recipient_semantic_pair; donor_reuse_reported_separately",
            "recipient_pair_id": recipient_pair, "donor_pair_id": donor_pair,
            "recipient_scene_id": recipient["scene_id"], "donor_scene_id": donor["scene_id"],
            "donor_variant_id": donor["variant_id"], "recipient_variant_id": recipient["variant_id"]}


def head_control_conditions(recipe: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Explicit subset: no restoration/clamping of images for unrelated inputs."""
    allowed = {"clean", "head_transfer", "self_head", "sham_clamp", "final_residual", "attention_reference"}
    selected = [dict(condition) for condition in recipe["conditions"] if condition["operation"] in allowed]
    if not selected or not any(c["operation"] == "head_transfer" for c in selected):
        raise ValueError("recipe has no head-control transfer conditions")
    return selected


def evaluate_head_control(session: Any, contrast: Mapping[str, Any], recipe: Mapping[str, Any], *,
                          calibration: Mapping[str, Any], identity: Mapping[str, Any], execute: bool = False,
                          cached_baseline: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate one explicitly supplied control, with no sampler or image captures.

    Scientific matching/sample direction choices belong to the frozen campaign
    manifest. Control rows remain separate from semantic-image effect summaries.
    """
    validate_runtime_context(session, recipe, calibration, identity, execute=execute)
    metadata = validate_head_control(contrast, session.variants, session.items)
    if contrast["split"] == "confirmation" and identity.get("confirmation_protocol_frozen") is not True:
        raise ValueError("confirmation controls require prior scientific protocol freeze")
    from ..core7.contrasts import _make_case, run_specs
    from ..core7.hooks import Intervention, Site
    case = _make_case(session, contrast)
    if case is None:
        return {"contrast_id": contrast["contrast_id"], "pair_id": contrast["pair_id"], **metadata,
                "status": "excluded_baseline_eligibility", "identity": dict(identity),
                "recipe_sha256": recipe["recipe_sha256"], "runtime_validation": "not_established"}
    if not case.donor_clean.correct or not case.recipient_clean.correct:
        raise ValueError("head controls require clean correct/correct baselines")
    _finite(case.donor_clean.logits, "control donor baseline")
    _finite(case.recipient_clean.logits, "control recipient baseline")
    donor_prepared = session.prepared(case.donor["variant_id"])
    if session.label_ids(donor_prepared, case.donor["labels"]) != case.label_ids:
        raise ValueError("donor/recipient label token IDs differ")
    specs = []
    for condition in head_control_conditions(recipe):
        operation = condition["operation"]
        selected = recipe["sets"].get(condition.get("head_set"), [])
        if operation == "clean":
            interventions = []
        elif operation == "head_transfer":
            interventions = _heads(case, selected, source="donor")
        elif operation in ("self_head", "sham_clamp"):
            interventions = _heads(case, selected, source="recipient")
        elif operation == "final_residual":
            key = f"resid_post:{condition['layer']}"
            interventions = [Intervention(Site("resid_post", condition["layer"]),
                                          [case.prepared.final_position], case.donor_clean.sites[key])]
        elif operation == "attention_reference":
            interventions = [Intervention(Site("attn", layer), [case.prepared.final_position],
                                          case.donor_clean.sites[f"attn:{layer}"]) for layer in condition["layers"]]
        else:
            raise ValueError("image operation forbidden in head-control evaluator")
        for intervention in interventions:
            _finite(intervention.values, intervention.site.key)
        specs.append(({"method": "stretch-head-control", **condition, **metadata}, interventions))
    checked_session = _FiniteSession(session, case)
    rows = run_specs(checked_session, case, specs)
    denominator = case.donor_diff - case.recipient_diff
    for row, readout in zip(rows, checked_session.adapter.readouts, strict=True):
        row["readout"] = readout
        if abs(denominator) < 1e-6:
            row["normalized_recovery"] = None
        row.update({"raw_margin_change": row["patched_diff"] - case.recipient_diff,
                    "recovery_denominator": denominator,
                    "flip_to_donor": row["predicted_symbol"] == contrast["donor_symbol"],
                    "baseline_source": "current_session_clean_replay", "image_values_source": "not_used",
                    "recipe_sha256": recipe["recipe_sha256"]})
    return {"contrast_id": contrast["contrast_id"], "pair_id": contrast["pair_id"], **metadata,
            "status": "evaluated", "rows": rows, "identity": dict(identity),
            "recipe_sha256": recipe["recipe_sha256"], "cached_baseline_supplied": cached_baseline is not None,
            "cached_baseline_use": "metadata_only; never used as activations",
            "cached_baseline_metadata": dict(cached_baseline or {}),
            "runtime_validation": "requires_separate_head_control_calibration"}
