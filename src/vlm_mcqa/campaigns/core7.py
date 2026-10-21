"""Preselected Pali route delegating unchanged Core-7 CLI implementations.

Uses the same method, freeze, confirm and plot functions as core7.cli, with an
exclusive campaign wrapper and verified local-only Session factory.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .config import canonical_hash
from .verification import read_object


class InsufficientSample(RuntimeError):
    pass


def method_eligibility(session: Any, split: str) -> dict:
    """Measure actual method-specific eligibility; fail before causal sweeps."""
    from ..core7.contrasts import select_cases
    from ..core7.components import ablation_pool
    from ..core7.patching import RESIDUAL_TYPES
    from ..core7.components import COMPONENT_TYPES

    profile = session.profile
    target = int(profile["pairs"])
    checks = {}
    methods = (("patching", RESIDUAL_TYPES, profile.get("contrast_limit_per_type")),
               ("components", COMPONENT_TYPES, profile.get("contrast_limit_per_type")),
               ("heads", ("position",), profile["head_screen_contrasts"]))
    for method, kinds, limit in methods:
        cases, info = select_cases(session, split, kinds, max_pairs=target, limit_per_type=limit)
        counts = {kind: len({c.contrast["pair_id"] for c in cases if c.kind == kind}) for kind in kinds}
        # Main methods require 64 distinct pairs per primary contrast; identity
        # and unrelated controls are audited separately, not promoted to primary.
        primary = [kind for kind in kinds if kind in {"position", "content", "content_and_position"}]
        passed = all(counts[kind] == target for kind in primary)
        if method == "heads":
            passed = passed and len(cases) == int(profile["head_screen_contrasts"])
        group_mismatches = 0
        if method == "patching":
            for case in cases:
                if case.kind in {"content", "content_and_position"}:
                    donor_groups = session.prepared(case.donor["variant_id"]).groups
                    for group in profile["token_groups"]:
                        if not donor_groups.get(group) or len(donor_groups[group]) != len(case.prepared.groups.get(group, [])):
                            group_mismatches += 1
            passed = passed and group_mismatches == 0
        pool = ablation_pool(session, split, info["pair_ids"])
        recipients = [v for v in sorted(pool, key=lambda v: v["variant_id"]) if session.clean(v["variant_id"]).correct]
        requested = int(profile["ablation_recipients"])
        if method in {"components", "heads"}:
            passed = passed and len(recipients) >= requested
        checks[method] = {"passed": passed, "target_pairs": target, "pairs_by_type": counts,
                          "directed_cases": len(cases), "selection": info,
                          "available_ablation_recipients": len(recipients),
                          "group_size_mismatches": group_mismatches,
                          "requested_ablation_recipients": requested}
    return {"schema": "core7-method-eligibility/v1", "split": split, "checks": checks,
            "all_passed": all(row["passed"] for row in checks.values()),
            "threshold_policy": profile.get("claim_scope", "unchanged_full_profile; no automatic sample reduction"),
            "denominator_policy": profile.get("denominator_policy", "full_profile_method_eligible_cases")}


def run_phase(session: Any, config: Mapping[str, Any], proof: Mapping[str, Any], *, split: str,
              freeze_path: Path | None = None) -> dict:
    from ..core7.aggregate import aggregate_run
    from ..core7.behavior import run_behavior
    from ..core7.calibration import run_calibration
    from ..core7.components import run_components
    from ..core7.heads import run_heads
    from ..core7.lens import run_lens
    from ..core7.patching import run_patching
    from ..core7.plots import plot_run

    frozen = read_object(freeze_path) if freeze_path else None
    if split == "confirmation":
        if frozen is None:
            raise ValueError("confirmation requires discovery freeze")
        for key, expected in (("model_key", config["model"]["key"]),
                              ("dataset_hash", proof["dataset_hash"]),
                              ("image_aware_release_hash", proof["image_aware_release_hash"]),
                              ("source_git_commit", proof["source"]["source_git_commit"])):
            if frozen.get(key) != expected:
                raise ValueError(f"frozen {key} mismatch")
    methods = ["behavior", "lens", "patching", "components", "heads"] if split == "discovery" else ["patching", "components", "heads"]
    run = {"schema": "core7-campaign-phase/v1", "model_key": config["model"]["key"],
           "dataset_hash": proof["dataset_hash"], "image_aware_release_hash": proof["image_aware_release_hash"],
           "source_git_commit": proof["source"]["source_git_commit"], "source_tree_dirty": None,
           "source_state": "manifested_local_snapshot; git cleanliness not asserted",
           "source_snapshot_sha256": proof["source"]["manifest_sha256"],
           "model_spec": proof["model_spec"], "checkpoint_manifest_sha256": proof["checkpoint"]["manifest_sha256"],
           "split": split, "methods": methods, "status": "running", "profile_name": config["profile"],
           "profile": dict(session.profile), "num_layers": session.adapter.num_layers,
           "num_heads": session.adapter.handle.num_heads,
           "selection_mode": config["protocol"]["selection_mode"], "screen_eligible": False,
           "protocol_decision_sha256": proof["protocol_decision_sha256"],
           "freeze_hash": frozen.get("freeze_hash") if frozen else None}
    session.write_json("run.json", run)
    try:
        calibration = run_calibration(session)
        if calibration.get("all_passed") is not True:
            raise RuntimeError("ordinary Core-7 calibration failed; no bypass permitted")
        # A restricted first-token content readout must distinguish all options;
        # do not count cases skipped for collapsed content tokens as full coverage.
        eligibility = method_eligibility(session, split)
        session.write_json("method_eligibility.json", eligibility)
        if not eligibility["all_passed"]:
            raise InsufficientSample(f"{split}: full-profile method eligibility insufficient")
        p = session.profile
        behavior_items = session.items_in("behavior", p["behavior_items"])
        if split == "discovery" and len(behavior_items) != p["behavior_items"]:
            raise InsufficientSample("behavior item target insufficient")
        if split == "discovery":
            n_variants = sum(len(session.variants_of(item["item_id"], schemes=p["schemes"])) for item in behavior_items)
            behavior_clean = n_variants
            n_letter = sum(len(session.variants_of(item["item_id"], schemes=["letters"])) for item in behavior_items)
            pair_rows = sum(len(session.variants_of(item["item_id"], schemes=["letters"], kind=kind))
                            for item in session.items.values() if item["split"] == split
                            for kind in ("position", "distractor_permuted"))
            run["expected_observational_rows"] = {
                "behavior_behavior.jsonl": behavior_clean + 2 * n_letter + pair_rows,
                "compliance_behavior.jsonl": min(p["compliance_items"], len(behavior_items)),
                "lens_behavior.jsonl": n_variants * (session.adapter.num_layers + 1),
                "component_projection_behavior.jsonl": n_variants * session.adapter.num_layers * 2}
        outputs = {}
        for method in methods:
            if method == "behavior":
                outputs[method] = run_behavior(session, split="behavior", n_items=p["behavior_items"],
                                              robust_items=p["behavior_items"],
                                              compliance_items=min(p["compliance_items"], p["behavior_items"]),
                                              controls=True, pair_splits=(split,))
            elif method == "lens":
                outputs[method] = run_lens(session, split="behavior", n_items=p["behavior_items"])
            elif method == "patching":
                outputs[method] = run_patching(session, split=split)
            elif method == "components":
                outputs[method] = run_components(session, split=split)
            elif method == "heads":
                outputs[method] = run_heads(session, split=split, freeze=frozen)
        aggregate = aggregate_run(session.out_dir, split)
        figures = plot_run(session.out_dir)
        # Historical method summaries can contain undefined ratios (NaN), e.g.
        # zero all-head effect. Hash their native JSON representation without
        # turning an undefined diagnostic into an execution failure or a zero.
        run.update(status="complete", figures=figures, method_result_hashes={
            k: hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest() for k, v in outputs.items()})
        session.write_json("run.json", run)
        return {"run": run, "aggregate": aggregate, "method_results": outputs}
    except BaseException as error:
        run.update(status="insufficient" if isinstance(error, InsufficientSample) else "failed",
                   error_type=type(error).__name__, error=str(error))
        session.write_json("run.json", run)
        raise


def run_pali(config: Mapping[str, Any], proof: Mapping[str, Any], output: Path, *, execute: bool) -> dict:
    from .session import create_session
    from ..core7.session import Session
    from ..core7.aggregate import freeze, confirm
    from .verification import verify_inputs

    profiles = read_object(config["profiles_config"])
    profile_name = config["profile"]
    profile = dict(profiles["profiles"][profile_name])
    baseline_profiles = read_object(Path(__file__).resolve().parents[3] / "configs/core7/profiles.json")
    expected_profile = baseline_profiles["profiles"][profile_name]
    if ({k: v for k, v in profile.items() if k != "batch_size"} !=
            {k: v for k, v in expected_profile.items() if k != "batch_size"}):
        raise ValueError("Pali profile differs from its preserved scientific configuration; batch size alone is deployment")
    if profile_name == "full_core7" and (profile.get("pairs"), profile.get("head_screen_contrasts"), profile.get("random_sets")) != (64, 96, 20):
        raise ValueError("full-profile scientific sampling differs from frozen baseline")
    if profile_name == "pali_correct_only_reduced" and (
            profile.get("pairs"), profile.get("confirmation_pairs"), profile.get("random_sets")) != (32, 24, 20):
        raise ValueError("correct-only Pali sensitivity scope differs from frozen route")
    if profiles["selection"] != baseline_profiles["selection"]:
        raise ValueError("selection thresholds differ from preserved policy")
    reuse = None
    if config.get("retry_from"):
        from .retry import plan_retry, copy_discovery
        reuse = plan_retry(config["retry_from"], config)
        with (output / "retry_link.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(reuse, sort_keys=True, indent=2) + "\n")
        copy_discovery(reuse, output / "discovery")
    reused = bool(reuse and reuse["reusable"])
    discovery, _ = create_session(config, output / ("retry_session" if reused else "discovery"),
                                   execute=execute, profile=profile, verified=proof)
    if reused:
        discovery.out_dir = output / "discovery"
        first = {"run": read_object(discovery.out_dir / "run.json")}
    else:
        first = run_phase(discovery, config, proof, split="discovery")
    freeze_path = output / "freeze.json"
    if freeze_path.exists():
        raise FileExistsError(freeze_path)
    frozen = freeze(discovery.out_dir, freeze_path)
    if verify_inputs(config) != proof:
        raise ValueError("inputs changed between discovery and confirmation")
    # Reuse one loaded adapter; independent Session caches/RNG and output phase.
    confirmation_profile = dict(discovery.profile)
    if profile_name == "pali_correct_only_reduced":
        confirmation_profile["pairs"] = int(profile["confirmation_pairs"])
        confirmation_profile["head_screen_contrasts"] = int(profile["confirmation_pairs"])
        confirmation_profile["ablation_recipients"] = int(profile["confirmation_pairs"])
    confirmation = Session(discovery.adapter, discovery.data_dir, output / "confirmation", confirmation_profile,
                           seed=config["seed"])
    second = run_phase(confirmation, config, proof, split="confirmation", freeze_path=freeze_path)
    report = confirm(confirmation.out_dir, freeze_path)
    verdict = "replicated" if report["checks"]["all_hypotheses_replicated"] else "non_replication"
    if profile_name == "pali_correct_only_reduced":
        verdict = "reduced_correct_only_sensitivity_" + verdict
    return {"schema": "pali-core7-campaign/v1", "status": "complete", "freeze": frozen,
            "discovery": first["run"], "confirmation": second["run"], "confirmation_report": report,
            "scientific_verdict": verdict}
