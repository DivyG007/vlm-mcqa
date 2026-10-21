"""Prospective E20 scalar analysis; standard library only, no import-time work.

Source implementation is unexecuted. Bootstrap uncertainty is across semantic
pairs, after averaging directions within each pair. It cannot validate hooks or
establish composed-intervention calibration.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import random
from typing import Any, Iterable, Mapping

from . import CONTRAST_TYPE


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return _number(math.fsum(value / len(values) for value in values), "computed mean")


def _quantile(ordered: list[float], q: float) -> float:
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    weight = index - low
    return _number((1 - weight) * ordered[low] + weight * ordered[high], "computed quantile")


def _intervals(pairs: list[dict[str, Any]], metrics: list[str], *, seed: int, samples: int) -> dict:
    """Use the same pair-resampling indices across all reported outcomes."""
    output = {}
    for metric in metrics:
        output[metric] = {"mean": _mean(p[metric] for p in pairs) if pairs else None,
                          "ci95": None, "n_pairs": len(pairs), "sampling_unit": "semantic_pair"}
    if len(pairs) < 2:
        return output
    draws = {metric: [] for metric in metrics}
    rng = random.Random(seed)
    for _ in range(samples):
        selected = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        for metric in metrics:
            draws[metric].append(_mean(p[metric] for p in selected))
    for metric in metrics:
        ordered = sorted(draws[metric])
        output[metric]["ci95"] = [_quantile(ordered, 0.025), _quantile(ordered, 0.975)]
    return output


def _recipe_conditions(recipe: Mapping[str, Any]) -> tuple[dict[str, dict], list[str]]:
    canonical = {k: v for k, v in recipe.items() if k != "recipe_sha256"}
    sha = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode()).hexdigest()
    if recipe.get("recipe_sha256") != sha:
        raise ValueError("recipe hash mismatch")
    for key in ("model_key", "dataset_hash", "freeze_hash", "head_source_sha256"):
        if not isinstance(recipe.get(key), str) or not recipe[key]:
            raise ValueError(f"recipe {key} identity missing")
    if recipe.get("random_sets") != 20:
        raise ValueError("primary E20 requires all twenty prespecified random4 sets")
    sets = recipe.get("sets", {})
    random_names = [f"random4_{i:02d}" for i in range(20)]
    if {name for name in sets if name.startswith("random4_")} != set(random_names):
        raise ValueError("random4 set identities/count mismatch")
    for name in ["top4", *random_names]:
        heads = sets.get(name)
        if (not isinstance(heads, list) or len(heads) != 4
                or any(not isinstance(h, (list, tuple)) or len(h) != 2
                       or any(type(v) is not int or v < 0 for v in h) for h in heads)
                or len({tuple(h) for h in heads}) != 4):
            raise ValueError(f"invalid four-head set: {name}")
    target = sets["top4"]
    for name in random_names:
        if Counter(h[0] for h in sets[name]) != Counter(h[0] for h in target):
            raise ValueError(f"layer matching failed: {name}")
        if {tuple(h) for h in sets[name]} & {tuple(h) for h in target}:
            raise ValueError(f"random4 overlaps top4: {name}")
    conditions = {}
    for condition in recipe.get("conditions", []):
        name = condition.get("name")
        if not isinstance(name, str) or not name or name in conditions:
            raise ValueError("invalid or duplicate recipe condition")
        conditions[name] = condition
    required = {"clean", "image_restore"}
    for head in ["top4", *random_names]:
        for operation in ("head_transfer", "restore_clamp"):
            name = f"{operation}/{head}"
            required.add(name)
            if conditions.get(name, {}).get("operation") != operation or conditions[name].get("head_set") != head:
                raise ValueError(f"missing or mismatched primary condition: {name}")
    if not required.issubset(conditions) or conditions["clean"].get("operation") != "clean" or conditions["image_restore"].get("operation") != "image_restore":
        raise ValueError("clean and shared image_restore references required")
    if recipe.get("condition_count") != len(conditions):
        raise ValueError("recipe condition count mismatch")
    return conditions, random_names


def validate_effect_recipe(recipe: Mapping[str, Any]) -> None:
    """Fail before output creation/model calls if primary analysis is invalid."""
    _recipe_conditions(recipe)


def summarize_semantic_effects(results: Iterable[dict[str, Any]], recipe: Mapping[str, Any], *,
                               expected_directions: list[dict[str, str]], expected_split: str, bootstrap_seed: int,
                               bootstrap_samples: int, denominator_min: float = 1.0) -> dict[str, Any]:
    """Summarize only prespecified semantic-image results, without altering recipe.

    Expected directions come from the frozen sample plan plus immutable contrast
    manifest. Missing/ineligible/partial directions remain in the coverage audit;
    pairs with any such direction do not enter the complete-pair descriptive
    summary. Any incomplete required scope disables the positive criterion.
    """
    if type(bootstrap_seed) is not int or type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError("integer bootstrap seed and at least 100 samples required")
    if expected_split not in ("discovery", "confirmation"):
        raise ValueError("explicit discovery or confirmation expected_split required")
    denominator_min = _number(denominator_min, "denominator_min")
    if denominator_min <= 0:
        raise ValueError("denominator_min must be positive")
    conditions, random_names = _recipe_conditions(recipe)
    expected = {}
    for entry in expected_directions:
        cid, pair = entry.get("contrast_id"), entry.get("pair_id")
        if not isinstance(cid, str) or not cid or not isinstance(pair, str) or not pair or cid in expected:
            raise ValueError("expected directions require unique nonempty contrast_id and pair_id")
        if entry.get("split", expected_split) != expected_split:
            raise ValueError("expected direction split mismatch")
        expected[cid] = pair
    if not expected:
        raise ValueError("explicit prospective expected_directions required")
    supplied = {}
    for result in results:
        if result.get("role") != "semantic" or result.get("contrast_type") != CONTRAST_TYPE:
            raise ValueError("semantic cohort requires explicit semantic role/type; summarize controls separately")
        if result.get("split") != expected_split:
            raise ValueError("result split differs from prospective analysis scope")
        cid = result.get("contrast_id")
        if cid not in expected or cid in supplied:
            raise ValueError("unexpected or duplicate result direction")
        if result.get("pair_id") != expected[cid]:
            raise ValueError("result pair differs from expected direction")
        identity = result.get("identity", {})
        recorded_hash = identity.get("recipe_sha256", result.get("recipe_sha256"))
        if recorded_hash != recipe["recipe_sha256"]:
            raise ValueError("result recipe identity missing or mismatched")
        if any(identity.get(key) != recipe[key] for key in ("model_key", "dataset_hash", "freeze_hash", "head_source_sha256")):
            raise ValueError("result model/data/freeze/head-source identity mismatch")
        supplied[cid] = result
    direction_audit = []
    effects = {}
    primary_metrics = ["head_transfer_minus_meanrandom", "clamp_attenuation_minus_meanrandom"]
    reference_metrics = ["top4_raw_transfer", "image_restoration_raw_effect", "top4_clamp_attenuation"]
    metrics = [*primary_metrics, *reference_metrics]
    for cid, pair in expected.items():
        result = supplied.get(cid)
        if result is None:
            direction_audit.append({"contrast_id": cid, "pair_id": pair, "status": "missing_result"})
            continue
        if result.get("status") != "evaluated":
            if result.get("rows"):
                raise ValueError("unevaluated result must not contain effect rows")
            direction_audit.append({"contrast_id": cid, "pair_id": pair,
                                    "status": result.get("status", "missing_status")})
            continue
        rows = {}
        for row in result.get("rows", []):
            name = row.get("name")
            if name not in conditions or name in rows:
                raise ValueError("unexpected or duplicate result condition")
            if row.get("recipe_sha256") != recipe["recipe_sha256"]:
                raise ValueError("row recipe identity missing or mismatched")
            for key, value in conditions[name].items():
                if row.get(key) != value:
                    raise ValueError(f"condition metadata mismatch: {name}/{key}")
            if row.get("contrast_id", cid) != cid or row.get("pair_id", pair) != pair:
                raise ValueError("condition belongs to different direction/pair")
            if (row.get("role", "semantic") != "semantic" or row.get("contrast_type", CONTRAST_TYPE) != CONTRAST_TYPE
                    or row.get("split", expected_split) != expected_split):
                raise ValueError("row semantic cohort/split mismatch")
            patched = _number(row.get("patched_diff"), f"{cid}/{name}/patched_diff")
            denominator = _number(row.get("recovery_denominator"), f"{cid}/{name}/denominator")
            normalized = row.get("normalized_recovery")
            if "normalized_recovery" not in row:
                raise ValueError("normalized_recovery field missing")
            if normalized is not None:
                _number(normalized, f"{cid}/{name}/normalized_recovery")
            elif abs(denominator) >= 1e-6:
                raise ValueError("normalized_recovery missing despite nonzero denominator")
            rows[name] = {"patched": patched, "denominator": denominator, "normalized": normalized}
        missing = sorted(set(conditions) - rows.keys())
        if missing:
            direction_audit.append({"contrast_id": cid, "pair_id": pair,
                                    "status": "incomplete_conditions", "missing_conditions": missing})
            continue
        denominators = [row["denominator"] for row in rows.values()]
        denominator = denominators[0]
        if any(not math.isclose(v, denominator, rel_tol=1e-10, abs_tol=1e-10) for v in denominators):
            raise ValueError("conditions do not share one donor/recipient denominator")
        clean = rows["clean"]["patched"]
        restore = rows["image_restore"]["patched"]
        top = rows["head_transfer/top4"]["patched"]
        random_transfer = _mean(rows[f"head_transfer/{name}"]["patched"] for name in random_names)
        top_attenuation = restore - rows["restore_clamp/top4"]["patched"]
        random_attenuation = _mean(restore - rows[f"restore_clamp/{name}"]["patched"] for name in random_names)
        effect = {"contrast_id": cid, "pair_id": pair,
                  "head_transfer_minus_meanrandom": top - random_transfer,
                  "clamp_attenuation_minus_meanrandom": top_attenuation - random_attenuation,
                  "top4_raw_transfer": top - clean,
                  "image_restoration_raw_effect": restore - clean,
                  "top4_clamp_attenuation": top_attenuation,
                  "recovery_denominator": denominator,
                  "stable_denominator": abs(denominator) >= denominator_min}
        for metric in metrics:
            _number(effect[metric], f"computed {cid}/{metric}")
        # Ratios are secondary sensitivity values, never raw-cohort eligibility.
        effect["normalized_sensitivity"] = {metric: _number(effect[metric] / denominator, f"computed normalized {metric}") for metric in metrics} if effect["stable_denominator"] else None
        effects[cid] = effect
        direction_audit.append({"contrast_id": cid, "pair_id": pair, "status": "complete"})
    by_pair = defaultdict(list)
    for cid, pair in expected.items():
        by_pair[pair].append(cid)
    complete_pairs, pair_audit, sensitivity_pairs = [], [], []
    for pair, ids in sorted(by_pair.items()):
        missing = [cid for cid in ids if cid not in effects]
        pair_audit.append({"pair_id": pair, "expected_directions": len(ids),
                           "complete_directions": len(ids) - len(missing), "incomplete_direction_ids": missing})
        if missing:
            continue
        averaged = {"pair_id": pair, "n_directions": len(ids),
                    **{metric: _mean(effects[cid][metric] for cid in ids) for metric in metrics}}
        complete_pairs.append(averaged)
        stable = [cid for cid in ids if effects[cid]["stable_denominator"]]
        # A pair enters sensitivity only when all its planned directions are stable.
        if len(stable) == len(ids):
            sensitivity_pairs.append({"pair_id": pair, "n_directions": len(ids),
                                      **{metric: _mean(effects[cid]["normalized_sensitivity"][metric] for cid in ids) for metric in metrics}})
    intervals = _intervals(complete_pairs, metrics, seed=bootstrap_seed, samples=bootstrap_samples)
    scope_complete = len(effects) == len(expected)
    enough_pairs = len(complete_pairs) >= 2
    positive = None
    reference_positive = None
    if scope_complete and enough_pairs:
        positive = all(intervals[m]["ci95"][0] > 0 for m in primary_metrics)
        reference_positive = all(intervals[m]["ci95"][0] > 0 for m in reference_metrics)
    return {"schema_version": "stretch.effects.v1", "recipe_sha256": recipe["recipe_sha256"], "split": expected_split,
            "status": "complete_scalar_summary" if scope_complete else "incomplete_required_scope",
            "coverage": {"expected_directions": len(expected), "supplied_directions": len(supplied),
                         "complete_directions": len(effects), "expected_pairs": len(by_pair),
                         "complete_pairs": len(complete_pairs), "direction_audit": direction_audit,
                         "pair_audit": pair_audit},
            "direction_effects": list(effects.values()), "pair_effects": complete_pairs,
            "pair_bootstrap": {"seed": bootstrap_seed, "samples": bootstrap_samples,
                               "method": "percentile_bootstrap_of_equal_weight_pair_means",
                               "intervals": intervals,
                               "partial_scope_interpretation": "exploratory_complete_pairs_only" if not scope_complete else None},
            "positive_criterion": {"both_primary_ci95_lower_bounds_above_zero": positive,
                                   "all_reference_ci95_lower_bounds_above_zero": reference_positive,
                                   "conjunctive_primary_and_reference_pattern": positive and reference_positive if positive is not None else None,
                                   "disabled_reason": "incomplete_scope" if not scope_complete else "fewer_than_two_pairs" if not enough_pairs else None,
                                   "scientific_confirmation": "not_established_by_scalar_summary"},
            "normalized_recovery_sensitivity": {"secondary_only": True, "denominator_min_abs": denominator_min,
                                                "n_pairs_all_directions_stable": len(sensitivity_pairs),
                                                "excluded_from_sensitivity_only": [p["pair_id"] for p in complete_pairs if p["pair_id"] not in {s["pair_id"] for s in sensitivity_pairs}],
                                                "pair_bootstrap": _intervals(sensitivity_pairs, metrics, seed=bootstrap_seed, samples=bootstrap_samples)},
            "limitations": ["Raw primary effects retain every complete pair regardless of denominator magnitude.",
                            "Random controls are averaged within direction; directions within pair; pairs have equal weight.",
                            "Clamps share the same image_restore reference; clean and restored baselines are not interchangeable.",
                            "Percentile intervals are marginal; conjunctive positivity is not a joint confidence region.",
                            "No random-p95 significance test or independent random-set sample claim is made.",
                            "Composed-intervention calibration, identity/sham controls and prospective scope audit remain separate gates.",
                            "Same-image position/unrelated-donor controls belong to separate descriptive summaries."]}


def descriptive_control_summary(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Report head controls separately; no pooling with semantic-image effects."""
    cohorts = defaultdict(list)
    seen = set()
    coverage = []
    for result in results:
        kind = result.get("contrast_type")
        if result.get("role") != "control" or kind not in ("position", "random_pair"):
            raise ValueError("only explicitly labelled position/random_pair controls accepted")
        cid = result.get("contrast_id")
        if not cid or cid in seen:
            raise ValueError("missing or duplicate control direction")
        seen.add(cid)
        coverage.append({"contrast_id": cid, "status": result.get("status")})
        if result.get("status") != "evaluated":
            continue
        names = set()
        for row in result.get("rows", []):
            name = row.get("name")
            if not name or name in names:
                raise ValueError("missing or duplicate control condition")
            names.add(name)
            pair = row.get("sample_unit", result.get("pair_id"))
            if not pair:
                raise ValueError("control recipient-pair sample unit missing")
            cohorts[(kind, name)].append((str(pair), _number(row.get("raw_margin_change"), "control margin change"),
                                         row.get("donor_variant_id")))
    summaries = []
    for (kind, name), values in sorted(cohorts.items()):
        paired = defaultdict(list)
        for pair, value, _ in values:
            paired[pair].append(value)
        summaries.append({"contrast_type": kind, "condition": name,
                          "mean_raw_margin_change": _mean(_mean(v) for v in paired.values()),
                          "n_recipient_pairs": len(paired), "n_direction_rows": len(values),
                          "n_unique_donor_variants": len({d for _, _, d in values if d is not None}),
                          "rows_missing_donor_identity": sum(d is None for _, _, d in values)})
    return {"schema_version": "stretch.controls.descriptive.v1", "cohorts": summaries,
            "coverage": coverage, "prospective_scope_completeness": "not_established_from_supplied_results",
            "missing_control_types": sorted({"position", "random_pair"} - {kind for kind, _ in cohorts}),
            "donor_reuse_dependence": "not_adjusted; descriptive recipient-pair means only",
            "inference": "descriptive_only; no semantic-cohort pooling or significance claim"}
