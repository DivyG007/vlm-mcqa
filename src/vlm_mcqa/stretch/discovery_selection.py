"""Deterministic discovery-only layer selection for the frozen Qwen studies."""
from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any, Mapping, Sequence


def _interval(values: Sequence[float], *, seed: int, draws: int) -> dict[str, Any]:
    if (not values or type(seed) is not int or type(draws) is not int or draws < 100
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError("finite nonempty unit effects and >=100 bootstrap draws required")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "n_units": 1, "ci95": None}
    rng = random.Random(seed)
    boot = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws))
    return {"mean": mean, "n_units": len(values),
            "ci95": [boot[int(.025 * (draws - 1))], boot[int(.975 * (draws - 1))]]}


def _choose(estimates: Mapping[int, Mapping[str, Any]], *, tie_tolerance: float,
            minimum_units: int) -> dict[str, Any]:
    if (type(tie_tolerance) not in (int, float) or not math.isfinite(tie_tolerance)
            or tie_tolerance < 0 or type(minimum_units) is not int or minimum_units < 1):
        raise ValueError("valid tie tolerance and minimum units required")
    eligible = {layer: row for layer, row in estimates.items()
                if row.get("gates_passed") is True and row.get("n_units", 0) >= minimum_units
                and row.get("ci95") is not None}
    if not eligible:
        return {"status": "insufficient", "selected_layer": None,
                "reason": "no_candidate_passed_prespecified_gates", "estimates": dict(estimates)}
    best_lower = max(row["ci95"][0] for row in eligible.values())
    tied = [layer for layer, row in eligible.items() if best_lower - row["ci95"][0] <= tie_tolerance]
    chosen = min(tied)
    return {"status": "ready", "selected_layer": chosen, "tie_tolerance": tie_tolerance,
            "selection_rule": "largest_bootstrap_lower_bound_then_earlier_layer_within_tolerance",
            "best_lower_bound": best_lower, "tied_layers": sorted(tied), "estimates": dict(estimates)}


def select_failure_option_layer(rows: Sequence[Mapping[str, Any]], *, candidate_layers: Sequence[int],
                                seed: int, draws: int, minimum_units: int,
                                tie_tolerance: float, identity_tolerance: float) -> dict[str, Any]:
    """Choose E22 option repair using matched failure/control scene effects."""
    if (not candidate_layers or len(set(candidate_layers)) != len(candidate_layers)
            or any(type(layer) is not int or layer < 0 for layer in candidate_layers)):
        raise ValueError("unique nonnegative E22 candidate layers required")
    estimates = {}
    for layer in candidate_layers:
        operation = f"options_candidate_L{layer}"
        self_operation = f"options_candidate_self_L{layer}"
        failures = defaultdict(list)
        controls = defaultdict(list)
        remap, copied, validity, identity = [], [], [], []
        for row in rows:
            if row.get("condition") == self_operation:
                identity.append(abs(row.get("raw_margin_change", float("inf"))))
            if row.get("condition") != operation:
                continue
            key = row.get("case_id") if row.get("role") == "failure" else row.get("matched_failure_case_id")
            # Donor matching expands one prospective failure case into one
            # attempt per donor class, appending ``:<donor_class>``.  Matched
            # controls retain the prospective base ID.  Pair on that base ID
            # so discovery estimates do not silently collapse to zero units.
            donor_class = row.get("donor_class")
            if row.get("role") == "failure" and donor_class and key.endswith(f":{donor_class}"):
                key = key[:-(len(donor_class) + 1)]
            if not key:
                continue
            target = failures if row.get("role") == "failure" else controls
            target[key].append(float(row["raw_margin_change"]))
            if row.get("role") == "failure":
                remap.append(float(bool(row.get("semantic_remapping_success"))))
                copied.append(float(bool(row.get("donor_symbol_copy"))))
                validity.append(row.get("readout", {}).get("global_top_is_label") is True)
        keys = sorted(set(failures) & set(controls))
        values = []
        for key in keys:
            failure_gain = sum(failures[key]) / len(failures[key])
            correct_change = sum(controls[key]) / len(controls[key])
            values.append(failure_gain - max(0.0, -correct_change))
        estimate = _interval(values, seed=seed + layer, draws=draws) if values else {
            "mean": None, "n_units": 0, "ci95": None}
        gates = {"identity": bool(identity) and max(identity) <= identity_tolerance,
                 "label_validity": bool(validity) and all(validity),
                 "content_over_literal_copy": bool(remap) and sum(remap) / len(remap) > sum(copied) / len(copied)}
        estimates[layer] = {**estimate, "gates": gates, "gates_passed": all(gates.values()),
                            "estimand": "failure_truth_margin_gain_minus_positive_matched_correct_harm"}
    return _choose(estimates, tie_tolerance=tie_tolerance, minimum_units=minimum_units)


def select_task_image_layer(study: str, rows: Sequence[Mapping[str, Any]], *,
                            candidate_layers: Sequence[int], seed: int, draws: int,
                            minimum_units: int, tie_tolerance: float,
                            identity_tolerance: float) -> dict[str, Any]:
    """Choose E23 region or E24 panel layer from parent-clustered effects."""
    if study not in {"spatial", "image_options"}:
        raise ValueError("task layer selection supports spatial or image_options")
    estimates = {}
    for layer in candidate_layers:
        by_parent = defaultdict(list)
        identity = []
        if study == "spatial":
            targets = defaultdict(list)
            controls = defaultdict(list)
            for row in rows:
                operation = row.get("operation", "")
                if operation == f"self-region-L{layer}" and "margin_change" in row:
                    identity.append(abs(float(row["margin_change"])))
                if operation != f"zero-region-L{layer}" or "margin_change" not in row:
                    continue
                collection = controls if str(row.get("region", "")).endswith("-matched") else targets
                collection[(row["pair_id"], str(row.get("region", "")).removesuffix("-matched"))].append(float(row["margin_change"]))
            for key in sorted(set(targets) & set(controls)):
                # Margin harm is negative: control change minus target change is positive specificity.
                by_parent[key[0]].append(sum(controls[key]) / len(controls[key]) -
                                         sum(targets[key]) / len(targets[key]))
        else:
            for row in rows:
                operation = row.get("operation", "")
                if operation == f"self-region-L{layer}" and "margin_change" in row:
                    identity.append(abs(float(row["margin_change"])))
                if (operation == f"donor-panel-L{layer}" and row.get("correct_correct_transfer") is True
                        and row.get("content_vs_copy_identifiable") is True):
                    by_parent[row["pair_id"]].append(float(row["content_vs_literal_logit_change"]))
        values = [sum(items) / len(items) for _, items in sorted(by_parent.items())]
        estimate = _interval(values, seed=seed + layer, draws=draws) if values else {
            "mean": None, "n_units": 0, "ci95": None}
        gates = {"identity": bool(identity) and max(identity) <= identity_tolerance}
        estimates[layer] = {**estimate, "gates": gates, "gates_passed": all(gates.values()),
                            "estimand": "region_specificity" if study == "spatial" else "content_minus_literal_copy"}
    return _choose(estimates, tie_tolerance=tie_tolerance, minimum_units=minimum_units)
