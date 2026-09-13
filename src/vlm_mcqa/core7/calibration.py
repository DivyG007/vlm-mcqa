"""Calibration gates that must pass before any scientific run.

Gates (all on real dataset variants):

- ``cached_matches_full``      decoder replay reproduces the full VLM forward;
- ``capture_is_inert``         capture-only hooks leave logits unchanged;
- ``identity_patch``           patching every site with its own value is a no-op;
- ``batched_matches_single``   batch-expanded replay matches batch-one replay;
- ``replay_deterministic``     two identical replays agree;
- ``residual_decomposition``   resid_pre + attn + mlp == resid_post (final token);
- ``lens_final_matches``       lens(final residual) == model logits (selects softcap);
- ``symbols_single_token``     every alphabet maps to four distinct single tokens;
- ``token_spans``              image tokens and the final position are resolved.

Informational: content first-token distinctness, question/option/label span
resolution rates, and additivity of direct component projections.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..clevr_mcq4.variants import ALPHABETS
from .hooks import Intervention, Site
from .session import Session

TOLERANCE = {"float32": 2e-3, "bfloat16": 0.25, "float16": 0.06}


UNIT_ROUNDOFF = {"bfloat16": 2.0 ** -8, "float16": 2.0 ** -11}


def _logit_error(a: torch.Tensor, b: torch.Tensor) -> float:
    delta = (a.float().cpu() - b.float().cpu()).abs()
    if not torch.isfinite(delta).all():
        return float("inf")
    return float(delta.max())


def _kernel_tolerance(tol: float, dtype: str, reference: torch.Tensor) -> float:
    """Limit for comparisons across different kernels (batched, cached, lens).

    Half-precision rounding accumulates through the decoder, so the difference
    scales with the logit magnitude. Allow eight units of relative roundoff at
    the largest logit (3.1% in bfloat16). Measured batched-vs-single maxima in
    bfloat16: Qwen2.5-VL-3B 0.375 at peak 35 (1.1%), InternVL3.5-2B 1.0 at peak
    54.75 (1.8%); in float32 the same checks agree to 3e-5 and 1.3e-4, so these
    are precision effects, and the answer-label argmax never changed.
    """
    if dtype not in UNIT_ROUNDOFF:
        return tol
    peak = float(reference.float().abs().max())
    if not math.isfinite(peak):
        return tol
    return max(tol, 8 * UNIT_ROUNDOFF[dtype] * peak)


def run_calibration(session: Session, *, num_variants: int = 4) -> dict[str, Any]:
    adapter, handle = session.adapter, session.adapter.handle
    tol = TOLERANCE[adapter.spec.dtype]
    split_order = ("screening", "discovery", "behavior", "iconqa", "calibration", "confirmation")
    chosen: list[dict[str, Any]] = []
    for split in split_order:
        for item in session.items_in(split, limit=num_variants):
            for scheme in ("letters", *[s for s in ALPHABETS if s != "letters"]):
                vs = session.variants_of(item["item_id"], schemes=[scheme])
                if vs:
                    chosen.append(vs[len(chosen) % len(vs)])
        if chosen:
            break
    letter_variants = [v for v in chosen if v["label_scheme"] == "letters"][:num_variants]
    if not letter_variants:
        raise RuntimeError("no variants available for calibration")
    gates: dict[str, dict[str, Any]] = {}

    def gate(name: str, value: float, limit: float, *, variant_id: str, **extra) -> None:
        # Keep each measured value beside its own limit. The top-level pair is
        # the worst value/limit ratio, not two extrema from different variants.
        previous = gates.get(name)
        check = {"variant_id": variant_id, "value": value, "limit": limit,
                 "passed": value <= limit}
        checks = [*(previous or {}).get("checks", []), check]
        worst = max(checks, key=lambda c: c["value"] / c["limit"] if c["limit"] else
                    (0.0 if c["value"] == 0 else float("inf")))
        gates[name] = {"value": worst["value"], "limit": worst["limit"],
                       "passed": all(c["passed"] for c in checks), "checks": checks, **extra}

    # Softcap selection: which final mapping reproduces the model's own logits?
    first = session.prepared(letter_variants[0]["variant_id"])
    logits, captured = adapter.decoder_forward(first, capture=[Site("resid_post", adapter.num_layers - 1)],
                                               capture_positions=[first.final_position])
    final_resid = captured[f"resid_post:{adapter.num_layers - 1}"][0, 0]
    candidate = handle.info.get("softcap_candidate")
    errors = {}
    for softcap in (None, candidate) if candidate else (None,):
        handle.softcap = softcap
        errors[str(softcap)] = _logit_error(handle.readout(final_resid), first.clean_logits)
    best = min(errors, key=errors.get)
    handle.softcap = None if best == "None" else float(best)
    gate("lens_final_matches", errors[best], _kernel_tolerance(tol, adapter.spec.dtype, first.clean_logits),
         variant_id=letter_variants[0]["variant_id"], softcap=handle.softcap, candidates=errors)

    span_rates = {"question": [], "options": [], "labels": []}
    content_distinct = []
    additivity = []
    for variant in letter_variants:
        prepared = session.prepared(variant["variant_id"])
        replay, _ = adapter.decoder_forward(prepared)
        kernel_tol = _kernel_tolerance(tol, adapter.spec.dtype, prepared.clean_logits)
        gate("cached_matches_full", _logit_error(replay[0], prepared.clean_logits), kernel_tol,
             variant_id=variant["variant_id"])
        again, _ = adapter.decoder_forward(prepared)
        gate("replay_deterministic", _logit_error(replay, again), tol / 10, variant_id=variant["variant_id"])

        clean = session.clean(variant["variant_id"])
        gate("capture_is_inert", _logit_error(clean.logits, replay[0]), tol / 10, variant_id=variant["variant_id"])

        interventions = [Intervention(site, [prepared.final_position], clean.sites[site.key])
                         for site in session.sites if site.kind != "resid_pre"]
        patched, _ = adapter.decoder_forward(prepared, interventions=interventions)
        gate("identity_patch", _logit_error(patched[0], replay[0]), tol / 10, variant_id=variant["variant_id"])

        batched, _ = adapter.decoder_forward(prepared, batch=3)
        gate("batched_matches_single", max(_logit_error(row, replay[0]) for row in batched), kernel_tol,
             variant_id=variant["variant_id"])

        worst_rel = 0.0
        direct_total = None
        final = clean.sites[f"resid_post:{adapter.num_layers - 1}"]
        for layer in range(adapter.num_layers):
            pre, post = clean.sites[f"resid_pre:{layer}"], clean.sites[f"resid_post:{layer}"]
            attn, mlp = clean.sites[f"attn:{layer}"], clean.sites[f"mlp:{layer}"]
            rel = float((pre + attn + mlp - post).norm() / post.norm().clamp_min(1e-6))
            if not math.isfinite(rel):
                rel = float("inf")
            worst_rel = max(worst_rel, rel)
            for update in (attn, mlp):
                part = handle.direct_projection(update, final).cpu()
                direct_total = part if direct_total is None else direct_total + part
        gate("residual_decomposition", worst_rel, 0.03 if adapter.spec.dtype != "float32" else 1e-4,
             variant_id=variant["variant_id"])
        direct_total = direct_total + handle.direct_projection(clean.sites["resid_pre:0"], final).cpu()
        unsoftcapped = handle.lm_head(handle.norm(final.to(handle.lm_head.weight.device,
                                                           handle.lm_head.weight.dtype))).float().cpu()
        ids = session.label_ids(prepared, variant["labels"])
        additivity_error = float((direct_total[ids] - unsoftcapped[ids]).abs().max())
        additivity.append(additivity_error if math.isfinite(additivity_error) else float("inf"))

        groups = prepared.groups
        gate("token_spans", 0.0 if groups["image"] and groups["final"] == [prepared.num_tokens - 1] else 1.0,
             0.0, variant_id=variant["variant_id"])
        span_rates["question"].append(bool(groups["question"]))
        span_rates["options"].append(len(groups["options"]) >= 4)
        span_rates["labels"].append(len(groups["labels"]) == 4)
        firsts = [prepared.content_ids.get(c) for c in variant["option_contents"]]
        content_distinct.append(None not in firsts and len(set(firsts)) == 4)

    symbol_report = {}
    for variant in chosen:
        prepared = session.prepared(variant["variant_id"])
        ids = [prepared.label_ids.get(label) for label in variant["labels"]]
        ok = None not in ids and len(set(ids)) == 4
        symbol_report[variant["label_scheme"]] = {"ids": ids, "ok": ok}
    required = [s for s in symbol_report if s != "punctuation"]
    gate("symbols_single_token", 0.0 if all(symbol_report[s]["ok"] for s in required) else 1.0, 0.0,
         variant_id="all-schemes", schemes=symbol_report)

    rate = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    report = {
        "gates": gates,
        "all_passed": all(g["passed"] for g in gates.values()),
        "tolerance": tol,
        "softcap": handle.softcap,
        "informational": {
            "span_resolution_rate": {k: rate(v) for k, v in span_rates.items()},
            "content_first_tokens_distinct_rate": rate(content_distinct),
            "direct_projection_additivity_max_error": max(additivity) if additivity else None,
        },
        "variants": [v["variant_id"] for v in chosen],
        "adapter": adapter.describe(),
    }
    session.write_json("calibration.json", report)
    return report
