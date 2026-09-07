"""Answer-symbol and content readouts, recovery metrics and intervals."""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

import torch

DEFAULT_LABELS = ("A", "B", "C", "D")


def symbol_readout(
    logits: torch.Tensor,
    label_token_ids: Sequence[int],
    correct_index: int,
    *,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Every Method-4 quantity for one vocabulary-logit vector.

    ``restricted_*`` values are a softmax over the four displayed labels only;
    ``label_mass_full`` is the full-vocabulary probability of the four labels.
    """
    logits = logits.float()
    ids = torch.tensor(list(label_token_ids), device=logits.device)
    scores = logits.index_select(0, ids)
    order = torch.argsort(scores, descending=True)
    correct = scores[correct_index]
    others = torch.cat([scores[:correct_index], scores[correct_index + 1:]])
    log_z = torch.logsumexp(logits, 0)
    masked = logits.clone()
    masked[ids] = -torch.inf
    top_other = int(masked.argmax())
    row = {
        "label_logits": [float(v) for v in scores],
        "predicted_index": int(order[0]),
        "correct": int(order[0]) == correct_index,
        "correct_logit": float(correct),
        "best_incorrect_logit": float(others.max()),
        "mean_label_logit": float(scores.mean()),
        "margin": float(correct - others.max()),
        "top_gap": float(scores[order[0]] - scores[order[1]]),
        "restricted_probs": [float(v) for v in torch.softmax(scores, 0)],
        "label_mass_full": float(torch.exp(torch.logsumexp(scores, 0) - log_z)),
        "correct_prob_full": float(torch.exp(correct - log_z)),
        "correct_rank_full": int((logits > correct).sum()) + 1,
        "global_top_is_label": int(logits.argmax()) in set(ids.tolist()),
        "top_non_label_id": top_other,
        "top_non_label_logit": float(logits[top_other]),
    }
    if tokenizer is not None:
        row["top_non_label_text"] = tokenizer.decode([top_other])
    return row


def pair_difference(logits: torch.Tensor, a: int, b: int) -> float:
    return float(logits[a] - logits[b])


def normalized_recovery(recipient: float, donor: float, patched: float) -> float:
    """0 = no effect, 1 = full transfer of the donor's readout difference."""
    denominator = donor - recipient
    if abs(denominator) < 1e-6:
        return float("nan")
    return (patched - recipient) / denominator


def bootstrap_ci(values: Sequence[float], *, seed: int = 0, samples: int = 2000) -> tuple[float, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(clean)
    means = sorted(sum(clean[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]


def mean(values: Sequence[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")


def summarize(values: Sequence[float], *, seed: int = 0) -> dict[str, float]:
    low, high = bootstrap_ci(values, seed=seed)
    return {"mean": mean(values), "ci_low": low, "ci_high": high,
            "n": sum(1 for v in values if not math.isnan(v))}


def summarize_clustered(values: Sequence[float], clusters: Sequence[str], *, seed: int = 0) -> dict[str, Any]:
    """Summarize dependent rows using the semantic pair as the sampling unit.

    Directed donor/recipient contrasts from one counterfactual pair are not
    independent observations.  Each pair therefore contributes one mean and
    the bootstrap resamples those pair means.  ``n`` remains the raw row
    count, while ``n_clusters`` makes the effective sample size explicit.
    """
    if len(values) != len(clusters):
        raise ValueError("values and clusters must have equal length")
    grouped: dict[str, list[float]] = {}
    for value, cluster in zip(values, clusters, strict=True):
        if not math.isnan(value):
            grouped.setdefault(str(cluster), []).append(value)
    cluster_means = [mean(grouped[key]) for key in sorted(grouped)]
    low, high = bootstrap_ci(cluster_means, seed=seed)
    return {"mean": mean(cluster_means), "ci_low": low, "ci_high": high,
            "n": sum(len(v) for v in grouped.values()), "n_clusters": len(cluster_means),
            "sampling_unit": "pair"}


def parse_generated_label(text: str, labels: Sequence[str]) -> str | None:
    """Strict format check: the first non-space characters must be a label."""
    stripped = text.strip().lstrip("(").strip()
    for prefix in ("Answer:", "answer:", "The answer is"):
        if stripped.startswith(prefix):
            return None  # verbose answers count as non-compliant
    for label in labels:
        if stripped.startswith(label) and (len(stripped) == len(label) or not stripped[len(label)].isalnum()):
            return label
    return None


def group_rates(rows: Sequence[Mapping[str, Any]], key: str, value: str = "correct") -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(float(row[value]))
    return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}
