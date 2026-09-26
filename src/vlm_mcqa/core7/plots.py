"""Figures derived from aggregate tables. Conclusions must cite the tables/JSONL."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"position": "#1f77b4", "content": "#d62728", "content_and_position": "#9467bd",
          "identity": "#7f7f7f", "random_pair": "#bcbd22", "attn": "#1f77b4", "mlp": "#ff7f0e",
          "letters": "#1f77b4", "numbers": "#2ca02c", "rare_letters": "#d62728", "punctuation": "#8c564b"}


def _csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def _lines(ax, rows, *, x: str, series: str, where=lambda r: True, band: bool = True):
    groups = defaultdict(list)
    for row in rows:
        if where(row):
            groups[row[series]].append(row)
    for name, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rs.sort(key=lambda r: r[x])
        xs = [r[x] for r in rs]
        ax.plot(xs, [r["mean"] for r in rs], marker="o", ms=3, label=str(name), color=COLORS.get(name))
        if band and "ci_low" in rs[0]:
            ax.fill_between(xs, [r["ci_low"] for r in rs], [r["ci_high"] for r in rs], alpha=0.15,
                            color=COLORS.get(name))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def plot_run(run_dir: Path) -> list[str]:
    run_dir = Path(run_dir)
    tables, figs = run_dir / "tables", run_dir / "figures"
    figs.mkdir(exist_ok=True)
    made: list[str] = []
    model = json.loads((run_dir / "run.json").read_text()).get("model_key", "") if (run_dir / "run.json").exists() else ""

    lens = _csv(tables / "m4_m5_m7_lens.csv")
    if lens:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        _lines(axes[0], lens, x="stage", series="scheme", where=lambda r: r["metric"] == "margin")
        axes[0].set(title="M4/M7 correct - best incorrect", xlabel="stage", ylabel="logit margin")
        _lines(axes[1], lens, x="stage", series="scheme", where=lambda r: r["metric"] == "correct_restricted_prob")
        axes[1].set(title="M4 restricted 4-way P(correct)", xlabel="stage")
        _lines(axes[2], lens, x="stage", series="scheme", where=lambda r: r["metric"] == "mean_label_logit")
        axes[2].set(title="M4 generic label signal (mean label logit)", xlabel="stage")
        fig.suptitle(model)
        made.append(_save(fig, figs / "m4_m7_lens.png"))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        _lines(axes[0], lens, x="stage", series="scheme", where=lambda r: r["metric"] == "requested_mass_8way")
        axes[0].set(title="M7 requested vs default A-D (8-way mass on requested)", xlabel="stage")
        _lines(axes[1], lens, x="stage", series="metric",
               where=lambda r: r["scheme"] == "letters" and r["metric"] in ("margin", "content_margin"))
        axes[1].set(title="M5 content-token vs symbol margin (letters)", xlabel="stage")
        made.append(_save(fig, figs / "m5_m7_content_default.png"))

    proj = _csv(tables / "m2_projection.csv")
    if proj:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric, title in ((axes[0], "direct_correct_minus_best_incorrect", "direct projection: margin"),
                                  (axes[1], "direct_mean_label", "direct projection: generic label")):
            _lines(ax, proj, x="layer", series="component",
                   where=lambda r, m=metric: r["metric"] == m and r["scheme"] == "letters")
            ax.set(title=f"M2 {title}", xlabel="layer")
        made.append(_save(fig, figs / "m2_projection.png"))

    for split in ("discovery", "confirmation"):
        m1 = _csv(tables / f"m1_residual_{split}.csv")
        if m1:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            _lines(axes[0], m1, x="layer", series="contrast_type",
                   where=lambda r: r["metric"] == "normalized_recovery")
            axes[0].axhline(0.5, ls="--", c="k", lw=0.7)
            axes[0].set(title=f"M1 final-token residual patching ({split})", xlabel="layer", ylabel="normalized recovery")
            _lines(axes[1], m1, x="layer", series="contrast_type", where=lambda r: r["metric"] == "flip_to_donor")
            axes[1].set(title="flip to donor symbol", xlabel="layer")
            made.append(_save(fig, figs / f"m1_residual_{split}.png"))
        outcomes = _csv(tables / f"m6_outcomes_{split}.csv")
        if outcomes:
            outcomes.sort(key=lambda r: r["layer"])
            fig, ax = plt.subplots(figsize=(7, 4))
            xs = [r["layer"] for r in outcomes]
            bottom = [0.0] * len(xs)
            for key, color in (("frac_none", "#7f7f7f"), ("frac_position", "#1f77b4"),
                               ("frac_content", "#d62728"), ("frac_other", "#bcbd22")):
                ax.bar(xs, [r[key] for r in outcomes], bottom=bottom, color=color, label=key[5:])
                bottom = [b + r[key] for b, r in zip(bottom, outcomes)]
            ax.set(title=f"M6 content-and-position outcome ({split})", xlabel="layer", ylabel="fraction")
            ax.legend(fontsize=7)
            made.append(_save(fig, figs / f"m6_outcomes_{split}.png"))
        groups = _csv(tables / f"m6_token_groups_{split}.csv")
        if groups:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax, kind in zip(axes, ("content", "content_and_position")):
                _lines(ax, groups, x="layer", series="group", where=lambda r, k=kind: r["contrast_type"] == k)
                ax.set(title=f"M6 token-group patching: {kind}", xlabel="layer")
            made.append(_save(fig, figs / f"m6_token_groups_{split}.png"))
        m2 = _csv(tables / f"m2_components_{split}.csv")
        if m2:
            kinds = sorted({r["contrast_type"] for r in m2})
            fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 4), squeeze=False)
            for ax, kind in zip(axes[0], kinds):
                _lines(ax, m2, x="layer", series="site", where=lambda r, k=kind: r["contrast_type"] == k)
                ax.set(title=f"M2 attn vs MLP patching: {kind}", xlabel="layer")
            made.append(_save(fig, figs / f"m2_components_{split}.png"))
        abl = _csv(tables / f"m2_ablation_{split}.csv")
        if abl:
            fig, ax = plt.subplots(figsize=(6, 4))
            _lines(ax, abl, x="layer", series="site")
            ax.set(title=f"M2 resampling ablation: margin drop ({split})", xlabel="layer")
            made.append(_save(fig, figs / f"m2_ablation_{split}.png"))
        heads = run_dir / f"heads_{split}_summary.json"
        if heads.exists():
            made += _plot_heads(json.loads(heads.read_text()), figs, split)
    return made


def _plot_heads(summary: dict[str, Any], figs: Path, split: str) -> list[str]:
    made = []
    ranking = summary.get("ranking")
    if ranking:
        layers = sorted({r["layer"] for r in ranking})
        n_heads = max(r["head"] for r in ranking) + 1
        grid = [[0.0] * n_heads for _ in layers]
        for r in ranking:
            grid[layers.index(r["layer"])][r["head"]] = r["mean_nr"]
        fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.35), 1 + 0.5 * len(layers)))
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        ax.set_yticks(range(len(layers)), [str(l) for l in layers])
        ax.set(xlabel="head", ylabel="layer", title=f"M3 head screen: mean recovery ({split})")
        fig.colorbar(im, ax=ax)
        made.append(_save(fig, figs / f"m3_head_screen_{split}.png"))
    per_k = summary.get("per_k") or {}
    if per_k:
        ks = sorted(per_k, key=int)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([int(k) for k in ks], [per_k[k]["top_k"]["mean"] for k in ks], marker="o", label="top-k")
        ax.plot([int(k) for k in ks], [per_k[k]["random_mean"] for k in ks], marker="s", label="matched random (mean)")
        ax.plot([int(k) for k in ks], [per_k[k]["random_p95"] for k in ks], ls="--", label="random p95")
        if summary.get("all_heads_mean_nr") is not None:
            ax.axhline(summary["all_heads_mean_nr"], c="k", lw=0.7, label="all heads in layers")
        ax.set_xscale("log", base=2)
        ax.set(title=f"M3 cumulative top-k recovery ({split}): {summary.get('verdict', [''])[0]}",
               xlabel="k", ylabel="normalized recovery")
        ax.legend(fontsize=7)
        made.append(_save(fig, figs / f"m3_topk_{split}.png"))
    return made


def plot_screening(selection: dict[str, Any], out_dir: Path) -> list[str]:
    rows = selection["table"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for family in sorted({r["family"] for r in rows}):
        rs = sorted((r for r in rows if r["family"] == family), key=lambda r: r["params_b"])
        axes[0].plot([r["params_b"] for r in rs], [r["worst_position_accuracy"] for r in rs], marker="o", label=family)
        axes[1].plot([r["params_b"] for r in rs], [r["clean_minus_shuffled"] for r in rs], marker="o", label=family)
    axes[0].axhline(selection["thresholds"]["min_worst_position_accuracy"], ls="--", c="k", lw=0.7)
    axes[0].set(xscale="log", xlabel="parameters (B)", ylabel="worst-position A/B/C/D accuracy",
                title="Twelve-model screening")
    axes[1].set(xscale="log", xlabel="parameters (B)", ylabel="clean - shuffled accuracy", title="Image dependence")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    made = [_save(fig, Path(out_dir) / "screening_accuracy_vs_params.png")]
    schemes = (("letters", "A/B/C/D"), ("numbers", "1/2/3/4"), ("rare_letters", "Q/Z/R/X"))
    if rows and all(r.get("robustness_cells") for r in rows):
        ordered = sorted(rows, key=lambda r: (r["family"], r["params_b"] or 0))
        labels = [r["model_key"] for r in ordered]
        matrix = [[r["robustness_cells"].get(f"{scheme}:p{position}", float("nan"))
                   for scheme, _ in schemes for position in range(4)] for r in ordered]
        fig, ax = plt.subplots(figsize=(11, 1.7 + 0.43 * len(ordered)))
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xticks(range(12), [f"{label} p{position}" for _, label in schemes for position in range(4)],
                      rotation=45, ha="right")
        ax.set(title="Forced-choice accuracy by position and alphabet", xlabel="scheme and answer position")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if value == value:
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if value >= 0.6 else "black")
        fig.colorbar(image, ax=ax, label="accuracy")
        made.append(_save(fig, Path(out_dir) / "screening_robustness_cells.png"))
    return made


def plot_compare(comparison: dict[str, Any], out_dir: Path) -> list[str]:
    models = comparison["models"]
    keys = ("position_transfer_layer", "content_transfer_layer", "attn_peak_layer", "mlp_peak_layer")
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.6 * max(1, len(models))))
    for i, m in enumerate(models):
        for j, key in enumerate(keys):
            for split, marker in (("discovery", "o"), ("confirmation", "x")):
                value = m.get(f"{split}:{key}:depth")
                if value is not None:
                    ax.scatter(value, i + 0.12 * (j - 1.5), marker=marker, color=f"C{j}",
                               label=f"{key} ({split})" if i == 0 else None)
        if m.get("letters_readable_depth") is not None:
            ax.scatter(m["letters_readable_depth"], i, marker="|", s=200, color="k",
                       label="letters readable (lens)" if i == 0 else None)
    ax.set_yticks(range(len(models)), [m["model_key"] for m in models])
    ax.set(xlim=(0, 1.02), xlabel="normalized depth (layer / num layers)", title="Cross-model transitions")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    made = [_save(fig, Path(out_dir) / "cross_model_normalized_depth.png")]
    completion = comparison["completion"]
    if completion:
        methods = [k for k in completion[0] if k not in ("model_key", "run_dir")]
        grid = [[sum(row[m].values()) / len(row[m]) for m in methods] for row in completion]
        fig, ax = plt.subplots(figsize=(1.2 * len(methods), 1 + 0.5 * len(completion)))
        ax.imshow(grid, cmap="Greens", vmin=0, vmax=1)
        ax.set_xticks(range(len(methods)), methods, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(completion)), [r["model_key"] for r in completion])
        ax.set_title("Core-7 outputs present (shared methods run once on behavior split)")
        made.append(_save(fig, Path(out_dir) / "core7_completion_matrix.png"))
    return made
