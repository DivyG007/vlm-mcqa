# Controlled IconQA: three-model synthesis

Read the [combined scientific report](../../docs/iconqa_core7_three_model_report_2026-10-30.md).
This package focuses on CF-small Qwen, InternVL and the completed A100 batch8
LLaVA experiment. The separate Ada batch2 LLaVA run is not substituted for it.
PaliGemma remains pending.

- `summary.json`: model identities, behavior, frozen decisions, causal-cohort
  overlap, paired behavior comparisons and shared-pair head sensitivity.
- `behavior.csv`: source-pair accuracy intervals and native/edited strata.
- `causal_curves.csv`: residual, attention/MLP and token-group pair estimates.
- `content_sensitivity.csv`: mean/median ratios, ratio of means and diagnostic
  denominator-filtered curves; original decisions remain unchanged.
- `heads.csv`: new IconQA versus historical CLEVR head-set effects.
- `ablation.csv`: donor-resampling disruption, clean margins and remaining accuracy.
- `fixed_text_controls.csv`: answer-changing/same-answer/paired image-donor shifts.
- `lens.csv`, `projections.csv`: observational diagnostics, not causal onset.
- `figures/`: seven comparison PNGs.
- `evidence/`: preserved LLaVA audit summary and final-lens qualification.
- `input_provenance.json`: hashes of source inputs used here and synthesis code.
- `MANIFEST.json`: package artifact hashes, excluding itself.

Reproduction: `python scripts/local/synthesize_iconqa_three_model.py`, optionally
`--llava-root PATH`. Requires NumPy/Matplotlib, preserved Ada Qwen/InternVL backups,
and the sibling LLaVA verified export/detailed audit. No model inference or raw
artifact rewrite. All newly computed intervals use source-pair means, 10,000
draws, NumPy seed20261030; pointwise, not multiplicity-corrected. Full prior
audits are reused; hashes of actually used inputs are reverified.

Main interpretation: sparse attention effects replicate in all three. Stable
content-before-position timing holds for InternVL/LLaVA under examined
normalization diagnostics, but Qwen's ordering is sensitive. Historical InternVL
top4 narrowly misses its own sparsity fraction cutoff. Causal cohorts, image
processing and fixed-text head counts differ. The ablation implementation uses
another clean donor's activation, correcting the detailed LLaVA narrative's
"mean replacement" wording without changing that original artifact.
