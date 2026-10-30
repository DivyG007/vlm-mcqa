# IconQA four-model synthesis

Read the [combined scientific report](../../docs/iconqa_core7_four_model_report_2026-10-30.md).
The original Qwen/InternVL/LLaVA matched-release comparison is preserved.
PaliGemma is explicitly an expanded-dataset, definition-prompt, BF16 exploratory
extension with failed original global behavior admission. It is not pooled into
an unchanged-protocol accuracy ranking.

- `summary.json`: prior matched-model results plus PaliGemma identities,
  amendment, behavior/causal findings, and common-image/protocol diagnostics.
- `behavior.csv`, `causal_curves.csv`, `content_sensitivity.csv`, `heads.csv`,
  `ablation.csv`, `fixed_text_controls.csv`, `lens.csv`, `projections.csv`:
  source-pair derived estimates. New Pali intervals use10,000draws/seed20261030.
- `figures/`: seven comparison PNGs, with protocol caveats visible.
- `evidence/`: preserved audit summaries and final-lens qualifications.
- `input_provenance.json`: source/dependency/input hashes, including reused
  three-model package hashes. Original full audits are not claimed newly rerun.
- `MANIFEST.json`: package artifact hashes, excluding itself.

Reproduce: `python scripts/local/synthesize_iconqa_four_model.py`, optionally
`--jarvis-root PATH`. Requires the preserved prior three-model package, local
Ada Qwen/InternVL backups, and the Jarvis worktree's verified Pali/LLaVA exports.
NumPy/Matplotlib are required. No model inference or original-result rewrite.

Common-image diagnostics verify identical image bytes and option mapping on
38confirmation source pairs, but Pali's executed prompt/precision remain different.
Only12confirmation causal pairs are eligible in all four. No model-only Pali
paired-significance claim is generated. A≥1content-denominator filter is undefined
for Pali; its additional≥0.1diagnostic is separately identified. Mean-replacement
wording in original detailed reports is qualified by executed donor-resampling
code and raw donor IDs. Raw datasets/images/backups/weights are not copied here.
