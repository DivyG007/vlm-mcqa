# IconQA transfer analysis — 30 October 2026

Read the [Qwen and InternVL report](../../docs/iconqa_core7_transfer_analysis_2026-10-30.md).
The final model comparison uses CF-small only; natural Qwen B1/B4 are historical
batch-stability evidence. LLaVA and PaliGemma are pending additions.

- `results_summary.json`: compact run provenance, inventory/count audit, replication outcomes,
  behavior strata, historical and new head sets, fixed-text controls and overlap.
- `behavior_strata.csv`: accuracy and source-pair bootstrap intervals.
- `causal_curves_pair_bootstrap.csv`: residual, component and token-group curves.
- `lens_curves.csv`: observational curves recovered from phase-suffixed raw files.
- `fixed_text_control_differences.csv`: paired answer-changing minus same-answer
  head-patch shifts; different head counts prohibit direct model ranking.
- `figures/`: selected PNG analysis figures, distinct from original run outputs.
- `accounting_recheck_20261030.txt`: read-only live Ada terminal-state recheck.
- `publication_manifest.json`: script, input-inventory and published-output SHA-256 bindings.

Detailed `metrics.json`, `analysis_manifest.json` and redundant SVG exports remain
local. They can be regenerated from the preserved raw backups. Raw predictions,
datasets/images, model weights, backup inventories and environments are excluded
from this publication.

Run `python scripts/local/analyze_iconqa_transfer_runs.py` from this checkout to
reproduce numerical outputs from `results/ada_completed_run_backups/`. Python,
NumPy and Matplotlib are required. The audit rehashes every inventory file,
including shared dataset assets, and can take a few minutes. It writes only this
analysis directory. Source-pair bootstrap: 10,000 draws, seed 20261030; saved M3
intervals retain the runner's bootstrap. Accounting is a separately captured
receipt and is not refreshed by the offline analysis script.

After a full analysis, `python scripts/local/analyze_iconqa_transfer_runs.py
--export-summary-only` refreshes the compact publication package from
hash-verified local metrics without repeating the numerical audit.

Reporting policy: CF-small is primary; larger Qwen batch4 supports natural-image
transfer; cancelled batch1 supports batch numerical stability. The larger/smaller
comparison changes several protocol factors and is not a dataset-size ablation.

No original run, cloud publication, model weight, GPU job or submitted report was
modified. The cancelled batch-1 confirmation remains incomplete.
