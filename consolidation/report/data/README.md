# Report input data not present in the curated release packages

These small tables feed `../scripts/plot_final_report.py`. They are copies, not
new results.

- `qwen25vl_7b_core7_wandb/`: per-layer tables from the audited Qwen2.5-VL-7B
  full Core-7 W&B result artifacts `vlm-mcqa/smoke-testing/qwen25vl_7b-confirmation-full_core7:v0`
  and `qwen25vl_7b-discovery-full_core7:v0` (the artifacts named in
  `docs/ai_handoff/qwen7b_full_result_audit_2026-10-01.md`). Downloaded
  2026-10-30; hashes in `SHA256SUMS`. The released Qwen package
  (`../../core7_release_results/qwen25vl_7b_core7/`) holds only figures and
  summaries. The M3 table in that artifact is the superseded seven-pair-weighted
  result and is not used; the corrected pair-balanced M3 values come from
  `docs/ai_handoff/qwen7b_m3_pairbalanced_rerun_audit_2026-10-01.md`.
- `tuned_lens_full_01/`: Tuned Lens versus Logit Lens behaviour-split tables and
  `summary.json` from branch `codex/divyanshu-tuned-lens` (commit `efb4eb4`,
  `docs/tuned_lens/results_2026-10-28/`). PaliGemma is provisional pending the
  quirk report.
