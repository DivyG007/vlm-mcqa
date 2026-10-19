# Curated Core-7 result snapshot

This directory is the compact, versionable evidence layer used for team
collaboration. Raw JSONL records, logs, W&B working directories, caches, and
machine-local mirrors remain under ignored `results/` paths and on the
execution hosts.

The original files here were synchronized on 2026-10-10 from the clean release
repository `../vlm-mcqa` at commit
`5f1e05d663fa078c16818db6c4bbd539338142f8`. They contain only selected
summaries, tables, figures, freeze records, confirmation reports, and run
provenance suitable for Git.

On 2026-10-19, the independently audited LLaVA pkgpu result was added from
execution source `051ea21e538497a8137f0cdf94f73543d3fbf063`. Its export manifest
records original/export hashes; this does not replace or relabel earlier results.

## Contents

- `screening/`: 12-checkpoint feasibility screen.
- `qwen25vl_3b_shapes_pilot/`: selected figures from the exploratory Shapes
  programme.
- `qwen25vl_7b_core7/`: full Qwen2.5-VL-7B discovery and confirmation package,
  including the corrected pair-balanced M3 result.
- `internvl35_8b_core7/`: full InternVL3.5-8B discovery and confirmation
  package.
- `llava_ov_7b_si_core7/`: full LLaVA discovery/confirmation, 16 figures,
  12 tables, freeze, raw-count/hash audit and W&B publication audit. Exact
  ordering does not replicate; attention dominance and sparse heads do.
- `cross_model/`: historical Qwen-versus-32-pair preliminary InternVL
  comparison figures. Their legends retain the preliminary label. They do
  **not** compare the two full runs and must be regenerated from the full E15
  package before use in a final report.

The earlier 32-pair InternVL package is retained separately at
`../internvl35_8b_preliminary_32pair/` for provenance. It is superseded for
final effect estimates by the full package here, but remains an independent
engineering and early-replication record.

Do not edit these copied result files by hand. Regenerate or re-export them
from audited raw evidence, then record the source commit and replacement
relationship in `docs/research_log.md` and `docs/run_ledger.md`.
