# Source audit for the mid-semester report

This index records the repository-wide documentation audit performed before
the current report rewrite. “Used” means that a result, qualification, design
choice, or provenance statement affected the report. “Checked / operational”
means the file was reviewed for consistency but contains commands or transfer
instructions rather than a new scientific result. “Superseded” means the file
was still read, but later evidence controls what is reported.

## Proposal, canonical memory, and evidence packages

| Source | Audit outcome | Treatment in report |
| --- | --- | --- |
| `out-of-tokens-Proposal.pdf` | Original research question, seven methods, families, intended timeline | Used as intent; later dataset/model changes are explained |
| `docs/research_log.md` | Complete cumulative experiment record through verified full InternVL completion | Primary interpretation source |
| `docs/run_ledger.md` | Retries, supersession, full-run closure, artifacts and external tracking identities | Used for status and replacement relationships |
| `evidence_report_20261001.md` | Independent Qwen package, calibration, rows, logs, W&B and checksum audit | Used for Qwen completion claims |
| `qwen25vl_7b_full_core7_results_20261001.tar.gz` | Raw Qwen discovery/confirmation evidence | Extracted only under ignored `tmp/`; tables/figures checked |
| `consolidation/core7_release_results/qwen25vl_7b_core7/` | Compact full Qwen discovery/confirmation evidence | Current Git-sized Qwen result package |
| `consolidation/core7_release_results/internvl35_8b_core7/` | Compact full InternVL discovery/confirmation evidence | Current Git-sized InternVL result package; supersedes preliminary estimates |
| `results/garuda/internvl35_8b_full_core7/` | Checksum-verified local raw InternVL mirror | Authoritative local raw evidence; intentionally ignored by Git |

## Qwen2.5-VL-3B exploratory programme

| Source | Audit outcome | Treatment in report |
| --- | --- | --- |
| `docs/experiment_00_shapes.md` | Controlled task and causal design | Used to explain why the pilot exists |
| `docs/preliminary_results_2721632.md` | First 8-pair behavior and stage-28 lens transition | Used as historical validation; 64-pair values reported |
| `docs/experiment_01_results.md` | 64-pair behavior, four-alphabet compliance, shared label rise | Used in pilot table and new figure |
| `docs/experiment_02_results.md` | Per-alphabet emergence timing and margins | Used in pilot alphabet comparison |
| `docs/causal_experiment_plan.md` | Intervention logic, controls, necessity/sufficiency caveats | Used to qualify causal claims |
| `docs/methods_1_4_7_audit.md` | Methods 1/4 completion and the original M7 gap | Used; later dual-alphabet run closes the gap |
| `docs/experiment_04_content_symbol.md` | Content-versus-symbol lens design | Used in pilot content/readability explanation |
| `docs/experiment_05_content_position.md` | Position-versus-content factorial | Used in pilot route table |
| `docs/literature_tooling.md` | Literature-to-method mapping and tooling decisions | Used in related work and experimental framing |
| `results/ada/...` pilot mirrors | Raw summaries, layerwise CSV/JSONL, figures and logs | Used for exact values and regenerated pilot chart |

## Core-7 design, smoke testing, and final dataset

| Source | Audit outcome | Treatment in report |
| --- | --- | --- |
| `docs/core7_roadmap.md` | Seven-method portable study and four-family plan | Used for method table and scope |
| `docs/core7_screening_policy.md` | A preregistered policy for a possible 300-item screen | Read but not presented as a hard boundary because that full screen was not run |
| `docs/core7_execution_checklist.md` | Frozen dataset, Qwen completion, remaining InternVL/full-screen work | Used for current status and limitations |
| `docs/core7_runbook.md` | Generic commands and artifact schema | Checked / operational |
| `docs/local_gpu_dataset_plan.md` | Final 1,252-item release lineage plus superseded build plans | Used only for final release facts; detailed renderer history omitted |
| `scripts/context.md` | Complete 12-model smoke table, smoke Core-7 results, fixes and caveats | Used for all 12 screening rows and smoke qualifications |
| `configs/core7/*.json` and config READMEs | Frozen model/dataset/profile definitions | Checked against reported model and split names |

## AI handoff package

| Source | Audit outcome | Treatment in report |
| --- | --- | --- |
| `docs/ai_handoff/README.md` | Reading order and authority rules | Checked |
| `PROJECT_CONTEXT.md` | Core computation and pilot interpretation | Used throughout the rewritten narrative |
| `EXPERIMENT_HISTORY.md` | Compact E00--E06 results and exact pilot route values | Used heavily in the expanded pilot section |
| `GLOSSARY_AND_METRICS.md` | Definitions of logits, restricted accuracy, mass, recovery and stage indexing | Used to make metrics readable and prevent overclaiming |
| `NEXT_STEPS.md` | Smoke audit, Core-7 status and superseded pilot priorities | Used for caveats; current checklist/run ledger override stale status |
| `CODEBASE_AND_OPERATIONS.md` | Repository structure, artifact map and run mechanics | Checked / operational |
| `2026-10-01_execution_handoff.md` | Final dataset lineage, Qwen audit summary and InternVL assignment | Used for matched-data and completion state |
| `continuation_checklist.md` | Evidence still required for comparison | Used for InternVL insertion gate |
| `qwen7b_full_result_audit_2026-10-01.md` | Authoritative M1/M2/M4--M7 result and original M3 caveat | Used; its original M3 numbers are excluded |
| `qwen7b_m3_pairbalanced_rerun_audit_2026-10-01.md` | Corrected 64-pair M3 result | Authoritative M3 source |
| `qwen7b_wandb_audit_2026-10-01.json` | Exact W&B configs, summaries and artifact manifests | Checked; W&B is not treated as sole evidence |
| `qwen7b_m3_correction_prompt.md`, `qwen_live_m3_fix_audit_prompt.md` | Requirements that produced/verified the corrected M3 run | Checked / operational; no independent result |
| `internvl8b_corrected_run.md`, `internvl8b_multigpu_agent_prompt.md`, `collaborator_internvl8b_commands.md` | InternVL provenance, placement and launch requirements | Checked / operational; current run ledger supersedes older commands |

## Presentations and literature

| Source | Audit outcome | Treatment in report |
| --- | --- | --- |
| `presentation/vlm_mcqa_experiments.pdf` plus provenance | Pilot results and explanatory figures | Read and visually inspected; exact values cross-checked to raw mirrors |
| `presentation/core7_smoke_update.pdf` plus provenance | 12-model smoke and two-model smoke Core-7 | Read and visually inspected; described only as preliminary |
| `presentation/core7_execution_handoff_2026-10-01.pdf` | Dataset/Qwen/InternVL handoff state | Read and visually inspected |
| `lit/AAA-LM-MCQ.pdf` | Text-only formatted-MCQA mechanism and Core-7 experimental template | Used in related work and method rationale |
| `lit/Mind's-eye.pdf` | Related VLM interpretability review and negative sparse-head context | Read; no result imported as our own |
| External papers in `references.bib` | CLEVR, MMMU, binding, patching best practices, and model-family reports | Cited only where they directly support the text |

## Reconciliation decisions

- The corrected pair-balanced Qwen M3 audit supersedes every original M3
  confidence interval and top-k value.
- The final `clevr_mcq4_v3_official3k_cuda45pairs` release supersedes the v2,
  small local trial, and the unstarted 1,500-scene build plan.
- The 12-model run was smoke-scale. The report includes all rows but does not
  call the two detailed models formal screen winners or apply a hard cutoff.
- Qwen and InternVL full runs use identical data and split hashes. Their source
  commits differ for adapter/head-sampling corrections; this is provenance,
  not a different scientific dataset.
- The full InternVL package completed and was audited on 2026-10-04. It
  supersedes deadline-preliminary values for final effect estimates; smoke and
  preliminary evidence remain labeled and are not silently merged with it.
