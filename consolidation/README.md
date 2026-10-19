# Evidence consolidation

This directory is the durable, human-readable evidence bundle for the
team. It does not replace raw experiment directories or
`docs/research_log.md`; it records compact audited evidence suitable for Git,
which evidence entered a report, and how it was interpreted.

## Contents

- `source_index.md`: source inventory and evidence status.
- `evidence_summary.md`: concise, claim-level synthesis of the project so far.
- `core7_release_results/`: curated result snapshot for the Shapes pilot,
  screen, full Qwen and full InternVL runs, synchronized from clean release
  commit `5f1e05d`.
- `internvl35_8b_preliminary_32pair/`: retained early InternVL package,
  superseded for final estimates by the full run.
- `report/`: LaTeX source, bibliography, reproducible figure source, copied
  scientific figures, and the rendered mid-semester report PDF. The submitted
  report is historical and is not silently rewritten when later runs finish.

## Evidence policy

1. Proposal text is treated as intent, not as an experimental result.
2. Smoke tests are used only for provisional model selection.
3. Discovery findings are marked as exploratory until they reproduce on the
   held-out confirmation split.
4. Logit-lens and direct-projection plots are observational. Patching and
   ablation support causal claims only for the specific intervention and
   metric used.
5. The full InternVL package supersedes the 32-pair preliminary package for
   final effect estimates. The preliminary package remains visible as an
   independent engineering and early-replication record.
6. Raw rows and logs remain outside Git; compact exports must retain enough
   provenance to reconnect them to the authoritative raw evidence.
