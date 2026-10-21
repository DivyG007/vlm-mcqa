# Ada Entry Points

Read the repository [`AGENTS.md`](../../AGENTS.md) and the installed Ada skill
before using these scripts.

## Scripts

- `qwen_shapes_3b.sbatch` — generates Shapes data, runs behavioral/logit-lens
  evaluation, renders figures, publishes W&B artifacts, and verifies outputs.
- `qwen_causal_abcd.sbatch` — generates Shapes data, runs calibrated causal
  interventions, renders figures, publishes W&B artifacts, and verifies
  outputs.
- `stretch_e19_saved.sbatch` — CPU-only saved-record E19 analysis, ordered Qwen
  then InternVL; it does not load a model or claim a causal failure diagnosis.
- `rsync-excludes.txt` — excludes Git metadata, environments, generated data,
  results, caches, logs, literature, and other files not needed by an execution
  snapshot.

## Required submission pattern

1. Commit a clean local source/config/script state.
2. Create an immutable Ada snapshot keyed by the full Git SHA.
3. Dry-run rsync, inspect it, then perform the real sync.
4. Inspect jobs, result destinations, node health, and storage.
5. Submit with `SOURCE_GIT_COMMIT` and `SOURCE_ROOT` set to the immutable
   snapshot.
6. Verify the allocated node and framework-level CUDA.
7. Monitor queue/history, both logs, progress, artifacts, W&B, and storage.
8. Verify completion, then dry-run and copy the whole result directory locally.

Comma-containing lists are unsafe in Slurm `--export`. The causal script accepts
`*_PLUS` variables in which `+` is converted to `,` inside the job.

Do not copy example commands blindly: node availability, caches, storage, and
the appropriate experiment config must be checked anew for every submission.
