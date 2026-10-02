# Tests

Deterministic unit and integration tests. None download weights or need a
GPU; tests that need optional dependencies (torch, transformers, wandb) are
skipped when those are not installed.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v                # whole suite
PYTHONPATH=src python -m unittest discover -s tests/core7 -t tests    # one folder
PYTHONPATH=src python -m unittest discover -s tests -p test_adapters.py   # one file
PYTHONPATH=src python -m pytest tests                                 # pytest also works
```

## Layout

| Folder / file | Covers |
| --- | --- |
| `pilot/test_shapes_generator.py` | `vlm_mcqa.data.generate_shapes` |
| `pilot/test_shapes_eval.py` | `vlm_mcqa.eval.qwen_shapes`, `vlm_mcqa.analysis.plot_shapes` |
| `pilot/test_causal_abcd.py` | `vlm_mcqa.eval.qwen_causal_abcd` |
| `pilot/test_intervention_utils.py` | `vlm_mcqa.interventions.core` |
| `pilot/test_wandb_report.py` | `vlm_mcqa.tracking.wandb_report` |
| `clevr_mcq4/test_scene_and_questions.py` | `vlm_mcqa.clevr_mcq4.scene`, `.questions` |
| `clevr_mcq4/test_variants.py` | `vlm_mcqa.clevr_mcq4.variants` |
| `clevr_mcq4/test_build_pipeline.py` | `vlm_mcqa.clevr_mcq4.build` (plan → stub render → finalize) |
| `core7/test_adapters.py` | `core7.adapters` (PaliGemma prompt, token-span location) |
| `core7/test_aggregate.py` | `core7.aggregate` |
| `core7/test_calibration.py` | `core7.calibration` (kernel tolerance) |
| `core7/test_cli.py` | `core7.cli` (multi-GPU placement gate) |
| `core7/test_pair_sampling.py` | `core7.sampling`, `core7.contrasts.select_cases` |
| `core7/test_scoring.py` | `core7.scoring` (pair-clustered summaries) |
| `core7/test_selection_policy.py` | `core7.report.select` |
| `core7/test_source_snapshot.py` | `core7.source_snapshot` |
| `core7/test_tracking.py` | `core7.tracking` |
| `core7/test_merge_shards_script.py` | `scripts/core7/merge_shards.py` |
| `core7/test_validate_screening_script.py` | `scripts/core7/validate_screening.py` |
| `integration/test_tiny_models.py` | All Core-7 stages on tiny random-weight models of every family |
| `clevr_mcq4/fixtures.py` | Shared random CLEVR-scene generator |
| `support.py` | `REPO_ROOT`, `HAS_*` dependency flags, `requires_*` skip decorators, `load_script` |

## Conventions

- File `test_<module>.py`, class `<Subject>Tests`, method `test_<expected behaviour>`.
- Each file starts with a one-line docstring naming what it tests; test
  methods and `setUp` are annotated `-> None`.
- Optional dependencies are guarded with `support.HAS_*` and a
  `support.requires_*` decorator.
- Repository scripts are imported with `support.load_script`.
