# Configurations

| File | Contents |
| --- | --- |
| `core7/clevr_mcq4_official3k.json` | Released CLEVR-MCQ-4 v3: seed, object counts, program depth, question-family weights, split sizes |
| `core7/clevr_mcq4_dataset.json` | Same question and split policy for a fully locally rendered build (default for `validate_screening.py` and the tests) |
| `core7/models.json` | The twelve candidate checkpoints: family, Hugging Face id, nominal size, backbone, processor settings |
| `core7/qwen25vl_7b_pinned.json` | Qwen2.5-VL-7B pinned to the snapshot used for the reported run |
| `core7/internvl35_8b_pinned.json` | InternVL3.5-8B pinned to the snapshot used for the reported run, with multi-GPU placement |
| `core7/profiles.json` | Run profiles (`smoke`, `report_prelim_32`, `full_core7`) and the model-selection thresholds |
| `experiment_00_shapes.json` | Pilot Shapes dataset, alphabets, behavioural gates and lens probes |
| `experiment_03_causal_abcd.json` | Pilot causal contrasts, discovery/confirmation pairs, patching modes and controls |
| `experiment_05_content_position.json` | Pilot content-versus-position factorial: readouts, layers and token groups |

Every run writes its fully resolved configuration and dataset hashes to
`run.json` in its result directory.
