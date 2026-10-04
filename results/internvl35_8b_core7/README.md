# InternVL3.5-8B full Core-7 results

This is the curated, repository-safe record of the completed InternVL3.5-8B discovery and held-out confirmation workflow. It contains the summaries, tables, figures, frozen hypotheses and sanitized run manifests needed to inspect or report the experiment without committing raw per-example records or machine logs.

## Provenance

| Field | Value |
| --- | --- |
| Model | `OpenGVLab/InternVL3_5-8B-HF`, 8.53B parameters, 36 language layers, 32 heads |
| Model revision | `741a7d03020411e666c6109218ab71e08151ef86` |
| Dataset | CLEVR-MCQ-4 v3; `dataset_hash=f5573e25f3aa2bf2`; image-aware release `eb7b79e56977bbaeed5fc981c5c8e09a4e26eaf8b1702283abe2d4bbfc718cbd` |
| Source snapshot | `9b2679d95ddcc0a424da4b9316253f412dfa5323`, clean gitless manifest |
| Profile | `full_core7`; 64 semantic pairs per discovery/confirmation phase, all seven methods |
| Frozen hypothesis record | `freeze_hash=84a69cc52774f28f` |
| W&B | [discovery `dxk3hym4`](https://wandb.ai/rachitsamirmehta-iiit-hyderabad/anlp-vlm-mcqa/runs/dxk3hym4), [confirmation `tfrx0zjz`](https://wandb.ai/rachitsamirmehta-iiit-hyderabad/anlp-vlm-mcqa/runs/tfrx0zjz) |
| Completion | 2026-10-04 12:24 IST; outer status 0; both run manifests complete; both calibrations passed |

## Included evidence

`discovery/` contains the behavioural and lens summaries, all discovery method summaries, seven tables and ten figures. `confirmation/` contains the held-out aggregate, frozen-hypothesis report, method summaries, five tables and six figures. `freeze.json` is the exact scientific freeze with machine paths sanitized; its stored hash refers to the original record.

Raw JSONL, W&B internals, caches and execution logs are intentionally excluded. The complete raw result tree remains in the project archive and was checksum-verified against Garuda before this package was prepared.

## Results

### Behaviour and answer alphabets

| Measure | Value |
| --- | ---: |
| A/B/C/D restricted accuracy | 91.21% |
| Accuracy by correct position A/B/C/D | 90.23% / 91.02% / 91.80% / 91.80% |
| 1/2/3/4 restricted accuracy | 78.32% |
| Q/Z/R/X restricted accuracy | 85.45% |
| Worst position × alphabet cell | 70.31% |
| Blank / shuffled image accuracy | 28.81% / 29.30% |
| Clean − shuffled accuracy | 61.91 percentage points |
| Global top token is a label: letters / numbers / rare | 100.00% / 86.82% / 99.12% |
| Full-vocabulary mass on four labels: letters / numbers / rare | 100.00% / 85.57% / 98.52% |

### Core-7 findings

- **M1 — residual patching:** position transfer is layer 29 in both phases; semantic-content transfer is layer 32 in discovery and 31 in confirmation. The frozen position-before-content order replicates.
- **M2 — attention versus MLP:** attention peaks at layer 24 with normalized recovery 0.3647/0.3525 (discovery/confirmation), versus MLP peaks at layer 31 of 0.0457/0.0491. Attention dominance replicates.
- **M3 — heads:** the categorical sparsity verdict changes from `distributed` to `sparse`. Four frozen heads recover 52.43% of the confirmation all-head effect; all tested top-k sets beat matched random controls. Treat this as a non-replicating sparsity hypothesis, not a failed run.
- **M4/M5/M7 — readouts and symbols:** the full layerwise answer-label, content-word and arbitrary-alphabet evidence is included. Letter labels become readable at stage 25, before the later causal content transfer.
- **M6 — token route:** for content-only contrasts, image-token recovery peaks at layer 4 (1.1347/1.0658) and option-token recovery at layer 22 (0.2102/0.2373), consistent with an image-to-option-to-final-token route.

## Held-out confirmation

| Frozen check | Discovery | Confirmation | Result |
| --- | --- | --- | --- |
| Dominant component | attention | attention | replicated |
| Position-transfer layer | 29 | 29 | replicated |
| Content-transfer layer | 32 | 31 | replicated within tolerance |
| Position before content | yes | yes | replicated |
| Head sparsity | distributed | sparse | **not replicated** |

The first four preregistered conclusions replicate. The overall report flag is false only because the categorical M3 sparsity verdict crosses the 50% threshold on held-out data.

## Raw evidence volume

Discovery contains 7,680 behavioural rows, 113,664 lens rows, 178,532 patching rows, 136,440 component-patching rows, 9,216 component ablations, 12,288 head-screen rows, 11,712 head-validation rows and 1,408 head ablations. Confirmation contains 179,408 patching rows, 137,376 component-patching rows, 9,216 component ablations, 11,712 head-validation rows and 1,408 head ablations.
