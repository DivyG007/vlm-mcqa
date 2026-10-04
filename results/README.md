# Results

Mid-submission results. Raw per-example records (`*.jsonl`) are not included;
the folders below hold the summaries, tables and figures derived from them.
Layer indices are zero-based. Absolute machine paths in provenance records
(`run.json`, `freeze.json`) have been replaced with repository-relative paths
(`data/`, `runs/`); each `freeze_hash` was computed on the original record.

| Folder | Model | Data | Status |
| --- | --- | --- | --- |
| [`screening/`](screening/) | 12 checkpoints, 4 families | CLEVR-MCQ-4 smoke build, 24 items | Feasibility screen |
| [`qwen25vl_3b_shapes_pilot/`](qwen25vl_3b_shapes_pilot/) | Qwen2.5-VL-3B-Instruct | Synthetic Shapes, 64 pairs | Exploratory, discovery split only |
| [`qwen25vl_7b_core7/`](qwen25vl_7b_core7/) | Qwen2.5-VL-7B-Instruct | CLEVR-MCQ-4 v3, 64 pairs per phase | Full Core-7, discovery + confirmation |
| [`internvl35_8b_core7/`](internvl35_8b_core7/) | InternVL3.5-8B | CLEVR-MCQ-4 v3, 64 pairs per phase | Full Core-7, discovery + confirmation |
| [`internvl35_8b_core7_prelim/`](internvl35_8b_core7_prelim/) | InternVL3.5-8B | CLEVR-MCQ-4 v3, 32 pairs per phase | Earlier preliminary run; retained for provenance |
| [`cross_model/`](cross_model/) | Qwen2.5-VL-7B vs InternVL3.5-8B | confirmation splits | Preliminary comparison |

## Screening

`screening_table.csv` / `.md` and `accuracy_vs_params.png`. The 24-item screen
was used for adapter calibration and provisional model choice only: no
checkpoint can meet the pair-yield gates at this size, so all family winners
are provisional. Provisional winners: Qwen2.5-VL-7B, InternVL3.5-8B,
LLaVA-OneVision-7B-SI and PaliGemma2-10B; Qwen2.5-VL-7B and InternVL3.5-8B
were taken forward to Core-7.

## Qwen2.5-VL-3B pilot (Shapes)

| Alphabet | Restricted acc. (%) | Global top is a label (%) | Label mass (%) |
| --- | ---: | ---: | ---: |
| A/B/C/D | 100.00 | 100.00 | 99.999 |
| 1/2/3/4 | 100.00 | 100.00 | 98.27 |
| Q/Z/R/X | 97.50 | 69.69 | 62.84 |
| !/@/#/$ | 95.94 | 25.47 | 30.83 |

- All four label logits rise together late in the network; the correct label
  then separates (`label_logit_decomposition.png`). The correct-label margin
  goes from −0.64 at stage 27 to 4.50 at stage 28 and 12.31 at stage 32.
- Patching (32 discovery pairs) localizes a sequence of causally useful sites:
  queried image tokens (layers 0–16) → option-content tokens (≈ 20–25) →
  option labels (26) → final token (27), where the write is attention-dominated
  (attention recovery 0.68 vs MLP 0.27). A final-token colour-word readout
  becomes transferable at layer 33 (`token_route_heatmap.png`,
  `factorial_*.png`).
- The answer symbol becomes readable in the lens before the answer-content
  word (`content_vs_symbol_lens.png`).

## Qwen2.5-VL-7B full Core-7

`run_config.json` (model, snapshot, commit, dataset hashes, profile),
`discovery/summary.json` and `confirmation/summary.json` (logged scalar
summaries per method), `freeze.json`, `confirmation/confirmation_report.json`,
`m3_heads.json` and `figures/`. M3 comes from a pair-balanced rerun (64 semantic
pairs per split) that supersedes the original run's M3; the superseded values
are omitted.

Behaviour split (256 items):

| Measure | Value |
| --- | ---: |
| A/B/C/D restricted accuracy | 0.837 |
| Accuracy by correct position | 0.867 / 0.836 / 0.832 / 0.813 |
| 1/2/3/4, Q/Z/R/X accuracy | 0.803, 0.822 |
| Worst of 12 position × alphabet cells | 0.746 |
| Blank / shuffled image accuracy | 0.285 / 0.296 |
| Clean − shuffled gap | 0.541 |

Mechanistic results (discovery → confirmation):

- **M1.** Position transfer at layer 24 (frozen 24, confirmed 24); content
  transfer at 26 (confirmed 25). The `random_pair` control also rises late, so
  M1 alone does not establish a position-specific state.
- **M2.** Attention patch recovery peaks at layer 24 (0.441 discovery, 0.439
  confirmation) versus ≈ 0.10 for MLPs. Ablating attention at layer 24 lowers
  the answer margin by 4.95 logits versus 0.59 for the MLP. Near the output,
  MLPs contribute strongly to the shared label rise.
- **M3.** Confirmation top-k recovery 0.116 / 0.244 / 0.417 / 0.684 / 0.895 for
  k = 1 / 2 / 4 / 8 / 16; every set beats its matched-random 95th percentile.
  Four heads recover 52.7% of the all-head effect → `sparse`.
- **M4/M5/M7.** The label lens becomes positive before the content-word lens;
  common alphabets are promoted more robustly than rare ones.
- **M6.** Image-token recovery ≈ 1.01 at layers 0–8 and 0.06 by layer 24;
  option-content recovery peaks ≈ 0.43 at layer 18.


## InternVL3.5-8B full Core-7

The full run uses 64 semantic pairs per phase and all seven methods. The curated folder contains provenance, summaries, 12 tables and 16 figures; raw per-example JSONL and machine logs remain outside Git.

| Frozen check | Discovery | Confirmation | Replicated |
| --- | --- | --- | --- |
| Dominant component | attention | attention | yes |
| Position-transfer layer | 29 | 29 | yes |
| Content-transfer layer | 32 | 31 | yes, within tolerance |
| Position before content | yes | yes | yes |
| Head verdict | distributed | sparse | **no** |

- **Behaviour:** A/B/C/D accuracy is 91.2%; blank/shuffled-image accuracy is 28.8%/29.3%, and the clean-minus-shuffled gap is 61.9 points.
- **M1:** Position transfer stays at layer 29; content transfer changes only from layer 32 to 31, so position-before-content replicates.
- **M2:** Attention peaks at layer 24 (0.365 → 0.352), far above the layer-31 MLP peak (0.046 → 0.049).
- **M3:** Four heads recover 52.43% of the confirmation all-head effect, but the categorical sparsity verdict does not replicate.
- **M4/M5/M7:** The complete answer-label, content-word and arbitrary-symbol logit-lens evidence is included.
- **M6:** Content-only image-token recovery peaks at layer 4 and option-token recovery at layer 22.

Full details and provenance: [`internvl35_8b_core7/README.md`](internvl35_8b_core7/README.md).

## Earlier InternVL3.5-8B preliminary Core-7

This 32-pair run is superseded by the full package above, but is retained as an independent early replication and engineering record.

32 pair-balanced cases per contrast in each phase; `freeze.json`,
`confirmation/confirmation_report.json`, per-method summaries, tables and
figures for both phases.

| Frozen check | Discovery | Confirmation | Replicated |
| --- | --- | --- | --- |
| Dominant component | attention | attention | yes |
| Position-transfer layer | 29 | 29 | yes |
| Content-transfer layer | 31 | 31 | yes |
| Position before content | yes | yes | yes |
| Head verdict | distributed | sparse | **no** |

- Attention peak recovery at layer 24: 0.342 / 0.350; MLP peak (layer 31)
  0.039 / 0.046.
- Top-4 heads recover 49.9% (discovery) and 50.9% (confirmation) of the
  all-head effect, on either side of the 50% threshold; every top-k set beats
  its matched-random 95th percentile.
- On confirmation, image-token recovery peaks at 1.03 (layer 4) and
  option-content recovery at 0.29 (layer 22); 90.6% of `content_and_position`
  cases follow the donor position at layer 31 versus 3.1% following donor
  content.

The lens methods (M4, M5, M7) are not part of this preliminary package.
