# What Is the Price of a Model's Eye? Mechanistic Analysis of Multiple-Choice QA in Vision-Language Models

Divyanshu Giri, Rachit Mehta, Yashav Bhatnagar 

IIIT Hyderabad

WandB link - https://forge.coreweave.com/wandb/vlm-mcqa/smoke-testing

Git Repo - https://github.com/DivyG007/vlm-mcqa/



How does a vision-language model (VLM) answering a formatted multiple-choice
question move from visual evidence to the answer content, bind that content to
an option position, and finally emit the requested answer symbol? This
repository extends the text-only analysis of Wiegreffe et al. (2025, *Answer,
Assemble, Ace*) to VLMs. It contains a controlled counterfactual dataset,
CLEVR-MCQ-4, and a seven-method pipeline (Core-7) that combines behavioural
controls, logit-lens readouts, activation patching, attention/MLP component
tests and attention-head localization.

```text
image tokens ──► option-content tokens ──► final token ──► answer symbol
   (early)            (middle layers)         (late, attention-written)
```

## Findings so far (mid-submission, preliminary)

1. In correct predictions, late layers raise the logits of all answer labels
   together; a more selective signal then separates the correct label.
2. Patching supports a staged route: answer information is available in image
   tokens early, carried by option-content tokens in the middle layers, and
   written to the final token late.
3. The late write is attention-dominated. On Qwen2.5-VL-7B, four heads recover
   52.7% of the all-head effect on held-out pairs; on InternVL3.5-8B
   (preliminary) attention dominance replicates but the sparse-head verdict
   does not.
4. Common alphabets are promoted far more robustly than rare symbols, even when
   forced-choice accuracy stays high.

Numbers and figures: [`results/README.md`](results/README.md).

## Repository structure

```text
├── src/vlm_mcqa/
│   ├── clevr_mcq4/      CLEVR-MCQ-4 builder: scenes, questions, variants, contrasts, splits
│   ├── core7/           Core-7 pipeline: model adapters, hooks, calibration, M1–M7, freeze/confirm, CLI
│   ├── data/            synthetic Shapes generator (pilot)
│   ├── eval/            pilot behavioural, lens and causal runners
│   ├── interventions/   pilot patching utilities
│   ├── analysis/        pilot figures
│   └── tracking/        W&B logging
├── configs/             dataset, model, profile and pilot configurations
├── scripts/
│   ├── dataset/         download_clevr_scenes · render_pairs · verify_* (CLEVR-MCQ-4 build)
│   └── core7/           setup_environment · screen_models · run_core7[_preselected] ·
│                        merge_shards · compare_models · validate_screening
├── tests/               unit and integration tests (CPU only)
├── results/             summaries, tables and figures for the reported runs
└── docs/
    ├── dataset.md       CLEVR-MCQ-4 and Shapes construction
    ├── methods.md       model selection, calibration gates, the seven methods, metrics
    └── reproduce.md     commands for every step
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # add '.[models]' for GPU model runs
PYTHONPATH=src python -m unittest discover -s tests -v
```

Full instructions for building the dataset and running screening and Core-7
are in [`docs/reproduce.md`](docs/reproduce.md).

## Models and data

| | |
| --- | --- |
| Screened | 12 checkpoints: Qwen2-VL / Qwen2.5-VL, InternVL3.5, LLaVA-OneVision, PaliGemma / PaliGemma2 |
| Core-7 | Qwen2.5-VL-7B-Instruct (full), InternVL3.5-8B (preliminary) |
| Pilot | Qwen2.5-VL-3B-Instruct on synthetic Shapes |
| Dataset | CLEVR-MCQ-4 v3: 1,252 items, 17,328 prompt variants, 14,400 contrasts;


## References

- S. Wiegreffe, O. Tafjord, Y. Belinkov, H. Hajishirzi, A. Sabharwal. *Answer, Assemble, Ace: Understanding How LMs Answer Multiple Choice Questions.* ICLR 2025.
- J. Johnson et al. *CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning.* CVPR 2017.