# vlm-mcqa

Mechanistic interpretability of multiple-choice vision-language models on the
CLEVR-MCQ-4 synthetic benchmark.

## Overview

We study how three frontier VLMs (Qwen2.5-VL-7B, InternVL2.5-8B,
LLaVA-OV-7B-SI) convert visual evidence into option-content states,
option-label states, and final answer symbols. The Core-7 experimental
pipeline applies seven complementary methods (behavioral, logit lens, spatial
patching, component projection, attention head identification, tuned lens, and
causal ablation) in discovery + confirmation phases across 72-item paired item
sets.

Stretch experiments E19–E24 extend the pipeline to three additional models and
two specialised tasks (semantic restoration, option rescue). The IconQA
transfer study checks whether the same mechanistic route holds when the visual
input switches from synthetic objects to icon-based quantity representations.

## Links

- **W&B project:** https://forge.coreweave.com/wandb/vlm-mcqa/projects
- **Model weights:** https://huggingface.co/DivyG007/vlm-mcqa
- **Final presentation:** https://docs.google.com/presentation/d/1WHqHC_upbVSNQdnXGuw89qG38uJydwbV/edit?usp=sharing

## Repository layout

```
consolidation/      curated result packages, evidence audits, and final report
configs/            campaign and experiment configuration files
docs/               runbooks, experiment summaries, and result write-ups
presentation/       final presentation slides and assets
scripts/            local analysis scripts and Ada HPC submission scripts
src/vlm_mcqa/      Python package (core-7 pipeline, stretch, campaigns, iconqa)
tests/              unit and integration tests
```

## Setup

```bash
pip install -e .
```

## Team

Rachit Mehta · Divyanshu Giri · Yashav Bhatnagar

IIIT Hyderabad / CVIT / KathaAI — Advanced NLP, Semester 5, 2026
