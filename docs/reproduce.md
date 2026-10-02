# Reproducing the experiments

All commands run from the repository root. Generated data goes to `data/` and
run outputs to `runs/` (both git-ignored); `results/` holds only the curated
results reported in the paper.

## Scripts

| Script | Purpose |
| --- | --- |
| **Dataset** (`scripts/dataset/`) | |
| `download_clevr_scenes.py` | Download 3,000 official CLEVR train images and match each to its scene graph |
| `verify_clevr_download.py` | Check the downloaded images, scene graphs and hashes |
| `render_pairs.py` | Blender 4.5 script: render both members of every counterfactual pair |
| `relocate_render_jobs.py` | Rewrite render-job output paths when rendering on another machine |
| `verify_pair_renders.py` | Check every rendered pair against the frozen render profile; writes `pair_validation.json` |
| `verify_release.py` | Check the finished dataset and freeze the image-aware release hash |
| **Experiments** (`scripts/core7/`) | |
| `env.sh` | Shared paths and helpers, sourced by every shell script below |
| `setup_environment.sh` | Create a Python environment with a CUDA build of torch |
| `download_models.sh` | Prefetch model weights into `HF_HOME` |
| `validate_screening.py` | Fail-closed check of the dataset, screening runs and model selection before any run |
| `calibrate_difficulty.sh` | Optional difficulty check on the unreported calibration split |
| `screen_models.sh` | Screen all twelve candidates and select two models |
| `run_core7.sh` | Full Core-7 for a model chosen by the screen |
| `run_core7_preselected.sh` | Full Core-7 for a model fixed in advance (how the reported runs were made) |
| `merge_shards.py` | Merge per-method runs (patching, components, heads) executed on separate GPUs |
| `compare_models.sh` | Cross-model comparison of finished Core-7 runs |

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                    # data generation, analysis, tests
# GPU runs: install torch/torchvision for your CUDA version first, then
pip install -e '.[models]'
PYTHONPATH=src python -m unittest discover -s tests -v
```

Alternatively: `VENV=$PWD/.venv TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/core7/setup_environment.sh`.
The reported runs used torch 2.5.1+cu121, torchvision 0.20.1 and
transformers 4.57.6, in bfloat16 with SDPA attention.

Paths default to locations inside the repository (`scripts/core7/env.sh`).
Override them in the environment or in an untracked `scripts/core7/local.env`:

```bash
PYTHON=$PWD/.venv/bin/python
HF_HOME=/path/to/hf_cache        # default: ~/.cache/huggingface
DATA_ROOT=/path/to/data          # default: ./data
RESULTS_ROOT=/path/to/runs       # default: ./runs
```

PaliGemma checkpoints are gated: accept the licence on the Hugging Face Hub and
run `huggingface-cli login`. Choose GPUs with `CUDA_VISIBLE_DEVICES`. Runs log
to W&B unless `CORE7_NO_WANDB=1` (or `EXTRA_ARGS=--no-wandb`) is set; the local
run directory is always the authoritative record.

## 2. Build CLEVR-MCQ-4

Requires the official CLEVR v1.0 scene file (`CLEVR_train_scenes.json`), the
[`facebookresearch/clevr-dataset-gen`](https://github.com/facebookresearch/clevr-dataset-gen)
repository (for its Blender shape and material files) and Blender 4.5.

```bash
export PYTHONPATH=src
ACQ=data/clevr_official_3k
OUT=data/clevr_mcq4_v3

# 2.1 Download 3,000 official train images and match them to scene graphs
python scripts/dataset/download_clevr_scenes.py \
  --source-scenes /path/to/CLEVR_v1.0/scenes/CLEVR_train_scenes.json --out-dir $ACQ --limit 3000
python scripts/dataset/verify_clevr_download.py $ACQ

# 2.2 Plan questions, options, counterfactual edits and splits
python -m vlm_mcqa.clevr_mcq4.build plan \
  --scenes $ACQ/selected_scenes.json --images-dir $ACQ/images \
  --config configs/core7/clevr_mcq4_official3k.json --out-dir $OUT

# 2.3 Render both members of every pair (576 images)
blender --background --python scripts/dataset/render_pairs.py -- \
  --jobs $OUT/render_jobs.json --clevr-root /path/to/clevr-dataset-gen/image_generation \
  --backend CUDA --samples 512 --denoise        # add --device-name <substring> to pick a GPU
python scripts/dataset/verify_pair_renders.py --jobs $OUT/render_jobs.json --out-dir $OUT

# 2.4 Expand variants and contrasts, write the manifest, freeze the release hash
python -m vlm_mcqa.clevr_mcq4.build finalize --out-dir $OUT
python scripts/dataset/verify_release.py --dataset $OUT --acquisition $ACQ
```

`render_jobs.json` stores absolute output paths. To render on a different
machine, run `python scripts/dataset/relocate_render_jobs.py --input-jobs
render_jobs.json --out-dir <dataset dir on that machine>`, render with the
`render_jobs_local.json` it writes, and copy `images/` and `rendered_scenes/`
back before verification.

A correct build reproduces `dataset_hash = f5573e25f3aa2bf2`.

## 3. Screen the twelve candidates

```bash
bash scripts/core7/download_models.sh          # optional prefetch
bash scripts/core7/screen_models.sh            # all twelve, or name a subset
cat runs/screening/_selection/screening_table.md
```

`calibrate_difficulty.sh` optionally checks task difficulty on the unreported
120-item calibration split before screening.

## 4. Core-7

```bash
# One of the two models selected by the screen
bash scripts/core7/run_core7.sh <model_key>

# As run for the reported results (models fixed in advance)
CORE7_MODELS_CONFIG=configs/core7/qwen25vl_7b_pinned.json \
  bash scripts/core7/run_core7_preselected.sh qwen25vl_7b
CORE7_MODELS_CONFIG=configs/core7/internvl35_8b_pinned.json \
CORE7_DEVICE_MAP=balanced CORE7_REQUIRE_MULTI_GPU=1 \
  bash scripts/core7/run_core7_preselected.sh internvl35_8b
```

Both scripts validate the inputs with `validate_screening.py`, run discovery →
freeze → confirmation → confirm → plots, and refuse to overwrite an existing
run directory. The same steps by hand:

```bash
core7() { python -m vlm_mcqa.core7 "$@"; }
core7 run --model <key> --data $OUT --out <dir>/discovery --profile full_core7 --split discovery
core7 freeze --run <dir>/discovery --out <dir>/freeze.json
core7 run --model <key> --data $OUT --out <dir>/confirmation --profile full_core7 \
  --split confirmation --freeze <dir>/freeze.json --methods patching,components,heads
core7 confirm --run <dir>/confirmation --freeze <dir>/freeze.json
core7 plot --run <dir>/confirmation
```

**Parallel shards.** The InternVL3.5-8B preliminary results use
`--profile report_prelim_32` (32 pairs per phase), with `patching`,
`components` and `heads` run as separate jobs on separate GPUs
(`--methods <name>`; discovery heads also take `--components-from <components run>`).
Merge each split before `freeze` / `confirm`:

```bash
python scripts/core7/merge_shards.py --split discovery --out <dir>/discovery \
  --patching <patching run> --components <components run> --heads <heads run>
```

**Comparison.** `bash scripts/core7/compare_models.sh qwen25vl_7b internvl35_8b`.

Each run directory contains `run.json` (provenance: commit, host, GPU,
library versions, dataset hashes, model spec, profile, timings),
`calibration.json`, raw `*.jsonl` records, `*_summary.json`, `tables/*.csv`
and `figures/*.png`.

## 5. Pilot (Qwen2.5-VL-3B, Shapes)

```bash
export PYTHONPATH=src
python -m vlm_mcqa.data.generate_shapes --output-dir data/shapes_v0 --num-pairs 64 --seed 42 --image-size 448

# Behaviour, full-vocabulary compliance and layerwise lens for all alphabets
python -m vlm_mcqa.eval.qwen_shapes --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --manifest data/shapes_v0/manifest.jsonl --output-dir runs/pilot_shapes --max-pixels 50176
python -m vlm_mcqa.analysis.plot_shapes --predictions runs/pilot_shapes/predictions.jsonl \
  --layerwise runs/pilot_shapes/layerwise_label_logits.jsonl --output-dir runs/pilot_shapes/figures

# Causal patching; contrasts, modes, layers and token groups are listed in
# configs/experiment_03_causal_abcd.json and configs/experiment_05_content_position.json
python -m vlm_mcqa.eval.qwen_causal_abcd --help
python -m vlm_mcqa.analysis.plot_causal_abcd --summary <run>/summary.json --output-dir <run>/figures
```
