# Datasets

Two controlled datasets are used. Both are built so that a single factor
(visual answer, option position, or answer alphabet) can be changed while
everything else is held fixed, which is what activation patching needs.

## CLEVR-MCQ-4 (main dataset)

Four-option multiple-choice questions about CLEVR scenes
(Johnson et al., 2017), with minimally different image pairs.

### Construction

1. **Source scenes.** 3,000 official CLEVR v1.0 *train* images are streamed from
   the Hugging Face mirror `dpdl-benchmark/clevr` (revision
   `a9b3cc07eacabb93d57ceef72374427982104e00`) and matched to their official
   scene graphs by 3D object coordinates
   (`scripts/dataset/download_clevr_scenes.py`, verified by `verify_clevr_download.py`).
2. **Planning.** `python -m vlm_mcqa.clevr_mcq4.build plan` shuffles the scenes
   (seed `20260929`), keeps scenes with 4–10 objects, and selects 964 source
   scenes while balancing question families and answer priors. For each item it
   generates a question, its functional program (depth 2–5), the semantic
   answer, and three type-matched distractors. Original CLEVR questions are not
   reused. For pair items it also searches for a single scene edit that changes
   the answer to another listed option without moving any object.
3. **Rendering pairs.** Both members (`x`, `y`) of every pair are re-rendered
   from their scene graphs with Blender 4.5 Cycles (CUDA, 512 samples,
   denoising on, adaptive sampling off, zero camera/lamp jitter) by
   `scripts/dataset/render_pairs.py`, so the two images differ only in
   the edited attribute; `verify_pair_renders.py` checks every render against
   the frozen render profile. Single items use the downloaded official images.
4. **Finalizing.** `python -m vlm_mcqa.clevr_mcq4.build finalize` expands every
   item into prompt variants and directed contrasts and writes a manifest with
   SHA-256 hashes; `scripts/dataset/verify_release.py` additionally hashes
   every image into an image-aware release hash.

The question policy (family weights, depth, split sizes) is frozen in
[`configs/core7/clevr_mcq4_official3k.json`](../configs/core7/clevr_mcq4_official3k.json).

### Question families

`query_color_relate`, `query_relation`, `count`, `count_relate`,
`count_compare`, `same_attr_query`, `same_attr_count`, `logic_and`,
`logic_or`. There are no yes/no questions.

### Prompt and variants

```text
<question>
A. <option 1>
B. <option 2>
C. <option 3>
D. <option 4>
Respond with only the option label.
Answer:
```

Each item has 12 variants: the correct answer rotated to each of the 4
positions × 3 alphabets (`A/B/C/D`, `1/2/3/4`, `Q/Z/R/X`). Pair items add 4
distractor-permuted controls.

### Contrast types

| Contrast | What changes between donor and recipient |
| --- | --- |
| `position` | Same image and answer; the correct option moves to another position, so the answer symbol changes |
| `content` | The paired image changes the answer content; the correct position and symbol stay the same |
| `content_and_position` | Both answer content and position change |
| `identity` | Same image and answer under a distractor permutation (control) |
| `random_pair` | An unrelated donor item (baseline for specificity) |

### Splits

Splits are scene-disjoint and fixed before any model is run.

| Split | Use | Single items | Pairs | Items |
| --- | --- | ---: | ---: | ---: |
| calibration | difficulty calibration (not reported) | 120 | – | 120 |
| screening | model screening | 300 | – | 300 |
| behavior | behaviour and logit-lens methods | 256 | – | 256 |
| discovery | choose layers/heads, then freeze | – | 160 | 320 |
| confirmation | held-out test of frozen choices | – | 128 | 256 |
| **total** | | **676** | **288** | **1,252** |

The release has 17,328 prompt variants and 14,400 directed contrasts.
Identifiers: `dataset_hash = f5573e25f3aa2bf2`, image-aware release hash
`eb7b79e56977bbaeed5fc981c5c8e09a4e26eaf8b1702283abe2d4bbfc718cbd`.

**Limitation.** The 676 official single images and the 576 Blender-4.5 pair
images differ slightly in rendering appearance. Within each pair, both images
share the same renderer.

## Synthetic Shapes (pilot dataset)

Used for the exploratory Qwen2.5-VL-3B study. Generated deterministically by
`python -m vlm_mcqa.data.generate_shapes` (64 pairs, seed 42, 448×448 px);
see [`configs/experiment_00_shapes.json`](../configs/experiment_00_shapes.json).

- Four distinct shapes in four distinct colours in a fixed 2×2 layout.
- Each pair swaps the colours of two objects, one of which is queried, so the
  answer changes for a visual reason while object identities, positions, the
  global colour histogram, image size, token count and prompt text stay fixed.
- Variants: cross-image pairs, every answer position, and four alphabets
  (`A/B/C/D`, `1/2/3/4`, `Q/Z/R/X`, `!/@/#/$`); 40 prompts per pair.

## Why not MMMU

The proposal planned to use MMMU (Yue et al., 2024). Sub-10B models are not
reliably correct on its expert-level questions, and causal contrasts require
inputs that the model answers correctly in both conditions; otherwise a failed
patch confounds missing domain knowledge with answer binding.
