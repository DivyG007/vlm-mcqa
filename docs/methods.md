# Methods

All analyses operate on the language model of the VLM. Once an image is
encoded and projected into the language model's embedding space it is a
sequence of tokens in the same residual stream as the text, so the vision
encoder is treated as a black box and never intervened on.

## Model selection

Candidates (12 checkpoints, 4 families; [`configs/core7/models.json`](../configs/core7/models.json)):
Qwen2-VL-2B, Qwen2.5-VL-3B/7B; InternVL3.5-1B/2B/8B; LLaVA-OneVision-0.5B-SI,
-7B-SI, -7B-OV; PaliGemma-3B, PaliGemma2-3B/10B.

Each screen first passes the calibration gates below, then measures A/B/C/D
accuracy at every answer position, accuracy on the other two alphabets,
blank- and shuffled-image accuracy, output-format compliance, and the number of
counterfactual pairs answered correctly in both members. A family winner is the
smallest checkpoint (< 10B) with worst-position accuracy > 0.90, a
clean-minus-shuffled gap ≥ 0.30, and ≥ 64 / 48 usable discovery / confirmation
pairs. An exact tie on the primary score is broken by the 12-cell robustness
score (minimum over positions × alphabets, pass at ≥ 0.45). Thresholds are in
[`configs/core7/profiles.json`](../configs/core7/profiles.json) (`selection`).

## Calibration gates

No scientific run starts unless all of these pass on the loaded model:
inert hooks (capture does not change logits), identity patch, cached forward =
full forward, residual decomposition (`resid_pre + attn + mlp = resid_post`),
logit lens at the last layer = model logits, batched = unbatched, deterministic
replay, single-token answer symbols, and resolved prompt token spans.
Tolerances scale with dtype and logit magnitude.

## The seven methods (Core-7)

| ID | Question | Procedure | Evidence type |
| --- | --- | --- | --- |
| M1 | At which layer can the answer be transferred? | Patch the final-token residual stream from donor to recipient at every layer, for every contrast type, in both directions | Causal |
| M2 | Attention or MLP? | Patch attention and MLP outputs separately at every layer; resampling ablation; direct vocabulary projection of each component update | Causal (patching, ablation) and observational (projection) |
| M3 | Is the answer write sparse? | Screen every head in the four layers localized by M2; freeze the top candidates; test top-k sets (k = 1, 2, 4, 8, 16, 32) against size- and layer-matched random head sets | Causal |
| M4 | When do answer labels emerge? | Logit lens over the four displayed labels at every layer | Observational |
| M5 | Answer content vs answer symbol | Logit lens of the answer-content word (e.g. "red") versus the answer symbol | Observational |
| M6 | Where does information travel? | Patch token groups (image, question, option content, option labels) at every second layer; for `content_and_position` contrasts, record whether the prediction follows donor position, donor content, neither, or something else | Causal |
| M7 | Does the alphabet matter? | Logit lens of the requested alphabet versus default A/B/C/D for each alphabet | Observational |

Causal methods use only pairs that the model answers correctly in both the
donor and the recipient condition. M3 samples cases round-robin across semantic
pairs, and summary statistics are clustered by pair, so a pair with many
directed contrasts is not over-weighted.

### Discovery → freeze → confirmation

1. **Discovery** runs every method and chooses transfer layers, component
   peaks, head layers and ranked head candidates.
2. **Freeze** writes these decisions to `freeze.json` with a content hash.
3. **Confirmation** reruns M1–M3 and M6 on untouched confirmation pairs using
   only the frozen decisions.
4. **Confirm** reports which frozen decisions replicated, with predeclared
   tolerances (±1 layer for position transfer, ±2 for content transfer).

## Metrics

| Metric | Definition |
| --- | --- |
| Restricted accuracy | Argmax over the four displayed labels only |
| Global-top validity | Whether the full-vocabulary argmax is one of the four labels |
| Label mass | Full-vocabulary softmax probability summed over the four labels |
| Logit margin | Correct-label logit − best incorrect-label logit |
| Normalized recovery | (patched − recipient) / (donor − recipient) on the readout difference; 0 = no effect, 1 = full transfer; not clipped |
| Flip rate | Fraction of patched cases whose prediction becomes the donor answer |
| Transfer layer | First layer whose mean normalized recovery reaches 0.5 |
| Sparse-head verdict | `sparse` if some set of ≤ 4 heads beats the matched-random 95th percentile and recovers ≥ 50% of the all-head effect |

Layer indices are zero-based. A logit-lens "stage" *s* is the output of layer
*s − 1*; stage 0 is the embedding.

## Interpretation rules

- Logit-lens and projection results show when information is aligned with the
  output vocabulary; they do not show when it is first computed.
- A patching result is causal only for its source, target, site and metric. A
  sequence of causally useful sites is not an edge-complete circuit.
- Restricted four-way probabilities are not full-vocabulary probabilities.
- At the final token, `random_pair` transfers about as strongly as `position`,
  so M1 alone does not show a position-specific code; M6 is the content-vs-
  position instrument.

## Pilot (Qwen2.5-VL-3B, Shapes)

The pilot used the same logic with separate runners:
`vlm_mcqa.eval.qwen_shapes` (behaviour, full-vocabulary compliance, layerwise
lens for all alphabets) and `vlm_mcqa.eval.qwen_causal_abcd` (residual and
token-group patching, attention/MLP output patching, content-vs-position
factorial). Configurations: `configs/experiment_00_shapes.json`,
`experiment_03_causal_abcd.json`, `experiment_05_content_position.json`.
