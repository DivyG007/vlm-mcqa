# Claim-level evidence summary

## Project question

The project asks how a VLM converts visual evidence and a textual question into
semantic answer content, binds that content to an option position, and finally
emits the requested answer symbol. The planned contribution is not another
accuracy leaderboard: it is a mechanistic comparison across two selected VLM
families using seven complementary observational and causal probes.

## Dataset and model selection

- The proposal named MMMU, but sub-10B models are not reliably correct enough
  on its heterogeneous expert-level questions for clean correct-to-correct
  causal contrasts. Wrong answers would mix visual/reasoning incapacity with
  answer-symbol binding failure.
- The final diagnostic dataset is **CLEVR-MCQ-4**, derived from official CLEVR
  scenes and controlled counterfactual pairs. It is not CLEVRER.
- Release inventory: 1,252 items and 17,328 prompt variants; 120 calibration,
  300 screening, 256 behaviour, 320 discovery members (160 pairs), and 256
  confirmation members (128 pairs). The release spans nine reasoning families
  and three answer alphabets (letters, numbers, rare letters).
- A 24-item smoke test covered 12 checkpoints across Qwen, InternVL,
  LLaVA-OneVision, and PaliGemma. It was a feasibility and adapter-calibration
  exercise, not a stable ranking or a hard selection boundary. The report now
  shows all 12 rows rather than only the four representatives.
- Provisional family winners were Qwen2.5-VL-7B, InternVL3.5-8B,
  LLaVA-OneVision-7B-SI, and PaliGemma2-10B. The two selected for full Core-7
  study were Qwen2.5-VL-7B and InternVL3.5-8B.

## Exploratory Qwen2.5-VL-3B pilot

- On the 64-pair synthetic shapes dataset, forced four-way accuracy was 100%
  for A/B/C/D and 1/2/3/4, 97.50% for Q/Z/R/X, and 95.94% for punctuation.
- Full-vocabulary behaviour was much less robust: the model's global top token
  was a valid label 100%, 100%, 69.69%, and 25.47% of the time, respectively;
  the four labels held 99.999%, 98.27%, 62.84%, and 30.83% of total vocabulary
  probability mass.
- All four A/B/C/D logits share a large late rise, consistent with a generic
  answer-label/output-format signal. The correct label then obtains an
  additional relative margin. This is a standing hypothesis about mechanism;
  the logit lens alone is not causal evidence.
- The A/B/C/D correct-label margin moved from -0.64 at residual stage 27 to
  4.50 at 28 and 12.31 at 32; final correct logit was 33.80 versus 24.16 mean
  incorrect. These are logits, not probabilities.
- In a content-vs-symbol lens, the correct symbol became four-way readable
  before the answer-content word. This locates vocabulary alignment, not the
  first computation of answer semantics.
- Exact patching made the pilot mechanistic rather than only descriptive:
  queried image-token regions carry answer-relevant state at layers 0--16,
  option-content tokens become the stronger carrier around layers 20--25,
  option-label effects appear around layer 26, and an attention-dominated write
  transfers the answer symbol to the final token at layer 27. A final-token
  semantic colour-word readout becomes transferable around layer 33. This is a
  causal sequence of useful sites, not yet an edge-complete circuit.

## Verified Qwen2.5-VL-7B Core-7 results

- Behaviour: A/B/C/D restricted accuracy 0.8369 over 256 items; position-wise
  accuracy 0.8672/0.8359/0.8320/0.8125; numbers 0.8027; rare letters 0.8223;
  worst 12-cell accuracy 0.7461. `logic_or` was weakest at 0.4474.
- Controls: blank-image 0.2852, shuffled-image 0.2959, and clean-vs-shuffled
  discrimination 0.5410 support real but incomplete image dependence.
- Method 1: frozen position transfer localized to layer 24 in discovery and
  confirmation; content transfer localized to layer 26/25. However, the
  random-pair control also rises late, so M1 alone does not prove a dedicated
  binding state.
- Method 2: attention patch recovery peaked at layer 24 (0.4406 discovery,
  0.4388 confirmation), while MLP peak recovery was about 0.10. At layer 24,
  attention ablation reduced the answer margin by about 4.95 logits versus
  0.59 for MLP ablation. Near the output, MLPs contribute strongly to the
  common label-logit rise, while attention is more discriminative.
- Method 3: the corrected pair-balanced rerun used all 64 semantic pairs per
  split. Confirmation top-k recovery was 0.1156, 0.2441, 0.4170, 0.6837, and
  0.8949 for k=1,2,4,8,16; every set beat its matched layer-controlled random
  p95. Four heads recover 52.7% of the all-head reference effect, supporting a
  sparse causal set under the predeclared criterion.
- Methods 4/5/7: the label lens becomes positive before the content-word lens;
  common alphabets are more robust than rare symbols. These are observational
  readouts and must not be phrased as the moment content is first computed.
- Method 6: token-group patching supports an early-image to mid-question/option
  to late-final-token route. Image recovery is approximately 1.01 at layers
  0--8 but 0.06 by layer 24; option-token recovery peaks around 0.43 at layer
  18. This is causal for the patched states and tested readout, not a complete
  circuit proof.

## Verified InternVL3.5-8B Core-7 results

- The full sequential workflow finished successfully on 2026-10-04. Both
  discovery and confirmation are complete, both calibrations passed, and the
  confirmation report records `execution_complete=true`.
- Behaviour: A/B/C/D restricted accuracy is 0.912; blank/shuffled-image
  accuracy is 0.288/0.293, producing a clean-minus-shuffled gap of 0.619.
- Method 1: position transfer remains at layer 29; content transfer moves from
  the frozen layer 32 to layer 31, within the predeclared tolerance. The
  position-before-content ordering therefore replicates.
- Method 2: confirmation attention recovery peaks at 0.3525 at layer 24,
  compared with the MLP peak of 0.0491 at layer 31.
- Method 3: four frozen heads recover 52.43% of the all-head effect and all
  tested top-k sets beat matched random controls. However, discovery is
  classified as distributed and confirmation as sparse, so the categorical
  sparsity hypothesis does not replicate.
- The full package supersedes the 32-pair preliminary package for final effect
  estimates. The earlier package remains valid as an independent preliminary
  replication and execution record.

## Current next work

1. Use the full Qwen and InternVL packages in the next paper/report version;
   preserve the submitted mid-semester report as a historical artifact.
2. Test path/edge continuity through the proposed image-token to option-content
   to final-token route. The current causal results localize useful sites but
   do not prove a complete circuit.
3. Generalize the frozen protocol beyond the current Qwen-lineage language
   backbones before making a family-wide mechanistic claim.

## Cross-model data comparability

- Qwen2.5-VL-7B and InternVL3.5-8B full workflows use the identical
  CLEVR-MCQ-4 release: `dataset_hash=f5573e25f3aa2bf2` and image-aware hash
  `eb7b79e56977bbaeed5fc981c5c8e09a4e26eaf8b1702283abe2d4bbfc718cbd`.
  Examples, image bytes, variants, contrasts and split membership are matched.
- Their source snapshots differ because InternVL required later adapter and
  pair-balanced-head fixes. This is retained as reproducibility provenance but
  is not a dataset difference and is not expected to dominate the scientific
  comparison.
