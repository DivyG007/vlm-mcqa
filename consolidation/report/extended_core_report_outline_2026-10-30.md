# Extended Core report outline and figure plan

This is a drafting scaffold, not a frozen submission. Numerical authority is
`../stretch_results/README.md` and `../core7_release_results/`; interpretation
authority is `../../docs/ai_handoff/2026-10-30_report_writer_extended_core_handoff.md`.

## Suggested paper structure

### 1. Introduction

- Motivation: VLM accuracy can remain similar while the internal route from
  visual evidence to an output symbol differs.
- Core question: how does visual answer content become an option position and
  then the requested label?
- Contributions:
  1. audited full Core-7 comparison on Qwen, InternVL and LLaVA;
  2. four vision-specific causal studies, completed on Qwen and partially
     replicated on InternVL/LLaVA;
  3. explicit positive, negative and insufficient outcomes rather than only
     favorable interventions.

Do not finalize the abstract until the running E22/E24 jobs are reconciled.

### 2. Related work

Reuse the Mid report's organization: MCQA format sensitivity, answer-to-label
binding, VLM representation/patching studies, and activation-patching caveats.
Update citations only if time permits; do not silently copy unresolved Mid
citations.

### 3. Dataset, models and evidence rules

- CLEVR-MCQ-4 v3: 1,252 items, 17,328 variants, 14,400 directed contrasts,
  dataset hash `f5573e25f3aa2bf2`.
- Models: Qwen2.5-VL-7B, InternVL3.5-8B, LLaVA-OneVision-7B-SI; PaliGemma2-10B
  pending.
- Correct-only conditioning for causal interventions.
- Discovery choices frozen before disjoint confirmation.
- Pair/parent is the independent unit; repeated layouts are clustered.
- Vocabulary projections are observational; patching/ablation/edge blocking are
  causal only for the tested intervention.

### 4. Core-7 recap

Keep this concise because the Mid report already explains M1-M7. Include:

- behavior and image-control gap;
- residual/token-group route;
- attention versus MLP;
- frozen heads versus matched random heads;
- content/symbol lens ordering;
- cross-model differences and failed replication checks.

### 5. Vision-specific extension methodology

Use one table with four rows:

| Study | Question | Intervention | Main control | Main outcome |
| --- | --- | --- | --- | --- |
| E20 | Can donor image information be restored and carried by frozen heads? | layer-0 image restoration; same-site frozen-head transfer/clamp | layer-matched random head sets | donor-margin restoration and selective attenuation |
| E22 | Can a wrong-symbol failure be repaired? | option-state/head/final-state repair within the frozen failure stratum | matched already-correct scenes and donor classes | correction without literal-symbol copying or harm |
| E23 | Do selected heads use the relevant spatial region? | selected-edge blocking and region zeroing | equal-size/equal-token matched regions | target harm minus matched-region harm |
| E24 | Do heads transfer visual answer content or a literal symbol? | panel-state and frozen-head donor transfers | random heads, all-layer heads and symbol/content contrasts | semantic content preference versus donor-symbol copying |

Explain donor, recipient, patch, final query position and same-site head transfer
before reporting numbers.

### 6. Results by study

Preferred organization is by study, with three model columns. This exposes
replication and divergence more clearly than four separate model narratives.

#### 6.1 E20

- Qwen and InternVL strong positives.
- LLaVA discovery only; held-out confirmation insufficient.
- Conclude that early image restoration and frozen-head mediation replicate
  across two models, not all three.

#### 6.2 E22

- Qwen: single-option negative, all-option same-image narrow positive.
- InternVL: placeholder until job `47606` finishes.
- LLaVA: bounded preparation retained 23 matched failure scenes, but baseline
  content reliability was `0.7159` and failed the frozen gate; no layer was
  patched and no repair effect was estimated.
- Emphasize that a repair location is not necessarily the origin of failure.

#### 6.3 E23

- Qwen negative after top-16 closure.
- InternVL option-region positive.
- LLaVA question-region positive.
- Interpret as model-dependent routing, not one universal edge.

#### 6.4 E24

- Qwen and InternVL both show selected-head effects but fail clean semantic
  binding because literal-copy metrics are strongly negative.
- LLaVA placeholder until finalized.

### 7. Cross-model discussion

Organize into:

1. robust shared findings;
2. model-dependent findings;
3. hypotheses rejected or narrowed;
4. what PaliGemma/IconQA would add.

### 8. Limitations

- Same controlled synthetic dataset for causal work.
- Correct-only conditioning.
- Qwen-lineage language backbones for all three completed Core-7 models.
- Geometry controls had to be changed from translated-shape to equal-token
  background regions for InternVL/LLaVA; label them sensitivities.
- No edge-complete circuit claim.
- PaliGemma and IconQA model studies pending.
- LLaVA E20 sample insufficiency.

### 9. Conclusion

Freeze only after running outputs are reconciled. A safe current conclusion is
that the image-to-option-to-symbol account is useful at the level of causal
sites and component effects, but the exact spatial edge and content-to-symbol
binding mechanism are model-dependent and often fail stricter controls.

## Recommended figures

Use ordinary dot/bar/line charts with explicit axis definitions. Avoid dense
multi-axis graphics and unexplained phrases such as "target minus control."

### Figure 1: experimental schematic

One donor/recipient example showing image tokens, option-content spans, answer
labels and final query position. Draw arrows for E20-E24 and label what is
replaced versus unchanged.

### Figure 2: Core-7 cross-model route

Three aligned normalized-depth panels:

- final-token position/content recovery;
- image/question/option token-group recovery;
- attention versus MLP recovery.

Regenerate from the full audited packages. Do not reuse the historical
`core7_release_results/cross_model/` plots because they contain preliminary
InternVL.

### Figure 3: Core-7 head concentration

For each model, show top-k held-out recovery for k=1/2/4/8/16 with the matched
random 95th percentile and all-head reference. Annotate InternVL's discovery to
confirmation sparsity-category change.

### Figure 4: E20 intervention bars

For Qwen and InternVL, grouped bars/dots for:

1. image restoration;
2. selected four-head transfer;
3. matched-random four-head transfer;
4. selected-head clamp attenuation;
5. matched-random clamp attenuation.

Do not put LLaVA on the held-out-effect panel; show it separately as
"confirmation underfilled: 14 < 16."

### Figure 5: E22 layer search and rescue

- Panel A: Qwen single-option effect by layer 16-26, all nonpositive.
- Panel B: layer-20 all-option held-out margin change and correction rate.
- Add InternVL/LLaVA panels only when finalized.

### Figure 6: E23 spatial specificity

Two region categories (question and option), three models, showing the direct
specificity estimate with 95% interval. Use a zero line. Label Qwen as negative,
InternVL option-positive and LLaVA question-positive. In the caption define the
estimate in words: "how much more blocking the named region hurt than blocking
an equal-token comparison region."

### Figure 7: E24 content versus symbol binding

For Qwen and InternVL, three dots with intervals:

- selected heads minus random heads;
- improvement toward donor content relative to literal-symbol copying;
- final patched preference for donor content versus donor symbol.

The first is positive while the latter two are negative. This makes the mixed
result visually obvious. Add LLaVA only after its report is complete.

### Figure 8: completion/claim matrix

A compact model-by-study matrix using text labels rather than color alone:
strong positive, narrow positive, negative, insufficient, running, not started.

## Recommended tables

1. Model/checkpoint/layer/head/frozen-source table.
2. Core-7 behavior and image-control table.
3. Core-7 discovery/confirmation replication table.
4. Stretch study design table.
5. Final model-by-study result table with sample size and claim limit.
6. Pending work table for PaliGemma and IconQA.

## Drafting checklist

- Every number has a model, phase, independent-unit count and source.
- Every sensitivity is labelled in the heading or first sentence.
- Negative and insufficient outcomes remain visible.
- No partial running-job number appears in the abstract/conclusion.
- Figure captions define every plotted difference in ordinary language.
- Abstract and conclusion are updated only after final reconciliation.
