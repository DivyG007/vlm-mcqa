# Curated stretch-result snapshot for report writing

Snapshot date: **2026-10-31**. This directory is the compact,
versionable result layer for Rachit's E19/E20/E22/E23/E24 work. It complements
the full Core-7 packages in `../core7_release_results/`. Raw JSONL, stdout,
stderr, checkpoints and large Ada result trees remain in ignored local storage
and on Ada; they must not be replaced by this summary.

Use the status words literally:

- **complete**: discovery/confirmation and required audits exist;
- **complete sensitivity**: a declared design modification was necessary and
  the conclusion is restricted to that modified design;
- **insufficient**: the preregistered sample or control gate did not pass; this
  is a valid outcome, not a missing result;
- **running**: partial output must not be used as a final result;
- **not started**: no scientific result exists.

The plain-language, report-writer interpretation is in
`../../docs/ai_handoff/2026-10-30_report_writer_extended_core_handoff.md`.
The graph-ready scalar table is `report_metrics_2026-10-30.csv`.

## Qwen2.5-VL-7B

Detailed authority: `../../docs/rachit_qwen_e20_e24_results_2026-10-27.md`.
The complete ignored local mirror is `results/ada_qwen_stretch_20261027/` plus
`results/ada/qwen_closure_*`.

| Study | Status | Main held-out result | Defensible conclusion |
| --- | --- | --- | --- |
| E19 | complete descriptive screen | `truth_aligned_content_with_wrong_final_symbol` was eligible for causal follow-up; the wrong-content proxy remained descriptive-only | E19 selects an error stratum; it does not itself diagnose or repair a failure |
| E20 | complete | 24 pairs; layer-0 image restoration `23.9774`, 95% CI `[22.6430,25.1566]`, normalized recovery `0.999823`; selected four-head transfer `9.742` versus matched-random mean `0.292`; selected-head clamp removed `9.738` versus random `0.287` | Strong positive: early image state is sufficient for nearly full donor transfer, and the frozen four heads carry a large, selective part of that effect |
| E22 | complete bounded successor | Original correct-option-only patches were negative at every tested layer 16-26. Same-image all-option transfer selected layer 20; on 35 evaluable held-out failures the truth-margin gain was `0.1901`, CI `[0.1102,0.2743]`, and `12.9%` were corrected. Matched-correct mean change was near zero | Narrow positive for whole-option-set rescue, not evidence that one answer token is sufficient or that layer 20 is where the failure originated |
| E23 | complete sensitivity negative | Layer 8 remained selected. Top-4 and top-16 closure runs did not satisfy the held-out target-versus-matched-region specificity rule | The selected heads matter globally, but this test did not establish the proposed selective question/option-region routing edge |
| E24 | complete sensitivity; binding claim failed | Layer 0 selected. Top-4 minus random `0.7474`, CI `[0.5619,0.9340]`; content-minus-literal-copy change `-1.7878`, CI `[-2.0898,-1.5118]`; patched content-minus-literal preference `-2.0512`, CI `[-2.2821,-1.8180]` | Early panel content transfers and selected heads beat random heads, but donor-symbol copying dominates; clean content-to-recipient-symbol binding is not supported |

## InternVL3.5-8B

Checkpoint: `OpenGVLab/InternVL3_5-8B-HF`, revision
`741a7d03020411e666c6109218ab71e08151ef86`. Its own frozen Core-7 heads were
used; Qwen head indices were not copied numerically.

| Study | Status | Main result | Defensible conclusion / evidence path |
| --- | --- | --- | --- |
| E20 | complete | 24 held-out pairs / 96 directions. Image restoration `40.1590`, CI `[36.6807,43.9877]`; selected-head transfer `17.7209`, CI `[16.0922,19.4361]`; transfer minus matched-random `17.8199`, CI `[16.1387,19.5870]`; selected-head clamp attenuation `17.7280`, CI `[16.0959,19.4420]`; normalized image recovery `1.00024`, selected-head transfer `0.44022` | Strong positive, closely paralleling Qwen. Ada job `47236`; exact science artifacts under `/home2/rsm/anlp-vlm-mcqa/results/next3_e20_internvl35_8b_ae33d6f7159580a75fc78b782b98e577ec7cbbce_47236/science/`. Slurm exit 1 came only from an outer wrapper checking the wrong summary path after all scientific artifacts were written |
| E22 | complete execution; discovery insufficient | Reliability passed at `0.9154`, CI `[0.8659,0.9527]`; 34 failure scenes and 54 matched controls entered causal discovery. The run completed 9,308 rows, but layers 26, 30 and 32 all failed the label-validity and content-over-literal-copy gates. Their adjusted mean effects were `-0.01775`, `-0.00656` and `-0.08355`; no layer was frozen and confirmation was not launched | Valid negative/insufficient outcome: this intervention did not establish a reliable content-based repair layer. Job `47606`; compact evidence in `internvl35_8b_e22_47606/`, full raw output on Ada under `/home2/rsm/anlp-vlm-mcqa/results/next3_e22_internvl35_8b_0d3fc6daea7980e2748bf38565de0ea637f3610b_47606/` |
| E23 | complete, geometry-relaxed token-count sensitivity | Discovery selected layer 5. Confirmation: 24 pairs, accuracy `1.0`, image gap `0.7604`, 3,744/3,744 operations. Option-region specificity `0.04997`, CI `[0.01774,0.08268]`; question-region specificity `-0.01172`, CI `[-0.04297,0.02132]` | Positive for the option-region edge under the declared equal-token background-control sensitivity; not a proof of a complete circuit. Job `45795`; `/home2/divyg/anlp-vlm-mcqa/results/next3_e23_internvl35_8b_cb0fce20a92e1a8fb9fb0668628c92dd99084196_45795/` |
| E24 | complete | Discovery selected layer 0. Confirmation: 32 pairs, 31 correct/correct transfer units, accuracy `0.921875`, 9,090 rows. Top-4 minus random `2.14776`, CI `[1.69115,2.66871]`; content-minus-literal-copy change `-4.49269`, CI `[-5.11014,-3.95060]`; patched content-minus-literal preference `-4.66457`, CI `[-5.29461,-4.00750]` | Selected heads have a strong intervention effect but the binding pattern fails even more clearly than Qwen: literal-symbol copying dominates. Job `46896`; `/home2/divyg/anlp-vlm-mcqa/results/next3_e24_internvl35_8b_27dfaa55e6b1b4524dcf6de2fc9b902ad30f8c2a_46896/` |

## LLaVA-OneVision-7B-SI

Checkpoint: `llava-hf/llava-onevision-qwen2-7b-si-hf`, revision
`cae7503dce88887557f73db1882328d21d9a7e21`. Its own frozen Core-7 heads were
used.

| Study | Status | Main result | Defensible conclusion / evidence path |
| --- | --- | --- | --- |
| E20 | discovery complete; confirmation insufficient | Reduced discovery completed 20 control-complete pairs, 80 semantic directions and 160 controls with zero unmatched groups. Confirmation had only 14 control-complete pairs, below the fixed minimum 16 and requested 24 | No held-out E20 effect estimate. Report a valid sample-insufficiency outcome, not a negative causal effect. Job `44637`; `/home2/rsm/anlp-vlm-mcqa/results/next3_e20_llava_ov_7b_si_c40701c1b7ca279bb48cf3606a525636f6e3b544_44637/` |
| E22 | complete execution; baseline reliability insufficient | The bounded preparation retained 23 matched failure scenes and passed the sample-count gate, but final-hidden-state content reliability was only `0.7159`, CI `[0.6259,0.8021]`; the frozen reliability gate failed before patching | No option layer was tested or selected, so this is not a negative repair effect. Job `49272`; compact evidence in `llava_ov_7b_si_e22_49272/`, full raw output on Ada |
| E23 | complete, geometry-relaxed token-count sensitivity | Discovery selected layer 4. Confirmation: 24 pairs, accuracy `0.9167`, 3,744/3,744 operations. Question-region specificity `0.05225`, CI `[0.03141,0.07617]`; option-region specificity `-0.01514`, CI `[-0.02490,-0.00570]` | Positive for the question-region edge, in contrast to InternVL's option-region result. Job `45609`; `/home2/rsm/anlp-vlm-mcqa/results/next3_e23_llava_ov_7b_si_1de2c1c1c9d3c126041ef91d60891b2119a55e92_45609/` |
| E24 | running | At 15:05 IST, 8,062 discovery rows, 15 fully processed parent pairs and pair 16 started. These are partial rows, not an estimate | Job `47421`, gnode079. Wait for a finalized report/freeze/confirmation or an explicitly labelled deadline-bounded replacement |

## PaliGemma2-10B and IconQA

- All three PaliGemma family screens completed, but none passed the frozen full
  Core-7 eligibility criteria. PaliGemma2-10B was the strongest family member.
- A separately labelled correct-only reduced Core-7 route (32 discovery / 24
  confirmation) is implemented but not scientifically run. The prompt must be
  physically one line because this model treats a newline as prompt termination.
- The checkpoint-access/runtime work is prepared on `divyg-ada`; job `46605`
  remained intentionally held at this snapshot. No Pali E20/E22/E23/E24 result
  exists.
- IconQA has a completed CPU-only audit and prospective generalization plan,
  not a frozen model-eligible causal subset and not a VLM result. Do not present
  its candidate counts as usable causal pairs.

## Cross-model interpretation that is currently defensible

1. **Early visual state transfer is robust where tested.** Qwen and InternVL
   both show nearly complete layer-0 image restoration.
2. **Frozen Core-7 heads carry substantial effects, but that does not make them
   a complete binding circuit.** E20 head transfer/clamp is strong; E24 shows
   that head transfer can still favor literal donor-symbol copying.
3. **Whole-option structure matters.** Qwen E22's single-answer-token design
   remained negative after a broad layer search, while copying the complete
   option set produced a small held-out rescue effect. The same kind of
   content-based option repair did not pass discovery gates in InternVL.
4. **Spatial routing is model-dependent.** Under the declared sensitivity,
   InternVL supports an option-region edge and LLaVA a question-region edge;
   Qwen's analogous closure remained negative.
5. **Negative and insufficient outcomes are part of the result.** The Qwen and
   InternVL E24 binding criteria fail; LLaVA E20 has only 14 held-out pairs and
   LLaVA E22 fails its baseline-reliability gate;
   these must not be rewritten as implementation failures or hidden.
6. **Universality is not established.** Qwen, InternVL and LLaVA all use
   Qwen-lineage language backbones. PaliGemma is the first planned substantially
   different language-family replication.

## Do not make these claims

- A logit lens identifies where information is first computed.
- A useful patch site proves an uninterrupted causal path.
- The frozen heads are the only or minimal sufficient circuit.
- A negative E24 content-over-copy contrast means visual content is absent.
- LLaVA E20 provides a held-out negative effect; it provides insufficient
  sample coverage.
- A completed E22 preparation that stops at a reliability gate is a negative
  layer-patching result.
- Partial E24 rows from a running job are final results.
