# InternVL3.5-8B E22 final evidence — Ada job 47606

Status: **complete execution; scientifically insufficient at discovery**.

The run finished successfully on `gnode090` at 2026-10-30 19:29 IST with
Slurm state `COMPLETED` and exit code `0:0`. It used the bounded 80-scene
discovery / 64-scene confirmation preparation, source commit
`0d3fc6daea7980e2748bf38565de0ea637f3610b`, the model's own frozen Core-7
heads, and candidate option layers 26, 30 and 32.

Preparation had already passed its reliability check (`0.9154`, 95% interval
`[0.8659, 0.9527]`) and retained 34 failure scenes with 54 matched correct
controls. The causal discovery sweep then wrote 9,308 rows and recorded 506
excluded attempts. All three candidate layers failed the prespecified gates:

| Layer | Mean failure improvement after accounting for harm to matched-correct cases | 95% interval | Why it could not be selected |
| --- | ---: | --- | --- |
| 26 | -0.01775 | [-0.13959, 0.09621] | Label-validity and content-over-literal-copy gates failed |
| 30 | -0.00656 | [-0.09987, 0.07277] | Label-validity and content-over-literal-copy gates failed |
| 32 | -0.08355 | [-0.17593, -0.00569] | Label-validity and content-over-literal-copy gates failed |

In plain language, copying the candidate option representation did not provide
a reliable content-based repair: it did not consistently preserve a valid
answer label and did not beat the simpler explanation that the intervention
was copying the donor's literal answer symbol. Therefore no option layer was
frozen and confirmation was correctly not launched. This is a valid negative /
insufficient scientific outcome, not an implementation failure and not evidence
that InternVL lacks the relevant information everywhere.

The full raw rows remain on Ada at:

`/home2/rsm/anlp-vlm-mcqa/results/next3_e22_internvl35_8b_0d3fc6daea7980e2748bf38565de0ea637f3610b_47606/`

This directory contains the compact final result, audit, summary, estimates and
artifact manifest needed for report writing. Do not describe this run as a
held-out confirmation result because the discovery gate intentionally stopped
the protocol before confirmation.
