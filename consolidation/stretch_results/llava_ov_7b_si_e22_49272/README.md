# LLaVA E22 compact evidence: job 49272

Ada job `49272` completed successfully on `gnode090` in 48 minutes. The
deadline-bounded preparation considered 64 discovery scenes, retained 23
matched failure scenes, and therefore passed the minimum sample-count gate.
However, the final-hidden-state content proxy had reliability `0.7159` with a
scene-bootstrap 95% interval of `[0.6259, 0.8021]`, below the frozen reliability
requirement. The campaign therefore stopped before activation patching, layer
selection, or held-out confirmation.

Report this as **complete execution; baseline reliability insufficient**. It is
not evidence that LLaVA cannot be repaired, and it supplies no layer-specific
repair effect.

Files:

- `completion.json`: terminal campaign status;
- `result.json`: complete campaign-level result;
- `discovery_result.json`: discovery-stage insufficiency record;
- `discovery_preparation_report.json`: exact reliability and matching evidence;
- `artifact_manifest.json`: hashes for this compact package.

Full raw evidence remains on Ada at
`/home2/rsm/anlp-vlm-mcqa/results/next3_e22_llava_ov_7b_si_0d3fc6daea7980e2748bf38565de0ea637f3610b_49272/`.
