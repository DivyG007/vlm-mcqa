# Final Beamer presentation

This directory contains the 26-slide final presentation for the VLM MCQA project.

## Build

From the project root:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd \
  presentation/final_beamer/vlm_mcqa_final_presentation.tex
```

The delivery PDF is copied to `output/pdf/vlm_mcqa_final_presentation.pdf`.

## Narrative and timing

- Motivation, models, and datasets: slides 1--10, about 6 minutes
- Methods and Core-7: slides 11--15, about 5 minutes
- Tuned Lens, E20--E24, and IconQA: slides 16--23, about 7 minutes
- Synthesis, interpretation guardrails, and conclusions: slides 24--26, about 2 minutes

## Figure and result provenance

| Asset | Source |
|---|---|
| `clevr_pair_x.png`, `clevr_pair_y.png` | Consolidated report Figure 6; CLEVR scene 047521 counterfactual pair |
| `core7_route.png` | Consolidated report Core-7 held-out confirmation figure |
| `iconqa_item_38.png` | Native IconQA train item 38 from `data/external/iconqa/iconqa_data.zip` |
| `iconqa_heads.png` | Yashav branch controlled three-model IconQA synthesis |
| `tuned_qwen_correct.png`, `tuned_qwen_alphabet.png` | Divyanshu branch Tuned Lens `full_01` Qwen plots |
| `tuned_qwen_alphabet_rare.png` | Presentation crop of the rare-letter panel from `tuned_qwen_alphabet.png` |

The Core-7, E20--E24, behavior, and synthesis charts drawn directly in LaTeX use the audited values recorded in the consolidated final report and branch handoffs. LLaVA E24 is intentionally labelled **running**; no result is inferred for it. PaliGemma Tuned Lens is intentionally labelled **provisional**.

## Verification notes

- Remote branch tips audited on 30 October 2026: `main` (`ec9910e`), Yashav (`227eec6`), Divyanshu (`bd0b6a1`), and Rachit (`6bd2a57`).
- Selected tests passed on all four lines of work; missing local `scipy` and `torch` prevented a small number of environment-dependent tests from running.
- The compact IconQA manifest hashes verified successfully.
- The three older branch tips retain an old syntax error in `src/vlm_mcqa/stretch/assets.py`; the Rachit integration branch contains the fix and compiles cleanly.
- The PDF was rendered page-by-page and visually inspected at 150 DPI. The final LaTeX log contains no overfull boxes.
