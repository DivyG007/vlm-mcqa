# PaliGemma2-10B Core-7 analysis and handoff

The detailed standalone [report](../consolidation/paligemma_core7_analysis_20261029_final/REPORT.md)
is ready for later incorporation into the complete model analysis. It covers
all M1–M7 methods, behavior/error profiles, actual cohort sizes, uncertainty,
runtime/provenance, the failed final audit, and report-ready scientific prose.

The [integration handoff](../consolidation/paligemma_core7_analysis_20261029_final/INTEGRATION_HANDOFF.md)
identifies mergeable tables, preserved negatives and remaining comparison gaps.
The package contains eight PNG/SVG figures, 3,601 original source-referenced
metric rows and 1,192 additional equal-pair causal endpoints. It checks
1,056,258 saved raw rows and retains the original input archive unchanged.

Main results: 54.39% restricted letter accuracy; +26.37 pp clean-versus-shuffled
image advantage; late attention dominance and sparse heads replicate. Frozen
top-four confirmation recovery is 0.61835 [0.60675, 0.63026], 92.95% of the
effect of all 64 heads in four selected layers. This is not whole-network
effect or task accuracy. M3 covers 96 discovery/95 confirmation pairs;
main interventions use 76/61 pairs.

Original position-first ordering does not replicate: L37/L39 becomes L37/L37.
Equal-pair weighting changes discovery to L39/L39; semantic-content recovery
has many small or negative denominators. Pali's absolute mean-label logits
also differ from the Qwen common-late-rise pattern. These caveats are central
to the later synthesis, not footnotes to remove.

All method blocks completed, but the historical wrapper failed its final
artifact audit. Figures exist under `figures/`, while the auditor checks
basenames at phase root; its later heads-eligibility assumption also differs
from the actual schema. Saved data were checked locally without rewriting
the failure or running the model again. The prior three-model synthesis,
campaign code, source outputs and frozen decisions were not modified.

Numerical generator: `scripts/local/analyze_paligemma_core7.py`. Saved-evidence
checks and hashes are in the analysis package. No GPU/CPU model tests, commit,
push or release promotion were performed for this task.
