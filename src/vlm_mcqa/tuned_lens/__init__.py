"""Tuned Lens (learned per-stage affine translators) for the Core-7 VLMs.

Isolated from Core-7: imports its adapters, hooks, sessions and scoring
read-only and never changes their algorithms. Pipeline:

1. ``cache``    capture residual stages at frozen text fit-positions of the
                held-out CLEVR corpus (screening + calibration scenes);
2. ``train``    fit one translator per stage, one stage at a time, against the
                model's own final next-token distribution (KL(final || lens));
3. ``evaluate`` re-read the Core-7 M4/M5/M7 behavior-split rows with both the
                Logit Lens and the Tuned Lens.

Stage convention is Core-7's: stage ``s`` is the residual entering layer ``s``
(``s = 0`` is the embedding with image features); stage ``num_layers`` is the
final residual, whose lens equals the model logits (identity, no translator).
"""

SCHEMA = "vlm-mcqa-tuned-lens/v1"
