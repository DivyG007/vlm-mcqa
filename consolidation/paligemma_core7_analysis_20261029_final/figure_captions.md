# Figure captions

All figures are available as PNG and SVG. Source records are the archived
PaliGemma campaign identified in REPORT.md. New intervals use 2,000 NumPy
PCG64 bootstrap draws, seed 20261027, except the explicitly retained original
M3 pair intervals. These are pointwise intervals, not simultaneous bands.

1. **Residual transfer (`01_residual_transfer`).** Final-token residual patches
   for position and semantic-content contrasts. Solid curves average within
   semantic pair; shading is the pair-bootstrap 95% interval. Dotted curves
   preserve the original directed-row means. Discovery/confirmation use 76/61
   pairs. The dashed 0.5 recovery threshold is operational. Near-threshold
   weighting changes move discovery's position crossing from L37 to L39;
   semantic-content intervals are wide because baseline differences can be
   small or negative. Full final-state transfer at L41 is an expected endpoint.
2. **Component transfer (`02_component_transfer`).** Attention (blue) and MLP
   (orange) final-token update patches on position contrasts, equal-pair means
   and 95% intervals, 76/61 pairs. Attention and MLP have distinct late peaks;
   peak selection is not accounted for in the pointwise intervals. Negative
   values, including the final MLP effect, are retained rather than clipped.
3. **Frozen head sets (`03_head_sets`).** Selected top-k sets on 96 directions
   across 96 discovery/95 confirmation pairs. Left: pair-mean normalized
   recovery with original 95% pair-bootstrap intervals; dotted lines show the
   empirical p95 of 20 layer-matched random-set means, not confidence bounds.
   Right: selected-set effect divided by all 64 heads in L28/L34/L39/L41,
   not all 672 network heads. No ratio interval was supplied. Two heads meet
   the frozen sparse rule in both phases. Ratios above one are allowed.
4. **Shared lens (`04_lens_readouts`).** Item-averaged mean label logit,
   correct-minus-best-incorrect label margin, and full-vocabulary mass on four
   displayed labels. Each scheme has 256 items × four layouts. Stage 0 is
   decoder input; stage 42 is final residual. The lens is untuned. High early
   absolute logits coexist with negligible label probability; curves do not
   locate first internal computation. No confirmation acquisition is implied.
5. **Content token groups (`05_token_groups`).** Equal-pair mean semantic-content
   recovery under image/question/options/labels residual-group swaps. Original
   76/61 main pair cohorts. Intervals are available in pair_reanalysis.csv but
   omitted from this overview for readability. Large/negative effects remain
   visible; selected peak intervals and denominator diagnostics must accompany
   quantitative interpretation. Group sizes differ; curves are not route edges.
6. **Behavior image controls (`06_behavior_controls`).** Restricted letter
   accuracy for clean, blank, and shuffled images, 256 items × four layouts per
   condition, with item-clustered 95% intervals. The dashed line is 25% chance.
   Paired clean-control difference intervals are in behavior_clustered.csv;
   comparing separate bar intervals is not a paired significance test.
7. **Joint-factorial outcomes (`07_factorial_outcomes`).** Original directed-row
   categorical fractions for content-and-position residual swaps: 420 discovery
   and 339 confirmation directions. Position-following, content-following,
   recipient-retaining (`none`), and other-label outcomes sum to one at each
   layer. No independence-based confidence bands are asserted. The full
   final-state swap yields donor-position following by construction.
8. **Alphabet and content diagnostics (`08_alphabets_content`).** Left:
   requested alphabet mass within the eight-label union of requested symbols
   and default letters, for numbers/rare letters. This is not full-vocabulary
   mass. Right: correct semantic first-token margin for three schemes. Curves
   average four layouts within each of 256 items; pointwise item intervals are
   supplied in lens_item_clustered.csv. Nonmonotonic competition and absent
   positive-lower-bound semantic margins limit strong stage/sequence claims.
