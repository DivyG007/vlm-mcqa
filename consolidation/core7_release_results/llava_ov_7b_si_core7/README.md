# LLaVA-OneVision-7B-SI full Core-7 evidence

Audited 2026-10-19; campaign `20261016_124107`, execution source
`051ea21e538497a8137f0cdf94f73543d3fbf063`, freeze `e7b66bcb947ff29c`.
All five stages exited 0; both calibrations passed; exact raw counts/hashes,
pair coverage, raw-derived means and W&B publication were independently checked.

Read the [full results report](../../../docs/llava_core7_results_2026-10-19.md).

- Behavior accuracy: 65.33%; worst position 60.55%; clean-minus-shuffled gap
  38.48 percentage points, on the 256-item behavior split.
- Position transfer: layer 24 in both phases. Content: 24 → 23, within tolerance.
- **Ordering non-replication:** discovery `same_layer`, confirmation `content_first`.
- Attention dominance replicates; held-out peak recovery 0.6439 versus MLP0.09565.
- Sparse heads replicate: frozen top4 recovery0.5799, randomp950.02575,
  71.28% of the all-head reference within the four selected layers.
- `execution_complete=true`, `all_hypotheses_replicated=false`; distinguish
  successful execution from a fully positive scientific replication.

`discovery/` and `confirmation/` contain summaries, run/calibration records,
tables, figures and raw-file digests. `audit/` contains independent local and
W&B read-back audits. `export_manifest.json` records original/export hashes.
No raw JSONL, console logs, credentials or W&B working caches are committed.

Machine paths in JSON are sanitized to `<pkgpu-project-root>`. Stored freeze
hashes refer to the original unsanitized records, preserved in the full archive.
Archive SHA256: `607d076d52ab244d50503f6f0f1e1371c80eb6efabfa0b2b952684096d800d00`.
The raw mirror is `results/pkgpu_complete/llava_audit_20261019/` (ignored).

Do not infer a complete circuit or a unique position code from final-token
patching: the random-pair control also transfers strongly. M3's all-head
reference is limited to selected layers, and its intervals are pair clustered.
