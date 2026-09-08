# Independent confirmation of the pooled-source control

The source-consensus PRIMARY study332332 failed its comparison with pooling.
That failure remains. Its secondary pooled-source control improved all9
development cells. This new, independently frozen test evaluates that existing
control, without modifying its scoring rule, cap, candidates or calibration
data recipe in response to any confirmation result.

Primary map: shared64 C4 +64 OpenWebMath +64 CodeParrot sequences,512tokens
each; pool the once-collected per-sequence CE/KL scores; rank by
max(mean CE+2SE CE, mean KL+2SE KL), take at most256 negative-scoring tiles.
All candidates remain FourOverSix/E0M3 alpha1, legal8x64 weights; activation
FourOverSix and the native nonhead-linear W4A4 scope are unchanged. The output
head is identified by get_output_embeddings(), not a model-specific name.

Replay the already-frozen pooled maps for Llama1B, OPT350M, Qwen0.6B. Add two
new model families: EleutherAI/pythia-1.4b and allenai/OLMo-1B-hf. Resolve model
revisions before scoring. For each new model collect the same single shared
192-sequence score table and export all controls before loading test examples.
No fitting-loss gate, backtracking, selected seed, cap sweep or checkpoint
selection. All actual fitting-loss audits occur after map export.

Controls: FourOverSix; weight-MSE; C4-only64 scored sequences; mixed64 (first22
web,21 math,21 code); primary pooled192. The three task-scored maps use the
same pooled-score election code and256 cap. They reuse subsets of ONE shared
table; no separate calibration per configuration. Mixed64 vsC4-only64 is the
matched-token diversity ablation. Primary192 vsMixed64 measures extra data.

Three previously uninspected evaluation families (only cards/schemas read):
emozilla/pg19-test text; ccdv/arxiv-summarization document/article;
ccdv/govreport-summarization document/report. Use test splits, first64 distinct
documents with>=512 tokens, one512-token crop per document from a fresh
RNG seed20260925 for each model/domain. Resolve and record dataset revisions
before reading examples. Exclude exact document hashes found in any of the
192 calibration documents; record exclusions. No loss-dependent filtering.
Original pretraining contamination or unrecognized near-duplicates are not
ruled out by this check. These domains were reserved for a previous method
that failed and was never evaluated on them; their losses remain unseen.

Five models x three domains =15 primary cells. Require>=0.01PPL gain versus
FourOverSix in>=12 cells; no supported harm; lower point NLL than weight-MSE
and C4-only64 in>=9 cells each. Both mean fitting CE and mean fitting KL must
fall on each of the two new models. Paired document2SE intervals are
descriptive; preserve every result. These criteria are frozen before any
confirmation-model or confirmation-domain loss is inspected.

Report the Mixed64 ablation regardless of direction. Do not claim that
diversity helps if its matched-token comparison does not support that claim.
Primary success would establish a useful common calibrated procedure on this
panel, not a universal weight-only rule, superiority on every task, a new
gradient-selection invention, or a complete academic paper.

Current dynamic activation tensor scales depend on the whole teacher-forced
window. Report matched fake-quantized reference-text NLL/PPL; do not claim a
certified causal sequence likelihood, decoding accuracy or kernel speed.
These evaluation limitations require separate work even if this panel passes.
