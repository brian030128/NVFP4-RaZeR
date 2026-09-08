# Frozen-map causal activation audit

The pooled-source confirmation332349 improves all15 baseline comparisons but
fails its full comparator screen (8/15 point wins over C4-only; required9).
That failed criterion is retained. This audit does not retune source weights,
tile counts or the election rule to obtain one more comparator win.

It asks a different necessary question: do the already-frozen maps retain
their gains with a causal activation-scale convention? The earlier activation
FP32 tensor factor depends on all tokens in the teacher-forced window.

For activations only, compute the FP32 factor independently for each token
row, then perform the same per16-element E4M3 scale quantization and E2M1
FourOverSix election. Weight candidates and maps remain EXACTLY those from
332349. This adds a per-row FP32 factor and changes the scaling convention;
it is not claimed as an unchanged native CUDA interface or free kernel change.
Both the FourOverSix control and selected maps use this same row convention.

Reuse the identical recorded64-document confirmation inputs per domain and
model, replaying pinned dataset revisions, document hashes, offsets and token
hashes. No training, score recollection, recalibration, new map, validation
gate or configuration selection. This is a matched robustness audit on already
inspected examples, not a second untouched-domain confirmation.

Models: Llama1B, OPT350M, Qwen0.6B, Pythia1.4B, OLMo1B. Domains: literature,
science, government. Replay FourOverSix, weight-MSE, C4-only64, mixed64,
pooled192. Report all comparisons, including C4-only, without changing the
failed mixture-superiority conclusion from332349.

Before evaluation, verify the row quantizer exactly equals calling canonical
FourOverSix independently on each row and is independent of appended/changed
rows. For each real model, hold the first128 input tokens fixed and replace
the remaining384 with a fixed valid token. At the same512-token shape,
the first128 output-logit vectors must be bitwise identical under row scaling,
for both FourOverSix and pooled192. Also record the older window-scaling
response to this intervention; it does not enter any selection decision.

The causal-robustness screen requires >=0.01PPL gain over its matched baseline
in>=12/15 cells, no supported harm, exact real-model prefix independence, and
at least half of the earlier convention's baseline NLL gain retained in>=12
cells. Since all15 earlier gains had descriptive paired2SE support, all are
included in this retention check. This is a new mechanism audit, not a rescue
of the failed C4 comparison. Paired2SE is descriptive, not a universal risk
certificate. No native-kernel or generation-accuracy result is implied.
