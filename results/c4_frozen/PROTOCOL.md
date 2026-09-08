# Held-out C4 evaluation of the frozen seven-model rule

Declared before submitting this evaluation. This follow-up measures held-out
C4 loss for the already frozen maps; it does not select or recalibrate maps.
C4 is a calibration source and a previously inspected data family, so this
is held-out within-source evaluation, not new-domain confirmation.

## Fixed inputs and maps

Use the five map bundles from results/pooled_confirmation/model_332349_*
and the two from results/pooled_scale/model_332389_* (Qwen4B, Llama8B).
Verify the model revision, each source linear weight hash, quantizer source
hash, map shape and selected tile count against the original record. Save
map-file and report-file hashes before loading any C4 evaluation examples.
No score tensors, gradients, tile election, loss backtracking or acceptance
gates are used.

C4: allenai/c4 revision 1588ec454efa1a09f29cd18ddd04fe05fc8653a2,
file en/c4-validation.00001-of-00008.json.gz. In stream order take the first
256 distinct documents with at least 512 model-tokenizer tokens, excluding
exact text hashes from all 192 calibration documents for that model. One
512-token crop per document, using Python Random(20260928). Preserve text
hashes, offsets and token hashes. Eligibility and tokenization can change
which documents are used across models; comparisons are paired within model.
Exact hash exclusion is not a near-duplicate or pretraining contamination audit.

## Matched evaluation

Evaluate FourOverSix, pooled192, c4_64, mixed64 and weight_mse using the saved
maps. Every policy uses canonical weight candidates and one FP32 activation
factor per token with per-16-element E4M3 scales. Nonhead linear weights and
inputs are fake quantized; other operations remain native BF16. Same model
revision, tokenizer, eager attention, 512-token windows, and all 511 next-token
labels for all policies. Check bitwise prefix independence for baseline and
pooled192 using a suffix intervention on the first document.

Report exp(mean document NLL), paired pooled192-minus-baseline NLL mean +/-
2 standard errors, absolute and relative PPL change, and all control PPLs.
All windows have 511 scored labels, so this equals token-weighted PPL.
Retain every model result regardless of sign. Descriptive two-SE intervals
are not simultaneous confidence guarantees. No result-dependent stopping,
extra seeds, changed document counts or tuning. No new overall pass/fail
criterion: this is a measurement extension of the existing frozen method.
