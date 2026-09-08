# Causal WikiText-2 evaluation of three frozen pooled maps

Declared before submission. Models: Qwen3.8-27B, Llama3.1-8B, Qwen3-4B.
Use the original map bundles in pooled_qwen27b/model_332840 (27B) and
pooled_scale/model_332389_{llama8b,qwen4b}. No new calibration, score pass,
map election, backtracking or validation acceptance. Verify model revisions,
source linear weight hashes, quantizer sources and map hashes against the
completed C4 records before evaluating WikiText.

Dataset: Salesforce/wikitext, wikitext-2-raw-v1, test split, revision
b08601e04326c79dfdd32d625aee71d232d685c3. Concatenate all rows in dataset order
with two newlines, tokenize with each pinned model tokenizer, and partition
from token0 into nonoverlapping512-token windows. Omit only the final incomplete
window; record its length. Score all511 next-token labels in every full window.
Record raw text and token hashes, window offsets and window token hashes.
Assert no nonempty WikiText row hash equals any of the192 calibration document
hashes; this exact row/document check does not establish near-duplicate or
pretraining-data independence. Do not drop test rows to obtain a favorable set.

All policies use simulated nonhead-text-linear W4A4 with per-token FP32
activation factors and per16-element E4M3 scales. Baseline is canonical
FourOverSix; alternate candidate is E0M3 alpha1 at8x64 weight tiles. Other
operations remain native. For27B use the pinned native hybrid architecture
with Transformers5.16.1; the other models use the existing4.57.3 environment.

Evaluate FourOverSix, pooled192, c4_64, mixed64 and weight_mse on identical
windows within each model. Check exact128-token prefix independence under a
suffix intervention for baseline and pooled192. Report PPL=exp(mean window
NLL), paired pooled192-minus-control mean NLL +/-2SE, absolute/relative PPL
changes, and all controls regardless of outcome. The two-SE intervals across
adjacent windows are descriptive and do not account for article dependence.
No result-dependent seed/window selection or stopping. Preserve the existing
inconclusive27B C4 result. WikiText is a previously inspected dataset family,
not a new untouched-domain confirmation.

Run the two smaller models with one H100 each, then the27B model with twoH100s
once both finish, keeping this experiment at most twoH100s concurrently. All
compute, tests, aggregation and table generation run through Slurm.
