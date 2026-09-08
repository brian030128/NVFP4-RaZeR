# Frozen source-consensus and leave-source-out study

The preceding current-model update experiment improves both fitting losses
on OPT, but harms held-out math. This study tests whether source-specific
gradient alignment, rather than only stale gradients, explains the failure.

Collect CE and BF16-teacher-KL tile derivatives ONCE at the FourOverSix
baseline on one shared table:64 C4 documents (the previous shard3 recipe),
64 OpenWebMath documents and64 CodeParrot Python files, each512 tokens.
First distinct sufficiently long documents from the first lexicographic
training shard (OpenWebMath parquet, CodeParrot json.gz); fixed crop seed20260926 separately for each new
source. Resolve source revisions before scoring and record document/token
hashes. CodeParrot is codeparrot/codeparrot-clean. New sources contain no
GSM8K/MBPP/WikiText benchmark input deliberately selected for calibration.
Absence of unrecognized near-duplicate contamination is not claimed.

Same legal FourOverSix/E0M3-alpha1 candidates,8x64 weights, W4A4 scope.
For each source and both CE and KL compute directional mean+2SE for switching
a tile. A tile is eligible only when every included source/objective has a
negative score. Rank by the worst score and take at most256 tiles. The256
cap is retained from the preceding sparse study, not tuned here. No candidate
loss, backtracking, checkpoint selection or per-configuration calibration.

All maps reuse this ONE shared score table. Export all-source consensus and
pooled controls, plus three leave-source-out consensus maps and their paired
pooled controls. Nothing chooses which map to report for a test outcome.

PRIMARY test: withhold web-source scores when evaluating WikiText; withhold
math-source scores for GSM8K; withhold code-source scores for MBPP. Thus each
reported primary map was elected without the corresponding source family.
Frozen fresh Wiki test windows384:416 and math/code rows400:432;32 examples
each, maximum512tokens. Models Llama1B, OPT350M, Qwen0.6B. Other exported
maps are declared secondary, not fallbacks that can rescue a failed primary.

Require >=0.01PPL improvement over FourOverSix in>=7/9primary cells, no
supported harm, and lower point NLL than the paired pooled-source map and
weight-MSE in>=6/9 each. Audit actual fitting CE/KL on each included source
after maps freeze; both source losses must improve for every primary map.
This stronger audit cannot be inferred from a negative directional estimate.
Paired2SE is descriptive, not a simultaneous certificate over selected tiles.

The exact limited statement: if a direction has negative TRUE directional
derivatives for every source risk, it has a negative derivative for every
convex mixture of these risks. This follows by linearity. Estimated gradients
with identity activation STE, finite tile flips and arbitrary unseen domains
do not inherit an unconditional guarantee. Source agreement and gradient
surgery are established ideas; novelty is not claimed from this elementary
convex-mixture observation. The primary benchmarks remain development data
families after prior study inspection; success requires additional untouched
model/domain confirmation under the unchanged rule.
