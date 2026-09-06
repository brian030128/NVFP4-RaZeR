# Capturing complementary gains beyond FourOverSix

The success criterion is meaningful gain retention, not finding a test-optimal
configuration. First require a direct comparison with FourOverSix, keeping the
native Qwen3.8-27B revision, tokenizer, activation quantizer, sequence length,
quantization scope, and held-out data identical to the previous target study.

Two fresh calibration seeds, 20260912 and 20260913, use 64 fit and 16 separate
validation windows. Each weight tile's default is the canonical
`quant_nvfp4_4over6` result: E2M1 with per-16-element scale selection between
normalizing the maximum to 4 and 6. The alternative remains E0M3 at alpha=1,
uniform over each 8x64 type tile. Thus useful E2M1 scaling remains available
while the type selector can contribute complementary gains.

Use the frozen gradient-score eligibility, 0.1 predicted-NLL trust budget,
fit-loss backtracking, and independent validation rule. Test results do not
choose the budget, tiles, or preferred seed. Evaluate FourOverSix first, then
the resulting proposal, including a rejected proposal. Compare the original
NVFP4-based maps to this stronger baseline using their saved paired test losses.

Exported maps explicitly record the FourOverSix default and E0M3 alpha.
They must not be mistaken for the earlier alpha=1 E2M1 maps.

Job 329279 runs the two seeds sequentially on two H100s. All compute remains
in Slurm, with worker-local downloads and temporary caches.

The current gain-retention reference set is plain NVFP4, FourOverSix, ordinary
alpha=1 MSE type selection, and the two prior sparse alpha=1 maps. This is not
an exhaustive claim about every reordering, rotation, or possible configuration.
Those transformations can also change activation quantization and require
their own matched comparison before extending the claim.

Array 329289 repeats the frozen FourOverSix-plus-type procedure on Qwen3-4B,
Llama3.1-8B, Qwen3-8B, and Llama3.1-8B-Instruct, with calibration seed 20260912.
It follows the mechanism job and runs at most two one-H100 tasks. Summary job
329294 follows the entire array, runs regression checks, and generates the
combined report and explicitly qualified guarantee diagnostics.

Separate questions:

1. Do type changes improve the FourOverSix model?
2. Which earlier beneficial changes become redundant after FourOverSix?
3. Can one rule retain a specified fraction of gains from multiple configurations?
4. Under what assumptions can that retention receive a finite-sample guarantee?

The existing mean ± 2 SE checks are empirical validation, not universal or
distribution-free guarantees. A stricter certificate must explicitly state
the loss bound, sampling distribution, independence, and multiple comparisons.

## Mechanism ablations

After both primary seeds, job 329286 applies the original seed-20260909 map
and new seed-20260912 map to both NVFP4 and FourOverSix backgrounds. For each
map evaluate four prespecified interventions: the full map; only tiles covering
input channels 3968–4031 in 5120-input projections; the complementary tiles;
and randomized output-row groups preserving each projection/input-column count
(random seed 20260914). Use the eight reserved WikiText probe windows in the
third training-data segment, with hashes checked against the original split.
These are diagnostic ablations, not candidates used to tune the primary maps.

Collect quantized-input channel energy on those probe windows to test whether
the prominent region carries unusual activation energy. Replay the exported
FourOverSix map against direct installation over every text weight exactly.
Run the mathematical certificate tests in the same allocation. Dependencies
prevent overlap with the primary two-H100 job.
