# Domain transfer versus predictor failure

Frozen before these new evaluations. This study tests the prior heuristic;
it does not assume that task gradients are the complete mechanism.

Target: pinned Qwen/Qwen3.8-27B revision from the completed native probe,
native Transformers 5.16.1, same text-linear W4A4 scope. Calibration chooses
only E0M3/E2M1 at 8x64. Scales follow fixed candidate formulas: canonical
FourOverSix for E2M1 and alpha=1 for E0M3. No scale optimization, permutation,
or rotation is added.

Two runs pair the existing WikiText calibration seeds 20260912/20260913 with
new C4 calibration seeds 20260914/20260915. Each compares three maps:

* WikiText: the already frozen 64-window map and scores from that paired seed.
* C4: 64 C4 train windows, using the same score, trust budget, backtracking,
  and once-only independent validation check as before.
* Mixed: first 32 WikiText and first 32 C4 fit windows, with pooled per-window
  score moments. Fit budget is again 64 windows; validation uses first eight
  independent validation windows from each domain. The aggregate check is
  unchanged. Per-domain checks are reported but do not tune the proposal.

WikiText first-32 scores are recomputed; the old full-64 scores and masks are
reused only after source-model, quantizer, fit-token hashes and baseline losses
match. C4 scores are computed in two 32-window chunks and pooled exactly from
sample moments. Independent single-domain and mixed proposals are all evaluated,
including any rejected candidate; rejected exports fall back to FourOverSix.

C4 calibration streams distinct documents from train shard 00000, shuffled
with the run's seed and a 1000-document buffer. One random 2048-token window
is drawn per qualifying document; 64 fit, 16 validation, and eight diagnostic
probe documents are disjoint. The dataset revision and document/token hashes
are recorded. No C4 validation document is used for calibration.

New evaluation: all nonoverlapping 2048-token windows of the official
WikiText-2 raw validation split (previously unused here), and 256 distinct
documents from C4 validation shard 00001 (old tests used shard 00000), sampled
with fixed seed 20260916. Evaluation is identical across both calibration runs.
No new held-out result selects a seed, map, threshold, or intervention.

Before held-out evaluation, perform isolated single-tile interventions on
the eight reserved WikiText and eight C4 training probes. Select up to four
tiles from each prespecified category: most negative WikiText score; most
negative C4 score; negative WikiText but positive C4 score; negative C4 but
positive WikiText score. Deduplicate their union. Cross-domain categories
require two-SE stable signs on both domains. This is a diagnostic sample,
not an unbiased estimate of all-tile prediction accuracy. Record each tile's
predicted domain effect and measured paired probe effect, with no map updates.

Interpretation: reliable within-domain C4 gains with weaker cross-domain
transfer support distribution dependence. Failed within-domain predictions,
or wrong-sign isolated interventions, implicate surrogate error. Mixed gains
on both untouched domains support complementary correction, but do not prove
a universal mechanism. No C4 gains from these particular proposals cannot
establish that all possible type maps lack gains.

All numerical mean +/- 2 SE intervals are descriptive; these windows are not
automatically IID certification samples. No universal guarantee is claimed.
All compute, data sampling, tests, and summaries run in Slurm on at most two
H100s, using worker-local caches.
