# Why reordering stops at the final MLP, and what to change

Written 2026-09-21 against `MIXFP4_REPORT.md` section 3-5. Every number below is
recomputed from artifacts already in the repository; no new model run produced
any figure in this document. The one new job is the CPU objective ablation
described in section 4, which reuses saved score shards and runs no forward pass.

## 1. The blocker is the search overfitting, not the layer depth

`results/task_reorder/cluster_20260919/fine_rows_v2_diagnosis.json` records, for
each of the 24 matrices in the Qwen layers 56-63 study, the ratio of the
**held-out election objective to the fit objective** of the same search:

| layer | gate_proj | up_proj | down_proj |
|---|---:|---:|---:|
| 56 | 0 | 0 | 0 |
| 57 | 0 | 0 | 0 |
| 58 | 0 | 0 | 0.00075 |
| 59 | 0 | 0 | 0 |
| 60 | 0 | 0.0000038 | 0 |
| 61 | 0 | 0 | 0 |
| 62 | 0 | 0.0171 | 0.0112 |
| **63** | 0.00048 | **0.0423** | **0.0584** |

Every matrix reaches a positive fit objective. The share that survives to the
disjoint election sequences is exactly zero below layer 62, and even the final
MLP keeps only 4-6%. Seventeen of the 24 matrices elected zero tiles on held-out
data. That is the mechanism: the layout search converts per-document score noise
into a confident-looking arrangement at every depth, and only at the last MLP is
the true effect large enough to leave a residue.

This reading is consistent with the depth diagnosis in
`results/task_reorder/llama_diagnosis_20260920/REPORT.md`, and it corrects a
tempting misreading of it. That report's 54.2% / 54.2% / 91.7% sign agreement at
layers 0 / 15 / 31 is often quoted as "early-layer scoring is unreliable". But the
frozen-residual control in the same table puts the true effect of the transferred
layout at **+0.000200 ± 0.000725 (layer 0)** and **-0.000268 ± 0.000507 (layer
15)** — statistically zero. Sign agreement on a null effect is 50% by
construction, so those two numbers carry no information about whether early-layer
scoring could work. The report says as much ("The early layouts were transferred
rather than optimized, so this does not demonstrate that only the last MLP can
benefit"); the headline should not say more than that.

What the depth table does establish is a **noise** problem: per-document ΔCE
spread for an early-layer perturbation is ±0.0077 (layer 0) and ±0.0065 (layer
15) against ±0.0017 at layer 31, because a tiny early weight change flips
activation-quantization codes throughout the suffix. Signal falls with depth and
noise rises. Both push the same way.

## 2. On switching to KL

**Measured, from the four saved Llama fresh-gate reports.** Those reports store
per-document CE and KL for all 64 documents, so this needed no GPU:

| candidate | ΔCE | SE | t | ΔKL | SE | t | ρ(ΔCE,ΔKL) |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint192 (passed) | -0.001675 | 0.000448 | **-3.74** | -0.001291 | 0.000137 | **-9.41** | 0.47 |
| tile_refine (rejected) | -0.000514 | 0.000394 | -1.31 | -0.000982 | 0.000141 | **-6.95** | 0.45 |
| gate_up (rejected) | -0.000531 | 0.000275 | -1.93 | -0.000197 | 0.000080 | -2.45 | 0.32 |
| first transfer (failed) | +0.005206 | 0.000886 | +5.88 | +0.004720 | 0.000626 | +7.55 | 0.59 |

KL detects effects of comparable magnitude with **3-5x smaller standard error**,
i.e. 9-25x fewer documents for the same power. `tile_refine` was rejected at
t = -1.31 on CE while sitting at t = -6.95 on KL. This is the expected
relationship: ΔCE reads the log-probability change at the one token that happened
to occur, while ΔKL averages the change over the teacher's full predictive
distribution at every position. KL is the Rao-Blackwellized reading of the same
perturbation, biased only by teacher-versus-data mismatch.

**The counter-evidence, which is decisive and must not be dropped.** Job 404650
measured the known-good fixed 3,787-tile 8x64 map — the map whose PPL we are
trying to recover — against raw256 on 64 development windows:

```
pooled CE  -0.004427  (SE 0.001374)   t = -3.22
pooled KL  -0.000189  (SE 0.000604)   t = -0.31      math KL  +0.0000638
```

The target improves CE clearly and KL not at all. A KL-primary gate would reject
it. Combined with ρ(ΔCE, ΔKL) of only 0.32-0.59, CE and KL are demonstrably not
measuring the same thing, and CE is the one aligned with the PPL objective.

**So do not swap KL in globally. Split it by role.**

| stage | decisions | binding constraint | instrument |
|---|---|---|---|
| arrangement search | ~10^5 coupled assignments | estimator noise | **KL** |
| format election | ~100 independent binary | alignment with PPL | CE (keep k=3 conjunction) |
| frozen fresh gate | 1 | alignment with PPL | CE primary, KL diagnostic (unchanged) |

The arrangement never changes which weights exist — it only changes which weights
share a type tile. That is a fidelity question, which is what KL measures, and it
is the stage where noise is the binding constraint. Election and gating decide
whether a change helps the actual task, which is what CE measures, and they make
few enough decisions to afford CE's variance. This keeps the frozen CE-primary
protocol and every prior failure intact.

Expected size of the effect: if the fit objective is noise-dominated, its
generalization ratio scales roughly as signal/noise. KL's 3-5x noise reduction at
~0.8x the effect size predicts a ~2.5-3x better ratio — layer 63's 4-6% to
perhaps 12-18%. **Necessary, not sufficient.** Sections 3 and 4 cover the rest.

## 3. Redesign proposals, ranked by value per GPU hour

### P1. Calibrate every search against a placebo (no GPU)

The fit objective is reported with no reference scale, so nobody can tell an
overfit layout from a real one before spending GPU on a finite-loss check. Run
the identical search on **balanced per-sequence sign-flipped scores**: exactly
half the sequences negated, which cancels any true atom mean while preserving
each sequence's internal covariance and each atom's spread. Whatever fit
objective the search reaches on that null is what it manufactures from noise.

Then require the real fit objective to beat its own null by a decisive margin
before any GPU is spent. This is `CLAUDE.md`'s "a rule that fires when it helps
loses; the same rule with a decisive margin wins" applied to the search itself
rather than to individual elections. Applied retroactively it would have screened
out most of the 404xxx series, where 17/24 matrices had a positive fit objective
and zero held-out election.

### P2. Restrict the hypothesis class to k-sparse permutations (no GPU)

The current pipeline searches a **full** row and column permutation, then compacts
afterwards — and compaction found that almost nothing needed to move: 39,934 → 5,001
rows and 27,216 → 1,392 columns. So the deployed layout lives in a tiny
neighbourhood of identity, but the search explores the full symmetric group to
find it. That is the worst of both: maximal selection bias during the search, and
a result that is nearly identity anyway.

Invert it. Search directly over layouts that relocate at most *m* rows and *m*
column groups from identity. The hypothesis class shrinks from N! to about
C(N,m)·m!, selection bias falls with it, and the output is by construction the
cheap deployment case the fusion work already supports. Sweep *m* and use P1 to
pick the largest *m* that still beats its null.

### P3. Shrink atom scores before searching (no GPU)

A 256x64 tile sums 1,024 atoms. Signal accumulates as 1024·m̄ but noise only as
√1024·s, so atoms whose mean is inside their own standard error contribute pure
noise — and the search actively selects on them. Positive-part James-Stein
shrinkage of each atom's mean, leaving the per-sequence deviations untouched,
erases those atoms without changing the noise the search must beat. Twenty lines,
zero cost. Implemented as `shrink_()` in `run_reorder_objective_ablation.py`.

### P4. Sketch the document axis so calibration can grow past 128 documents

The noise floor is set by 64 fit documents. Storage is why: per-document float32
CE+KL atoms cost ~64 bytes per weight, about 1 GiB per 4096x4096 matrix at 128
sequences, and `/work` has already been exhausted once by exactly this.

But the search and election only ever need the **mean and variance of linear
combinations** of atoms across documents. That does not require keeping every
document. Keep the per-atom mean plus a rank-S random ±1 sketch of the
document-axis residuals; S ≈ 32-64 gives unbiased variance estimates for any tile
sum, independent of how many documents were scored. That buys 1,000-document
calibration at the storage cost of about 64, cutting the estimator noise by ~4x —
which attacks the root cause directly rather than working around it. Costs one
calibration pass; no change to the search's interface.

### P5. Sequential back-to-front layer sweep with prefix caching

Part of why the scope is the final MLP is pure cost, not statistics: a final-MLP
trial replays only that MLP, the final norm and the head. But a change at layer L
cannot affect layers before L, so the prefix up to L can be cached once and shared
across every candidate at that layer, making a layer-L trial cost (32-L)/32 of a
forward. A full back-to-front sweep is therefore ≈16 full-model forwards per
candidate pass — affordable. Walk layers from the last backwards, freezing each
accepted layout, with one re-verification pass at the end because accepting at L
perturbs the inputs of L+1..31.

No previous experiment did this. The wider scopes that failed (last-four, last-eight,
the 96-matrix packing search, the 218-tile Fisher extension) searched many matrices
at once against a single pooled gate, which multiplies the selection bias in P1
across matrices instead of isolating it.

### P6. Spend the search budget where sensitivity says to

`k=3` is applied uniformly, so tile counts follow each layer's noise level rather
than where the gain is. Measure each layer's end-to-end sensitivity once — inject
calibrated random format perturbations at layer L and regress ΔCE on perturbation
norm, ~32 forwards total — and use it to rank layers before searching them. This
predicts which layers can repay a search, and gives the per-layer budget that the
uniform threshold currently lacks.

### P7. Orthogonal, and probably the cheapest real win: per-tile alpha on E2M1

This has nothing to do with reordering and should be evaluated separately.

About 99% of 256x64 tiles stay E2M1, and their block scale is
`alpha · block_max / grid_max`. FourOverSix is the two-point search
`alpha ∈ {1, 1.5}`. `CLAUDE.md` records the five-point `headx` set
`{1, 1.25, 1.5, 2, 3}` as **neutral-to-positive and never harmful** across three
models. Alpha only changes the value written into the ue4m3 scale field that
already exists, so it costs **no metadata, no kernel change, and no permutation,
at every layer**. The current MixFP4 baseline is FourOverSix, so this is
unclaimed.

Before acting on it, confirm against `../mixfp4/src/mixed_nvfp4_gemm_sm100.cu`
that the SM100 scale path really is agnostic to how the stored ue4m3 scale was
chosen. It should be, but the kernel is the authority, not this note.

### Do not retry

Already measured and failed, per `CLAUDE.md` and the 404xxx ledger: Hadamard
rotation at every scope tried, row permutation by E0M3 preference, Fisher /
logit-Gauss-Newton row grouping (404794, failed fresh CE), the cached CE/KL
disagreement penalty (404720, failed), MAE/Lp selection losses, `alpha < 1`
clipping, `corr<r>`, and calibration-free `diag(S)` proxies.

## 4. The first experiment, already submitted

`run_reorder_objective_ablation.py` measures which objective generalizes, using
only the surviving 1x16 score shards. It runs no model and no teacher.

Surviving shards, 128 sequences each:

```
/work/u4320956/task_reorder/transfer_20260920/llama8b/calibration/reorder_scores  layer 31 gate/up/down
/work/u4320956/task_reorder/pilot_20260919/qwen27b/calibration/reorder_scores     layer 63 gate/up/down
```

The layers 56-62 scores that produced the section 1 table are gone; only their
layouts and reports remain. Extending this to earlier layers needs a calibration
re-run, which is the point of doing the cheap version first.

Five variants, all implemented as input transforms so `quantize/task_reorder.py`
is byte-identical and the `ce_kl` arm reproduces the deployed rule exactly:

| variant | change | tests |
|---|---|---|
| `ce_kl` | none — deployed conjunction | P0 reproduction |
| `kl` | KL in both channels | P1 |
| `ce` | CE in both channels | P1 |
| `shrunk` | James-Stein on fit means only | P3 |
| `placebo` | balanced sign flip on both splits | P1 null |

Headline metric is `election_fit_ratio`, directly comparable to the section 1
table, alongside whether the searched layout beats the identity control on
held-out sequences at all.

```bash
sbatch --array=0-14 slurm/reorder_objective_ablation.sbatch \
  /work/u4320956/task_reorder/transfer_20260920/llama8b/calibration/reorder_scores \
  /work/u4320956/task_reorder/objective_ablation_20260921/llama8b
python summarize_reorder_objective_ablation.py --root <OUT_ROOT>
```

**What it can and cannot settle.** It compares surrogate objectives on held-out
*scores*. It is not model loss, not PPL, and not a substitute for the frozen
fresh-document gate. A variant winning here earns a finite-loss replay, nothing
more. If `kl` roughly triples the ratio and `placebo` lands near zero, sections 2
and 3 are on track and P2/P4/P5 are worth GPU. If `placebo` reaches a fit
objective close to the real arms, then the fit objective is uninformative at this
depth too and P1 becomes mandatory before anything else is searched.
