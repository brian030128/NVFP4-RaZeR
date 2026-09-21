# Next experiments: proposed execution specifications

Status: **proposed, not frozen and not authorized for execution by this delivery**.
No new evaluation, model download, or bootstrap campaign was run for this document.
P0 is complete locally; publication remains blocked by repository READ permission.
Boundary/corruption and its near-versus-random control are complete and are not
pending work. Negative outcomes do not become unfinished experiments.

## Shared resolved configuration and launch prerequisites

All P1/P2/P3 studies below use weight-only N16K64 type blocks (16 output rows ×
64 input columns), FourOverSix-E2M1 versus E0M3 alpha=1, K16 UE4M3 scales,
non-head Linear weights, causal per-token FourOverSix activations, and
BF16-dequantized fake-quant evaluation. Exclude head, embeddings, norms and KV
cache. No reorder, new selector, native performance or broad k sweep is implied.
Student reference for calibration is all-FourOverSix; teacher is pristine BF16.
Use SDPA, eval mode and the historical quantizer implementation, not an
unverified replacement from a newer main.

| Model key | Exact model ID | Model and tokenizer revision |
|---|---|---|
| llama8b | meta-llama/Llama-3.1-8B | d04e592bb4f6aa9cfee91e2e20afa771667e1d4b |
| qwen4b | Qwen/Qwen3-4B | 1cfa9a7208912126459214e8b04321603b3df60c |
| mistral7b | mistralai/Mistral-7B-v0.3 | caa1feb0e54d415e2df31207e5f4e273e33509b1 |

Dataset inputs, from `software/primary/campaign/data.py`:

| Use | Repository | Revision | File/split |
|---|---|---|---|
| Calibration math | open-web-math/open-web-math | fde8ef8de2300f5e778f56261843dab89f230815 | data/train-00000-of-00114-5a023365406cb9c4.parquet, text |
| Calibration code | codeparrot/codeparrot-clean | 35a59fb025bc0a102f7d96eac09d145b896d487b | file-000000000001.json.gz, content |
| C4 evaluation | allenai/c4 | 1588ec454efa1a09f29cd18ddd04fe05fc8653a2 | en/c4-validation.00000-of-00008.json.gz |
| WikiText evaluation | Salesforce/wikitext | b08601e04326c79dfdd32d625aee71d232d685c3 | wikitext-2-raw-v1/test-00000-of-00001.parquet |
| Existing PG19 | emozilla/pg19-test | c5e39bf32e33f9111323aa68d7d9000d22722035 | test books in file order |

P1 calibration has exactly 64 math + 64 code sequences, 512 tokens each.
`seed0` for Llama/Qwen is the archived document-hash/token-offset manifest;
Mistral uses file order, document deduplication, >=512-token filtering and
`Random(20260926)` crop offsets. Draws 1–4 use SHA256(`drawN:document_sha`)
ordering, exclude seed0 and prior draws, and use SHA256(`drawN:offset:document_sha`)
modulo eligible crop starts. Each tokenizer's actual sample and token identities
are in [15 calibration manifests](campaigns/primary/calibration_manifests/).
Their original SHA-256 and the 45 natural-map hashes are resolved in
[CURRENT_EVIDENCE_AUDIT.json](validation/CURRENT_EVIDENCE_AUDIT.json).
Draw labels alone never authorize reuse. Do not silently rebuild seed0 differently.

No sequence packing across documents or padding to manufacture valid tokens:
use the pinned tokenizer's historical default special-token behavior and exact
512-token crops; next-token CE/KL uses 511 target positions. Evaluation uses the
original 2048-token windows (2047 targets), C4 256 sampled crops with
`Random(0)`, and concatenated WikiText `\n\n` text with the historical tail rule.
Verify each token-window hash and cluster identity before reuse. C4 documents
and Wiki articles are the inference units (231/53 for Llama/Qwen; 235/57 for
Mistral in the verified boundary arrays). Windows spanning articles retain the
historical assignment; changing assignment requires a separate declared analysis.

Execution prerequisites for every future campaign:

1. Create a new append-only root. Resolve all source/map/config/data hashes;
   retain failed, invalid and superseded attempts. Never write to sealed campaigns.
2. Freeze protocol, matrix, reuse ledger, code hashes, sample manifests,
   endpoint families and seeds before reading new outcomes. Proposed seed root
   for new analyses: 20260921; derive endpoint seeds from the first 64 bits of
   SHA256 of `20260921:<study>:<model>:<corpus>:<endpoint>`.
3. Audit evaluation/calibration overlap by document and token hashes. Missing
   hashes, tokenizer changes, non-finite moments or unresolved ownership stop
   the affected arm; no replacement samples chosen from outcomes.
4. Cap this N16 study at three physical GPUs, homogeneous A6000 preferred,
   never mixed with Ada in one run. Verify UUID/PID/owner before, every <=60 s,
   at phase boundaries and after runs; ambiguous/foreign usage fails closed.
   Site scheduling requirements remain applicable. The current task launches none.
5. Freeze a resource budget from existing valid-run durations before any launch.
   Missing approved budget is a launch blocker. Counts below are exact design
   budgets, not invented GPU-hour estimates. Do not repeat an expensive pilot.

Planning proxy from the three authoritative boundary A6000 runs (GPU-policy
audit): 8.3626/6.3130/8.5872 GPU-hours for 56 two-corpus policies on
Llama/Qwen/Mistral, respectively. Linear amortized evaluation estimates are
0.1493/0.1127/0.1533 GPU-hours per two-corpus policy. These are estimates from
a different map workload, not measured costs for new studies; startup, memory,
invalid retries and calibration can change them. P1-B natural missing-cell
ceiling is approximately 3.32 GPU-hours, matched-control ceiling 4.98; P2 full
evaluation approximately 3.99 plus unestimated calibration; P3 full matrix
approximately 5.82. Reuse lowers them. Freeze an actual contingency allowance
and calibration cost estimate before launch; no extra hardware lease follows
from this planning table. P1-A consumes zero GPU-hours.

New inferential analyses use token-weighted delta NLL, `exp(delta_NLL)-1`, paired
natural-cluster bootstrap B=10,000, percentile 95% CI and centered finite Monte
Carlo plus-one two-sided p-values. Share a bootstrap draw across policies in a
panel; CI and Holm-adjusted decisions are separate. Draws/corpora/models are
not interchangeable independent replicates. Preserve historical B=2000 output
as such; do not promote its zero p-values. Missing cells remain missing; no
imputation or reduced family chosen after observing results.

## P1-A — Existing five-draw multi-map analysis (CPU first)

Question: how much exact selection and module allocation recur across the
five draws, after accounting for sparsity? Hypothesis: observed common selection
and pairwise overlap exceed a density/quota-matched random reference. A null or
low overlap result is valid; high module rank similarity alone is insufficient.
This is retrospective descriptive analysis, not an untouched confirmatory test.

Matrix: 3 models × 5 conjunction maps = 15 maps; all 30 C4/Wiki PPL cells and
15 four-task accuracy panels already exist. Reuse all; no GPU evaluation.
Primary outputs: per-model frequency counts 0/5 through 5/5 on the common
eligible module/grid universe; 5/5 core; 1/5 draw-specific sets; all ten pairs'
Jaccard. Secondary: layer/module counts and shares, allocation Spearman,
L1/2 total variation between share vectors, PPL/delta-NLL and accuracy ranges.

Procedure:

1. Read the 15 conjunction hashes from the current audit and load maps with
   `campaign.mapio.read_map(expected_sha256=...)`; assert identical module order,
   shapes, eligible universe and tokenizer/model revision across each model.
2. Sum boolean masks over draws; retain zeros in the eligible universe. Check
   frequency-weighted counts equal total selections across all five maps.
3. Compute ten intersections/unions. Empty/empty Jaccard is 1 with an explicit
   empty flag; only one empty map gives 0. Constant-vector Spearman is null with
   reason, not zero or one. Report module shares as well as ranks.
4. For each module with universe U and counts Ka,Kb, use hypergeometric overlap
   `I ~ Hypergeom(U,Ka,Kb)`. Report expected intersection Ka*Kb/U; simulate 10,000
   deterministic quota-preserving random map sets for model-wide overlap
   reference intervals. Sum module intersections before computing Jaccard;
   the ratio of expectations is not the expected ratio. These are reference
   distributions, not ten independent experimental replicates.
5. Join the existing seed0 primary and V50 draw1–4 PPL/accuracy records using map
   hashes. Report all values, mean, median, sample SD and min/max, no five-draw
   population-tail claims. Only paired prediction files justify paired accuracy
   CIs; aggregate accuracy alone supports descriptive distributions.
6. Frequency versus same-draw score is mechanically coupled. If complete moments
   exist, for each held-out draw correlate the other four maps' frequency with
   that draw's CE mean, SE, U_CE, U_KL and margin, on the same eligible universe.
   Report five descriptive panels; they are dependent, not five independent
   validation trials. Missing moments block only this secondary analysis.

Implementation: add a CPU-only `analyze_draw_stability.py` in a new analysis
directory using `campaign.mapio`, NumPy and existing compact JSON; never change
the historical map reader. Its CLI contract is proposed:
`--input-audit FILE --primary-root DIR --out NEW_DIR --seed 20260921 --random-replicates 10000`.
It is **not implemented by this documentation task**. Acceptance fixtures must
cover empty sets, constant vectors, unequal universes and a known sparse random
case; do not replace the reference with independent Bernoulli draws changing quotas.

Artifacts: `INPUTS.json`, frequency CSV/JSON, ten-pair overlap table, per-module
allocation table, draw outcome table, optional held-out-score table, SVG/PNG
frequency/overlap panels, test report and hash manifest. No strong stability
threshold is preregistered: report distributions, not a post-hoc PASS cutoff.

## P1-B — Objective robustness across draws

Question: does the frozen conjunction change mean outcomes and observed draw
variation relative to CE-only and KL-only? Hypotheses may fail. CE+KL remains a
preselected operating rule even if it is not uniformly better or more stable.
Seed0 follow-up is complete; it is not cross-draw robustness evidence.

Natural-rule matrix: Llama/Qwen/Mistral × seed0/draw1–4 × three rules = **45 maps,
90 model/corpus cells**. All 45 maps already exist and were hash/header checked.
Reuse the 30 conjunction cells plus seed0 natural-objective cells only after
payload, window, model, activation and validity matching. At least 12 seed0
single-objective cells are represented by the completed follow-up; at most 48
cross-draw single-objective cells remain if no other exact evaluations match.
Primary selector-control outputs may provide further reuse: audit them first.
No calibration regeneration is needed for available complete verified moments.

For the paired 128 sequence observations per tile, mean=sum/n,
SE=sqrt(max((sum_sq-sum²/n)/(n-1),0)/n). CE and KL refer to the same sequences:
CE-only accepts mean_CE+3*SE_CE<0; KL-only analogously; conjunction accepts both.
Do not combine CE/KL probabilities or treat the upper scores as simultaneous CIs.
Zero SE uses the mean sign; non-finite scores fail closed before election.

Primary comparisons: conjunction-minus-CE and conjunction-minus-KL, within
model/draw/corpus. For each model/corpus, report equal-weight five-draw mean and
median delta NLL, observed worst delta, sample SD/range, number with delta>0,
and selected counts. Proposed practical regression threshold is +0.0025 delta
NLL (separate count from delta>0); freeze it before launch. Never label an
observed maximum a population tail estimate, or a five-draw 10th percentile a
robust risk bound. Accuracy is not a primary endpoint of this PPL study.

Proposed inference: bootstrap the shared evaluation clusters jointly across all
five draws, estimate the mean-over-these-five-draw contrast. This is conditional
on these maps, not an estimate over all possible calibrations. One natural-rule
family has 3 models × 2 corpora × 2 comparisons = 12 Holm tests. Individual-draw
tables are descriptive; no pooled raw-effect result dominated by Qwen. Standardize
within model/corpus only as a labeled sensitivity, with leave-Qwen displayed.

Matched-count stage (separate interpretation): use every conjunction module's
exact quota K. CE-ranked candidates must pass U_CE<0, sorted ascending U_CE;
KL-ranked must pass U_KL<0, sorted ascending U_KL; ties ascending flat tile index.
Random candidates are the union of the two pass sets, SHA256 keyed order,
without replacement. Follow `followup_maps.objective_maps` and its seed
20260917; use exactly one matched-random map per model/draw for continuity,
and label its randomness limitation. Use a draw-qualified model label in the
hash key except seed0, whose existing key is preserved for exact reuse.
No nonpassing tiles are allowed in the informed matched maps. Infeasible
quotas stop the affected design before outcomes; no silent support relaxation.

Matrix: 15 × (conjunction, CE-ranked, KL-ranked, random) = 60 maps / 120 cells.
Conjunction overlaps natural stage. Existing seed0 matched controls provide
18 cells besides the conjunction; absent further reuse, 72 new cross-draw
matched-control cells remain. Six pairwise policy contrasts averaged over the
five draws per panel form a distinct 36-test family (3×2×6), frozen separately
from the natural-rule family. Counts and module allocation are controlled;
this does not uniquely identify every mechanism called conservatism.

Steps: audit all stored maps/evaluations; adapt the pure CPU `objective_maps`
function to accept each manifest-resolved draw; emit immutable maps plus a
reuse ledger; generate an `evaluate_ppl` plan containing all unreused policies;
hash-freeze; run approved missing cells via the existing GPU-policy launcher;
retain paired cluster/token arrays; analyze all valid cells, not only favorable ones.
The draw-general plan adapter and approved runtime budget remain launch blockers.

Artifacts: exact maps, natural/matched plan files, quota and Jaccard audits,
map-to-calibration manifest, all attempt records, paired arrays, per-draw tables,
12/36-family results, regression-count and density figures, limitations report.
Support robustness only in the tested five maps and measured endpoints. A
universal conjunction-superiority claim requires evidence this design cannot supply.
Call a particular model/corpus mean contrast favorable only when its predeclared
adjusted test rejects and its estimate is negative; retain the unadjusted CI as
a separate field. Do not call conjunction universally more stable based on
descriptive SD/range or one random control. If the mean conditions fail, report
the estimates and retain the rule's frozen operating-point status.

Conditional k=2 study: launch only if a separately frozen practical deployment
decision requires that comparison after P1-B is complete and reviewed. Reuse
the existing post-hoc extension first; specify the missing endpoint and cost.
Keep it separate from objective inference, mark it post hoc, and do not expand
to a k grid. No trigger is currently satisfied by this TODO alone.

## P2 — Matched-budget composition × sampling

Proposed minimum: Qwen/Mistral × (Math64, Code64, Math32+Code32) × five draws
= **30 maps / 60 C4-Wiki cells**, each 64 sequences ×512 tokens. This differs
from P1's 128-sequence budget. Existing V51/V52 single-draw cells are completed
controls; reuse requires exact new sample identity, never merely the same label.

Question: how large are the fixed-composition contrasts relative to observed
within-composition draw variation? Hypothesis: fixed-budget composition changes
mean delta NLL. Null results are valid. Primary: all three pairwise composition
contrasts, averaged over five blocks, per model/corpus (12-test Holm family).
Secondary: within-composition SD/range, selected counts/module shares; descriptive
spread of the three means. Only three deliberately chosen compositions exist;
they are not a random sample of all calibration domains.

Sampling design is **paired source blocks**, not independent composition draws.
For each model and block b=1..5, create one 64-math and one 64-code document/crop
list using the pinned repositories above. Keyed order label
`composition64-v1:block<b>:<domain>`; keyed offsets use the same label plus
`:offset:<document_sha>`. Exclude evaluation matches and documents used in any
other new block. Filter <512-token texts; deduplicate by document hash, preserve
source field/file order before hashing, no extra undocumented language filter.
Math64 and Code64 use complete lists; mixture uses the first 32 of each list
under that outcome-blind ordering. Shared halves induce dependence and must be
reported. No cross-document packing; 32,768 input tokens and 32,704 target
positions per map. Pin actual tokenizer assets and save token SHA per sequence.

Steps: implement a new manifest-only paired-block sampler using `data.source_texts`
and the pinned tokenizer; run overlap checks before model scoring; write all 30
sample manifests; freeze map plans and inference; calibrate only missing blocks;
derive the three conjunctions using k=3; evaluate all cells in a common protocol.
Scoring may reuse per-sequence scores within a block only with proven identical
student/teacher state; aggregate moments separately for 64 and 32+32 sets.
Current `calibrate --n-per-domain` does not implement three compositions by
itself. A reviewed explicit-manifest adapter and budget are blockers, not a
reason to guess CLI flags or alter historical calibrate.py.

Analysis: paired evaluation-cluster bootstrap, shared across all cells, for the
five-block mean contrasts; also report five paired block contrasts separately.
Their variation is descriptive with n=5. Do not subtract noisy variances and
claim pure composition variance. No variance-component model is primary.
Optional normalized spread: common denominator is absolute mean N16 conjunction
gain of the existing P1 five draws for that model/corpus, fixed before outcomes;
report Range/denominator and sample SD/denominator of the three composition
means, alongside raw delta NLL. If denominator=0 or missing, normalized values
are null. This compares scales, not explained variance.

Stop for missing revisions, overlap, insufficient disjoint documents, invalid
score values or undefined budget. Artifacts: paired-block/sample/token manifests,
moments/maps, exact reuse matrix, cluster arrays, 12 corrected contrasts,
within-composition distributions, SVG/PNG plots and hash audit. Claims remain
limited to these two models, three compositions and five selected blocks.
Evidence for a particular composition difference requires its two-sided
Holm-adjusted p<.05, reported with direction and raw delta NLL; lack of rejection
does not prove equivalence. Do not select a composition for deployment using
these same evaluation outcomes without a separately frozen validation set.

## P3 — Optional good-map interpolation

Only justify this experiment if connectivity between existing maps is a real
research question after P1. It is not needed to finish the boundary campaign.
Different endpoint quotas make exact A, exact B and constant quotas jointly
impossible; do not specify all three.

Proposed feasible design: per model use seed0 versus draw1 conjunction maps,
chosen without outcome ranking. In stratum s choose quota min(K_A,K_B). Form
A' and B' by retaining the most favorable own-draw U=max(U_CE,U_KL) tiles,
stable flat-index ties. Empty quotas are retained as empty and reported. These
new endpoints have no inherited success status. First evaluate A' and B' versus
FourOverSix under a frozen endpoint non-inferiority gate: upper 95% delta-NLL
bound < log(1.005), Holm across 3×2×2=12 endpoints. If either endpoint fails,
stop that model's paths and report failure; no alternative pair search.

For passing models freeze common core A'∩B', equal-sized A'\B' and B'\A', and
four SHA256-keyed random swap paths using labels `interp:20260921:<model>:<s>:<path>:<side>`.
At fractions {0,.25,.5,.75,1}, swap floor(f*D_s+.5) in each stratum. This keeps
exact per-stratum counts, nested prefixes and exact A'/B' endpoints. Never reorder
swaps from observed intermediate PPL. Interior empty/duplicate maps are deduplicated
by payload but their planned labels remain in accounting.

Upper bound: 2 endpoints + 4×3 interiors = 14 maps/model, 42 maps / 84 cells,
minus exact reuse/duplicates. Two endpoints mean four cells/model. Primary path outcome
is each tested map minus FourOverSix; secondary is minus A'. Holm family covers
the planned 3 models × 4 paths × 3 interiors × 2 corpora = 72 interior tests;
absent models remain absent, not an outcome-selected smaller family. Pair cluster
draws across paths and fractions. Four paths are construction sensitivity, not
four model replications. A path adapter, endpoint manifests and budget must be
implemented/frozen before launch.

Required artifacts include endpoint-selection audit, removed tiles, common core,
all swaps, payload hashes, duplicate ledger, arrays, interval/adjusted-test tables
and path plots. Even all sampled fractions passing supports only the tested
maps on those paths; it does not establish a broad connected flat optimum.

## Supplementary downstream and long context

Already completed: seed0 eight-task confirmation on Mistral/OLMo2/Phi-4;
five-draw representative four-task accuracy on Llama/Qwen/Mistral; primary
GSM8K on Llama/Qwen/Mistral; PG19 4K and 8K on those three models. The primary
PG19 analysis has only four book clusters, not the later extension's proposed
20-book coverage. The extension downstream gate stopped its promotion.
These are results to interpret, not a new run list.

No new downstream experiment is specified until a precise remaining claim is
named and its power/budget frozen. PPL-selected best/median/worst draws would
be a purposive diagnostic subset, not unbiased all-draw accuracy robustness.
Never fabricate paired accuracy uncertainty from task aggregates.

## Command and implementation handoff

Existing safe inspection commands (from repository root):

```bash
python3 research/n16k64/tools/verify_public_snapshot.py
PYTHONDONTWRITEBYTECODE=1 python3 research/n16k64/tools/audit_current_evidence.py --workspace "$PWD"
PYTHONPATH="$PWD/research/n16k64/software/boundary/support:$PWD/research/n16k64/software/boundary:$PWD" \
  CUDA_VISIBLE_DEVICES='' python3 -m campaign.evaluate_ppl --help
```

Future evaluator CLI (template only; must be launched through validated wrapper,
not directly to bypass GPU ownership checks): `python -m campaign.evaluate_ppl
--model MODEL --plan FROZEN_PLAN --domains wiki,c4 --length 2048
--protocol-id aligned-primary --freeze ORIGINAL_FREEZE --freeze-sha256 ORIGINAL_SHA
--teacher none --attn sdpa --evaluation-stage full`. Map protocol identity must
match the frozen plan; never change original hashes to hashes of redacted copies.
The new campaign protocol is an additional seal. Keep token arrays enabled.
Plans list name, kind, map_path, map_policy, map_sha256, protocol_id, type_block
and expected_total_tiles; use completed `job_specs` as schema examples.

P1-A analysis, draw-general P1-B plan generation, P2 manifest adapter and P3 path
builder remain proposed code work. Their missing implementation is explicit;
the commands above do not claim those future studies already exist. Before any
GPU work, freeze their tests, endpoint families, resource estimate and input
access. Do not launch merely because an example command appears in this file.
