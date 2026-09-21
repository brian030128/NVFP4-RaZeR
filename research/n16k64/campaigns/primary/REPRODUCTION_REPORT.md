# Reproduction report (V20-V23)

Protocol freeze `df78f1fbbd034cb8d2bc8e6f6a264e7e221e1b5f2b2ff99a3fe821369c5ac88f`.

## Inputs

- Handoff ZIP `<USER_HOME>/mixfp4/research_artifacts/mixfp4_n16k64_top_tier_agent_handoff.zip` sha256 `b4ba1a4b1af25dfa1dfc5e07429760af7ae0974918c5b43a8a12670bd48c2ce4` (23815495 bytes); unchanged after extraction.
- `SHA256SUMS.txt` (CRLF line endings) verified 1650/1650 files; failures: [].
- Inventory: `runs/V00_inventory_attempt2/inventory/INVENTORY.json` (sha256 `f2f58d7ffeb3863b…`); archived score shards present: False; original cluster paths available: False.
- Models were fetched at the archived revisions (Llama-3.1-8B d04e592, Qwen3-4B 1cfa9a7, Qwen3.8-27B 1d4bf0f) and every matrix weight was checked against the archived per-matrix SHA-256 before use.
- Calibration crops (64 OpenWebMath + 64 CodeParrot, 512 tokens) and WikiText/C4 evaluation windows were rebuilt from pinned local files; their token SHA-256 lists equal the archived lists for all three development models.

## Environment differences from the archived jobs

| item | archived | this campaign |
|---|---|---|
| GPU | NVIDIA H100 / H200 (Slurm, gov113008) | RTX A6000 (Ampere sm86); RTX 6000 Ada for portability |
| scheduler | Slurm | campaign-local lease + Docker device cgroup (D01) |
| Python / torch | 3.11.11 / 2.9.0+cu128 | 3.11.11 / 2.9.0+cu128 (uv lock `env/hist.lock.txt`, `env/main.lock.txt`) |
| transformers | 4.57.3 (4B/8B), 5.16.1 (27B) | same, by environment |
| datasets | 4.8.5 (released reproduction) | 4.8.5; files read locally and token-hash verified |
| driver | not recorded | 565.57.01 (CUDA 12.7 API) |
| deviations | – | D-H1..D-H5 in `campaign/historical.py` (inputs from local files, local snapshot, BF16 teacher logits in RAM, no SLURM_JOB_ID, 27B streaming moments) |

## Score shards and N8 anchors (V20/V21)

Score manifest: `runs/V90_analyze_historical_attempt3/analysis_historical/SCORE_MANIFEST.json` (sha256 `4459c842d6bca34a…`). Anchor checks: `runs/V90_analyze_historical_attempt3/analysis_historical/N8_ANCHOR_RESULTS.json` (sha256 `ddff22e6f00c3995…`).

### llama8b (`V21_hist_calib_llama8b_attempt1`)

| check | value | passed |
|---|---|---|
| N8_k2_score_identity (tierA) | True | True |
| adaptive_maps_bitwise (tierA, count) | 2 / 10 | None |
| bf16_fit_nll_max_abs (tierB <= 0.002) | +0.004869 | False |
| bf16_fit_nll_mean_abs (tierB <= 0.0005) | +0.001189 | False |
| fit_kl_max_abs (tierB <= 0.02) | +0.056129 | False |
| fit_kl_mean_abs (tierB <= 0.005) | +0.005432 | False |
| fixed256_jaccard (tierB >= 0.8) | +0.630573 | False |
| fixed256_map_bitwise_vs_archived (tierA) | False | False |
| k3_count_relative (tierB <= 0.05) | -0.011659 | True |
| n256_prefix_equals_regenerated_fixed256 (internal) | True | True |
| w4a4_fit_ce_max_abs (tierB <= 0.05) | +0.054281 | False |
| w4a4_fit_ce_mean_abs (tierB <= 0.01) | +0.014954 | False |
| weight_mse_map_bitwise (tierA) | True | True |

Election counts regenerated vs archived: `{"k2": {"archived": 99024, "selected": 95214}, "k3": {"archived": 3345, "selected": 3306}, "k4": {"archived": 541, "selected": 535}, "k5": {"archived": 267, "selected": 270}, "k6": {"archived": 145, "selected": 143}, "n256": {"archived": 256, "selected": 256}}`.
Archived fixed-256 tiles in the regenerated k=2 ranking: `{"max": 13631326, "within_256": 198, "within_512": 211}`.
Same-GPU kernel perturbation (eager to SDPA, identical inputs/weights): `"not run"`.

### qwen27b (`V21_hist_calib_qwen27b_attempt1`)

| check | value | passed |
|---|---|---|
| N8_k2_score_identity (tierA) | not bitwise testable in streaming mode (float64 moments) | False |
| adaptive_maps_bitwise (tierA, count) | 0 / 10 | None |
| bf16_fit_nll_max_abs (tierB <= 0.002) | +0.008323 | False |
| bf16_fit_nll_mean_abs (tierB <= 0.0005) | +0.001317 | False |
| fit_kl_max_abs (tierB <= 0.02) | +0.028410 | False |
| fit_kl_mean_abs (tierB <= 0.005) | +0.004269 | True |
| fixed256_jaccard (tierB >= 0.8) | +0.450425 | False |
| fixed256_map_bitwise_vs_archived (tierA) | False | False |
| k3_count_relative (tierB <= 0.05) | -0.054954 | False |
| n256_prefix_equals_regenerated_fixed256 (internal) | True | True |
| w4a4_fit_ce_max_abs (tierB <= 0.05) | +0.053353 | False |
| w4a4_fit_ce_mean_abs (tierB <= 0.01) | +0.011015 | False |
| weight_mse_map_bitwise (tierA) | True | True |

Election counts regenerated vs archived: `{"k2": {"archived": 149033, "selected": 155512}, "k3": {"archived": 3785, "selected": 3577}, "k4": {"archived": 593, "selected": 432}, "k5": {"archived": 165, "selected": 126}, "k6": {"archived": 47, "selected": 34}, "n256": {"archived": 256, "selected": 256}}`.
Archived fixed-256 tiles in the regenerated k=2 ranking: `{"max": 47559326, "within_256": 159, "within_512": 181}`.
Same-GPU kernel perturbation (eager to SDPA, identical inputs/weights): `"not run"`.

### qwen4b (`V21_hist_calib_qwen4b_attempt1`)

| check | value | passed |
|---|---|---|
| N8_k2_score_identity (tierA) | True | True |
| adaptive_maps_bitwise (tierA, count) | 6 / 10 | None |
| bf16_fit_nll_max_abs (tierB <= 0.002) | +0.010792 | False |
| bf16_fit_nll_mean_abs (tierB <= 0.0005) | +0.002561 | False |
| fit_kl_max_abs (tierB <= 0.02) | +0.053124 | False |
| fit_kl_mean_abs (tierB <= 0.005) | +0.010200 | False |
| fixed256_jaccard (tierB >= 0.8) | +0.528358 | False |
| fixed256_map_bitwise_vs_archived (tierA) | False | False |
| k3_count_relative (tierB <= 0.05) | +0.068504 | False |
| n256_prefix_equals_regenerated_fixed256 (internal) | True | True |
| w4a4_fit_ce_max_abs (tierB <= 0.05) | +0.081666 | False |
| w4a4_fit_ce_mean_abs (tierB <= 0.01) | +0.019848 | False |
| weight_mse_map_bitwise (tierA) | True | True |

Election counts regenerated vs archived: `{"k2": {"archived": 66856, "selected": 64866}, "k3": {"archived": 7912, "selected": 8454}, "k4": {"archived": 1837, "selected": 2036}, "k5": {"archived": 576, "selected": 669}, "k6": {"archived": 222, "selected": 256}, "n256": {"archived": 256, "selected": 256}}`.
Archived fixed-256 tiles in the regenerated k=2 ranking: `{"max": 7095745, "within_256": 177, "within_512": 213}`.
Same-GPU kernel perturbation (eager to SDPA, identical inputs/weights): `{"bf16_fit_nll_vs_other": {"max_abs": 0.013623237609863281}, "fit_ce_vs_other": {"max_abs": 0.08753514289855957}, "fixed256_prefix_k2": {"intersection": 167, "jaccard": 0.48405797101449277}, "k2": {"intersection": 17642, "jaccard": 0.16190668940544128, "other": 64866, "this": 61740}, "k3": {"intersection": 3596, "jaccard": 0.29259559512138367, "other": 8454, "this": 7432}, "run": "V21_hist_calib_qwen4b"}`.

## PPL anchors (V22)

Source: `runs/V90_analyze_historical_attempt3/analysis_historical/HISTORICAL_PPL_ANCHORS.json` (sha256 `2beb2ed3f8045e2e…`).

| model | policy | dataset | archived PPL | regenerated PPL | rel. diff | window NLL max abs diff | tier B (<=0.5%) | tier C effect |
|---|---|---|---:|---:|---:|---:|---|---|
| llama8b | archived_fixed256 | c4 | 9.764387 | 9.782736 | +0.00188 | 0.1152 | True | True |
| llama8b | archived_fixed256 | wiki | 6.848383 | 6.851241 | +0.00042 | 0.0205 | True | True |
| llama8b | four_over_six | c4 | 9.823733 | 9.831496 | +0.00079 | 0.0672 | True | – |
| llama8b | four_over_six | wiki | 6.875525 | 6.877294 | +0.00026 | 0.0344 | True | – |
| llama8b | hist_n8_k2 | c4 | 9.842249 | 9.798667 | -0.00443 | 0.1373 | True | False |
| llama8b | hist_n8_k2 | wiki | 6.907813 | 6.873041 | -0.00503 | 0.0416 | False | False |
| llama8b | hist_n8_k3 | c4 | 9.773040 | 9.785801 | +0.00131 | 0.0674 | True | True |
| llama8b | hist_n8_k3 | wiki | 6.849275 | 6.852539 | +0.00048 | 0.0241 | True | True |
| llama8b | hist_n8_k4 | c4 | 9.777571 | 9.788438 | +0.00111 | 0.1999 | True | True |
| llama8b | hist_n8_k4 | wiki | 6.852014 | 6.852955 | +0.00014 | 0.0338 | True | True |
| llama8b | hist_n8_k5 | c4 | 9.795584 | 9.783451 | -0.00124 | 0.1187 | True | False |
| llama8b | hist_n8_k5 | wiki | 6.861062 | 6.858176 | -0.00042 | 0.0220 | True | True |
| llama8b | hist_n8_k6 | c4 | 9.800744 | 9.797669 | -0.00031 | 0.0949 | True | True |
| llama8b | hist_n8_k6 | wiki | 6.860541 | 6.860852 | +0.00005 | 0.0284 | True | True |
| llama8b | hist_n8_n256 | c4 | 9.764387 | 9.789117 | +0.00253 | 0.1831 | True | True |
| llama8b | hist_n8_n256 | wiki | 6.848383 | 6.847544 | -0.00012 | 0.0399 | True | True |
| qwen27b | archived_fixed256 | c4 | 10.167717 | 10.171121 | +0.00033 | 0.0220 | True | True |
| qwen27b | archived_fixed256 | wiki | 7.292438 | 7.289835 | -0.00036 | 0.0853 | True | False |
| qwen27b | four_over_six | c4 | 10.188365 | 10.184717 | -0.00036 | 0.0245 | True | – |
| qwen27b | four_over_six | wiki | 7.287076 | 7.301374 | +0.00196 | 0.1056 | True | – |
| qwen27b | hist_n8_k3 | c4 | 10.149866 | 10.151931 | +0.00020 | 0.0348 | True | True |
| qwen27b | hist_n8_k3 | wiki | 7.214750 | 7.235154 | +0.00283 | 0.0763 | True | True |
| qwen4b | archived_fixed256 | c4 | 16.615953 | 16.604809 | -0.00067 | 0.0391 | True | True |
| qwen4b | archived_fixed256 | wiki | 13.040957 | 13.012235 | -0.00220 | 0.0601 | True | True |
| qwen4b | four_over_six | c4 | 17.326633 | 17.313040 | -0.00078 | 0.0439 | True | – |
| qwen4b | four_over_six | wiki | 14.269062 | 14.217286 | -0.00363 | 0.0682 | True | – |
| qwen4b | hist_n8_k2 | c4 | 15.162447 | 15.165126 | +0.00018 | 0.0261 | True | True |
| qwen4b | hist_n8_k2 | wiki | 10.850905 | 10.827130 | -0.00219 | 0.0268 | True | True |
| qwen4b | hist_n8_k3 | c4 | 15.824034 | 15.782464 | -0.00263 | 0.0298 | True | True |
| qwen4b | hist_n8_k3 | wiki | 11.862908 | 11.805826 | -0.00481 | 0.0462 | True | True |
| qwen4b | hist_n8_k4 | c4 | 16.358000 | 16.358511 | +0.00003 | 0.0280 | True | True |
| qwen4b | hist_n8_k4 | wiki | 12.715484 | 12.750967 | +0.00279 | 0.0503 | True | True |
| qwen4b | hist_n8_k5 | c4 | 16.781254 | 16.745323 | -0.00214 | 0.0405 | True | True |
| qwen4b | hist_n8_k5 | wiki | 13.410134 | 13.366496 | -0.00325 | 0.0744 | True | True |
| qwen4b | hist_n8_k6 | c4 | 17.037529 | 16.966421 | -0.00417 | 0.0541 | True | True |
| qwen4b | hist_n8_k6 | wiki | 13.822810 | 13.727053 | -0.00693 | 0.0635 | False | True |
| qwen4b | hist_n8_n256 | c4 | 16.615953 | 16.587502 | -0.00171 | 0.0319 | True | True |
| qwen4b | hist_n8_n256 | wiki | 13.040957 | 12.992857 | -0.00369 | 0.0556 | True | True |

## Cross-GPU archived anchor (V23)

Source: `runs/V90_analyze_historical_attempt3/analysis_historical/CROSS_GPU_ARCHIVE.json` (sha256 `9991966c2349edb2…`).

```
{
 "a6000_run": "V22_hist_eval_qwen4b_attempt1",
 "ada_run": "V23_hist_eval_qwen4b_ada_attempt3",
 "gpus": {
  "a6000": [
   "NVIDIA RTX A6000"
  ],
  "ada": [
   "NVIDIA RTX 6000 Ada Generation"
  ]
 },
 "policies": {
  "archived_fixed256": {
   "c4": {
    "exact_equal": false,
    "ppl_a6000": 16.604808807373047,
    "ppl_ada": 16.612071990966797,
    "ppl_rel": 0.00043741446697809216,
    "window_nll_max_abs": 0.03334546089172363,
    "window_nll_mean_abs": 0.008558167610317469
   },
   "wiki": {
    "exact_equal": false,
    "ppl_a6000": 13.012234687805176,
    "ppl_ada": 13.060771942138672,
    "ppl_rel": 0.0037301244173673087,
    "window_nll_max_abs": 0.06860566139221191,
    "window_nll_mean_abs": 0.014963618696552433
   }
  },
  "four_over_six": {
   "c4": {
    "exact_equal": false,
    "ppl_a6000": 17.313039779663086,
    "ppl_ada": 17.331790924072266,
    "ppl_rel": 0.0010830648255777398,
    "window_nll_max_abs": 0.05499076843261719,
    "window_nll_mean_abs": 0.010718311183154583
   },
   "wiki": {
    "exact_equal": false,
    "ppl_a6000": 14.217286109924316,
    "ppl_ada": 14.224836349487305,
    "ppl_rel": 0.0005310605346626751,
    "window_nll_max_abs": 0.08366823196411133,
    "window_nll_mean_abs": 0.018280889073463334
   }
  },
  "hist_n8_k3": {
   "c4": {
    "exact_equal": false,
    "ppl_a6000": 15.782464027404785,
    "ppl_ada": 15.775300979614258,
    "ppl_rel": -0.0004538611827715755,
    "window_nll_max_abs": 0.03476870059967041,
    "window_nll_mean_abs": 0.007707879645749927
   },
   "wiki": {
    "exact_equal": false,
    "ppl_a6000": 11.805826187133789,
    "ppl_ada": 11.800321578979492,
    "ppl_rel": -0.0004662620020864372,
    "window_nll_max_abs": 0.04751253128051758,
    "window_nll_mean_abs": 0.010359108448028564
   }
  }
 }
}
```

## Findings

The archived N8 campaign reproduces **at the level of paired perplexity effects** and **not** at the level of individual
tile selections. The two statements come from different checks and must be reported separately.

**1. What reproduces.** With the archived calibration manifest, archived model/tokenizer revisions and the archived
(historical) evaluation protocol, the FourOverSix baseline and the archived N8 k=3 map land within the frozen tier-B
tolerance on all three development models and both corpora: relative PPL differences are +0.00026 (Wiki) / +0.00079 (C4)
for FourOverSix on Llama-3.1-8B, −0.00363 / −0.00078 on Qwen3-4B and +0.00196 / −0.00036 on Qwen3.8-27B, against the
frozen 0.5% tolerance (`ppl_rel_w4a4` = 0.005). The paired *effect* of the
regenerated N8 k=3 map reproduces the archived effect within tier C on every model×corpus, now including the third
development model: Llama-3.1-8B −0.00361 vs archived −0.00383 (Wiki) and −0.00466 vs −0.00517 (C4); Qwen3-4B −0.18587 vs
−0.18468 (Wiki) and −0.09256 vs −0.09071 (C4); Qwen3.8-27B −0.00911 vs −0.00997 (Wiki) and −0.00322 vs −0.00379 (C4).
Re-evaluating the *archived fixed-256 map file itself* reproduces its archived effect on Llama-3.1-8B and Qwen3-4B on
both corpora and on Qwen3.8-27B's C4. There is one exception, and it is a disagreement about the sign of approximately
zero rather than about a reproducible effect: on Qwen3.8-27B WikiText the archived effect is +0.00074 and the regenerated
−0.00158, so the sign flips and the difference (0.00232) exceeds tier C's absolute floor of 0.002 by 16%. That floor is
what binds, because the archived effect is far too small for the 25%-relative half of the frozen tier-C rule to apply;
both numbers sit inside the kernel-noise band quantified in section 3.
Two internal identities hold bitwise on Llama-3.1-8B and Qwen3-4B: the N8 k=2 directional score identity and the
weight-MSE map.
On Qwen3.8-27B the weight-MSE map also matches exactly, but the k=2 identity is not bitwise testable there (section 6).

**2. What does not reproduce.** Score shards are absent from the handoff package (`INVENTORY.json`), so every score was
regenerated rather than recovered. The regenerated tile sets differ from the archived ones: the archived fixed-256 map is
not bitwise reproducible (Jaccard 0.528 on Qwen3-4B, 0.631 on Llama-3.1-8B and 0.450 on Qwen3.8-27B, against a 0.8
tolerance), the archived adaptive maps are bitwise equal in only 6/10, 2/10 and 0/10 cases respectively — the 27B
reproduces none of them — and the k=3 selected count misses the ±5% tolerance on two of the three development
models (Qwen3-4B 8,454 vs archived 7,912, +6.9%; Qwen3.8-27B 3,577 vs 3,785, −5.5%; Llama-3.1-8B passes at 3,306 vs
3,345, −1.2%). On Qwen3.8-27B the archived `maps.json` and the archived fixed-256 prefix also fail to reproduce, only
159 of the 256 archived tiles fall inside the regenerated top 256, and its archived fixed-256 presets reproduce at
Jaccard 0.051–0.455. Per-sequence BF16 and W4A4 fit losses miss tier B by factors of 1.07–5.4
(e.g. Qwen3-4B BF16 NLL max |Δ| 0.0108 against a 0.002 tolerance). **The frozen tolerances were not widened.**

**3. Why: this is kernel noise, not a coding difference.** Two controlled experiments on a single A6000 bound the effect
without changing GPU. (a) The V20 diagnostic re-runs the identical calibration inputs under same-GPU kernel variants
(batch 1 vs 2, eager vs SDPA, float32 accumulation) and moves per-sequence BF16 NLL by up to 0.0086 (Llama-3.1-8B) and
0.0197 (Qwen3-4B) — as large as the A6000-vs-archive difference that fails tier B. (b) The V21 SDPA investigation repeats
the archived calibration changing *only* the attention backend and yields k=3 Jaccard 0.29, k=2 Jaccard 0.16 and
fixed-256 Jaccard 0.48 against the eager run on the same card — i.e. a backend switch alone perturbs the selection as much
as a different datacentre GPU does. The mechanism is visible in V14: under W4A4 the same prompt evaluated as a 48- vs
64-token prefix differs by up to 13.7 logits (BF16 model: 0.44), because fake quantization turns tiny kernel-order
differences into discrete rounding flips of the 4-bit codes.

**4. Consequence for the paper.** Selected tile *identities* from the archived campaign are not portable across kernels or
GPUs and must not be presented as a reproducible artifact; the reproducible quantity is the aggregate paired effect of a
map that is loaded from disk and hash-verified. This is why every headline evaluation in this campaign re-reads the exact
map file and verifies its digest, model/tokenizer revision, module names/shapes, type block, protocol id and source
manifest before installing it, and why per-tile statistics are reported with the sign-flip FDP estimate rather than as
stable per-tile claims.

**5. One archived claim does not survive.** The archived Llama-3.1-8B N8 k=2 regression (archived 6.9078 vs FourOverSix
6.8755 on Wiki, i.e. k=2 hurts) does not reproduce: the regenerated k=2 map gives 6.8730, slightly *better* than
FourOverSix, and the paired effect flips sign on both corpora (tier C fails: Wiki archived +0.00469 vs regenerated
−0.00062; C4 archived +0.00188 vs regenerated −0.00334). The historical narrative "k=3 is the first threshold that is
positive on every development model" therefore rests on a k=2 data point that is within kernel noise of zero. k=3 remains
the frozen primary rule, but it must be justified by the pre-registered freeze and the k-sweep in this campaign, not by
the archived k=2 regression. Three further isolated tier failures are reported for completeness, and all three are
marginal: Llama-3.1-8B k=5 on C4 (tier C, |Δ| 0.002029 against the 0.002 floor — a miss of 1.4%), Qwen3-4B k=6 on Wiki
(tier B, −0.0069 against 0.005) and Qwen3.8-27B's archived fixed-256 map on Wiki (tier C, a sign flip on an archived
effect of +0.00074; see section 1).

**6. Coverage, and one check that is unavailable rather than failed.** All three development models now carry
archived-anchor evidence through both V21 and V22. The Qwen3.8-27B historical evaluation
(`runs/V22_hist_eval_qwen27b_attempt1`, 2× RTX A6000, 5.88 GPU-hours) completed with no failures and no foreign
co-tenancy, and it covers every perplexity anchor the freeze requires — `required_anchors` names FourOverSix, the
archived fixed-256 map and N8 k=3, which are exactly the three policies it evaluated; the k=2–k=6 sweep carried by the
other two models is supplementary rather than a required anchor. All six of its policy×corpus cells pass tier B
(relative PPL +0.00196 / −0.00036 for FourOverSix, −0.00036 / +0.00033 for archived fixed-256 and +0.00283 / +0.00020
for N8 k=3, Wiki/C4). The quantity the paper actually relies on — the N8 k=3 paired effect — reproduces within tier C on
both corpora, so the headline historical claim now reproduces on all three development models. Its single tier failure
is the archived fixed-256 effect on Wiki, reported in section 1.
The cross-GPU archived anchor (V23) completed on an RTX 6000 Ada after two attempts were invalidated by another user's
process, and it gives the campaign's clearest portability statement — on **Qwen3-4B**, which the freeze designates as the
portability and determinism model and which is the only model evaluated on both GPU generations under the historical
protocol: with the same archived maps and windows, Ada and A6000 agree to within 0.37% relative perplexity on all six
policy×corpus cells (three policies × two corpora), while per-window NLL differs by up to 0.084. Aggregate paired effects port across GPU generations; per-window values do not.
One check is unavailable rather than failed, and is reported as such: the N8 k=2 directional-score identity is verified
bitwise on Llama-3.1-8B and Qwen3-4B but cannot be tested bitwise on Qwen3.8-27B, which is scored in streaming mode with
float64 moments. The strongest internal correctness check in the historical protocol therefore does not cover the
largest model, and no claim is made that it does.

**7. The boundary of reproducibility is now measured, and it is sharp.** Sections 2–4 attribute the archived-reproduction
failures to kernel noise. That is correct but incomplete, and the missing half changes what a reproducer should expect. Under a
*fixed* configuration the aligned W4A4 evaluation is not noisy at all: it is **bitwise deterministic**. The three V43 noise
controls (`runs/V43_fidelity_noise_{llama8b,mistral7b,qwen4b}_attempt1`) each performed 24 baseline
evaluations over the same 128 sequences — plain repeats, repeats separated by a weight switch-on/switch-off cycle, repeats
interleaved through a tile-intervention block, and repeats at the end of a ~30-minute run — and on all three models every one
is bitwise identical, with single-measurement SD and end-of-run drift both exactly 0.0, on two *different* A6000 cards
(`GPU-2e6984d6` for Llama-3.1-8B and Mistral-7B, `GPU-d0ab9929` for Qwen3-4B). The reproduction also crosses processes, containers and cards, and this has
now been observed twice independently: Mistral-7B's V43 run executed on `GPU-9cec7336` and its noise control on
`GPU-2e6984d6`, hours apart and under different source manifests, yet the full 128-sequence baseline agrees bitwise on both
objectives (CE 1.3765774966450408, KL 0.04167576700238247); and Qwen3-4B's V43 run executed on `GPU-9cec7336` against its
noise control on `GPU-d0ab9929`, with the same top tile measuring -7.7346281614e-04 in both, difference exactly zero.
So the instability documented above is attributable specifically to **changing the kernel configuration**, not to inherent
nondeterminism: the attention backend (V21: eager→SDPA alone gives k=3 Jaccard 0.29 on the same card), the batch composition
(V20), or the GPU generation (V23/V81: per-window NLL differs by up to 0.084 between A6000 and RTX 6000 Ada while aggregate
paired effects port). The practical statement for a reproducer is therefore precise: pin the model and tokenizer revisions,
the window construction, the attention backend, the batch size and the GPU architecture, and this campaign's aligned numbers
reproduce **exactly**; change any one of them and expect per-window differences of the magnitude reported in sections 3 and 6,
with only the aggregate paired effect preserved.
One field must be disregarded rather than trusted while reading the V43 artifacts: `baseline_restored_exact` is `False` in
every block, but that is an artifact of the check itself (`campaign/fidelity.py:184` evaluates a single batch of 8 sequences
and compares it against a baseline in which those sequences sat in a batch of 16 — the very batch-composition sensitivity
described above), not evidence of a restore failure. The restore path is verified bitwise on CPU
(`campaign/tests/test_fidelity_noise.py`) and on device by the noise controls' switch-cycle block.

One further expectation must be set for anyone reproducing this work: a reproducer who **regenerates the maps** rather than
reloading them will not obtain an identical selection. The V82 determinism repeat re-ran the seed0 calibration under the same
protocol and reproduced every *forward* quantity exactly (`bf16_fit_nll_equal` and `fit_losses_equal` both true), while the
gradient-derived score stream and map payloads did not match: Jaccard 0.9789-0.9908 across the five policies, with selected
counts differing by a handful (n16 k3 4,077 vs 4,083; n8 k3 7,349 vs 7,340). The non-determinism therefore enters through the
backward pass, not the forward evaluation. It is a far smaller perturbation than any configuration change - a backend switch
or a change of GPU generation moves the same selection to Jaccard ~0.29 - but it does mean the published maps are not bitwise
regenerable, which is precisely why every headline evaluation in this campaign reloads a stored, hash-verified map file
instead of recomputing one (findings_log item 36; risk-register row C16, which is unsupported for this reason). One caveat on
that measurement: the repeat ran on a different card and under a different source manifest, so run-to-run backward
non-determinism is confounded with cross-card variation, and the queue cannot pin a GPU UUID to separate them.

A second, independent measurement extends that boundary to the task suite, and it is the half a reviewer is most likely to
care about: with the map **reloaded** rather than regenerated, the downstream evaluation reproduces at the level of
individual examples. The V40 and V80 representative-accuracy runs install the same N16 k3 map file (identical SHA-256,
`map_reloaded_for_evaluation` true on both sides) under an identical harness, and **all 22,641 per-example outcomes agree
exactly** — 7,547 examples on each of Llama-3.1-8B, Mistral-7B and Qwen3-4B over arc_challenge, boolq, piqa and winogrande,
agreement 1.0000 in every one of the twelve model×task cells, with all twelve task accuracies identical to four decimals.
The two runs of each pair are 2.5 to 4.2 hours apart, in different containers, under different source manifests, and for
Mistral-7B and Qwen3-4B on different A6000 cards. Like the statements above this is intra-architecture, and it concerns
*decisions* rather than logits — but it means a reproducer who installs the published map obtains not merely the same
aggregate score but the same answer on every individual question (findings_log item 39).

