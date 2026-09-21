# Statistical validity report (V73)

Protocol freeze `df78f1fbbd034cb8d2bc8e6f6a264e7e221e1b5f2b2ff99a3fe821369c5ac88f`; statistics plan: `PROTOCOL_FREEZE.json#statistics`.

## 1. Sampling units

- WikiText-2: 2048-token windows of the joined test split; cluster = article containing the window's first token (sensitivity: contiguous blocks of 5 windows).
- C4: 256 crops from documents drawn with replacement (Random(0)); cluster = document SHA-256 (231-235 unique documents per tokenizer).
- Accuracy/GSM8K: examples (MMLU: questions pooled over 57 subjects); paired example bootstrap; macro = mean of task differences with independent task bootstraps.
- Calibration: 128 sequences; tile statistics use per-sequence scores; N16 per-sequence scores are sums of the two N8 children.
- Bootstrap: B=10000 for primary endpoints, 2000 exploratory, percentile intervals, seed 20260911.

## 2. Evaluation-sample uncertainty (primary endpoints)

Sources: `runs/V90_analyze_ppl_attempt5/analysis_ppl/LEGACY_PANEL_PPL.json` (sha256 `ea5f6e743d720e5b…`), `runs/V90_analyze_ppl_attempt5/analysis_ppl/CONFIRMATORY_PPL.json` (sha256 `f68cba72f8632910…`).

- development llama8b wiki: dlogPPL(N16k3-4/6) -0.0047 CI [-0.0063, -0.0031] over 53 clusters / 141 windows; block-5 CI [-0.0060, -0.0034]
- development llama8b c4: dlogPPL(N16k3-4/6) -0.0050 CI [-0.0067, -0.0035] over 231 clusters / 256 windows;
- development qwen27b wiki: dlogPPL(N16k3-4/6) -0.0053 CI [-0.0089, -0.0003] over 54 clusters / 145 windows; block-5 CI [-0.0093, -0.0004]
- development qwen27b c4: dlogPPL(N16k3-4/6) -0.0032 CI [-0.0040, -0.0025] over 233 clusters / 256 windows;
- development qwen4b wiki: dlogPPL(N16k3-4/6) -0.1525 CI [-0.1617, -0.1441] over 53 clusters / 146 windows; block-5 CI [-0.1607, -0.1448]
- development qwen4b c4: dlogPPL(N16k3-4/6) -0.0752 CI [-0.0784, -0.0721] over 231 clusters / 256 windows;
- confirmatory mistral7b wiki: dlogPPL(N16k3-4/6) -0.0039 CI [-0.0049, -0.0030] over 57 clusters / 163 windows; block-5 CI [-0.0049, -0.0030]
- confirmatory mistral7b c4: dlogPPL(N16k3-4/6) -0.0025 CI [-0.0036, -0.0015] over 235 clusters / 256 windows;
- confirmatory olmo2_13b wiki: dlogPPL(N16k3-4/6) -0.0017 CI [-0.0030, -0.0004] over 54 clusters / 141 windows; block-5 CI [-0.0032, -0.0004]
- confirmatory olmo2_13b c4: dlogPPL(N16k3-4/6) -0.0010 CI [-0.0016, -0.0003] over 231 clusters / 256 windows;
- confirmatory phi4 wiki: dlogPPL(N16k3-4/6) -0.0053 CI [-0.0066, -0.0041] over 54 clusters / 141 windows; block-5 CI [-0.0064, -0.0043]
- confirmatory phi4 c4: dlogPPL(N16k3-4/6) -0.0039 CI [-0.0047, -0.0031] over 231 clusters / 256 windows;

Holm-adjusted one-sided non-inferiority (margin log 1.005):

- mistral7b/wiki: p=0, Holm p=0, non-inferior=True
- mistral7b/c4: p=0, Holm p=0, non-inferior=True
- phi4/wiki: p=0, Holm p=0, non-inferior=True
- phi4/c4: p=0, Holm p=0, non-inferior=True
- olmo2_13b/wiki: p=0, Holm p=0, non-inferior=True
- olmo2_13b/c4: p=0, Holm p=0, non-inferior=True

## 3. Calibration-draw variability versus evaluation variance

Source: `runs/V90_analyze_ppl_attempt5/analysis_ppl/CALIBRATION_SEED_STABILITY_PPL.json` (sha256 `55a6eec5ca167018…`).

- llama8b n8 wiki: draws(+seed0) estimates [-0.0052, -0.0033, -0.0040, -0.0035, -0.0050]; between-draw variance 7.164e-07; mean within-draw bootstrap variance 7.126e-07
- llama8b n8 c4: draws(+seed0) estimates [-0.0054, -0.0054, -0.0039, -0.0056, -0.0050]; between-draw variance 4.611e-07; mean within-draw bootstrap variance 9.266e-07
- llama8b n16 wiki: draws(+seed0) estimates [-0.0053, -0.0044, -0.0039, -0.0053, -0.0047]; between-draw variance 3.338e-07; mean within-draw bootstrap variance 7.354e-07
- llama8b n16 c4: draws(+seed0) estimates [-0.0061, -0.0049, -0.0046, -0.0049, -0.0050]; between-draw variance 3.390e-07; mean within-draw bootstrap variance 8.670e-07
- mistral7b n8 wiki: draws(+seed0) estimates [-0.0040, -0.0033, -0.0043, -0.0039, -0.0045]; between-draw variance 1.869e-07; mean within-draw bootstrap variance 2.314e-07
- mistral7b n8 c4: draws(+seed0) estimates [-0.0023, -0.0025, -0.0026, -0.0022, -0.0027]; between-draw variance 4.051e-08; mean within-draw bootstrap variance 1.652e-07
- mistral7b n16 wiki: draws(+seed0) estimates [-0.0044, -0.0036, -0.0038, -0.0033, -0.0039]; between-draw variance 1.684e-07; mean within-draw bootstrap variance 2.237e-07
- mistral7b n16 c4: draws(+seed0) estimates [-0.0018, -0.0024, -0.0024, -0.0024, -0.0025]; between-draw variance 6.617e-08; mean within-draw bootstrap variance 1.331e-07
- qwen4b n8 wiki: draws(+seed0) estimates [-0.1953, -0.1742, -0.1548, -0.1495, -0.1785]; between-draw variance 3.449e-04; mean within-draw bootstrap variance 1.854e-05
- qwen4b n8 c4: draws(+seed0) estimates [-0.0986, -0.0886, -0.0802, -0.0743, -0.0878]; between-draw variance 8.499e-05; mean within-draw bootstrap variance 3.690e-06
- qwen4b n16 wiki: draws(+seed0) estimates [-0.1678, -0.1476, -0.1289, -0.1247, -0.1525]; between-draw variance 3.136e-04; mean within-draw bootstrap variance 1.493e-05
- qwen4b n16 c4: draws(+seed0) estimates [-0.0841, -0.0781, -0.0695, -0.0632, -0.0752]; between-draw variance 6.460e-05; mean within-draw bootstrap variance 2.736e-06

## 4. Selection statistics: dependence, joint null and multiplicity

Source: `runs/V90_analyze_selection_attempt2/analysis_selection/SELECTION_STATISTICS.json` (sha256 `255cea8aa73c44e4…`).

- **llama8b N8**: tiles 13,631,488, selected k=3 3,130; per-tile CE/KL correlation median 0.65 (p05 0.34, p95 0.85); across-tile corr(t_CE,t_KL) 0.64. Null false-positive bounds: independence Phi(-3)^2 24.8 (invalid), measured-rho 2789.2, IUT any-dependence 18401. BH q=.05 330 / BY q=.05 221 (t-dist BH 283). Sign-flip (R=1000, 136,256 sampled tiles): observed 29 vs permuted mean 8.04 (p95 13.0), est. FDP +0.277, global p 0.0010.
- **llama8b N16**: tiles 6,815,744, selected k=3 1,781; per-tile CE/KL correlation median 0.65 (p05 0.34, p95 0.84); across-tile corr(t_CE,t_KL) 0.64. Null false-positive bounds: independence Phi(-3)^2 12.4 (invalid), measured-rho 1391.2, IUT any-dependence 9201. BH q=.05 203 / BY q=.05 146 (t-dist BH 181). Sign-flip (R=1000, 68,128 sampled tiles): observed 13 vs permuted mean 4.12 (p95 8.0), est. FDP +0.317, global p 0.0010.
- **mistral7b N8**: tiles 13,631,488, selected k=3 7,501; per-tile CE/KL correlation median 0.32 (p05 0.10, p95 0.53); across-tile corr(t_CE,t_KL) 0.35. Null false-positive bounds: independence Phi(-3)^2 24.8 (invalid), measured-rho 498.6, IUT any-dependence 18401. BH q=.05 2,685 / BY q=.05 1,683 (t-dist BH 2,326). Sign-flip (R=1000, 136,256 sampled tiles): observed 79 vs permuted mean 2.77 (p95 6.0), est. FDP +0.035, global p 0.0010.
- **mistral7b N16**: tiles 6,815,744, selected k=3 4,179; per-tile CE/KL correlation median 0.32 (p05 0.10, p95 0.53); across-tile corr(t_CE,t_KL) 0.36. Null false-positive bounds: independence Phi(-3)^2 12.4 (invalid), measured-rho 245.9, IUT any-dependence 9201. BH q=.05 1,545 / BY q=.05 997 (t-dist BH 1,355). Sign-flip (R=1000, 68,128 sampled tiles): observed 54 vs permuted mean 1.44 (p95 4.0), est. FDP +0.027, global p 0.0010.
- **olmo2_13b N8**: tiles 24,780,800, selected k=3 3,826; per-tile CE/KL correlation median 0.33 (p05 0.05, p95 0.67); across-tile corr(t_CE,t_KL) 0.34. Null false-positive bounds: independence Phi(-3)^2 45.2 (invalid), measured-rho 1376.3, IUT any-dependence 33452. BH q=.05 21 / BY q=.05 8 (t-dist BH 12). Sign-flip (R=1000, 247,760 sampled tiles): observed 23 vs permuted mean 2.92 (p95 6.0), est. FDP +0.127, global p 0.0010.
- **olmo2_13b N16**: tiles 12,390,400, selected k=3 2,181; per-tile CE/KL correlation median 0.33 (p05 0.06, p95 0.67); across-tile corr(t_CE,t_KL) 0.34. Null false-positive bounds: independence Phi(-3)^2 22.6 (invalid), measured-rho 687.5, IUT any-dependence 16726. BH q=.05 20 / BY q=.05 7 (t-dist BH 10). Sign-flip (R=1000, 123,880 sampled tiles): observed 14 vs permuted mean 1.56 (p95 4.0), est. FDP +0.111, global p 0.0010.
- **phi4 N8**: tiles 26,624,000, selected k=3 4,201; per-tile CE/KL correlation median 0.32 (p05 0.02, p95 0.60); across-tile corr(t_CE,t_KL) 0.36. Null false-positive bounds: independence Phi(-3)^2 48.5 (invalid), measured-rho 1174.2, IUT any-dependence 35940. BH q=.05 22 / BY q=.05 6 (t-dist BH 6). Sign-flip (R=1000, 266,240 sampled tiles): observed 39 vs permuted mean 3.69 (p95 7.0), est. FDP +0.095, global p 0.0010.
- **phi4 N16**: tiles 13,312,000, selected k=3 2,184; per-tile CE/KL correlation median 0.32 (p05 0.03, p95 0.59); across-tile corr(t_CE,t_KL) 0.36. Null false-positive bounds: independence Phi(-3)^2 24.3 (invalid), measured-rho 582.0, IUT any-dependence 17970. BH q=.05 12 / BY q=.05 3 (t-dist BH 3). Sign-flip (R=1000, 133,120 sampled tiles): observed 19 vs permuted mean 1.94 (p95 4.0), est. FDP +0.102, global p 0.0010.
- **qwen27b N8**: tiles 47,559,680, selected k=3 3,946; per-tile CE/KL correlation median 0.43 (p05 0.10, p95 0.78); across-tile corr(t_CE,t_KL) 0.44. Null false-positive bounds: independence Phi(-3)^2 86.7 (invalid), measured-rho 4640.8, IUT any-dependence 64201. BH q=.05 163 / BY q=.05 78 (t-dist BH 108). Sign-flip (R=1000, 481,056 sampled tiles): observed 45 vs permuted mean 7.63 (p95 12.0), est. FDP +0.170, global p 0.0010.
- **qwen27b N16**: tiles 23,779,840, selected k=3 2,168; per-tile CE/KL correlation median 0.43 (p05 0.11, p95 0.78); across-tile corr(t_CE,t_KL) 0.45. Null false-positive bounds: independence Phi(-3)^2 43.3 (invalid), measured-rho 2316.8, IUT any-dependence 32100. BH q=.05 108 / BY q=.05 52 (t-dist BH 76). Sign-flip (R=1000, 240,528 sampled tiles): observed 19 vs permuted mean 3.92 (p95 7.0), est. FDP +0.206, global p 0.0010.
- **qwen4b N8**: tiles 7,096,320, selected k=3 7,349; per-tile CE/KL correlation median 0.45 (p05 0.21, p95 0.70); across-tile corr(t_CE,t_KL) 0.42. Null false-positive bounds: independence Phi(-3)^2 12.9 (invalid), measured-rho 600.7, IUT any-dependence 9579. BH q=.05 1,140 / BY q=.05 514 (t-dist BH 862). Sign-flip (R=1000, 71,784 sampled tiles): observed 87 vs permuted mean 3.03 (p95 6.0), est. FDP +0.035, global p 0.0010.
- **qwen4b N16**: tiles 3,548,160, selected k=3 4,077; per-tile CE/KL correlation median 0.45 (p05 0.21, p95 0.69); across-tile corr(t_CE,t_KL) 0.42. Null false-positive bounds: independence Phi(-3)^2 6.5 (invalid), measured-rho 297.3, IUT any-dependence 4790. BH q=.05 705 / BY q=.05 303 (t-dist BH 539). Sign-flip (R=1000, 35,892 sampled tiles): observed 53 vs permuted mean 1.66 (p95 4.0), est. FDP +0.031, global p 0.0010.

## 5. N8 versus N16 structure

Source: `runs/V90_analyze_selection_attempt2/analysis_selection/N8_N16_STRUCTURE.json` (sha256 `a19a42806b7d1b72…`).

- llama8b: N8 3,130 tiles (1,602,560 weights), N16 1,781 (1,823,744); N8-any 3,067, N8-both 63; Jaccard(N16,any) 0.203, (N16,both) 0.035; selected children per selected parent {'0': 964, '1': 754, '2': 63}; cancellation/amplification {"both_children_selected_parent_rejected": 0, "child_selected_parent_rejected": 2250, "parent_selected_both": 63, "parent_selected_no_child": 964, "parent_selected_one_child": 754}
- mistral7b: N8 7,501 tiles (3,840,512 weights), N16 4,179 (4,279,296); N8-any 7,182, N8-both 319; Jaccard(N16,any) 0.418, (N16,both) 0.076; selected children per selected parent {'0': 829, '1': 3031, '2': 319}; cancellation/amplification {"both_children_selected_parent_rejected": 0, "child_selected_parent_rejected": 3832, "parent_selected_both": 319, "parent_selected_no_child": 829, "parent_selected_one_child": 3031}
- olmo2_13b: N8 3,826 tiles (1,958,912 weights), N16 2,181 (2,233,344); N8-any 3,737, N8-both 89; Jaccard(N16,any) 0.272, (N16,both) 0.041; selected children per selected parent {'0': 915, '1': 1177, '2': 89}; cancellation/amplification {"both_children_selected_parent_rejected": 0, "child_selected_parent_rejected": 2471, "parent_selected_both": 89, "parent_selected_no_child": 915, "parent_selected_one_child": 1177}
- phi4: N8 4,201 tiles (2,150,912 weights), N16 2,184 (2,236,416); N8-any 4,168, N8-both 33; Jaccard(N16,any) 0.212, (N16,both) 0.015; selected children per selected parent {'0': 1072, '1': 1079, '2': 33}; cancellation/amplification {"both_children_selected_parent_rejected": 0, "child_selected_parent_rejected": 3056, "parent_selected_both": 33, "parent_selected_no_child": 1072, "parent_selected_one_child": 1079}
- qwen27b: N8 3,946 tiles (2,020,352 weights), N16 2,168 (2,220,032); N8-any 3,925, N8-both 21; Jaccard(N16,any) 0.266, (N16,both) 0.010; selected children per selected parent {'0': 887, '1': 1260, '2': 21}; cancellation/amplification {"both_children_selected_parent_rejected": 0, "child_selected_parent_rejected": 2644, "parent_selected_both": 21, "parent_selected_no_child": 887, "parent_selected_one_child": 1260}
- qwen4b: N8 7,349 tiles (3,762,688 weights), N16 4,077 (4,174,848); N8-any 7,198, N8-both 151; Jaccard(N16,any) 0.278, (N16,both) 0.037; selected children per selected parent {'0': 1621, '1': 2305, '2': 151}; cancellation/amplification {"both_children_selected_parent_rejected": 0, "child_selected_parent_rejected": 4742, "parent_selected_both": 151, "parent_selected_no_child": 1621, "parent_selected_one_child": 2305}

## 6. Numerical noise floor

W4A4 fake quantization amplifies kernel-level floating-point differences into discrete rounding flips: on Qwen3-4B the same prompt
evaluated as a 48- vs 64-token prefix differs by up to 13.7 logits (bf16) and 6.1 logits (float32) with no KV cache (`runs/V14_causality_diag_qwen4b_attempt3/diag/causality_diag.json`).
Per-window and per-tile quantities are therefore not portable across kernels/GPUs; aggregate paired endpoints are the unit of inference.

## 7. Confirmatory versus exploratory

- Confirmatory: the 6 primary endpoints listed in the freeze (confirmatory panel x {WikiText, C4}, N16k3 vs FourOverSix).
- Secondary (pre-declared, not multiplicity-controlled): accuracy, GSM8K, retained fraction, development-panel endpoints.
- Exploratory: k sweep, objective ablations, selector controls, draws, size/domain, fidelity, long context, baselines, structure.

## Findings

**1. The primary endpoints are statistically sound, and small.** All six frozen confirmatory endpoints are non-inferior
after Holm (bootstrap p = 0 at B = 10,000) and every upper 95% CI is below zero, so N16K64 k=3 is not merely
non-inferior to FourOverSix but significantly better on all three unseen families and both corpora. The effect sizes are
-0.0039/-0.0025 (Mistral-7B Wiki/C4), -0.0053/-0.0039 (Phi-4) and -0.0017/-0.0010 (OLMo-2-13B) in log-PPL, i.e. 0.1-0.5%
perplexity. The three development models agree in sign and significance (Llama-3.1-8B -0.0047/-0.0050, Qwen3.8-27B
-0.0053/-0.0032, Qwen3-4B -0.153/-0.075), so the direction of the result holds on all six models. Anything reported at
this magnitude must be paired and clustered; an unpaired comparison of two PPL numbers at this scale is
indistinguishable from the kernel noise measured in V14/V20.

**2. Cluster counts, not window counts, set the resolution.** WikiText-2 contributes only 53-57 article clusters (141-163
windows) and C4 231-235 document clusters (256 windows). The effective sample size for the Wiki endpoints is therefore
~55, not ~150, and the frozen block-5 sensitivity analysis agrees with the article clustering on every primary endpoint
(e.g. Mistral Wiki article-cluster CI [-0.00486, -0.00296] vs block-5 [-0.00487, -0.00298]), so the conclusions do not
depend on the choice of cluster definition. With ~55 clusters the percentile bootstrap is usable but not precise; CI
widths of +/-0.001 log-PPL are the resolution floor of this protocol, which is why the accuracy suite cannot be expected
to resolve the same effect (see the risk audit, C07).
The same arithmetic disqualifies the long-context row's intervals, and this report is the right place to say so. V71 draws
all 64 (4,096-token) and 32 (8,192-token) windows from **four** PG19 books, so every interval in `LONG_CONTEXT.json` rests on
k=4 clusters. At that count the percentile bootstrap is not merely imprecise, it is **anti-conservative**: the book-clustered
CIs come out *narrower* than naive window-level CIs in 16 of the 24 intervals (width ratios 0.32-0.99, seven of them below
0.6), which is the reverse of what clustering for positive intra-cluster correlation must do; the eight exceptions are
dominated by the six large BF16-minus-FourOverSix cells, where real between-book heterogeneity widens the clustered
interval to 1.4-2.6x the naive one. The small N16-minus-N8 contrast flips both sign and significance between
adjacent context lengths on two models (Llama +0.00032 spanning zero at 4k versus -0.00090 excluding zero at 8k; Mistral the
reverse). Only the **sign and point estimate** are reportable there for Llama-3.1-8B and Mistral-7B. Qwen3-4B's long-context
effect is roughly twenty-five times larger and is unaffected by the cluster deficiency. This is a pre-registration defect -
the freeze pinned window counts and the cluster definition but no minimum number of clusters - and it is recorded rather than
repaired, because redrawing the windows after seeing the results would defeat the freeze (findings_log item 37).

**3. N16 is significantly worse than N8 on most models - report it.** The N16-minus-N8 contrast is significantly positive
(worse) on four of the six models, on at least one corpus each: Qwen3-4B Wiki +0.02597 [+0.02319, +0.02861] and C4
+0.01256 [+0.01118, +0.01393]; Phi-4 Wiki +0.00206 [+0.00105, +0.00311] and C4 +0.00070 [+0.00002, +0.00136]; OLMo-2
Wiki +0.00134 [+0.00001, +0.00271]; Qwen3.8-27B C4 +0.00072 [+0.00003, +0.00141]. It is indistinguishable from zero on
Llama-3.1-8B and Mistral-7B (both corpora), on OLMo-2 C4 and on Qwen3.8-27B Wiki. Retained fraction R of the N8 log-PPL
gain: Llama 0.97, Mistral 0.89, Qwen3-4B 0.86, Phi-4 0.77, Qwen3.8-27B 0.71, OLMo-2 0.63. The honest statement is
"N16K64 keeps most, but not all, of the N8K64 gain; the loss is model-dependent, ranges from 3% to 37% of the gain, and
is statistically detectable on four of six models". Coarsening the type block from N8K64 to N16K64 is therefore a real
quality concession, not a free change, and the paper must not present N16 as equivalent to N8.
There is one dimension on which the coarser type block is **better**, and it must be reported alongside rather than omitted.
V43's batched interventions measure how much of a map's predicted first-order sum is actually realised when all of its tiles
are switched together on the same 128 calibration sequences, and N16K64 is markedly more additive than N8K64 on all three
models measured. Realised/predicted CE is 0.791 (N16) versus 0.632 (N8) on Mistral-7B, 0.739 versus 0.362 on Llama-3.1-8B and
0.664 versus 0.578 on Qwen3-4B; realised/predicted KL is 0.688 versus 0.552, 0.621 versus 0.329, and 0.151 versus -0.026.
N8 predicts the larger total gain and delivers a smaller fraction of it, which is what interacting row-tiles imply: summing
per-sequence scores over twice as many half-height tiles double-counts interactions that do not survive simultaneous
switching. The ratio ordering does not carry over to the realised effect, and the two must be reported together: on
Qwen3-4B the less additive N8 map still produces the larger actual calibration-set change (-8.41e-2 against N16's
-7.71e-2). Two further caveats bound this. The Qwen3-4B KL column is an outlier - at N8 the realised KL change carries the *opposite
sign* to its prediction (-0.026), so the aggregate KL prediction fails outright on that model, consistent with it moving away
from the BF16 teacher while improving CE (section 5 and risk-audit rows C07, C12). And this is a calibration-set quantity,
not a held-out one: it says the N16 map gives a more faithful first-order account of itself, **not** that N16 generalises
better - the held-out retained fractions above still show N16 keeping only 63-97% of the N8 gain.

**4. The archived multiplicity argument is invalid; the replacement is weaker than it looks.** The archived rule assumed
CE and KL scores are independent, giving an expected false-positive bound of Phi(-3)^2 x tiles = 6.5-43.3 tiles. The
measured per-tile CE/KL correlation over sequences is strongly positive on every model (median 0.32 Mistral and Phi-4,
0.33 OLMo-2, 0.43 Qwen3.8-27B, 0.45 Qwen3-4B, 0.65 Llama-3.1-8B; across-tile correlation of the t statistics 0.34-0.64),
so that bound does not hold. Under the measured correlation the bound rises to 246-2,317 tiles, and the assumption-free
intersection-union bound (any dependence) to 4,790-32,100 tiles - which exceeds the number of tiles k=3 selects on **all
six** models, not only the three with the largest bounds: Qwen3.8-27B 32,100 vs 2,168, Phi-4 17,970 vs 2,184, OLMo-2
16,726 vs 2,181, Llama-3.1-8B 9,201 vs 1,781, Mistral-7B 9,201 vs 4,179 and Qwen3-4B 4,790 vs 4,077. The assumption-free
bound is therefore **vacuous on every model measured**: it permits more false positives than the rule selects tiles, so it
cannot distinguish the selected set from noise anywhere in the panel.
Dependency-robust FDR control keeps far fewer tiles than k=3: BH q=0.05 retains 1,545/4,179 (Mistral), 705/4,077
(Qwen3-4B), 203/1,781 (Llama), 108/2,168 (Qwen3.8-27B), 20/2,181 (OLMo-2) and 12/2,184 (Phi-4); BY is stricter still
(3-997). The empirical sequence-level sign-flip test is the most defensible statement: it rejects the global null on
every model (p = 0.001, R = 1,000) with estimated false-discovery proportions of 0.027 (Mistral), 0.031 (Qwen3-4B),
0.102 (Phi-4), 0.111 (OLMo-2), 0.206 (Qwen3.8-27B) and 0.317 (Llama-3.1-8B). Those estimates rest on small observed
counts in the stored stratified sample (13-54 selected tiles per model) and are themselves noisy. **k=3 should be
described as a conservative screening threshold whose empirical FDP is a few percent on some models and ~10-32% on
others, never as a per-tile significance guarantee.**

**5. Selection is real but the individual selections are not the claim.** Point 4 above and the reproduction report
(kernel-noise sensitivity of tile identities) point the same way: the evidence supports "the directional score selects a
set of tiles whose aggregate effect is reproducibly negative", not "tile X is a genuine improvement". Both the FDP
estimates and the aligned/archived Jaccard values should be quoted whenever a per-tile or per-layer analysis is shown.
This is no longer an inference from instability; it has now been measured directly, and the direct measurement is stronger
than the inference. The V43 noise controls re-measured **nine** N16 tiles - three on each of Llama-3.1-8B, Mistral-7B and
Qwen3-4B - with four replicates apiece, and every replicate is bitwise identical, so each tile's effect on the 128 calibration
sequences is an exact number with no error bar to argue about. The predicted CE sign is correct in only **3 of 9** cases
(KL 7 of 9); the tile the k=3 rule *rejects* bears no relation to its prediction on any model, helping on Llama-3.1-8B
(-8.24e-4) and Mistral-7B (-5.31e-4) but hurting on Qwen3-4B (+1.12e-3), in every case by two to three orders of magnitude
more than predicted; and near the median prediction the measured effect is 41x to 202x larger with essentially arbitrary
sign. Prediction is credible only for the rare tile whose predicted effect is unusually large (top-tile ratio 0.80 on
Mistral-7B and 0.54 on Qwen3-4B, though -0.27 on Llama-3.1-8B). The decisive observation for this report is that **the model
with the highest single-tile rank correlation of all six model x type-block cells - Qwen3-4B at Spearman +0.239 - still
mispredicts its median and rejected tiles by -146x and -1,657x**, so a favourable aggregate correlation confers no per-tile
validity. The aggregate arm behaves completely differently and remains sound: the full k=3 map realises 0.664-0.791 (N16) and
0.362-0.632 (N8) of its predicted first-order sum, with the correct sign in every model x type-block cell.
The consequence for the paper is concrete: every per-tile or per-layer statement must be presented as a property of the
selected **set**, and the first-order score must never be described as identifying individually beneficial tiles.

**6. N16 is not recoverable from N8 masks, as the frozen construction requires.** Jaccard(N16, N8-any) is 0.20-0.42 and
Jaccard(N16, N8-both) is 0.010-0.076 across the six models. Of the tiles N16 selects, a large fraction have no selected
N8 child (Qwen3-4B 1,621 of 4,077 parents; Phi-4 1,072 of 2,184; Llama 964 of 1,781; OLMo-2 915 of 2,181; Qwen3.8-27B
887 of 2,168; Mistral 829 of 4,179), and very few have two (21-319 across models, the extreme being Qwen3.8-27B with 21
of 2,168). Any OR/AND/count-based derivation of N16 from N8 masks would therefore select a materially different set; the
per-sequence score summation followed by recomputed mean/SD/SE is not a formality.

**7. Calibration-draw variance is reported separately from evaluation variance.** The five-draw design (seed0 + draw1-4 on
three models) is the only way to separate "this map is good" from "this calibration sample is lucky"; the comparison of
between-draw variance against mean within-draw bootstrap variance is in section 3 of this report. Endpoints computed on a
single calibration draw - which is what the archived campaign reported - cannot distinguish the two.

