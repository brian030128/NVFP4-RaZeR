# Experiment A: IF4 and MixFP4 (Zou et al.) per-block selection, coarsened to hardware tiles — protocol

Written 2026-09-29 on branch `tm-opt`, before any A GPU run; the hashes and time are in `registration.json`. Deviations are
appended at the end. Requested by the user (paper §2.3, §5.5; relayed by nvfp4-razer-c9). Approved for Llama-3.1-8B;
Mistral-7B-v0.3 and Phi-4 come later, after the higher-priority items.

## Question and hypothesis

The purpose (the user, 2026-09-29): to show whether the per-block selection rules (IF4, Zou's MixFP4) lose quality when
they are coarsened to hardware tile granularity. It is not a kernel validation.

**Hypothesis** (stated before the runs; the result is reported whatever it is):
- A tile must elect one format, so the within-tile mixing that gives the per-block rules their 1x16 gain is averaged
  away. R@g should then be worse than R@1x16, and the gain over the rule's own base should shrink at g.
- At the same realizable granularity, OURS (a calibrated, loss-aware tile selection) may do better than the rules.

## A1: the formats, checked against the originals (`A1_CHECK.md`, `a1_check.json`)

- **IF4** (Cook et al., arXiv 2603.28765). Official code: github.com/mit-han-lab/fouroversix @ dadfad69,
  `quantize/pytorch/reference.py`.
  - α = amax / (6·448).
  - One E4M3 block scale for both candidates, e4m3(bmax / 6), relative to α.
  - FP4 candidate: E2M1, round to nearest even.
  - INT4 candidate: round(clamp(x_b · 1.16666666, −7, 7)), half to even, dequantized × 0.8571428571 (the official
    truncated 7/6 and 6/7).
  - Per block, the lower sum of squared errors; ties keep FP4.
  - Implemented exactly (`quantize/adaptive_formats.py`) and verified bitwise against the official reference on real
    Llama weights: choice, codes, scales and dequantized values.
- **MixFP4** (Zou et al., arXiv 2605.31035, Algorithm 1; no official code).
  - s32 = amax / 2688.
  - E2M1 scale e4m3(bmax / 6); E1M2 scale e4m3(bmax / 7), with the ×2 remap to the integers −7..7.
  - Both rounded to nearest even.
  - Per block, the lower squared error; ties go to E1M2 (Algorithm 1's else branch).
  - A zero (underflowed) scale gives a zero block, and such blocks are counted (the paper does not say).
- **The repo's `quant_nvif4` and `mixfp4` 1x16** are Zou's construction, not IF4. They differ from Zou in E2M1 rounding
  (half away from zero), tie-breaking (to FP4) and a [2^-9, 448] scale clamp. They are not used here, and the
  differences are counted in `a1_check.json`.

## A2: arms

- **Rule × granularity:** {IF4, Zou, Zou + FourOverSix E2M1} × {1x16, 8x64, 16x64, 256x64} = 12 arms.
- **Zou + FourOverSix E2M1** (rule `zoufo6`) is requested by the user (2026-09-29).
  - **Its purpose:** to separate the selection rule from the FourOverSix base. OURS vs zoufo6 isolates the selection
    method, since both use FourOverSix E2M1 and the same uniform format up to details. zoufo6 vs Zou isolates the
    FourOverSix contribution.
  - **E2M1 candidate:** the repo's FourOverSix (`quant_nvfp4_4over6`: per 16-block the max/6 or max/4 scale, whichever
    has the lower squared error), the E2M1 base of our maps.
  - **Uniform candidate:** Zou's E1M2 path exactly as above.
  - **Selection:** the lower squared error per block, or tile sum at the coarse granularities; ties go to the uniform
    format, as in Zou.
  - **Units:** `quant_nvfp4_4over6` returns BF16, so both candidates' errors are taken on the weights as installed (BF16),
    in the original units.
- **1x16** is the original per-block rule.
- **Coarse tiles** (rows × 64 columns of the weight [out, in]): for each format, the squared errors of all elements of
  the tile are summed, and the tile takes the smaller, with the original rule's tie-breaking. Every 16-block inside keeps
  that format's own scale rule. Rows are padded with zeros to a whole tile.

## A3: weights only

The rules apply to the weights only. Activations are per-token FourOverSix (convention (c)), as in every paper number.
This is deliberate: the question is the weight format's granularity, not the activation format.

## A4: evaluation

- **One fake simulator** for every row (1x16 cannot run natively): `run_ppl_deploy.py` fake (c), one process per
  (model, policy) (`experiments/paper_extra/A_formats.py`). The released protocol windows: WikiText-2 test in 2048-token
  windows, and 256 C4 validation crops.
- **The arms:** `fake:format:<rule>:<unit>`.
- **The references, in the same simulator:**
  - BF16;
  - NVFP4: `fake:nvfp4`, with NVFP4 per-token activations as in the paper's NVFP4 row;
  - FourOverSix: `fake:four_over_six`;
  - the committed TM-OPT+TC maps at 8x64, 16x64 and 256x64: `fake:map:`, the paper run's .mixfp4map.
  - **e2m1 and e2m1z:** each rule's own E2M1 base, with the arms' FourOverSix activations. e2m1 is IF4's FP candidate
    everywhere (`fake:format:e2m1:1x16`); e2m1z is MixFP4 (Zou et al.)'s E2M1 candidate everywhere
    (`fake:format:e2m1zou:1x16`). Both are NVFP4 weights.
    - The paper's NVFP4 row uses NVFP4 activations, so comparing an arm with it would mix the weight rule with the
      activation quantizer.
    - The A1 check found that the two candidates differ in 0.33 % of BF16 elements (up to 1.36 % in one module, layers.31.mlp.down_proj), from
      the operation order of the scale and dequantization. So each rule gets its own base.
- **Check:** the re-run references reproduce the Parts 2–3 fake (c) records window by window, where those exist.
- **The native Zou 16x64 check is dropped** (the user): A is about the rules' quality, not the kernel.

## A5: models

Llama-3.1-8B now. Mistral-7B-v0.3 and Phi-4 later, under this protocol, after the higher-priority items.

## A6: the PRIMARY comparisons (fixed before the runs; `experiments/paper_extra/A_analyze.py`)

All are paired over the same windows (token hashes compared), on WikiText-2 and C4, as ΔNLL (nats per token = Δ log
PPL): ± 2 SE in `A.md`, 1 SE in the CSVs. Negative means the first policy is better. For R in {IF4, Zou, Zou + FO6}:

- **(a) Degradation from coarsening:** R@g − R@1x16, for g ∈ {8x64, 16x64, 256x64}.
- **(b) Gain over the rule's own base,** at g ∈ {1x16, 8x64, 16x64, 256x64}: R@g − e2m1 for IF4; R@g − e2m1z for
  MixFP4 (Zou et al.); R@g − FourOverSix for Zou + FO6. Does the per-block advantage survive coarsening?
- **(c) OURS against the rule at the same realizable granularity:** tc@g − R@g, for g ∈ {8x64, 16x64, 256x64}.
- **(d) Retention of the 1x16 gain over FourOverSix** (the paper's §2.3 sentence: "how much of each rule's 1x16 gain
  over FourOverSix survives at 8x64, 16x64 and 256x64"):
  - gain_g = NLL(FourOverSix) − NLL(R@g), paired, ± 2 SE, on both corpora;
  - retained = gain_g / gain_1x16, with a 95 % paired bootstrap interval over windows (10,000 resamples, seed 0);
  - if R@1x16 is not better than FourOverSix by 2 SE, that is reported plainly and no fraction is given.
  - FourOverSix has the same FourOverSix activations as the arms, so there is no activation confound.

## A7: mechanism and secondary records

- **Within-tile mixing** (the paper's "fine-grained preferences are genuinely mixed within a weight matrix"):
  - computed from each rule's own 1x16 per-block choices, recorded inside the 1x16 arm's run, so no extra GPU work;
  - at 8x64, 16x64 and 256x64: the share of tiles whose 16-blocks disagree (some prefer E2M1, some the uniform grid),
    and the mean minority share inside those tiles;
  - per projection type and overall, for IF4, MixFP4 (Zou et al.) and MixFP4 (Zou et al.) + FourOverSix
    (`A_mixing.csv`).
- **Weight reconstruction error:**
  - per policy, the total squared error of the installed BF16 weights of the quantized Linears against the model's;
  - reported relative to FourOverSix and to NVFP4, side by side with ΔNLL, per arm and granularity. A lower
    tile-summed reconstruction error can still cost perplexity.
- WikiText-2 and C4 perplexity of every policy.
- Every arm against FourOverSix and against NVFP4 (the paper's NVFP4 row, with NVFP4 activations).
- **The mechanism evidence:** the share of weights in the uniform format (INT4 / E1M2 blocks for IF4 / Zou / Zou +
  FO6, E0M3 tiles for the maps), overall and per projection (q/k/v/o/gate/up/down), at 1x16, 8x64, 16x64 and 256x64.
  Also the count of zero-scale blocks.
- **`A.csv`, one row per policy:** model, rule, granularity, dnll_wiki, se_wiki, dnll_c4, se_c4, uniform_frac (all
  against FourOverSix). Then:
  - the ΔNLL columns against NVFP4;
  - the degradation against the same rule at 1x16 (dnll_*_vs_1x16, se_*_vs_1x16);
  - the base and the gain over it (dnll_*_vs_base, se_*_vs_base);
  - the two PPLs.
- **`A_primary.csv`:** every primary comparison in long form (model, comparison, rule, granularity, policy,
  reference, dnll_wiki, se_wiki, dnll_c4, se_c4). **`A_retention.csv`:** (d).
- **The paper tables** (`A_table_<model>_<corpus>.tex`, LaTeX, in the user's layout):
  - rows at 1x16 / 8x64 / 16x64 / 256x64: IF4 (Cook et al.); MixFP4 (Zou et al.); MixFP4 (Zou et al.) + FourOverSix
    (our variant); FlipQuant (ours), with "--" at 1x16 (never run at 1x16);
  - a reference line: NVFP4, NVFP4 weights + FourOverSix act., FourOverSix, BF16;
  - all from the same fake (c) path, with a note that ours is simulated there and differs slightly from the native
    main-table numbers;
  - Llama's WikiText-2 table for the paper; C4 and the other models for the appendix.
- **Names:** every output calls Zou et al.'s method "MixFP4 (Zou et al.)" and ours "FlipQuant (ours)" (the paper name
  since 2026-09-29; "TM-OPT+TC" only in parentheses where the code path matters), never a bare "MixFP4".
- **The A1 identity check** (IF4's FP candidate against MixFP4 (Zou et al.)'s E2M1 candidate on real weights) is
  reported with the other A1 checks.
- **Never selected or tuned on WikiText-2 or C4:** the formats are fixed rules, and the maps come from calibration data.

## Deviations (append-only)

1. **2026-09-29, after the Llama-3.1-8B runs: three fixes to the analysis script `A_analyze.py`. No GPU run was repeated
   and no number changed.**
   - **A crash.** The registered script (sha256 9c67ae37…) stopped at its window-identity check with a TypeError. The
     check put each policy's (WikiText-2, C4) token hashes into a set, and the reports store those hashes as lists,
     which cannot go into a set. The fix compares their JSON serialization. The check is the same: every policy must
     have evaluated the same windows, and it passes.
   - **A wrong sentence in `A.md`.** The A1 line said "the e2m1 reference is the E2M1 base of both". That contradicts
     A4: each rule has its own E2M1 base, e2m1 for IF4 and e2m1z for MixFP4 (Zou et al.). The text now says so. The
     computation already used the per-rule bases (`BASE = dict(if4='e2m1', zou='e2m1z', zoufo6='fo6')`).
   - **A missing line in `A.md`.** A7 asks for the count of zero-scale blocks. The script stored it in `A.json` but
     did not print it; `A.md` now does (0 in every Llama arm).
   - **New sha256 of `A_analyze.py`:** 5033e227f045bbe84ce833934d2a373ac95b6a96d7ec5a3cb0c3bddde775a91b. Every other
     registered file is unchanged (hashes re-checked).
   - **Not a deviation, for the record:** the 3-policy smoke test before registration found a variable-shadowing bug
     in `run_ppl_deploy.py`'s mixing aggregation (`tile` rebound inside the loop). It was fixed before registration;
     the registered hash includes the fix.

2. **2026-09-30, amendment before any of its GPU runs: a new arm, IF4 (Cook et al.) + FourOverSix (our variant),
   rule `if4fo6`,** for Llama-3.1-8B, Mistral-7B-v0.3 and Phi-4 at 1x16, 8x64, 16x64 and 256x64: 12 fake (c)
   policies. Requested by the user and relayed by nvfp4-razer-c9. Registered in `registration_if4fo6.json`.
   - **Our variant, not Cook et al.'s.** The user notes that Cook et al. position IF4 as an alternative to FourOverSix,
     not a combination; their code offers IF4 and FourOverSix as separate quantization schemes. The arm mirrors
     MixFP4 (Zou et al.) + FourOverSix.
   - **Definition** (`quantize/adaptive_formats.py`, rule `if4fo6`):
     - **FP candidate:** the repo's FourOverSix per 16-block (`quant_nvfp4_4over6`: the max/6 or max/4 scale by
       squared error). It is the same candidate as zoufo6's and the E2M1 base of our maps.
     - **Uniform candidate:** IF4's INT4, exactly as implemented and verified against the official reference in A1.
       The shared scale Δ = e4m3(max|x| / 6) relative to α = amax / (6·448); round(clamp(x / Δ · 1.16666666, −7, 7)),
       half to even; dequantized × Δ · 0.8571428571.
     - **Selection:** the lower squared error per block at 1x16, the tile sum at the coarse granularities. Ties keep
       the FP candidate (IF4's rule).
     - **Errors:** taken on the weights as installed (BF16), as for zoufo6.
     - **Activations:** FourOverSix per token, as in every A arm.
   - **Registered checks** (`check_if4fo6.py`, CPU, before the GPU runs; a failure stops them). On the 11 A1 modules:
     - the FP candidate equals zoufo6's bitwise;
     - the INT4 candidate equals IF4's as installed bitwise;
     - INT4 is chosen exactly where its error is strictly lower;
     - the tile rule at 1x16 equals the per-block rule.
     On 3 modules, every existing rule's installed weights at 1x16 and 16x64 equal those of the registered code
     (c923f32).
   - **During the runs:** the window hashes must equal the existing policies' (A_analyze asserts this).
   - **The existing references and arms are not re-run:** they are deterministic, and their records stand.
   - **Primary comparisons:** the same four as for the other rules.
     - (a) R@g − R@1x16;
     - (b) R@g − FourOverSix, its own base;
     - (c) FlipQuant (ours) − R@g;
     - (d) retention of the 1x16 gain over FourOverSix, with bootstrap intervals.
     - **Added:** the contrast IF4 + FO6 − Zou + FO6 at every g. It compares the two uniform candidates on the same
       FourOverSix base, each with its own tie rule.
   - **Mechanism and records:** within-tile mixing, the uniform share per projection, and the weight reconstruction
     error, as for the other rules.
   - **Outputs:**
     - A.csv, A_primary.csv, A_retention.csv and A_mixing.csv are extended;
     - every A_table_<model>_<corpus>.tex gains the row "IF4 (Cook et al.) + FourOverSix (our variant)";
     - A.md and SUMMARY.md are updated, and so is the A section of `results/paper_extra/SUMMARY_ACD_zh.md`.
   - **Changed files:**
     - `quantize/adaptive_formats.py`: the rule, and IF4's tie rule extended to it;
     - `experiments/paper_extra/A_formats.py`: the 4 policies per model;
     - `experiments/paper_extra/A_analyze.py`: the rule, its base, the contrast, the LaTeX row, and the check's line in
       A.md. Without the new records its CSVs are byte-identical, and the LaTeX tables only gain the row, as "--".
     - The new `check_if4fo6.py`. Hashes are in `registration_if4fo6.json`.

   **Follow-up, 2026-09-30 08:00 UTC: amendment 2 run.**
   - **The CPU check** (`if4fo6_check.json`) passed on all 11 modules. The existing rules' outputs are unchanged (30 of
     30).
   - **The runs:** 12 policies, 06:57–07:59 UTC, all exited 0.
   - **Registered checks:** the window hashes equal every other policy's (asserted by A_analyze). The re-run references
     still equal the Parts 2–3 records.
   - **Outputs updated:** the A outputs and the LaTeX tables with the new row, SUMMARY.md, and the A section of
     SUMMARY_ACD_zh.md.
   - **The records:** `ppl/<model>/if4fo6-<unit>.json`. The command logs (`commands.log` here and for D and the
     paper run) are now force-added; `*.log` is git-ignored, and they had not been committed before.
