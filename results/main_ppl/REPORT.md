# The main W4A4 perplexity table (tab:main-ppl): report

2026-10-02, NVFP4-RaZeR branch `main-ppl`. The protocol is `PROTOCOL.md`:
- registered at 3758782 before any GPU run;
- amendment 1 at 48acae6;
- deviation 3 and amendment 2 at 3d9ba43.
All runs ran on one RTX PRO 6000, 06:27–14:17 UTC.

## Scope of the delivered table

- **Native rows:** BF16, NVFP4 and FourOverSix, in convention (c) (per-token activations, NativeLinear).
- **Simulated rows** (fake (c), BF16 GEMM), in both variants:
  - IF4 (Cook et al.) 1x16 and MixFP4 (Zou et al.) 1x16, **W+A**: the rule on the weights and on the activations, with
    a per-token tensor scale for the activations;
  - IF4 and MixFP4 **(W) + FO6 act**: the rule on the weights, with per-token FourOverSix activations.
- **Columns:** 7. The six models are Qwen3-1.7B, Qwen3-8B, Mistral-7B, Nemotron-Nano-9B-v2, Phi-4 and Qwen3.8-27B. For
  Mistral there are two variants: Instruct-v0.3 (the primary column) and base v0.3 (the secondary column).
- **Deferred:** the FlipQuant rows (deviation 3; the user will change the calibration). Their records are kept and
  listed in `deferred.md`, never tabulated.

**Files:**
- `table_main_instruct.tex` and `table_main_base.tex`: the main table, two decimals. Bold marks the lower PPL of the
  native NVFP4 and FourOverSix.
- `table_dnll_appendix.tex`: Appendix D, ΔNLL ± 2 SE vs native FourOverSix, plus vs fake FourOverSix in brackets for the
  simulated rows.
- `tables.md`: every number, with four decimals.
- `main_ppl.json` and `main_ppl.csv`: every PPL, and every paired ΔNLL with its SE.
- `deferred.md`, `runs/` (all the records), `exports/`, `commands.log`.

## The numbers (PPL; WikiText-2 / C4)

| row | Qwen3-1.7B | Qwen3-8B | Mistral-7B-Instruct | Mistral-7B (base) | Nemotron-Nano-9B-v2 | Phi-4 | Qwen3.8-27B |
|---|---|---|---|---|---|---|---|
| BF16 | 16.7162 / 19.2463 | 9.7251 / 13.3004 | 5.4961 / 8.1351 | 5.3182 / 7.8306 | 8.0863 / 11.1535 | 6.4615 / 10.3098 | 7.0509 / 9.8935 |
| NVFP4 | 18.8713 / 21.0349 | 10.0484 / 13.7705 | 5.7304 / 8.4156 | 5.5506 / 8.0895 | 8.4596 / 11.5420 | 6.6934 / 10.5795 | 7.5506 / 10.2185 |
| FourOverSix | 19.4745 / 20.9072 | 10.0071 / 13.7575 | 5.7083 / 8.3987 | 5.5224 / 8.0623 | 8.4227 / 11.5093 | 6.6617 / 10.5441 | 7.3215 / 10.1869 |
| IF4 1x16 (W+A)* | 19.6007 / 20.9536 | 9.8852 / 13.6027 | 5.6703 / 8.3621 | 5.4989 / 8.0396 | 8.3930 / 11.4530 | 6.6346 / 10.5108 | 7.3353 / 10.1580 |
| MixFP4 (Zou) 1x16 (W+A)* | 19.5027 / 21.5547 | 9.8615 / 13.5916 | 5.6629 / 8.3689 | 5.4978 / 8.0417 | 8.4083 / 11.4555 | 6.6392 / 10.5066 | 7.3569 / 10.1608 |
| IF4 1x16 (W) + FO6 act* | 19.5443 / 21.0106 | 9.8902 / 13.6027 | 5.6809 / 8.3673 | 5.5092 / 8.0531 | 8.4136 / 11.4707 | 6.6451 / 10.5195 | 7.3463 / 10.1759 |
| MixFP4 (Zou) 1x16 (W) + FO6 act* | 19.5332 / 21.5259 | 9.8987 / 13.6112 | 5.6706 / 8.3746 | 5.5078 / 8.0541 | 8.4246 / 11.4769 | 6.6551 / 10.5184 | 7.3338 / 10.1711 |
| FourOverSix, simulated (appendix)* | 19.4312 / 20.8863 | 10.0110 / 13.7516 | 5.7090 / 8.3986 | 5.5210 / 8.0681 | 8.4212 / 11.5075 | 6.6627 / 10.5473 | 7.3016 / 10.1882 |

\* simulated (fake (c)).

**Loss recovered**, the share of NVFP4's log-PPL loss vs BF16 that a row removes, summed over the 12 model–corpus pairs
(NVFP4 0 %, BF16 100 %):

| row | Instruct variant | base variant |
|---|---:|---:|
| FourOverSix | 5.8 % | 6.3 % |
| IF4 1x16 (W+A)* | 13.0 % | 12.8 % |
| MixFP4 (Zou) 1x16 (W+A)* | 8.8 % | 8.5 % |
| IF4 1x16 (W) + FO6 act* | 10.9 % | 10.5 % |
| MixFP4 (Zou) 1x16 (W) + FO6 act* | 6.7 % | 6.1 % |

## What the numbers say

### NVFP4 vs FourOverSix (Appendix D; ΔNLL = NVFP4 − FourOverSix, nats per token, ± 2 SE)

- **FourOverSix is better in 12 of the 14 pairs.**
- **Qwen3-8B C4:** the two are not distinguishable (+0.0009 ± 0.0018).
- **Qwen3-1.7B WikiText-2:** NVFP4 is significantly better (−0.0315 ± 0.0112).

That one pair is why FourOverSix's loss recovered is low:
- On Qwen3-1.7B WikiText-2, FourOverSix "recovers" −25.9 %, and that pair's NVFP4 loss (0.121 nats) is the largest in
  the table.
- **Without Qwen3-1.7B,** FourOverSix's loss recovered is 15.7 % (Instruct variant) and 16.3 % (base variant).
- In the other direction, FourOverSix recovers 45 % of NVFP4's loss on Qwen3.8-27B WikiText-2.

### The simulated rows

**The simulator is checked.** The simulated FourOverSix equals the native FourOverSix within 2 SE in all 14 pairs. So
fake (c) and the native kernels agree at this resolution, and comparing the simulated rows with native FourOverSix is
fair. Every comparison with the simulated FourOverSix is also in the appendix table.

How each simulated row compares with native FourOverSix, over the 14 pairs:

| row | significantly better | not significant | significantly worse |
|---|---:|---:|---:|
| IF4 (W+A) | 11 | 3 | 0 |
| MixFP4 (W+A) | 11 | 2 | 1 |
| IF4 (W) + FO6 act | 10 | 3 | 1 |
| MixFP4 (W) + FO6 act | 9 | 4 | 1 |

- **The exceptions are mostly Qwen3-1.7B (both corpora) and Qwen3.8-27B WikiText-2.** The W-only rows are also not
  significant on Nemotron WikiText-2, and MixFP4 (W) + FO6 act on Phi-4 WikiText-2.
- **On Qwen3-1.7B C4,** MixFP4 is much worse than FourOverSix in both variants (+0.0305 and +0.0292 nats).
- **Applying the rule to the activations too helps both rules.** W+A beats W-only in loss recovered: IF4 13.0 % vs
  10.9 %, MixFP4 8.8 % vs 6.7 % (Instruct variant).
- **IF4 vs MixFP4:** IF4 is the stronger rule in both variants.

### Notes for the paper text

- **The per-token tensor scale of the W+A activations** was the coordinator's protocol decision, for parity with the
  per-token activations of the native rows. Both papers compute that scale over the whole tensor (§4 of the protocol).
- **The reuse rule held.** The Phi-4, Qwen3.8-27B and base-Mistral native and BF16 rows are the paper's step 03
  records. They were reused because re-runs were bitwise equal per window: Phi-4 bf16/fo6, base Mistral bf16/fo6, and
  Qwen3.8-27B fo6.
- **The Mistral column:** the two variants differ only in it. The primary is the Instruct model of ~/flipquant's
  registry; base v0.3 is the paper's earlier model.

## Checks (all passed)

1. **Activation rules, before registration** (CPU; `act_rules_check.json`).
   - Per-token IF4 equals Cook et al.'s official reference (fouroversix @ dadfad69) run on each row alone. This holds
     bitwise, in values and choices, on 3,076 rows: 12 real Qwen3-1.7B activations plus 4 edge rows.
   - Per-token IF4 and Zou both equal Experiment A's per-tensor rule on each row.
2. **Exports of the four new models** (export_map_artifact.py --ownership, as step 02):
   - the weights equal their calibration records;
   - the packed weights equal the fake-quant weights bitwise;
   - the ownership check is exact, with 0 format mismatches.
3. **Coverage:** every native row ran every quantized Linear natively in every forward (run_ppl_deploy's coverage
   check), including Nemotron's Mamba in_proj/out_proj.
4. **Windows:** within each model, every row evaluated the same windows (token hashes; asserted by analyze.py).
5. **Reuse rechecks** (`RECHECK` in commands.log): bitwise equal per window on both corpora:
   - Phi-4 bf16 and fo6;
   - base Mistral bf16 and fo6;
   - Qwen3.8-27B fo6. Its recheck moved from ours-8x64 under deviation 3.
6. **Experiment A cross-checks** (`CROSSCHECK-A`): our records of the identical setting equal A's bitwise on both
   corpora:
   - if4w, zouw and fo6-fake for Phi-4;
   - if4w, zouw and fo6-fake for base Mistral.
7. **No run failed.** The Nemotron `zou` run was left running when the queue stopped (deviation 3), and it completed.

## Deferred records (`deferred.md`; not part of this table)

- **FlipQuant 8x64 / 16x64 / 256x64:** the rows of Qwen3-1.7B, Qwen3-8B, Mistral-7B-Instruct and Nemotron, from the
  delivered maps. They will be superseded once the new calibration exists.
  - **Observed in them:** on both post-trained Qwen3 models, every FlipQuant map gives a lower WikiText-2 PPL than BF16.
  - **It comes from the maps, not the kernels:** the Qwen3-1.7B fake-map diagnostic reproduces it.
  - **For the new calibration:** check whether this reappears.
- **The Qwen3-1.7B fake-map diagnostic,** and the Nemotron feasibility smoke (2 windows per corpus).

## Deviations and amendments (PROTOCOL.md)

1. `analyze.py` formatting: decimals, nats in the appendix, tables.md.
2. A diagnostic outside the table (the Qwen3-1.7B fake map), and analysis robustness.
3. **The scope cut:** FlipQuant deferred, and Qwen3.8-27B's recheck moved to fo6.
- **Amendment 1:** the W-only rule rows, with the Experiment A cross-check.
- **Amendment 2:** the IF4 / MixFP4 rows restored in both variants, with fake FourOverSix, for all columns.
