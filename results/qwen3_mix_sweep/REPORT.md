# Qwen3 W4A16: nvfp4_4over6 vs mix_4_6 @ 8x64 (real perplexity)

The `results/decide_*` perplexity sweeps only ever covered the Llama models. This fills the gap on
the **Qwen3** family (qwen3-4b/8b/14b/32b), running the same real wikitext-2 + c4 perplexity eval
(`run_ppl_sweep.py`, seq 2048, W4A16, groupsize 16) that the Llama study used.

**Configs** (weights quantized, activations FP16):

- `fp16` — unquantized baseline
- `nvfp4_4over6` — plain NVFP4 with the FourOverSix block-scale search **← baseline to beat**
- `mix_4_6 @ 8x64` — E2M1(4over6)-vs-E0M3 per 8x64 type block, **argmin** MSE election
- `mix_4_6_m1 @ 8x64` — same, **margin** election z=1 (the decisive-margin rule)
- `mix_4_6_h1.5 @ 8x64` — same, **harm-ratio** election lambda=1.5

Env: layered venv `/home/u4320956/razer_venv` (transformers **4.57.3** to match the repo's
`qmodule_qwen3`, reusing cuda128's torch 2.6; cuda128 untouched). 1xH100 per model.

## Wikitext-2 perplexity

| model | fp16 | nvfp4_4over6 | mix_4_6 8x64 | mix_4_6_m1 8x64 | mix_4_6_h1.5 8x64 |
|---|---|---|---|---|---|
| qwen3-4b  | 13.6625 | 14.0407 | 14.5969 | 14.5071 | 14.5257 |
| qwen3-8b  |  9.7251 |  9.8898 |  9.9369 |  9.8933 |  9.8920 |
| qwen3-14b |  8.6477 |  8.7210 |  8.7469 |  8.7059 |  8.7152 |
| qwen3-32b |  7.6103 |  7.7995 |  7.8027 |  7.7837 |  7.7853 |

**Δ vs nvfp4_4over6** (negative = better than the 4over6 baseline):

| model | mix_4_6 8x64 | mix_4_6_m1 8x64 | mix_4_6_h1.5 8x64 |
|---|---|---|---|
| qwen3-4b  | **+0.5562** | +0.4664 | +0.4850 |
| qwen3-8b  | +0.0471 | +0.0035 | +0.0021 |
| qwen3-14b | +0.0259 | **-0.0150** | -0.0058 |
| qwen3-32b | +0.0032 | **-0.0157** | -0.0142 |

## C4 perplexity

| model | fp16 | nvfp4_4over6 | mix_4_6 8x64 | mix_4_6_m1 8x64 | mix_4_6_h1.5 8x64 |
|---|---|---|---|---|---|
| qwen3-4b  | 16.6425 | 17.0142 | 17.1092 | 17.1787 | 17.1435 |
| qwen3-8b  | 13.3004 | 13.5430 | 13.5709 | 13.5466 | 13.5456 |
| qwen3-14b | 12.0171 | 12.1947 | 12.1943 | 12.1716 | 12.1759 |
| qwen3-32b | 10.7823 | 11.0449 | 11.1226 | 11.0515 | 11.0581 |

**Δ vs nvfp4_4over6** (c4):

| model | mix_4_6 8x64 | mix_4_6_m1 8x64 | mix_4_6_h1.5 8x64 |
|---|---|---|---|
| qwen3-4b  | +0.0949 | +0.1645 | +0.1292 |
| qwen3-8b  | +0.0279 | +0.0036 | +0.0026 |
| qwen3-14b | -0.0004 | **-0.0231** | -0.0188 |
| qwen3-32b | +0.0777 | +0.0066 | +0.0132 |

## Findings — the Llama conclusion replicates on Qwen3

1. **Plain `mix_4_6` @ 8x64 (argmin) never meaningfully beats `nvfp4_4over6`, and usually loses.**
   Wikitext: +0.006 to +0.556; c4: -0.0004 to +0.095. The single "win" (14b c4, -0.0004) is noise.
   This is the same result the Llama W4A16 study found: an argmin E0M3 election on a realizable
   type block does not help.

2. **The decisive-margin election recovers the loss on the larger models, but the win is tiny.**
   `mix_4_6_m1` / `_h1.5` turn the argmin's deficit into ~neutral (8b) or a marginal win on 14b
   (-0.015 wiki / -0.023 c4) and 32b (-0.016 wiki, but +0.007 c4). This is exactly the
   "argmin loses, margin wins" pattern from `CLAUDE.md` — reproduced on a third model family — and,
   as there, the margin win is at the ~0.01-0.02 ppl level, i.e. small.

3. **qwen3-4b is hurt by the coarse type block regardless of election rule.** Every mix variant is
   +0.47 to +0.56 wikitext and +0.09 to +0.16 c4 worse than 4over6; the margin rules do not rescue
   it. The smallest model has the least redundancy, so committing an 8x64 tile to E0M3 costs the
   most. (The margin rules elect E0M3 *less*, which is why m1/h1.5 are a hair better than argmin on
   4b — but all remain far worse than staying on E2M1, i.e. `nvfp4_4over6`.)

4. **`nvfp4_4over6` remains the robust choice on Qwen3.** Straight quantization cost (fp16 ->
   nvfp4_4over6) is modest and monotone with size: +0.38 (4b) down to +0.07-0.19 (larger) wikitext.
   Adding the 8x64 E0M3 type block on top buys nothing reliable and risks a large regression on
   small models.

## Bottom line

Across four Qwen3 models and two datasets, the **8x64 `mix_4_6` type block does not beat plain
`nvfp4_4over6`**: argmin election loses (badly on qwen3-4b), and the decisive-margin variants only
reach a wash-to-marginal (~-0.02 ppl) improvement on the 8B+ models. Same verdict as the Llama
family — the free, robust win is the FourOverSix block-scale search (`nvfp4_4over6`); the E0M3 type
block is a model-dependent extra that, on Qwen3, does not pay off.

## Reproduce

```bash
for M in qwen3-4b qwen3-8b qwen3-14b qwen3-32b; do
  sbatch --export=ALL,MODEL=$M,DATASETS=wikitext+c4,TAG=full slurm/qwen3_mix_sweep.sbatch
done
# results: results/qwen3_mix_sweep/<model>_full.json
```
