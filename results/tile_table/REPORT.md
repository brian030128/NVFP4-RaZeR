# SM120 tile table for the RTX PRO 6000 — report

Protocol: `PROTOCOL.md`, registered 2026-09-28T13:25:44Z (`registration.json`), before the tuning.
- **One deviation:** a same-process A/B was added after the re-measurement. See the protocol.

## Summary

**The table.** `sm120/configs/nvidia_rtx_pro_6000_blackwell_workstation_edition.json`, with the raw timings in `.raw.json`.
- **What was tuned:** `sm120/bench/tune_tiles.py`, families mixed and stock, all 14 buckets (T = 1 to 8192).
  - Shapes: the 16 distinct text-Linear shapes of Llama-3.1-8B, Mistral-7B-v0.3, Phi-4 and Qwen3.8-27B. Phi-4 and Qwen
    were added to `MODEL_SHAPES`.
  - Each cell: every width, CUPTI kernel time, median of 20 calls. The run took 4 min on the idle GPU.
- **Mixed-family E0M3 tags:** per projection, the module with the most E0M3 tiles in the final TM-OPT+TC 16x64 map (the
  new `--maps` option). The tags per shape are in the table's `meta.tags`.
- **Kernels:** the repository's sm120 builds. Their SASS is identical, config for config, to the copy the Part R latency
  harness uses: 10 of 10 configurations. The library files differ only in host code.

**Verification, all passed.**
1. `KernelSet('mixed')` and `KernelSet('stock')` load the table on this GPU (`table_source` set, 16 shapes).
2. `tests/test_select.py`: 12 passed, none skipped.
3. **Logits, table against fallback:** bitwise identical in all 24 forwards per policy (`check_logits.json`).
   - Policies: Llama-3.1-8B TM-OPT+TC 16x64 (`auto`) and FourOverSix (`auto_stock`).
   - Inputs: 3 WikiText-2 windows at prefix lengths 1 to 2048.
   - The table run exercised other widths. For TC 16x64:

     | GEMM calls | width 16 | 32 | 64 | 128 |
     |---|---:|---:|---:|---:|
     | table | 1,536 | 1,152 | 768 | 1,920 |
     | fallback | 672 | 672 | 672 | 3,360 |

**Where the table departs from the fallback** (the narrowest width holding the tokens):
- **How often:** mixed departs in 53 of 224 (shape, bucket) cells, stock in 48. All departures are at buckets 32 to
  4096; there are none at T ≤ 16 or at T = 8192.
- **Kernel-time gain:** median 24 % (mixed) and 29 % (stock); maximum 56 % and 61 %.
- **Mixed and stock choose different widths** in 9 cells. The full list is below.
- **Summed over a forward** (kernel times × module counts), the GEMM time saved:
  - T = 512: 0.38 / 0.40 ms on Llama and Mistral (5.5 / 6.0 % of their GEMM time, mixed / stock); 1.10 / 1.35 ms on
    Qwen (5.1 / 6.5 %); none on Phi-4.
  - T = 2048: only Qwen, 0.62 / 0.73 ms (0.8 / 1.0 %).
  - T = 8192 and T = 1: none, for any model.

**Part R's operating points.**
- **Decode at batch 1 (T = 1) and prefill 4x2048 (T = 8192): no width changes, for any model.** Those Part R numbers
  stand as measured.
- **Prefill 1x2048 (T = 2048): one change.** Qwen's 48x5120 linear-attention projections (`in_proj_a`, `in_proj_b`)
  go from 128 to 64, in both families. Nothing else changes at T = 2048.
  - **Re-measured with the table:** the same harness and Part R's settings for Qwen, its six width-selecting policies,
    5 shuffled rounds (`remeasure.md`).
  - **Result: the effect is not resolvable.**
    - The table's expected change is about −0.1 % of the ~760 ms forward.
    - The new medians moved from +0.5 % to +14.6 %, and even 4x2048, where no width changed, moved −0.2 % on every
      policy.
    - Qwen's single-sequence prefill is bimodal in both sessions: repetitions land near 760 ms or near 880 ms.
  - **Same-process A/B** (deviation 1; `ab/`): one load, table and fallback alternated 5 times each.
    - The block medians still scatter by tens of ms.
    - The block minima differ by −2.7 ms (FourOverSix, table faster) and +2.5 ms (TC 16x64), i.e. within ±0.35 %.
      This is consistent with the 0.6–0.7 ms expected from the kernel timings.
  - **So the Part R numbers for Qwen at 1x2048 stand within their noise.** The change applies to both families
    alike, so the MixFP4-vs-stock overheads are not affected.
- **Prefill 1x512 (T = 512)** is in Part R's tables but not among its operating points. It changes for Llama and
  Mistral (`k_proj`/`v_proj`) and for Qwen, with the GEMM savings above: about 1 % of Llama's 31 ms forward. It was not
  re-measured, per the task.

**8x64 (`n8k64_wB`) has no width variants:** a single build, so the table does not apply. Every 8x64 number, Part R's
included, is unaffected.

**Estimate for a decode/prefill sweep (not run).** Decode at batch 1/8/32/64, prefill 1x512, 1x2048 and 4x2048; the Part R
design (one process per model × policy × round, 5 shuffled rounds).
- **Per-process time:** Part R's recorded medians (`commands_r2.log`) are 24 s (Llama, Mistral), 39 s (Phi-4; these
  include batch-1 decode) and 83 s (Qwen, prefill only). Decode at three more batch sizes adds about 12 s (7–14B) and
  about 25 s (Qwen).
- **Qwen:** its decode would need HF generate (eager), since the SM120 decode does not support its hybrid cache.

| policy set | Llama | Mistral | Phi-4 | Qwen | total |
|---|---:|---:|---:|---:|---:|
| 8 paper policies (BF16; NVFP4 and FourOverSix on stock_wA and stock_wB; TC 8x64, 16x64, 256x64) | ≈ 24 min | ≈ 24 min | ≈ 36 min | ≈ 72 min | ≈ 2.6 h |
| Part R's 11 policies (the 8 plus TM-OPT 8x64, 16x64, 256x64) | | | | | ≈ 3.6 h |

Three rounds instead of five: ×0.6.

GPU NVIDIA RTX PRO 6000 Blackwell Workstation Edition; 16 shapes x 14 buckets; kernel time = CUPTI median of 20 calls. The fallback is select.fallback_width: the narrowest width holding the tokens.

### mixed: departures from the fallback (53 of 224 (shape, bucket) cells)

| shape (out x in) | T bucket | fallback | table | us fallback | us table | gain | used by |
|---|---:|---:|---:|---:|---:|---:|---|
| 10240x5120 | 128 | 128 | 64 | 19.42 | 15.84 | 18.4 % | qwen27b.in_proj_qkv |
| 1024x4096 | 32 | 32 | 16 | 8.10 | 7.20 | 11.1 % | llama8b.k_proj, llama8b.v_proj, mistral7b.k_proj, mistral7b.v_proj |
| 1024x4096 | 64 | 64 | 16 | 10.32 | 7.23 | 29.9 % | llama8b.k_proj, llama8b.v_proj, mistral7b.k_proj, mistral7b.v_proj |
| 1024x4096 | 128 | 128 | 16 | 15.52 | 7.42 | 52.2 % | llama8b.k_proj, llama8b.v_proj, mistral7b.k_proj, mistral7b.v_proj |
| 1024x4096 | 256 | 128 | 32 | 15.58 | 8.35 | 46.4 % | llama8b.k_proj, llama8b.v_proj, mistral7b.k_proj, mistral7b.v_proj |
| 1024x4096 | 512 | 128 | 32 | 15.62 | 9.73 | 37.7 % | llama8b.k_proj, llama8b.v_proj, mistral7b.k_proj, mistral7b.v_proj |
| 1024x4096 | 1024 | 128 | 64 | 15.81 | 11.66 | 26.2 % | llama8b.k_proj, llama8b.v_proj, mistral7b.k_proj, mistral7b.v_proj |
| 1024x5120 | 32 | 32 | 16 | 9.44 | 8.35 | 11.5 % | qwen27b.k_proj, qwen27b.v_proj |
| 1024x5120 | 64 | 64 | 16 | 12.06 | 8.43 | 30.1 % | qwen27b.k_proj, qwen27b.v_proj |
| 1024x5120 | 128 | 128 | 16 | 18.58 | 8.70 | 53.2 % | qwen27b.k_proj, qwen27b.v_proj |
| 1024x5120 | 256 | 128 | 32 | 18.61 | 9.76 | 47.6 % | qwen27b.k_proj, qwen27b.v_proj |
| 1024x5120 | 512 | 128 | 32 | 18.69 | 11.86 | 36.5 % | qwen27b.k_proj, qwen27b.v_proj |
| 1024x5120 | 1024 | 128 | 64 | 18.99 | 13.70 | 27.9 % | qwen27b.k_proj, qwen27b.v_proj |
| 12288x5120 | 256 | 128 | 64 | 38.27 | 37.97 | 0.8 % | qwen27b.q_proj |
| 4096x14336 | 32 | 32 | 16 | 22.78 | 19.95 | 12.4 % | llama8b.down_proj, mistral7b.down_proj |
| 4096x14336 | 64 | 64 | 32 | 30.05 | 23.14 | 23.0 % | llama8b.down_proj, mistral7b.down_proj |
| 4096x14336 | 128 | 128 | 32 | 46.99 | 28.21 | 40.0 % | llama8b.down_proj, mistral7b.down_proj |
| 4096x14336 | 256 | 128 | 64 | 47.31 | 31.81 | 32.8 % | llama8b.down_proj, mistral7b.down_proj |
| 4096x14336 | 1024 | 128 | 64 | 117.50 | 117.36 | 0.1 % | llama8b.down_proj, mistral7b.down_proj |
| 4096x4096 | 32 | 32 | 16 | 8.13 | 7.39 | 9.1 % | llama8b.q_proj, llama8b.o_proj, mistral7b.q_proj, mistral7b.o_proj |
| 4096x4096 | 64 | 64 | 32 | 10.43 | 8.38 | 19.7 % | llama8b.q_proj, llama8b.o_proj, mistral7b.q_proj, mistral7b.o_proj |
| 4096x4096 | 128 | 128 | 32 | 15.57 | 10.50 | 32.6 % | llama8b.q_proj, llama8b.o_proj, mistral7b.q_proj, mistral7b.o_proj |
| 4096x4096 | 256 | 128 | 64 | 15.71 | 11.42 | 27.3 % | llama8b.q_proj, llama8b.o_proj, mistral7b.q_proj, mistral7b.o_proj |
| 48x5120 | 32 | 32 | 16 | 9.38 | 8.19 | 12.7 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 48x5120 | 64 | 64 | 16 | 11.89 | 8.22 | 30.9 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 48x5120 | 128 | 128 | 16 | 18.56 | 8.26 | 55.5 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 48x5120 | 256 | 128 | 16 | 18.46 | 8.25 | 55.3 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 48x5120 | 512 | 128 | 32 | 18.50 | 9.46 | 48.9 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 48x5120 | 1024 | 128 | 32 | 18.50 | 9.95 | 46.2 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 48x5120 | 2048 | 128 | 64 | 18.56 | 12.06 | 35.0 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 48x5120 | 4096 | 128 | 64 | 18.66 | 16.77 | 10.1 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 5120x17408 | 32 | 32 | 16 | 27.15 | 23.97 | 11.7 % | qwen27b.down_proj |
| 5120x17408 | 64 | 64 | 32 | 35.71 | 27.17 | 23.9 % | qwen27b.down_proj |
| 5120x17408 | 128 | 128 | 64 | 56.42 | 36.18 | 35.9 % | qwen27b.down_proj |
| 5120x17408 | 256 | 128 | 64 | 56.75 | 45.81 | 19.3 % | qwen27b.down_proj |
| 5120x17920 | 32 | 32 | 16 | 27.71 | 24.45 | 11.8 % | phi4.down_proj |
| 5120x17920 | 64 | 64 | 32 | 36.19 | 27.92 | 22.9 % | phi4.down_proj |
| 5120x17920 | 128 | 128 | 64 | 57.47 | 36.80 | 36.0 % | phi4.down_proj |
| 5120x17920 | 256 | 128 | 64 | 58.67 | 46.97 | 19.9 % | phi4.down_proj |
| 5120x5120 | 32 | 32 | 16 | 9.68 | 8.90 | 8.1 % | phi4.o_proj |
| 5120x5120 | 64 | 64 | 32 | 12.40 | 9.98 | 19.5 % | phi4.o_proj |
| 5120x5120 | 128 | 128 | 64 | 18.80 | 12.80 | 31.9 % | phi4.o_proj |
| 5120x5120 | 256 | 128 | 64 | 19.15 | 15.30 | 20.1 % | phi4.o_proj |
| 5120x6144 | 32 | 32 | 16 | 11.10 | 10.11 | 8.9 % | qwen27b.o_proj, qwen27b.out_proj |
| 5120x6144 | 64 | 64 | 32 | 14.37 | 11.62 | 19.1 % | qwen27b.o_proj, qwen27b.out_proj |
| 5120x6144 | 128 | 128 | 64 | 21.89 | 14.88 | 32.0 % | qwen27b.o_proj, qwen27b.out_proj |
| 5120x6144 | 256 | 128 | 64 | 22.38 | 17.95 | 19.8 % | qwen27b.o_proj, qwen27b.out_proj |
| 6144x5120 | 32 | 32 | 16 | 9.76 | 9.22 | 5.5 % | qwen27b.in_proj_z |
| 6144x5120 | 64 | 64 | 32 | 12.45 | 10.18 | 18.2 % | qwen27b.in_proj_z |
| 6144x5120 | 128 | 128 | 64 | 18.83 | 12.99 | 31.0 % | qwen27b.in_proj_z |
| 6144x5120 | 512 | 128 | 64 | 38.27 | 37.90 | 1.0 % | qwen27b.in_proj_z |
| 7680x5120 | 64 | 64 | 32 | 12.51 | 11.33 | 9.4 % | phi4.qkv_proj |
| 7680x5120 | 128 | 128 | 64 | 19.09 | 13.39 | 29.9 % | phi4.qkv_proj |

### stock: departures from the fallback (48 of 224 (shape, bucket) cells)

| shape (out x in) | T bucket | fallback | table | us fallback | us table | gain | used by |
|---|---:|---:|---:|---:|---:|---:|---|
| 10240x5120 | 128 | 128 | 64 | 18.88 | 15.36 | 18.6 % | qwen27b.in_proj_qkv |
| 1024x4096 | 32 | 32 | 16 | 6.70 | 6.24 | 6.9 % | llama8b.k_proj, llama8b.v_proj, mistral7b.k_proj, mistral7b.v_proj |
| 1024x4096 | 64 | 64 | 16 | 8.83 | 6.30 | 28.7 % | llama8b.k_proj, llama8b.v_proj, mistral7b.k_proj, mistral7b.v_proj |
| 1024x4096 | 128 | 128 | 16 | 15.17 | 6.53 | 57.0 % | llama8b.k_proj, llama8b.v_proj, mistral7b.k_proj, mistral7b.v_proj |
| 1024x4096 | 256 | 128 | 32 | 15.20 | 6.94 | 54.3 % | llama8b.k_proj, llama8b.v_proj, mistral7b.k_proj, mistral7b.v_proj |
| 1024x4096 | 512 | 128 | 64 | 15.33 | 9.12 | 40.5 % | llama8b.k_proj, llama8b.v_proj, mistral7b.k_proj, mistral7b.v_proj |
| 1024x4096 | 1024 | 128 | 64 | 15.52 | 11.26 | 27.4 % | llama8b.k_proj, llama8b.v_proj, mistral7b.k_proj, mistral7b.v_proj |
| 1024x5120 | 32 | 32 | 16 | 7.81 | 7.23 | 7.4 % | qwen27b.k_proj, qwen27b.v_proj |
| 1024x5120 | 64 | 64 | 16 | 10.37 | 7.33 | 29.3 % | qwen27b.k_proj, qwen27b.v_proj |
| 1024x5120 | 128 | 128 | 16 | 18.24 | 7.78 | 57.3 % | qwen27b.k_proj, qwen27b.v_proj |
| 1024x5120 | 256 | 128 | 32 | 18.29 | 8.06 | 55.9 % | qwen27b.k_proj, qwen27b.v_proj |
| 1024x5120 | 512 | 128 | 64 | 18.34 | 10.72 | 41.5 % | qwen27b.k_proj, qwen27b.v_proj |
| 1024x5120 | 1024 | 128 | 64 | 18.58 | 13.14 | 29.3 % | qwen27b.k_proj, qwen27b.v_proj |
| 12288x5120 | 256 | 128 | 64 | 37.34 | 35.49 | 5.0 % | qwen27b.q_proj |
| 4096x14336 | 32 | 32 | 16 | 17.89 | 16.99 | 5.0 % | llama8b.down_proj, mistral7b.down_proj |
| 4096x14336 | 64 | 64 | 32 | 24.70 | 18.43 | 25.4 % | llama8b.down_proj, mistral7b.down_proj |
| 4096x14336 | 128 | 128 | 64 | 45.68 | 25.17 | 44.9 % | llama8b.down_proj, mistral7b.down_proj |
| 4096x14336 | 256 | 128 | 64 | 45.95 | 31.57 | 31.3 % | llama8b.down_proj, mistral7b.down_proj |
| 4096x14336 | 1024 | 128 | 64 | 114.65 | 112.85 | 1.6 % | llama8b.down_proj, mistral7b.down_proj |
| 4096x4096 | 32 | 32 | 16 | 6.78 | 6.66 | 1.8 % | llama8b.q_proj, llama8b.o_proj, mistral7b.q_proj, mistral7b.o_proj |
| 4096x4096 | 64 | 64 | 32 | 8.90 | 7.04 | 20.9 % | llama8b.q_proj, llama8b.o_proj, mistral7b.q_proj, mistral7b.o_proj |
| 4096x4096 | 128 | 128 | 64 | 15.23 | 9.18 | 39.7 % | llama8b.q_proj, llama8b.o_proj, mistral7b.q_proj, mistral7b.o_proj |
| 4096x4096 | 256 | 128 | 64 | 15.38 | 10.91 | 29.1 % | llama8b.q_proj, llama8b.o_proj, mistral7b.q_proj, mistral7b.o_proj |
| 48x5120 | 32 | 32 | 16 | 7.74 | 7.10 | 8.3 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 48x5120 | 64 | 64 | 16 | 10.30 | 7.10 | 31.1 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 48x5120 | 128 | 128 | 16 | 18.18 | 7.10 | 60.9 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 48x5120 | 256 | 128 | 16 | 18.11 | 7.14 | 60.6 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 48x5120 | 512 | 128 | 32 | 18.11 | 7.71 | 57.4 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 48x5120 | 1024 | 128 | 32 | 18.18 | 9.31 | 48.8 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 48x5120 | 2048 | 128 | 64 | 18.18 | 10.59 | 41.7 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 48x5120 | 4096 | 128 | 64 | 18.27 | 16.96 | 7.2 % | qwen27b.in_proj_a, qwen27b.in_proj_b |
| 5120x17408 | 64 | 64 | 32 | 29.57 | 23.14 | 21.7 % | qwen27b.down_proj |
| 5120x17408 | 128 | 128 | 64 | 55.07 | 30.34 | 44.9 % | qwen27b.down_proj |
| 5120x17408 | 256 | 128 | 64 | 55.90 | 44.32 | 20.7 % | qwen27b.down_proj |
| 5120x17920 | 64 | 64 | 32 | 30.40 | 23.55 | 22.5 % | phi4.down_proj |
| 5120x17920 | 128 | 128 | 64 | 56.64 | 31.07 | 45.1 % | phi4.down_proj |
| 5120x17920 | 256 | 128 | 64 | 57.30 | 45.79 | 20.1 % | phi4.down_proj |
| 5120x5120 | 64 | 64 | 32 | 10.56 | 8.83 | 16.4 % | phi4.o_proj |
| 5120x5120 | 128 | 128 | 64 | 18.46 | 11.04 | 40.2 % | phi4.o_proj |
| 5120x5120 | 256 | 128 | 64 | 18.75 | 14.91 | 20.5 % | phi4.o_proj |
| 5120x6144 | 64 | 64 | 32 | 12.22 | 10.40 | 14.9 % | qwen27b.o_proj, qwen27b.out_proj |
| 5120x6144 | 128 | 128 | 64 | 21.47 | 12.78 | 40.5 % | qwen27b.o_proj, qwen27b.out_proj |
| 5120x6144 | 256 | 128 | 64 | 21.95 | 17.63 | 19.7 % | qwen27b.o_proj, qwen27b.out_proj |
| 6144x5120 | 64 | 64 | 32 | 10.69 | 9.60 | 10.2 % | qwen27b.in_proj_z |
| 6144x5120 | 128 | 128 | 64 | 18.58 | 11.47 | 38.3 % | qwen27b.in_proj_z |
| 6144x5120 | 512 | 128 | 64 | 37.39 | 35.06 | 6.2 % | qwen27b.in_proj_z |
| 7680x5120 | 64 | 64 | 32 | 10.78 | 10.72 | 0.6 % | phi4.qkv_proj |
| 7680x5120 | 128 | 128 | 64 | 18.72 | 12.93 | 30.9 % | phi4.qkv_proj |

### Where mixed and stock choose different widths

| shape | T bucket | mixed | stock | used by |
|---|---:|---:|---:|---|
| 1024x4096 | 512 | 32 | 64 | llama8b.k_proj, llama8b.v_proj, mistral7b.k_proj, mistral7b.v_proj |
| 1024x5120 | 512 | 32 | 64 | qwen27b.k_proj, qwen27b.v_proj |
| 4096x14336 | 128 | 32 | 64 | llama8b.down_proj, mistral7b.down_proj |
| 4096x4096 | 128 | 32 | 64 | llama8b.q_proj, llama8b.o_proj, mistral7b.q_proj, mistral7b.o_proj |
| 5120x17408 | 32 | 16 | 32 | qwen27b.down_proj |
| 5120x17920 | 32 | 16 | 32 | phi4.down_proj |
| 5120x5120 | 32 | 16 | 32 | phi4.o_proj |
| 5120x6144 | 32 | 16 | 32 | qwen27b.o_proj, qwen27b.out_proj |
| 6144x5120 | 32 | 16 | 32 | qwen27b.in_proj_z |

### Part R's operating points

| point | family | changed (shape: fallback -> table) |
|---|---|---|
| decode batch 1 (T = 1) | mixed | none |
| decode batch 1 (T = 1) | stock | none |
| prefill 1x512 | mixed | 1024x4096: 128 -> 32; 1024x5120: 128 -> 32; 48x5120: 128 -> 32; 6144x5120: 128 -> 64 |
| prefill 1x512 | stock | 1024x4096: 128 -> 64; 1024x5120: 128 -> 64; 48x5120: 128 -> 32; 6144x5120: 128 -> 64 |
| prefill 1x2048 | mixed | 48x5120: 128 -> 64 |
| prefill 1x2048 | stock | 48x5120: 128 -> 64 |
| prefill 4x2048 | mixed | none |
| prefill 4x2048 | stock | none |

The 8x64 policy (n8k64_wB) is a single build with no width variants; the table does not apply to it.

## Re-measurement (Qwen3.8-27B)

Median over 5 rounds of the per-round median ms (Part R harness, bench_latency.py). The table changes the 48x5120 linear-attention projections at T = 512 (128 -> 32) and T = 2048 (128 -> 64), and at T = 512 also 1024x5120 and 6144x5120; nothing at T = 8192.

| policy | 1x512 old | 1x512 new | change | 1x2048 old | 1x2048 new | change | 4x2048 old | 4x2048 new | change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nvfp4@stock_wA | 438.27 | 445.75 | +1.71 % | 763.47 | 772.94 | +1.24 % | 1640.68 | 1636.55 | -0.25 % |
| fo6@stock_wA | 433.78 | 454.50 | +4.78 % | 754.59 | 864.80 | +14.61 % | 1641.44 | 1637.11 | -0.26 % |
| tmopt-16x64@n16k64_wA | 442.58 | 449.60 | +1.59 % | 769.18 | 775.23 | +0.79 % | 1655.04 | 1651.01 | -0.24 % |
| tmopt-256x64@n16k64_wA | 469.90 | 443.59 | -5.60 % | 758.10 | 778.38 | +2.67 % | 1654.90 | 1651.82 | -0.19 % |
| tc-16x64@n16k64_wA | 447.52 | 444.44 | -0.69 % | 767.72 | 771.19 | +0.45 % | 1655.45 | 1651.85 | -0.22 % |
| tc-256x64@n16k64_wA | 439.48 | 454.24 | +3.36 % | 757.22 | 771.53 | +1.89 % | 1656.27 | 1652.09 | -0.25 % |

Overhead paired within a round (median over rounds), old -> new:

| map | vs | 1x512 | 1x2048 | 4x2048 |
|---|---|---|---|---|
| tc-16x64@n16k64_wA | fo6@stock_wA | +0.85 % -> -2.79 % | +2.25 % -> -1.94 % | +0.91 % -> +0.92 % |
| tc-16x64@n16k64_wA | nvfp4@stock_wA | +0.51 % -> -0.89 % | -1.23 % -> -0.14 % | +1.03 % -> +1.01 % |
| tc-256x64@n16k64_wA | fo6@stock_wA | -0.50 % -> -2.39 % | +0.28 % -> -11.79 % | +0.89 % -> +0.92 % |
| tc-256x64@n16k64_wA | nvfp4@stock_wA | +1.31 % -> +3.74 % | -0.90 % -> -0.18 % | +0.94 % -> +1.01 % |
| tmopt-16x64@n16k64_wA | fo6@stock_wA | +2.15 % -> -1.08 % | +1.93 % -> -10.93 % | +0.83 % -> +0.87 % |
| tmopt-16x64@n16k64_wA | nvfp4@stock_wA | +0.98 % -> +1.90 % | +0.63 % -> +1.36 % | +0.89 % -> +0.98 % |
| tmopt-256x64@n16k64_wA | fo6@stock_wA | +1.40 % -> -2.00 % | +0.40 % -> -11.16 % | +0.82 % -> +0.92 % |
| tmopt-256x64@n16k64_wA | nvfp4@stock_wA | +7.22 % -> -0.57 % | -0.55 % -> -0.19 % | +1.01 % -> +1.01 % |

Table source recorded by the new runs: {"nvfp4@stock_wA": ["/home/dev/n16k64_campaign/sm120_bench/sm120/configs/nvidia_rtx_pro_6000_blackwell_workstation_edition.json"], "fo6@stock_wA": ["/home/dev/n16k64_campaign/sm120_bench/sm120/configs/nvidia_rtx_pro_6000_blackwell_workstation_edition.json"], "tmopt-16x64@n16k64_wA": ["/home/dev/n16k64_campaign/sm120_bench/sm120/configs/nvidia_rtx_pro_6000_blackwell_workstation_edition.json"], "tmopt-256x64@n16k64_wA": ["/home/dev/n16k64_campaign/sm120_bench/sm120/configs/nvidia_rtx_pro_6000_blackwell_workstation_edition.json"], "tc-16x64@n16k64_wA": ["/home/dev/n16k64_campaign/sm120_bench/sm120/configs/nvidia_rtx_pro_6000_blackwell_workstation_edition.json"], "tc-256x64@n16k64_wA": ["/home/dev/n16k64_campaign/sm120_bench/sm120/configs/nvidia_rtx_pro_6000_blackwell_workstation_edition.json"]}

## Files

- **The table:** `sm120/configs/nvidia_rtx_pro_6000_blackwell_workstation_edition{,.raw}.json`, from the tuning log
  `tune_log.txt`.
- **Code:**
  - `sm120/bench/tune_tiles.py`: the `--maps` option, and the tags recorded in the table;
  - `sm120/bench/common.py`: Phi-4 and Qwen3.8-27B added to `MODEL_SHAPES`.
- **Analysis:** `analyze.py` → `tile_table.{md,json}`. The per-forward savings above are the raw timings × module counts,
  computed as in `analyze.py`.
- **Checks:** `check_logits.py` → `check_logits.json`.
- **The re-measurement:**
  - `remeasure/` holds the 30 run records, the queue, the round order and the log;
  - `compare_remeasure.py` → `remeasure.{md,json}`;
  - the old records are Part R's, in `/home/dev/n16k64_campaign/sm120_bench/results_r2/bench/qwen27b`.
  - The table was copied into that harness's sm120 copy (`/home/dev/n16k64_campaign/sm120_bench/sm120/configs`).
- **The same-process A/B:** `ab_same_process.py` → `ab/*.json`.
