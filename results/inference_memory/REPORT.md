# Inference GPU memory of the deployment policies — report

Protocol: `PROTOCOL.md`, registered 2026-09-28T14:26:56Z (`registration.json`), before any measurement.
- **Deviations:** none.
- **Runs:** 28 of 28 (4 models × 7 policies), one fresh process each, default caching allocator.
- **Kernel widths:** the new RTX PRO 6000 tile table was in place (`results/tile_table`).

## Summary

**The claim holds.** MixFP4 (TM-OPT+TC 8x64, 16x64, 256x64) has zero memory overhead over NVFP4.
- **Static:** on every model, its installed weight bytes, allocated memory after install and allocator slack equal
  NVFP4's and FourOverSix's to the byte, on both placements.
- **Dynamic:** every peak (allocated and reserved; prefill 1x2048, 4x2048, decode) is equal to the byte, with one
  exception.
  - The exception: Phi-4 prefill 1x2048, where FourOverSix on `stock_wA` peaks 1.1 MiB (0.01 %) below NVFP4 and
    TM-OPT+TC, which are equal to each other.
  - So it is not a MixFP4 cost. It is presumably an allocator-level effect, not investigated further.
- **Why:** the E0M3 flag is bit 7 of the UE4M3 scale byte NVFP4 already stores; the kernels need no workspace
  (0 B for every policy).

| model | policy | static allocated | of which quantized Linears | prefill 1x2048 peak (alloc / reserved) | prefill 4x2048 peak | decode b1 peak |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | BF16 | 14.96 GiB | 13.00 GiB | 15.72 / 16.04 | 17.99 / 19.25 | 15.77 / 16.20 |
| | every FP4 policy | 5.64 GiB | 3.66 GiB | 6.47 / 19.01 | 8.95 / 19.01 | 6.49 / 19.09 |
| Mistral-7B-v0.3 | BF16 | 13.50 GiB | 13.00 GiB | 13.99 / 14.21 | 15.42 / 16.34 | 14.00 / 14.41 |
| | every FP4 policy | 4.18 GiB | 3.66 GiB | 4.67 / 17.56 | 6.11 / 17.56 | 4.69 / 17.63 |
| Phi-4 | BF16 | 27.31 GiB | 25.39 GiB | 28.11 / 28.51 | 30.49 / 32.11 | 28.17 / 28.73 |
| | every FP4 policy | 9.09 GiB | 7.14 GiB | 9.99 / 34.81 | 12.62 / 34.81 | 10.01 / 34.90 |
| Qwen3.8-27B | BF16 | 50.96 GiB | 45.36 GiB | 52.23 / 52.99 | 55.91 / 58.97 | 51.87 / 52.04 |
| | every FP4 policy | 18.50 GiB | 12.76 GiB | 19.84 / 64.41 | 23.79 / 64.41 | 19.41 / 64.41 |

"Every FP4 policy" is NVFP4 (`stock_wA`), FourOverSix (`stock_wA`, `stock_wB`) and TM-OPT+TC 8x64 (`n8k64_wB`), 16x64 and
256x64 (`auto`). Peaks are in GiB.

**Static memory.**
- **The quantized Linears take 0.28125 of their BF16 bytes:** 4 bits of code plus one UE4M3 byte per 16 weights. That
  is 4.5 of 16 bits.
  - Qwen's is 0.2813: its 96 48-row linear-attention projections pad their scale layout to 128 rows, 2.3 MiB in total
    (`scales_padding`).
  - The global scales are host floats.
- **What stays BF16:** embeddings and `lm_head` (0.98 + 0.98 GiB on Llama, 0.25 + 0.25 on Mistral, 0.96 + 0.96 on
  Phi-4, 2.37 + 2.37 on Qwen), the norms, and on Qwen the vision tower (0.86 GiB) and the linear-attention parameters
  outside the Linears (4 MiB).
- **Whole model, ratio to BF16:** 0.377 (Llama), 0.310 (Mistral), 0.333 (Phi-4), 0.363 (Qwen).
- **Weights on A against weights on B: identical bytes.** The weight scale layout (`sf_buffer_size`, atoms of
  128 rows × 4 scale blocks) is the same for both placements. The only padding, Qwen's 48-row modules, is the same on
  both.

**The replaced BF16 weights are freed.** In every run, every active allocator block starts at a parameter or buffer:
0 stray blocks, so no lingering copy.
- The difference between the allocated memory and the expected one (BF16 total − replaced BF16 Linears + installed
  bytes) is exactly the allocator's slack: block sizes minus tensor bytes, i.e. unsplit remainders.
- The slack is 24.0 / 24.0 / 31.5 / 139.7 MiB (Llama / Mistral / Phi-4 / Qwen), identical across the FP4 policies,
  against 512 B after the BF16 load.

**Dynamic memory.**
- **KV cache:**
  - the StaticCache of the CUDA-graph decode (prompt 2048 + 128 + 8): 0.267 GiB (Llama, Mistral) and 0.417 GiB (Phi-4);
  - Qwen's hybrid cache after HF generate: 0.277 GiB (16 attention layers' KV plus the linear-attention states).
- **Activation-quantization scratch** per NativeLinear call, at the largest K: one allocation (codes | scale bytes |
  token scales). The quantization cache keeps one more between calls.

  | T | Llama, Mistral | Phi-4 | Qwen |
  |---|---:|---:|---:|
  | 2048 | 15.8 MiB | 19.7 MiB | 19.1 MiB |
  | 8192 | 63.0 MiB | 78.8 MiB | 76.5 MiB |
  | 1 (decode) | 119 KiB | 149 KiB | 145 KiB |

- **FP4 against BF16:** the FP4 policies' peaks rise by 0.003–0.35 GiB more over their static memory than BF16's do.
  - The source is the scratch plus the quantization cache's reference to the last quantized input.
  - It is largest at 4x2048: Llama +0.28 GiB, the last `down_proj` input 0.22 GiB plus its scratch; Phi-4 +0.35 GiB.
  - It is the same for every FP4 policy, so it is not MixFP4-specific. Clearing the cache after the last layer would
    remove it (not measured).
- **Reserved memory (an allocator effect of this flow):**
  - After the in-place install over a loaded BF16 model, the caching allocator keeps 19.0 / 17.6 / 34.8 / 64.4 GiB
    reserved against 5.6 / 4.2 / 9.1 / 18.5 GiB allocated. The segments the BF16 weights occupied stay partly in use,
    and during install the packed artifact is loaded onto the GPU next to the BF16 model.
  - Identical across the FP4 policies. A deployment that never materializes the BF16 Linears would not show it (not
    measured).
  - The allocated peaks are the requirement.
- **Decode harness:** Llama, Mistral and Phi-4 used Part R's CUDA-graph decode (`sm120/bench/model.py decode_graph`).
  Qwen used HF generate (greedy, exactly 128 new tokens), because the SM120 decode functions do not support its hybrid
  cache.

## Per-model tables and checks (generated)

### Llama-3.1-8B

| policy | quantized Linears | BF16 rest | allocated after install | ratio to BF16 (Linears / model) | prefill 1x2048 peak alloc / reserved | prefill 4x2048 peak alloc / reserved | decode b1 peak alloc / reserved |
|---|---:|---:|---:|---|---:|---:|---:|
| BF16 | 13.000 GiB | 1.958 GiB | 14.958 GiB | — | 15.72 / 16.04 | 17.99 / 19.25 | 15.77 / 16.20 |
| NVFP4 (stock_wA) | 3.656 GiB | 1.958 GiB | 5.637 GiB | 0.2812 / 0.3769 | 6.47 / 19.01 | 8.95 / 19.01 | 6.49 / 19.09 |
| FourOverSix (stock_wA) | 3.656 GiB | 1.958 GiB | 5.637 GiB | 0.2812 / 0.3769 | 6.47 / 19.01 | 8.95 / 19.01 | 6.49 / 19.09 |
| FourOverSix (stock_wB) | 3.656 GiB | 1.958 GiB | 5.637 GiB | 0.2812 / 0.3769 | 6.47 / 19.01 | 8.95 / 19.01 | 6.49 / 19.09 |
| TM-OPT+TC 8x64 (n8k64_wB) | 3.656 GiB | 1.958 GiB | 5.637 GiB | 0.2812 / 0.3769 | 6.47 / 19.01 | 8.95 / 19.01 | 6.49 / 19.09 |
| TM-OPT+TC 16x64 (auto) | 3.656 GiB | 1.958 GiB | 5.637 GiB | 0.2812 / 0.3769 | 6.47 / 19.01 | 8.95 / 19.01 | 6.49 / 19.09 |
| TM-OPT+TC 256x64 (auto) | 3.656 GiB | 1.958 GiB | 5.637 GiB | 0.2812 / 0.3769 | 6.47 / 19.01 | 8.95 / 19.01 | 6.49 / 19.09 |

Checks and details:

- NVFP4 (stock_wA): installed {'packed': 3489660928, 'scales_placed': 436207616, 'scales_padding': 0, 'bias': 0}; freed check difference 25165824 B; allocated outside parameters and buffers 25166336 B; kernel workspaces 0 B.
- FourOverSix (stock_wA): installed {'packed': 3489660928, 'scales_placed': 436207616, 'scales_padding': 0, 'bias': 0}; freed check difference 25165824 B; allocated outside parameters and buffers 25166336 B; kernel workspaces 0 B.
- FourOverSix (stock_wB): installed {'packed': 3489660928, 'scales_placed': 436207616, 'scales_padding': 0, 'bias': 0}; freed check difference 25165824 B; allocated outside parameters and buffers 25166336 B; kernel workspaces 0 B.
- TM-OPT+TC 8x64 (n8k64_wB): installed {'packed': 3489660928, 'scales_placed': 436207616, 'scales_padding': 0, 'bias': 0}; freed check difference 25165824 B; allocated outside parameters and buffers 25166336 B; kernel workspaces 0 B.
- TM-OPT+TC 16x64 (auto): installed {'packed': 3489660928, 'scales_placed': 436207616, 'scales_padding': 0, 'bias': 0}; freed check difference 25165824 B; allocated outside parameters and buffers 25166336 B; kernel workspaces 0 B.
- TM-OPT+TC 256x64 (auto): installed {'packed': 3489660928, 'scales_placed': 436207616, 'scales_padding': 0, 'bias': 0}; freed check difference 25165824 B; allocated outside parameters and buffers 25166336 B; kernel workspaces 0 B.
- Same placement, tc-16x64 vs fo6@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-16x64 vs nvfp4@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-256x64 vs fo6@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-256x64 vs nvfp4@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-8x64 vs fo6@stock_wB: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.

KV cache (decode, prompt 2048 + 128): 0.267 GiB (sm120/bench/model.py decode_graph (CUDA graph, StaticCache)). Activation-quantization scratch per NativeLinear call, largest K: prefill 1x2048 15.76 MiB, 4x2048 63.03 MiB, decode 119.0 KiB (one more is held by the quantization cache between calls). Logits: prefill 1x2048 0.489 GiB, 4x2048 1.957 GiB.

### Mistral-7B-v0.3

| policy | quantized Linears | BF16 rest | allocated after install | ratio to BF16 (Linears / model) | prefill 1x2048 peak alloc / reserved | prefill 4x2048 peak alloc / reserved | decode b1 peak alloc / reserved |
|---|---:|---:|---:|---|---:|---:|---:|
| BF16 | 13.000 GiB | 0.500 GiB | 13.500 GiB | — | 13.99 / 14.21 | 15.42 / 16.34 | 14.00 / 14.41 |
| NVFP4 (stock_wA) | 3.656 GiB | 0.500 GiB | 4.180 GiB | 0.2812 / 0.3096 | 4.67 / 17.56 | 6.11 / 17.56 | 4.69 / 17.63 |
| FourOverSix (stock_wA) | 3.656 GiB | 0.500 GiB | 4.180 GiB | 0.2812 / 0.3096 | 4.67 / 17.56 | 6.11 / 17.56 | 4.69 / 17.63 |
| FourOverSix (stock_wB) | 3.656 GiB | 0.500 GiB | 4.180 GiB | 0.2812 / 0.3096 | 4.67 / 17.56 | 6.11 / 17.56 | 4.69 / 17.63 |
| TM-OPT+TC 8x64 (n8k64_wB) | 3.656 GiB | 0.500 GiB | 4.180 GiB | 0.2812 / 0.3096 | 4.67 / 17.56 | 6.11 / 17.56 | 4.69 / 17.63 |
| TM-OPT+TC 16x64 (auto) | 3.656 GiB | 0.500 GiB | 4.180 GiB | 0.2812 / 0.3096 | 4.67 / 17.56 | 6.11 / 17.56 | 4.69 / 17.63 |
| TM-OPT+TC 256x64 (auto) | 3.656 GiB | 0.500 GiB | 4.180 GiB | 0.2812 / 0.3096 | 4.67 / 17.56 | 6.11 / 17.56 | 4.69 / 17.63 |

Checks and details:

- NVFP4 (stock_wA): installed {'packed': 3489660928, 'scales_placed': 436207616, 'scales_padding': 0, 'bias': 0}; freed check difference 25165824 B; allocated outside parameters and buffers 25166336 B; kernel workspaces 0 B.
- FourOverSix (stock_wA): installed {'packed': 3489660928, 'scales_placed': 436207616, 'scales_padding': 0, 'bias': 0}; freed check difference 25165824 B; allocated outside parameters and buffers 25166336 B; kernel workspaces 0 B.
- FourOverSix (stock_wB): installed {'packed': 3489660928, 'scales_placed': 436207616, 'scales_padding': 0, 'bias': 0}; freed check difference 25165824 B; allocated outside parameters and buffers 25166336 B; kernel workspaces 0 B.
- TM-OPT+TC 8x64 (n8k64_wB): installed {'packed': 3489660928, 'scales_placed': 436207616, 'scales_padding': 0, 'bias': 0}; freed check difference 25165824 B; allocated outside parameters and buffers 25166336 B; kernel workspaces 0 B.
- TM-OPT+TC 16x64 (auto): installed {'packed': 3489660928, 'scales_placed': 436207616, 'scales_padding': 0, 'bias': 0}; freed check difference 25165824 B; allocated outside parameters and buffers 25166336 B; kernel workspaces 0 B.
- TM-OPT+TC 256x64 (auto): installed {'packed': 3489660928, 'scales_placed': 436207616, 'scales_padding': 0, 'bias': 0}; freed check difference 25165824 B; allocated outside parameters and buffers 25166336 B; kernel workspaces 0 B.
- Same placement, tc-16x64 vs fo6@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-16x64 vs nvfp4@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-256x64 vs fo6@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-256x64 vs nvfp4@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-8x64 vs fo6@stock_wB: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.

KV cache (decode, prompt 2048 + 128): 0.267 GiB (sm120/bench/model.py decode_graph (CUDA graph, StaticCache)). Activation-quantization scratch per NativeLinear call, largest K: prefill 1x2048 15.76 MiB, 4x2048 63.03 MiB, decode 119.0 KiB (one more is held by the quantization cache between calls). Logits: prefill 1x2048 0.125 GiB, 4x2048 0.500 GiB.

### Phi-4

| policy | quantized Linears | BF16 rest | allocated after install | ratio to BF16 (Linears / model) | prefill 1x2048 peak alloc / reserved | prefill 4x2048 peak alloc / reserved | decode b1 peak alloc / reserved |
|---|---:|---:|---:|---|---:|---:|---:|
| BF16 | 25.391 GiB | 1.915 GiB | 27.305 GiB | — | 28.11 / 28.51 | 30.49 / 32.11 | 28.17 / 28.73 |
| NVFP4 (stock_wA) | 7.141 GiB | 1.915 GiB | 9.087 GiB | 0.2812 / 0.3328 | 9.99 / 34.81 | 12.62 / 34.81 | 10.01 / 34.90 |
| FourOverSix (stock_wA) | 7.141 GiB | 1.915 GiB | 9.087 GiB | 0.2812 / 0.3328 | 9.99 / 34.81 | 12.62 / 34.81 | 10.01 / 34.90 |
| FourOverSix (stock_wB) | 7.141 GiB | 1.915 GiB | 9.087 GiB | 0.2812 / 0.3328 | 9.99 / 34.81 | 12.62 / 34.81 | 10.01 / 34.90 |
| TM-OPT+TC 8x64 (n8k64_wB) | 7.141 GiB | 1.915 GiB | 9.087 GiB | 0.2812 / 0.3328 | 9.99 / 34.81 | 12.62 / 34.81 | 10.01 / 34.90 |
| TM-OPT+TC 16x64 (auto) | 7.141 GiB | 1.915 GiB | 9.087 GiB | 0.2812 / 0.3328 | 9.99 / 34.81 | 12.62 / 34.81 | 10.01 / 34.90 |
| TM-OPT+TC 256x64 (auto) | 7.141 GiB | 1.915 GiB | 9.087 GiB | 0.2812 / 0.3328 | 9.99 / 34.81 | 12.62 / 34.81 | 10.01 / 34.90 |

Checks and details:

- NVFP4 (stock_wA): installed {'packed': 6815744000, 'scales_placed': 851968000, 'scales_padding': 0, 'bias': 0}; freed check difference 32997376 B; allocated outside parameters and buffers 32997888 B; kernel workspaces 0 B.
- FourOverSix (stock_wA): installed {'packed': 6815744000, 'scales_placed': 851968000, 'scales_padding': 0, 'bias': 0}; freed check difference 32997376 B; allocated outside parameters and buffers 32997888 B; kernel workspaces 0 B.
- FourOverSix (stock_wB): installed {'packed': 6815744000, 'scales_placed': 851968000, 'scales_padding': 0, 'bias': 0}; freed check difference 32997376 B; allocated outside parameters and buffers 32997888 B; kernel workspaces 0 B.
- TM-OPT+TC 8x64 (n8k64_wB): installed {'packed': 6815744000, 'scales_placed': 851968000, 'scales_padding': 0, 'bias': 0}; freed check difference 32997376 B; allocated outside parameters and buffers 32997888 B; kernel workspaces 0 B.
- TM-OPT+TC 16x64 (auto): installed {'packed': 6815744000, 'scales_placed': 851968000, 'scales_padding': 0, 'bias': 0}; freed check difference 32997376 B; allocated outside parameters and buffers 32997888 B; kernel workspaces 0 B.
- TM-OPT+TC 256x64 (auto): installed {'packed': 6815744000, 'scales_placed': 851968000, 'scales_padding': 0, 'bias': 0}; freed check difference 32997376 B; allocated outside parameters and buffers 32997888 B; kernel workspaces 0 B.
- Same placement, tc-16x64 vs fo6@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 1155072, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-16x64 vs nvfp4@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-256x64 vs fo6@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 1155072, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-256x64 vs nvfp4@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-8x64 vs fo6@stock_wB: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.

KV cache (decode, prompt 2048 + 128): 0.417 GiB (sm120/bench/model.py decode_graph (CUDA graph, StaticCache)). Activation-quantization scratch per NativeLinear call, largest K: prefill 1x2048 19.70 MiB, 4x2048 78.78 MiB, decode 148.8 KiB (one more is held by the quantization cache between calls). Logits: prefill 1x2048 0.383 GiB, 4x2048 1.531 GiB.

### Qwen3.8-27B

| policy | quantized Linears | BF16 rest | allocated after install | ratio to BF16 (Linears / model) | prefill 1x2048 peak alloc / reserved | prefill 4x2048 peak alloc / reserved | decode b1 peak alloc / reserved |
|---|---:|---:|---:|---|---:|---:|---:|
| BF16 | 45.356 GiB | 5.599 GiB | 50.956 GiB | — | 52.23 / 52.99 | 55.91 / 58.97 | 51.87 / 52.04 |
| NVFP4 (stock_wA) | 12.759 GiB | 5.599 GiB | 18.495 GiB | 0.2813 / 0.3630 | 19.84 / 64.41 | 23.79 / 64.41 | 19.41 / 64.41 |
| FourOverSix (stock_wA) | 12.759 GiB | 5.599 GiB | 18.495 GiB | 0.2813 / 0.3630 | 19.84 / 64.41 | 23.79 / 64.41 | 19.41 / 64.41 |
| FourOverSix (stock_wB) | 12.759 GiB | 5.599 GiB | 18.495 GiB | 0.2813 / 0.3630 | 19.84 / 64.41 | 23.79 / 64.41 | 19.41 / 64.41 |
| TM-OPT+TC 8x64 (n8k64_wB) | 12.759 GiB | 5.599 GiB | 18.495 GiB | 0.2813 / 0.3630 | 19.84 / 64.41 | 23.79 / 64.41 | 19.41 / 64.41 |
| TM-OPT+TC 16x64 (auto) | 12.759 GiB | 5.599 GiB | 18.495 GiB | 0.2813 / 0.3630 | 19.84 / 64.41 | 23.79 / 64.41 | 19.41 / 64.41 |
| TM-OPT+TC 256x64 (auto) | 12.759 GiB | 5.599 GiB | 18.495 GiB | 0.2813 / 0.3630 | 19.84 / 64.41 | 23.79 / 64.41 | 19.41 / 64.41 |

Checks and details:

- NVFP4 (stock_wA): installed {'packed': 12175278080, 'scales_placed': 1524367360, 'scales_padding': 2457600, 'bias': 0}; freed check difference 146341888 B; allocated outside parameters and buffers 146447064 B; kernel workspaces 0 B.
- FourOverSix (stock_wA): installed {'packed': 12175278080, 'scales_placed': 1524367360, 'scales_padding': 2457600, 'bias': 0}; freed check difference 146341888 B; allocated outside parameters and buffers 146447064 B; kernel workspaces 0 B.
- FourOverSix (stock_wB): installed {'packed': 12175278080, 'scales_placed': 1524367360, 'scales_padding': 2457600, 'bias': 0}; freed check difference 146341888 B; allocated outside parameters and buffers 146447064 B; kernel workspaces 0 B.
- TM-OPT+TC 8x64 (n8k64_wB): installed {'packed': 12175278080, 'scales_placed': 1524367360, 'scales_padding': 2457600, 'bias': 0}; freed check difference 146341888 B; allocated outside parameters and buffers 146447064 B; kernel workspaces 0 B.
- TM-OPT+TC 16x64 (auto): installed {'packed': 12175278080, 'scales_placed': 1524367360, 'scales_padding': 2457600, 'bias': 0}; freed check difference 146341888 B; allocated outside parameters and buffers 146447064 B; kernel workspaces 0 B.
- TM-OPT+TC 256x64 (auto): installed {'packed': 12175278080, 'scales_placed': 1524367360, 'scales_padding': 2457600, 'bias': 0}; freed check difference 146341888 B; allocated outside parameters and buffers 146447064 B; kernel workspaces 0 B.
- Same placement, tc-16x64 vs fo6@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-16x64 vs nvfp4@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-256x64 vs fo6@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-256x64 vs nvfp4@stock_wA: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.
- Same placement, tc-8x64 vs fo6@stock_wB: installed bytes difference 0 B (device_bytes equal: True); allocated difference 0 B; peak allocated difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}; peak reserved difference {'prefill_1x2048': 0, 'prefill_4x2048': 0, 'decode_b1': 0}.

KV cache (decode, prompt 2048 + 128): 0.277 GiB (HF generate (greedy; the SM120 decode functions do not support the hybrid cache)). Activation-quantization scratch per NativeLinear call, largest K: prefill 1x2048 19.13 MiB, 4x2048 76.53 MiB, decode 144.5 KiB (one more is held by the quantization cache between calls). Logits: prefill 1x2048 0.947 GiB, 4x2048 3.789 GiB.


## Files

- `measure_memory.py` (one process per model and policy) → `runs/<model>/<policy>.json`.
- The queue and its log: `runs/queue.sh`, `runs/commands.txt`.
- `analyze.py` → `memory.{md,json}`, reproduced above.
