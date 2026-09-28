# Inference GPU memory of the deployment policies — protocol

Written 2026-09-28 on branch `tm-opt`, before any measurement; the hash and time are in `registration.json`.
Deviations are appended at the end. Task (user-approved, relayed by nvfp4-razer-c9). It runs after the tile table
(`results/tile_table`), with nothing else on the GPU.

**The claim to test:** MixFP4 (TM-OPT+TC) has zero memory overhead over NVFP4. The E0M3 flag is bit 7 of the UE4M3
scale byte that NVFP4 already stores. Any deviation is reported plainly, allocator and workspace effects included.

## Models and policies

- **Models:** Llama-3.1-8B, Mistral-7B-v0.3, Phi-4 and Qwen3.8-27B, loaded as the SM120 evaluation loads them
  (`sm120/eval/common.load_model`, BF16, SDPA, on the GPU).
- **Policies (the Parts 2–3 artifacts):**

  | policy | artifact | kernel | placement |
  |---|---|---|---|
  | BF16 | none | none | none |
  | NVFP4 | `<model>_nvfp4` | `auto_stock` | weights on A |
  | FourOverSix | `<model>_fo6` | `auto_stock` | weights on A |
  | FourOverSix | `<model>_fo6` | `stock_wB` | weights on B; the 8x64 same-placement reference |
  | TM-OPT+TC 8x64 | `<model>_tc_8x64` | `n8k64_wB` | weights on B |
  | TM-OPT+TC 16x64 | `<model>_tc_16x64` | `auto` | weights on A |
  | TM-OPT+TC 256x64 | `<model>_tc_256x64` | `auto` | weights on A |

- **One fresh process per (model, policy).** The default caching allocator (`PYTORCH_CUDA_ALLOC_CONF` unset, as for
  Part R), recorded with the versions.

## Static memory after install (`measure_memory.py`)

- **Every parameter and buffer, by group:** quantized Linears, embeddings, lm_head, norms, Qwen's vision tower, Qwen's
  linear-attention parameters outside its Linears (conv, A, dt, norm), other.
  - The quantized Linears are counted as installed: packed FP4 codes, scale bytes as placed including layout padding
    (`InstallReport.device_bytes`), and bias. The global scales are host floats.
- **`torch.cuda.memory_allocated`** after load and after install (after `gc.collect()` and `empty_cache()`).
- **Freed check:**
  - the allocated memory after install is compared with the BF16 total, minus the replaced Linears' BF16 bytes, plus
    the installed bytes;
  - **stray blocks must be zero:** every active allocator block must start at a parameter or buffer
    (`torch.cuda.memory_snapshot`), so no replaced weight lingers. A pre-registration smoke test (Mistral, 8x64)
    confirmed this;
  - **the remainder is allocator slack,** reported as such: block sizes minus tensor bytes, i.e. the caching
    allocator's rounding and unsplit remainders. The smoke test found 24 MiB after install, 512 B after load; it
    closes the freed check exactly;
  - no scoped `nn.Linear` may remain.
- **Same-placement equality:** the TM-OPT+TC maps' installed weight bytes must equal FourOverSix's and NVFP4's on the
  same placement: 16x64 and 256x64 against `stock_wA`, 8x64 against FourOverSix on `stock_wB`. A difference between
  weights-on-A and weights-on-B placement is explained.
- **Ratio to BF16:** for the quantized Linears alone, and for the whole model.

## Dynamic peaks

- **Each scenario is measured separately:** the quantization cache is cleared, then `gc.collect()`, `empty_cache()`
  and `reset_peak_memory_stats()`. Reported per scenario: peak allocated and reserved, and the increase over the static
  allocation.
- **Prefill 1x2048 and 4x2048:** `sm120/bench/model.py prefill`, the Part R harness. These are forwards with the KV
  cache written and the logits returned, over random tokens.
- **Decode at batch 1:** a 2048-token prompt, then 128 generated tokens with the KV cache.
  - Llama, Mistral, Phi-4: `sm120/bench/model.py decode_graph`, the Part R harness. It runs the CUDA-graph decode over
    a StaticCache of prompt + 128 + 8 positions.
  - Qwen3.8-27B: HF `generate` (greedy, exactly 128 new tokens, its hybrid cache). The SM120 decode functions do not
    support that cache, so this is a different harness, and it is stated as such.
- **Reported separately:**
  - the KV cache: the StaticCache's size from the model config, or the bytes of every tensor in Qwen's returned cache;
  - the activation-quantization scratch. Per `NativeLinear` call it is one allocation of 256-byte-aligned codes
    (T·K/2), scale bytes (`sf_buffer_size(T, K)`) and 4T bytes of token scales. The largest K at the scenario's T is
    given; one scratch is kept by the quantization cache between calls. Also the kernel workspaces the kernels cache
    per GEMM shape (measured).

## Report

- `REPORT.md` and `memory.{md,json}` from `analyze.py`.
- A per-model table: static weight GiB and peak prefill / decode GiB per policy.
- The zero-overhead check: MixFP4 against NVFP4 and FourOverSix on the same placement, for static bytes and for each
  peak.
- Commit and push on `tm-opt`; report back; stop.

## Deviations (append-only)

(none yet)
