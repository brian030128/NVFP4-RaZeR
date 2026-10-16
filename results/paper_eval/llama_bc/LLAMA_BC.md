# Part Z: memory (B) and ownership (C1) for the two Llama-Instruct models

B's and C1's protocols (run_gpu.memory / run_gpu.ownership, a30df07 / C1), run with part P's maps on flipquant paper-sm120-runs (its code = 4bbe0b1; main 120173a's registry lacks both models) and build_V. PROTOCOL.md part Z. The six models of B and C1 are in results/paper_eval/memory and results/paper_eval/verify.

## B: memory on the RTX PRO 6000

The columns are B's (MEMORY.md):
- **Weights:** every parameter and buffer on the GPU after the load and install.
- **After load:** torch.cuda.memory_allocated.
- **Peak:** torch.cuda.max_memory_allocated during one 1x2048 eager prefill.

| model | policy | weights | quantized linears / BF16 | whole model / BF16 | after load | 1x2048 peak |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | bf16 | 2.30 GiB | 1.0000 | 1.000 | 2.30 GiB | 2.87 GiB |
| Llama-3.2-1B-Instruct | nvfp4 | 1.00 GiB | 0.2812 | 0.434 | 1.00 GiB | 1.61 GiB |
| Llama-3.2-1B-Instruct | fo6 | 1.00 GiB | 0.2812 | 0.434 | 1.00 GiB | 1.61 GiB |
| Llama-3.2-1B-Instruct | fq-8x64 | 1.00 GiB | 0.2812 | 0.434 | 1.00 GiB | 1.61 GiB |
| Llama-3.2-1B-Instruct | fq-16x64 | 1.00 GiB | 0.2812 | 0.434 | 1.00 GiB | 1.61 GiB |
| Llama-3.2-1B-Instruct | fq-256x64 | 1.00 GiB | 0.2812 | 0.434 | 1.00 GiB | 1.61 GiB |
| Llama-3.1-8B-Instruct | bf16 | 14.96 GiB | 1.0000 | 1.000 | 14.96 GiB | 15.72 GiB |
| Llama-3.1-8B-Instruct | nvfp4 | 5.61 GiB | 0.2812 | 0.375 | 5.65 GiB | 6.49 GiB |
| Llama-3.1-8B-Instruct | fo6 | 5.61 GiB | 0.2812 | 0.375 | 5.65 GiB | 6.49 GiB |
| Llama-3.1-8B-Instruct | fq-8x64 | 5.61 GiB | 0.2812 | 0.375 | 5.65 GiB | 6.48 GiB |
| Llama-3.1-8B-Instruct | fq-16x64 | 5.61 GiB | 0.2812 | 0.375 | 5.65 GiB | 6.48 GiB |
| Llama-3.1-8B-Instruct | fq-256x64 | 5.61 GiB | 0.2812 | 0.375 | 5.65 GiB | 6.48 GiB |

**The paper's memory claims, per model:**

- **Llama-3.2-1B-Instruct:**
  - FlipQuant's installed quantized linears and whole model equal NVFP4's and FourOverSix's to the byte: **yes**. Quantized linears: 547,356,672 B for NVFP4, 547,356,672 B for fo6, 547,356,672 B for fq-8x64, 547,356,672 B for fq-16x64, 547,356,672 B for fq-256x64.
  - After load and at the 1x2048 peak, FlipQuant differs from NVFP4 / FourOverSix by at most 0.0000 / 0.0010 GiB. Within 0.02 GiB: **yes**.
- **Llama-3.1-8B-Instruct:**
  - FlipQuant's installed quantized linears and whole model equal NVFP4's and FourOverSix's to the byte: **yes**. Quantized linears: 3,925,868,544 B for NVFP4, 3,925,868,544 B for fo6, 3,925,868,544 B for fq-8x64, 3,925,868,544 B for fq-16x64, 3,925,868,544 B for fq-256x64.
  - After load and at the 1x2048 peak, FlipQuant differs from NVFP4 / FourOverSix by at most 0.0073 / 0.0079 GiB. Within 0.02 GiB: **yes**.

## C1: ownership check of the 6 Llama-Instruct maps

C1's check and columns (VERIFY.md):
- Every weight element is decoded by the kernel itself (an identity activation through the GEMM), and must equal its stored value under the map's format (flipquant `--ownership-check`; the install raises on any mismatch).
- Weight elements: the checked modules' out x in, from the checkpoint's safetensors headers.

| model | unit | modules | weight elements checked | informative elements | E0M3 tiles | E0M3 elements observed | exact | format mismatches |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Llama-3.2-1B-Instruct | 8x64 | 112 of 112 | 973,078,528 | 890,461,232 | 38,587 | 17,343,859 | True | 0 |
| Llama-3.2-1B-Instruct | 16x64 | 112 of 112 | 973,078,528 | 890,180,527 | 26,656 | 23,959,766 | True | 0 |
| Llama-3.2-1B-Instruct | 256x64 | 112 of 112 | 973,078,528 | 887,078,947 | 108,112 | 97,230,318 | True | 0 |
| Llama-3.1-8B-Instruct | 8x64 | 224 of 224 | 6,979,321,856 | 6,393,003,850 | 83,771 | 37,665,182 | True | 0 |
| Llama-3.1-8B-Instruct | 16x64 | 224 of 224 | 6,979,321,856 | 6,392,594,186 | 52,719 | 47,400,946 | True | 0 |
| Llama-3.1-8B-Instruct | 256x64 | 224 of 224 | 6,979,321,856 | 6,388,881,568 | 151,152 | 135,943,718 | True | 0 |

**Totals.**
- The 6 Llama maps: 23,857,201,152 weight elements checked.
- With C1's 18 maps (183,084,122,112): 24 maps, 206,941,323,264 weight elements.
- Every map exact: **yes**. Format mismatches: 0. Every module of every map checked: yes.
