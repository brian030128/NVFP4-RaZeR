# B: memory on the RTX PRO 6000 (mem_probe.py, flipquant's loader, build_V)

Weights: every parameter and buffer the model holds on the GPU after the load and install. Quantized linears: packed FP4 + placed UE4M3 scales + BF16 bias, against the same linears' BF16 size. After load: torch.cuda.memory_allocated. Peak: torch.cuda.max_memory_allocated during one 1x2048 prefill (an eager forward that writes the KV cache and returns the logits).

| model | policy | weights | quantized linears / BF16 | whole model / BF16 | after load | 1x2048 peak |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | bf16 | 3.20 GiB | 1.0000 | 1.000 | 3.20 GiB | 4.02 GiB |
| Qwen3-1.7B | nvfp4 | 1.32 GiB | 0.2812 | 0.411 | 1.32 GiB | 2.16 GiB |
| Qwen3-1.7B | fo6 | 1.32 GiB | 0.2812 | 0.411 | 1.32 GiB | 2.17 GiB |
| Qwen3-1.7B | fq-8x64 | 1.32 GiB | 0.2812 | 0.411 | 1.32 GiB | 2.17 GiB |
| Qwen3-1.7B | fq-16x64 | 1.32 GiB | 0.2812 | 0.411 | 1.32 GiB | 2.17 GiB |
| Qwen3-1.7B | fq-256x64 | 1.32 GiB | 0.2812 | 0.411 | 1.32 GiB | 2.17 GiB |
| Qwen3-8B | bf16 | 15.26 GiB | 1.0000 | 1.000 | 15.26 GiB | 16.14 GiB |
| Qwen3-8B | nvfp4 | 5.96 GiB | 0.2812 | 0.390 | 5.98 GiB | 6.92 GiB |
| Qwen3-8B | fo6 | 5.96 GiB | 0.2812 | 0.390 | 5.99 GiB | 6.94 GiB |
| Qwen3-8B | fq-8x64 | 5.96 GiB | 0.2812 | 0.390 | 5.98 GiB | 6.93 GiB |
| Qwen3-8B | fq-16x64 | 5.96 GiB | 0.2812 | 0.390 | 5.98 GiB | 6.93 GiB |
| Qwen3-8B | fq-256x64 | 5.96 GiB | 0.2812 | 0.390 | 5.97 GiB | 6.92 GiB |
| Mistral-7B-Instruct-v0.3 | bf16 | 13.50 GiB | 1.0000 | 1.000 | 13.50 GiB | 13.99 GiB |
| Mistral-7B-Instruct-v0.3 | nvfp4 | 4.16 GiB | 0.2812 | 0.308 | 4.19 GiB | 4.68 GiB |
| Mistral-7B-Instruct-v0.3 | fo6 | 4.16 GiB | 0.2812 | 0.308 | 4.19 GiB | 4.68 GiB |
| Mistral-7B-Instruct-v0.3 | fq-8x64 | 4.16 GiB | 0.2812 | 0.308 | 4.19 GiB | 4.68 GiB |
| Mistral-7B-Instruct-v0.3 | fq-16x64 | 4.16 GiB | 0.2812 | 0.308 | 4.19 GiB | 4.68 GiB |
| Mistral-7B-Instruct-v0.3 | fq-256x64 | 4.16 GiB | 0.2812 | 0.308 | 4.19 GiB | 4.68 GiB |
| Phi-4 | bf16 | 27.31 GiB | 1.0000 | 1.000 | 27.31 GiB | 28.11 GiB |
| Phi-4 | nvfp4 | 9.06 GiB | 0.2812 | 0.332 | 9.08 GiB | 9.97 GiB |
| Phi-4 | fo6 | 9.06 GiB | 0.2812 | 0.332 | 9.07 GiB | 9.96 GiB |
| Phi-4 | fq-8x64 | 9.06 GiB | 0.2812 | 0.332 | 9.07 GiB | 9.96 GiB |
| Phi-4 | fq-16x64 | 9.06 GiB | 0.2812 | 0.332 | 9.07 GiB | 9.96 GiB |
| Phi-4 | fq-256x64 | 9.06 GiB | 0.2812 | 0.332 | 9.08 GiB | 9.97 GiB |
