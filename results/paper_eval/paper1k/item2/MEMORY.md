# Part AC item 2: memory on the RTX PRO 6000 at 1k (the paper's memory paragraph)

Part B's probe (mem_probe.py, flipquant's loader and install, build_V) on flipquant calib-size d940fbf, with the 1k maps; fo6-wB added to B's six policies. PROTOCOL.md part AC item 2.

- **Weights:** every parameter and buffer on the GPU after the load and install.
- **After load:** torch.cuda.memory_allocated.
- **Peak:** torch.cuda.max_memory_allocated during one 1x2048 eager prefill.

**PENDING.** The memory probe stopped when the GPU stopped responding (2026-10-24 18:53 UTC; PROTOCOL.md part AC item 2, incident). Done: the anchor (below) and 1 of the 42 panel jobs (Llama-3.2-1B-Instruct BF16). The rest resumes at Llama-3.2-1B-Instruct NVFP4 after the GPU is restored; the paper's memory claims below are not evaluated until then.

## Step 0: the memory anchor (Mistral, FlipQuant 16x64 with the release map, against part B's record)

- Installed quantized linears equal: True; whole model equal: True.
- After load Δ +0.0000 GiB; 1x2048 peak Δ +0.0000 GiB (tolerance ±0.02 GiB). **PASS.**

## The panel

| model | policy | weights | quantized linears / BF16 | whole model / BF16 | after load | 1x2048 peak |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | bf16 | 2.30 GiB | 1.0000 | 1.000 | 2.30 GiB | 2.87 GiB |

## The paper's memory claims, per model

- Llama-3.2-1B-Instruct: incomplete
- Llama-3.1-8B-Instruct: incomplete
- Mistral-7B-Instruct-v0.3: incomplete
- Nemotron-Nano-9B-v2: incomplete
- Phi-4: incomplete
- Qwen3.8-27B: incomplete
