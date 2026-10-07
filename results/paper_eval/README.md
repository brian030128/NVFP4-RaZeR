# Paper evaluation on the RTX PRO 6000 (SM120): BF16, NVFP4, FourOverSix and FlipQuant

The user's request of 2026-10-07 (relayed by the coordinator): the paper's SM120 evaluation for BF16, NVFP4, FourOverSix
and FlipQuant only, on the six paper models (Qwen3-1.7B, Qwen3-8B, Mistral-7B-Instruct-v0.3, Nemotron-Nano-9B-v2,
Phi-4, Qwen3.8-27B) at 8x64 / 16x64 / 256x64. FlipQuant = the 5-epoch release maps (256 windows, top-1000).

- `cpu_tables/`: tables from existing records (main-ppl, tile granularity, calibration cost).
- The GPU parts (A prefill latency, B memory, C verification, D generative accuracy) follow their registered protocol.
