# P: prefill harness parity on Phi-4 (flipquant evaluation.latency vs NVFP4-RaZeR bench_prefill.py)

**Verdict: FAIL** (PROTOCOL.md: per policy, flipquant / RaZeR within ±1 % at the median over the 8 shapes and ±3 % at every shape; the FlipQuant 16x64 / FourOverSix ratio within 0.5 points at the median).

CUDA-graph ms, the median over 3 rounds of each process's median; both harnesses on build_V, the same paper 16x64 map.

| shape | bf16 RaZeR | bf16 flipquant | Δ % | fo6 RaZeR | fo6 flipquant | Δ % | fq-16x64 RaZeR | fq-16x64 flipquant | Δ % | FQ/FO6 RaZeR | FQ/FO6 flipquant | gap pt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | 28.46 | 28.41 | -0.17 | 11.37 | 11.38 | +0.07 | 11.41 | 11.43 | +0.18 | +0.32 % | +0.49 % | +0.17 |
| 1x256 | 38.43 | 38.07 | -0.95 | 14.94 | 15.24 | +2.02 | 15.16 | 15.39 | +1.56 | +1.45 % | +1.62 % | +0.17 |
| 1x512 | 68.54 | 68.33 | -0.30 | 27.62 | 27.85 | +0.85 | 27.84 | 28.20 | +1.30 | +1.34 % | +1.02 % | -0.32 |
| 1x1024 | 128.41 | 128.12 | -0.22 | 55.26 | 55.73 | +0.85 | 55.63 | 56.29 | +1.19 | +0.91 % | +1.06 % | +0.15 |
| 1x2048 | 248.38 | 247.65 | -0.29 | 111.12 | 111.94 | +0.74 | 111.75 | 112.90 | +1.03 | +0.92 % | +0.83 % | -0.09 |
| 1x4096 | 513.11 | 512.19 | -0.18 | 238.13 | 239.88 | +0.73 | 238.90 | 241.31 | +1.01 | +0.51 % | +0.59 % | +0.08 |
| 1x8192 | 1120.42 | 1117.87 | -0.23 | 561.84 | 564.95 | +0.55 | 564.62 | 568.53 | +0.69 | +0.56 % | +0.59 % | +0.03 |
| 4x2048 | 1044.97 | 1043.28 | -0.16 | 488.38 | 491.42 | +0.62 | 491.65 | 494.76 | +0.63 | +0.71 % | +0.75 % | +0.05 |

| criterion | value | pass |
|---|---:|---|
| bf16: median Δ over shapes / worst |Δ| | -0.23 % / 0.95 % | True |
| fo6: median Δ over shapes / worst |Δ| | +0.74 % / 2.02 % | True |
| fq-16x64: median Δ over shapes / worst |Δ| | +1.02 % / 1.56 % | False |
| FQ/FO6 gap, median over shapes | +0.06 pt | True |
