# Prefill latency (latency_rounds_smoke)

Medians over rounds of each process's median (ms); ratios paired within rounds, median [min, max] over rounds. Harness: evaluation.latency prefill (scripts/run_prefill_latency.py).

## Llama-3.2-1B-Instruct

| shape | FourOverSix | FlipQuant 16x64 |
|---|---:|---:|
| 1x128 | 2.93 | 2.93 |
| 1x2048 | 11.84 | 11.91 |

| shape | FlipQuant 16x64 vs FourOverSix |
|---|---:|
| 1x128 | -0.0 % [-0.1, +0.1] |
| 1x2048 | +0.6 % [+0.5, +0.6] |

## Summary: the median over the 2 shapes of the paired ratio

| model | FlipQuant 16x64 vs FourOverSix | FlipQuant 8x64 vs FourOverSix | FlipQuant 256x64 vs FourOverSix | FlipQuant 8x64 vs FourOverSix (wB) | NVFP4 vs FourOverSix | FourOverSix (wB) vs FourOverSix | BF16 vs FourOverSix |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | +0.3 % | — | — | — | — | — | — |
