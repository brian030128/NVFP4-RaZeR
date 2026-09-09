## Direct reproduction with the February RaZeR release

The archived released evaluator at commit `e230099` was run directly for 14 cases (28 separate WikiText-2/C4 cells). **15/28 cells match Table 3 at its published two-decimal precision.** No calibration or parameter search was used. The evaluator uses 2048-token windows, seed 0, 256 sampled C4 windows, and its original float32 PPL aggregation.

Signed Δ is reproduced PPL minus published PPL; negative means lower perplexity. A displayed-precision match is not a claim of bitwise agreement with unpublished author outputs.

| Model | Method | Paper Wiki | Reproduced Wiki | Δ Wiki | Paper C4 | Reproduced C4 | Δ C4 | Matches |
|---|---|---:|---:|---:|---:|---:|---:|---|
| llama-3.1-8b | BF16 (paper: FP16) | 6.24 | 6.240087 | +0.000087 | 8.96 | 8.958212 | -0.001788 | wikitext, c4 |
| llama-3.1-8b | NVFP4 W4A4 | 6.95 | 6.941772 | -0.008228 | 9.94 | 9.940895 | +0.000895 | c4 |
| llama-3.1-8b | FourOverSix W4A4 | 6.88 | 6.875525 | -0.004475 | 9.83 | 9.823733 | -0.006267 | wikitext |
| llama-3.1-8b | RaZeR W4A4 | 6.74 | 6.744553 | +0.004553 | 9.63 | 9.630655 | +0.000655 | wikitext, c4 |
| llama-3.1-8b | NVFP4 W4A16 | 6.63 | 6.625925 | -0.004075 | 9.48 | 9.480332 | +0.000332 | wikitext, c4 |
| llama-3.1-8b | FourOverSix W4A16 | 6.60 | 6.598704 | -0.001296 | 9.42 | 9.423216 | +0.003216 | wikitext, c4 |
| llama-3.1-8b | RaZeR W4A16 | 6.50 | 6.500746 | +0.000746 | 9.29 | 9.292561 | +0.002561 | wikitext, c4 |
| qwen3-4b | BF16 (paper: FP16) | 13.66 | 13.662473 | +0.002473 | 16.65 | 16.643564 | -0.006436 | wikitext |
| qwen3-4b | NVFP4 W4A4 | 13.88 | 13.935143 | +0.055143 | 17.21 | 17.209829 | -0.000171 | c4 |
| qwen3-4b | FourOverSix W4A4 | 13.88 | 14.201942 | +0.321942 | 17.21 | 17.280434 | +0.070434 | none |
| qwen3-4b | RaZeR W4A4 | 13.82 | 14.104443 | +0.284443 | 17.11 | 17.260754 | +0.150754 | none |
| qwen3-4b | NVFP4 W4A16 | 13.83 | 13.631868 | -0.198132 | 16.85 | 16.851393 | +0.001393 | c4 |
| qwen3-4b | FourOverSix W4A16 | 13.83 | 14.040705 | +0.210705 | 16.85 | 17.015335 | +0.165335 | none |
| qwen3-4b | RaZeR W4A16 | 13.83 | 13.969723 | +0.139723 | 16.85 | 17.033857 | +0.183857 | none |

Execution: job `335302`, account `gov113008`; NVIDIA H200. Independent one-GPU Slurm steps share one eight-GPU allocation. Python 3.10.18, Torch 2.7.1+cu126, Transformers 4.57.3, datasets 4.4.1; recorded attention backend(s): `sdpa`. All policies within each model passed exact input-token-hash equality checks; recorded losses reproduce the original evaluator’s PPL exactly.

**Historical behavior and limits.** The archived Qwen wrapper leaves `o_proj` inputs unquantized, despite calculating a quantized copy. Its W4A4 labels therefore describe the release’s command-line setting, with this omission; they must not replace a corrected full-W4A4 baseline without disclosure. The original NVFP4 midpoint lookup also differs from the current arithmetic quantizer. The first January evaluator could load author-local cached C4 tokens; those tokens are unavailable, and the February release always regenerates seed-0 windows. The released environment omits a Torch version, so exact environment reconstruction is not established. These are reproduction limits, not explanations proven to account for every residual.

This audit does not change or validate the existing 512-token E0M3 calibration gains at 2048 tokens.

[Published Table 3](https://arxiv.org/html/2501.04052v2#S4.T3).

### Fixed environment diagnostic

Both complete attempts are retained. The second uses Python 3.10.18, the released core package pins, and Torch 2.7.1/CUDA 12.6 inferred from the listed Triton/CUDA dependencies. The first uses Python 3.11/Torch 2.9. This does not establish the authors’ exact environment.

All input windows match exactly between attempts. Maximum absolute PPL change: **0.000004**. Δ below is second environment minus first; every case is shown.

| Case | Δ Wiki | Δ C4 | Changed weight matrices |
|---|---:|---:|---:|
| llama-3.1-8b_bf16 | +0.000000 | +0.000000 | 0/224 |
| llama-3.1-8b_nvfp4_w4a4 | +0.000000 | +0.000000 | 0/224 |
| llama-3.1-8b_four_over_six_w4a4 | +0.000000 | +0.000000 | 0/224 |
| llama-3.1-8b_razer_w4a4 | +0.000000 | +0.000000 | 0/224 |
| llama-3.1-8b_nvfp4_w4a16 | +0.000000 | +0.000000 | 0/224 |
| llama-3.1-8b_four_over_six_w4a16 | +0.000000 | +0.000002 | 0/224 |
| llama-3.1-8b_razer_w4a16 | +0.000000 | +0.000000 | 0/224 |
| qwen3-4b_bf16 | +0.000000 | +0.000004 | 0/252 |
| qwen3-4b_nvfp4_w4a4 | +0.000000 | +0.000000 | 0/252 |
| qwen3-4b_four_over_six_w4a4 | +0.000000 | -0.000004 | 0/252 |
| qwen3-4b_razer_w4a4 | +0.000000 | +0.000000 | 0/252 |
| qwen3-4b_nvfp4_w4a16 | -0.000003 | +0.000004 | 0/252 |
| qwen3-4b_four_over_six_w4a16 | +0.000000 | +0.000000 | 0/252 |
| qwen3-4b_razer_w4a16 | +0.000000 | +0.000000 | 0/252 |

Qwen3-4B NVFP4 weight-only WikiText is listed as **13.63 in Table 1** and **13.83 in Table 3**. The primary table above keeps the Table 3 target. This internal difference is documented separately rather than changing the reproduction target.
