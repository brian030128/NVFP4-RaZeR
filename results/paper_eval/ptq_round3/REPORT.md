# Round 3 of tab:ptq (part O)

PROTOCOL.md part O (registered 2026-10-10 07:34 UTC; amendment 1: GPTQ code files on the user's vault, the trainer in n16k64). Native sm120 (build_V, `--kernel-set auto`), n16k64-fast, release revisions; flipquant paper-sm120-runs 276ce86. ΔNLL: per-window paired difference x 1e-3 ± 2 SE (* beyond 2 SE). GSM8K: part N's protocol (lm-eval 0.4.11 gsm8k_llama, chat template, greedy, 1024 tokens, thinking off, 1,319 problems); accuracy ± 2 SE (binomial); paired per problem: A − B in percentage points ± 2 SE (A-only / B-only, McNemar exact p).

## O0: the BF16 reference row

| model | batch | strict-match (%) | flexible-extract (%) | cannot extract (strict / flexible) | used the 1024 budget | mean / max tokens | WikiText-2 | C4 |
|---|---:|---:|---:|---|---:|---|---:|---:|
| Nemotron-Nano-9B-v2 | 64 | 87.41 ± 1.83 | 90.52 ± 1.61 | 54 / 0 | 0 | 114 / 503 | 8.0863 | 11.1528 |
| Qwen3.8-27B | 48 | 95.68 ± 1.12 | 95.83 ± 1.10 | 20 / 0 | 20 | 232 / 1024 | 7.0523 | 9.8931 |

BF16 PPL: part F's records (`RUN/pplfast/<model>/bf16.json`; n16k64-fast, release revisions). Qwen3.8-27B's BF16 GSM8K ran at batch 48 (batch 64 failed the O0 memory rule); every quantized row ran at 64.

Part N's 18 configurations vs BF16 (A = the configuration, B = BF16):

| model | configuration | strict-match Δ (pp) | flexible-extract Δ (pp) |
|---|---|---:|---:|
| Nemotron-Nano-9B-v2 | RTN NVFP4 | -0.08 ± 1.64 (58 / 59, p 1) | -1.67 ± 1.35 * (29 / 51, p 0.0183) |
| Nemotron-Nano-9B-v2 | RTN FourOverSix | +1.52 ± 1.57 (64 / 44, p 0.067) | +0.15 ± 1.19 (32 / 30, p 0.899) |
| Nemotron-Nano-9B-v2 | RTN FlipQuant 16x64 | +1.90 ± 1.54 * (64 / 39, p 0.0176) | +0.76 ± 1.23 (38 / 28, p 0.268) |
| Nemotron-Nano-9B-v2 | GPTQ NVFP4 | -3.41 ± 1.82 * (50 / 95, p 0.000232) | -1.36 ± 1.35 * (31 / 49, p 0.0567) |
| Nemotron-Nano-9B-v2 | GPTQ FourOverSix | -1.29 ± 1.67 (52 / 69, p 0.145) | -1.74 ± 1.33 * (27 / 50, p 0.0117) |
| Nemotron-Nano-9B-v2 | GPTQ FlipQuant 16x64 | -1.82 ± 1.77 * (56 / 80, p 0.0482) | -1.67 ± 1.45 * (35 / 57, p 0.028) |
| Nemotron-Nano-9B-v2 | Hadamard NVFP4 | -2.50 ± 1.86 * (59 / 92, p 0.00899) | -2.65 ± 1.53 * (34 / 69, p 0.000728) |
| Nemotron-Nano-9B-v2 | Hadamard FourOverSix | +0.76 ± 1.67 (66 / 56, p 0.415) | -1.52 ± 1.47 * (37 / 57, p 0.0495) |
| Nemotron-Nano-9B-v2 | Hadamard FlipQuant 16x64 | +1.52 ± 1.65 (69 / 49, p 0.0798) | -0.99 ± 1.38 (35 / 48, p 0.187) |
| Qwen3.8-27B | RTN NVFP4 | -0.30 ± 1.03 (21 / 25, p 0.659) | -0.23 ± 1.02 (21 / 24, p 0.766) |
| Qwen3.8-27B | RTN FourOverSix | -1.06 ± 1.03 * (16 / 30, p 0.0541) | -0.68 ± 0.92 (14 / 23, p 0.188) |
| Qwen3.8-27B | RTN FlipQuant 16x64 | -0.53 ± 0.92 (15 / 22, p 0.324) | -0.23 ± 0.87 (15 / 18, p 0.728) |
| Qwen3.8-27B | GPTQ NVFP4 | +0.15 ± 0.83 (16 / 14, p 0.856) | +0.00 ± 0.83 (15 / 15, p 1) |
| Qwen3.8-27B | GPTQ FourOverSix | +0.91 ± 0.86 * (22 / 10, p 0.0501) | +0.83 ± 0.84 (21 / 10, p 0.0708) |
| Qwen3.8-27B | GPTQ FlipQuant 16x64 | +0.91 ± 0.83 * (21 / 9, p 0.0428) | +0.68 ± 0.82 (19 / 10, p 0.136) |
| Qwen3.8-27B | Hadamard NVFP4 | +0.08 ± 0.82 (15 / 14, p 1) | +0.23 ± 0.87 (18 / 15, p 0.728) |
| Qwen3.8-27B | Hadamard FourOverSix | +0.53 ± 0.92 (22 / 15, p 0.324) | +0.68 ± 0.95 (24 / 15, p 0.2) |
| Qwen3.8-27B | Hadamard FlipQuant 16x64 | +0.08 ± 0.97 (21 / 20, p 1) | -0.15 ± 1.01 (21 / 23, p 0.88) |

## O1: E0M3 shares, Hadamard-basis vs release maps

See `O1_SHARES.md` (all rows, per projection type). In all: Nemotron-Nano-9B-v2 1.066 % vs 1.064 %, Qwen3.8-27B 0.427 % vs 0.443 %; Jaccard 0.010 / 0.005 (1.6x / 1.8x chance). Env caveat: part H trained the Hadamard maps in n16k64-fast, the release maps were trained in n16k64.

## O2: GPTQ with activation quantization on during calibration

| model | format | WikiText-2 | C4 | ΔNLL wiki vs RTN FO6 | ΔNLL C4 vs RTN FO6 | ΔNLL wiki vs part H GPTQ | ΔNLL C4 vs part H GPTQ | propagate / Hessian input | codes_sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Nemotron-Nano-9B-v2 | NVFP4 | TBD | TBD | | | | | | |
| Nemotron-Nano-9B-v2 | FourOverSix | TBD | TBD | | | | | | |
| Nemotron-Nano-9B-v2 | FlipQuant 16x64 | TBD | TBD | | | | | | |
| Qwen3.8-27B | NVFP4 | TBD | TBD | | | | | | |
| Qwen3.8-27B | FourOverSix | TBD | TBD | | | | | | |
| Qwen3.8-27B | FlipQuant 16x64 | TBD | TBD | | | | | | |

## O3: GPTQ candidates and a retrained map

### Nemotron-Nano-9B-v2


### Qwen3.8-27B


## O4: GSM8K for the new rows (batch 64)

| model | row | strict-match (%) | flexible-extract (%) | cannot extract | used the budget | mean / max tokens |
|---|---|---:|---:|---|---:|---|
| Nemotron-Nano-9B-v2 | GPTQ, act. quant. on: NVFP4 | TBD | TBD | | | |
| Nemotron-Nano-9B-v2 | GPTQ, act. quant. on: FourOverSix | TBD | TBD | | | |
| Nemotron-Nano-9B-v2 | GPTQ, act. quant. on: FlipQuant 16x64 (release map fixed) | TBD | TBD | | | |
| Nemotron-Nano-9B-v2 | GPTQ candidates + retrained map: FlipQuant 16x64 | TBD | TBD | | | |
| Qwen3.8-27B | GPTQ, act. quant. on: NVFP4 | TBD | TBD | | | |
| Qwen3.8-27B | GPTQ, act. quant. on: FourOverSix | TBD | TBD | | | |
| Qwen3.8-27B | GPTQ, act. quant. on: FlipQuant 16x64 (release map fixed) | TBD | TBD | | | |
| Qwen3.8-27B | GPTQ candidates + retrained map: FlipQuant 16x64 | TBD | TBD | | | |

| model | comparison (A vs B) | strict-match Δ (pp) | flexible-extract Δ (pp) |
|---|---|---:|---:|

## Checks

| check | passed |
|---|---:|
| N1 | 2 / 2 |
| N4 | 2 / 2 |

GPTQ code files (`/vault/flipquant_paper_eval/ptq_round3`), sha256:

