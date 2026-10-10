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
| Nemotron-Nano-9B-v2 | NVFP4 | 8.4494 | 11.5223 | +2.77 ± 1.60* | +1.64 ± 1.06* | +0.36 ± 1.41 | -0.68 ± 0.92 | quantized / quantized | `6a9c9d292d49a82e` |
| Nemotron-Nano-9B-v2 | FourOverSix | 8.3950 | 11.4884 | -3.68 ± 1.38* | -1.31 ± 0.96* | +0.02 ± 1.24 | +0.32 ± 0.81 | quantized / quantized | `e2aba63fe972b433` |
| Nemotron-Nano-9B-v2 | FlipQuant 16x64 | 8.3936 | 11.4920 | -3.85 ± 1.39* | -0.99 ± 1.03 | -0.31 ± 1.33 | +0.45 ± 0.84 | quantized / quantized | `7d2ff4f579498991` |
| Qwen3.8-27B | NVFP4 | 7.3706 | 10.1799 | +9.31 ± 5.14* | -1.02 ± 1.26 | -20.70 ± 5.62* | +0.71 ± 0.93 | quantized / quantized | `1eb8cda90ce0109f` |
| Qwen3.8-27B | FourOverSix | 7.4282 | 10.1433 | +17.10 ± 7.10* | -4.62 ± 1.08* | +13.37 ± 5.24* | +0.09 ± 0.93 | quantized / quantized | `78a99adb2bf508bf` |
| Qwen3.8-27B | FlipQuant 16x64 | 7.2270 | 10.1390 | -10.37 ± 4.87* | -5.05 ± 1.08* | -11.69 ± 3.84* | -0.89 ± 0.95 | quantized / quantized | `38e8ef0f596953d4` |

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
| O-1 ppl | 6 / 6 |
| O-2 | 2 / 2 |

GPTQ code files (`/vault/flipquant_paper_eval/ptq_round3`), sha256:

- `/vault/flipquant_paper_eval/ptq_round3/nemotron-nano-9b-v2/codes_fo6.pt`: `8b2d569662247801d45138063c98beb74d6c8eefa33d0e52678c3483f8435168`
- `/vault/flipquant_paper_eval/ptq_round3/nemotron-nano-9b-v2/codes_fq-16x64.pt`: `0663cc237730702c9b397037450c1933f22a184446618736f4e3bd648e7f064e`
- `/vault/flipquant_paper_eval/ptq_round3/nemotron-nano-9b-v2/codes_nvfp4.pt`: `b22d36018b755f43a90aa38384d20195eba13c733fd73407533b2bae50ea2abf`
- `/vault/flipquant_paper_eval/ptq_round3/qwen3.8-27b/codes_fo6.pt`: `3207d39c5f4f17b06f4fa16f1a8d51cc6c140c3d06e00c206438846bf1ccaea0`
- `/vault/flipquant_paper_eval/ptq_round3/qwen3.8-27b/codes_fq-16x64.pt`: `0abc74ed3b47eef7137ec057e4a76caca8f2072c11fdd5768d70a58162bb4e8e`
- `/vault/flipquant_paper_eval/ptq_round3/qwen3.8-27b/codes_nvfp4.pt`: `4f969bf8c6c4e0ec802ff7c5715bbd95326d9132cc80c1f7d6108992dbbaeb43`
