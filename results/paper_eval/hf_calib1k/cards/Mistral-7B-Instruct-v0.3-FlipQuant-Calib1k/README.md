---
license: apache-2.0
base_model: mistralai/Mistral-7B-Instruct-v0.3
tags:
- quantization
- fp4
- nvfp4
- mixfp4
- flipquant
---

# Mistral-7B-Instruct-v0.3 FlipQuant MixFP4 maps, 1,024 calibration windows

## What this is

Format maps for [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) at revision `c170c708c41dac9275d15a8fff4eca08d52bab71`, calibrated with FlipQuant (TM-OPT+TC) on **1,024 calibration windows**, four times the release's 256 (`edgeai-lab/Mistral-7B-Instruct-v0.3-FlipQuant`). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 557,943 of 13,631,488 (4.09 %) | `e3fa4365e2f30b3a24b6aa3a5f0568f55f65524ccb69edeccc71e2e1a45c45cd` |
| `flipquant_16x64.pt` | 16 × 64 | 359,840 of 6,815,744 (5.28 %) | `8bb3d25d704d87ed1fb9f23b0958b5d87e23159769567dc08f3b22f4bf960960` |
| `flipquant_256x64.pt` | 256 × 64 | 63,099 of 425,984 (14.81 %) | `dbec75c2fc97dafce36a5bcc24a2789188052dd3166c46af73a65c6d7cd0dd7f` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

With more windows every map elects more E0M3 tiles. The E0M3 shares of the release's 256-window maps are 8x64 0.86 %, 16x64 1.11 %, 256x64 2.70 %.

**The 256x64 map** has the highest E0M3 share of the three (14.81 %; 16x64: 5.28 %). Against the 16x64 map its PPL is not significantly different (ΔNLL WikiText-2 +0.0004 ± 0.0010, C4 +0.0011 ± 0.0012; neither beyond 2 SE); GSM8K +0.15 (142/140, p=0.95), zero-shot -0.21 (431/476, p=0.14). Pooled over the study's eight models, 256x64 at 1,024 windows is the weakest of the three tiles (GSM8K −1.39*, zero-shot −0.24* against 16x64), and more windows did not improve it over its 256-window version (GSM8K −0.26 (n.s.)).

The FlipQuant code will be released with the paper.

## Calibration

`python -m calibration.train_map --model mistral-7b --unit <unit> --epochs 5 --teacher-topk 1000 --fit-windows 1024`, the same settings and the same 1,024 windows for the three tiles (FlipQuant revisions: 8x64 `394e6838825c23f234fe0a3ef4c252b0b6866be1`, 16x64 `898d00e102cb5eb0f9ce7614d8d0db8a7cf1a6f7`, 256x64 `394e6838825c23f234fe0a3ef4c252b0b6866be1`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs: **640 steps** for each tile (the release: 160). Deterministic.
- Fit set: 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 0 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `15803812b91f9d8cea000d889bd6dfb8adc4fc10f3c5788d1fd367a90f1bdbe9`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item.
- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| unit | setup | training | train_map end to end | GPU peak allocated | GPU peak reserved | trainer peak RSS | teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x64 | 1.8 min | 12.5 min | 14.5 min | 39.0 GiB | 39.9 GiB | 14.5 GiB | 2.93 GiB (cuda:0) | 40.6 GiB |
| 16x64 | 1.7 min | 12.7 min | 14.6 min | 38.8 GiB | 39.7 GiB | 14.4 GiB | 2.93 GiB (cuda:0) | 40.4 GiB |
| 256x64 | 1.9 min | 12.5 min | 14.6 min | 38.7 GiB | 39.6 GiB | 14.5 GiB | 2.93 GiB (cuda:0) | 40.3 GiB |

## Results

From the FlipQuant paper's calibration-size study. The numbers are also in `ppl_summary.json`.

Native sm_120 kernels on one NVIDIA RTX PRO 6000 Blackwell. Every quantized row uses per-token FourOverSix activations, and every row is evaluated on the same data:

- PPL: WikiText-2 test (141 windows of 2,048 tokens) and C4 validation (256 windows of 2,048 tokens), FlipQuant's `--paper-convention`;
- zero-shot: PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge and LAMBADA (lm-eval 0.4.11, full evaluation splits; acc_norm for PIQA, HellaSwag and ARC, acc for WinoGrande and LAMBADA). Avg. is their mean;
- GSM8K: lm-eval's `gsm8k_llama`: 8-shot chain of thought with the chat template, greedy decoding, at most 1,024 new tokens, strict-match accuracy ± 2 SE.

| method | zero-shot Avg. (%) | GSM8K strict (%) | WikiText-2 PPL | C4 PPL |
|---|---:|---:|---:|---:|
| BF16 | 76.10 | 47.01 ± 2.75 | 5.496 | 8.135 |
| FourOverSix (reference) | 75.43 | 44.50 ± 2.74 | 5.708 | 8.399 |
| FlipQuant 8x64, 256 windows (`Mistral-7B-Instruct-v0.3-FlipQuant`) | 75.52 | 45.11 ± 2.74 | 5.650 | 8.319 |
| FlipQuant 8x64, 1,024 windows (**this repository**) | 75.53 | 46.63 ± 2.75 | 5.634 | 8.304 |
| FlipQuant 16x64, 256 windows (`Mistral-7B-Instruct-v0.3-FlipQuant`) | 75.77 | 44.81 ± 2.74 | 5.653 | 8.324 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 75.69 | 45.79 ± 2.74 | 5.636 | 8.304 |
| FlipQuant 256x64, 256 windows (`Mistral-7B-Instruct-v0.3-FlipQuant`) | 75.40 | 44.73 ± 2.74 | 5.662 | 8.340 |
| FlipQuant 256x64, 1,024 windows (**this repository**) | 75.20 | 45.94 ± 2.74 | 5.638 | 8.313 |

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 83.51 | 83.32 | 74.59 | 81.73 | 59.90 | 73.53 |
| FourOverSix (reference) | 82.75 | 82.86 | 74.82 | 80.64 | 58.36 | 73.12 |
| FlipQuant 8x64, 256 windows (`Mistral-7B-Instruct-v0.3-FlipQuant`) | 82.37 | 82.71 | 75.45 | 80.77 | 58.19 | 73.61 |
| FlipQuant 8x64, 1,024 windows (**this repository**) | 82.86 | 82.65 | 75.53 | 80.93 | 57.51 | 73.69 |
| FlipQuant 16x64, 256 windows (`Mistral-7B-Instruct-v0.3-FlipQuant`) | 82.86 | 82.94 | 74.66 | 80.64 | 59.56 | 73.96 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 83.24 | 82.82 | 74.98 | 80.85 | 58.45 | 73.80 |
| FlipQuant 256x64, 256 windows (`Mistral-7B-Instruct-v0.3-FlipQuant`) | 83.24 | 82.71 | 73.32 | 81.27 | 58.11 | 73.76 |
| FlipQuant 256x64, 1,024 windows (**this repository**) | 82.92 | 82.82 | 73.40 | 80.98 | 57.51 | 73.59 |

**The maps against their references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this map gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| 8x64 vs FourOverSix | +0.07 (483/467, p=0.63) | +2.12 (155/127, p=0.11) | -0.0131 ± 0.0011* | -0.0113 ± 0.0011* |
| 8x64 vs its 256-window version | +0.02 (438/434, p=0.92) | +1.52 (148/128, p=0.25) | -0.0028 ± 0.0011* | -0.0017 ± 0.0009* |
| 8x64 vs the 16x64 map (this repository) | -0.15 (416/448, p=0.29) | +0.83 (142/131, p=0.55) | -0.0003 ± 0.0010 | +0.0001 ± 0.0007 |
| 16x64 vs FourOverSix | +0.22 (491/443, p=0.12) | +1.29 (147/130, p=0.34) | -0.0128 ± 0.0012* | -0.0114 ± 0.0011* |
| 16x64 vs its 256-window version | -0.08 (421/438, p=0.59) | +0.99 (148/135, p=0.48) | -0.0031 ± 0.0011* | -0.0024 ± 0.0009* |
| 256x64 vs FourOverSix | +0.01 (494/491, p=0.95) | +1.44 (149/130, p=0.28) | -0.0124 ± 0.0012* | -0.0103 ± 0.0018* |
| 256x64 vs its 256-window version | -0.08 (466/483, p=0.6) | +1.21 (159/143, p=0.39) | -0.0043 ± 0.0009* | -0.0033 ± 0.0011* |
| 256x64 vs the 16x64 map (this repository) | -0.21 (431/476, p=0.14) | +0.15 (142/140, p=0.95) | +0.0004 ± 0.0010 | +0.0011 ± 0.0012 |

## How to use

The FlipQuant code will be released with the paper. With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels (any of the three maps):

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model mistral-7b --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model mistral-7b --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the PPL above.

## License

These maps are released under the same license as the base model, the Apache License 2.0.

The base model card declares Apache-2.0, and the base repository has no license file; `LICENSE` is the Apache License 2.0 text from https://www.apache.org/licenses/LICENSE-2.0.txt.

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `flipquant_16x64.pt` | 6,890,735 | `8bb3d25d704d87ed1fb9f23b0958b5d87e23159769567dc08f3b22f4bf960960` |
| `flipquant_256x64.pt` | 501,077 | `dbec75c2fc97dafce36a5bcc24a2789188052dd3166c46af73a65c6d7cd0dd7f` |
| `flipquant_8x64.pt` | 13,706,441 | `e3fa4365e2f30b3a24b6aa3a5f0568f55f65524ccb69edeccc71e2e1a45c45cd` |
| `ppl_summary.json` | 16,881 | `4e32efc5e106fed6e79eca636273b4b525013b17875c3de3892dcd2d169a9165` |

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.

**Provenance of the map files.** They are the calibration outputs with only their `meta` changed: the run directory and calibration-record paths are relative instead of local absolute paths, the data-extension code reference names the file without its repository owner, and the internal branch and commit of the reference implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the calibration output → this repository's file:

- `flipquant_8x64.pt`: `a8f262ac643ff255f2ee00387c58896e7a4e38882d697b56e5f17847841aa914` → `e3fa4365e2f30b3a24b6aa3a5f0568f55f65524ccb69edeccc71e2e1a45c45cd`
- `flipquant_16x64.pt`: `b6adf69a31000018a5c5ac2e137cf0b66373f0d150f1a40c6470b6da6d0ad1f5` → `8bb3d25d704d87ed1fb9f23b0958b5d87e23159769567dc08f3b22f4bf960960`
- `flipquant_256x64.pt`: `73e93f19f2a57b4696ccf2e92c5b7d508998bbdd94bbf4c97381ad8e8e9b2a89` → `dbec75c2fc97dafce36a5bcc24a2789188052dd3166c46af73a65c6d7cd0dd7f`
