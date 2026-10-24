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

# Mistral-7B-Instruct-v0.3 FlipQuant MixFP4 map, 1,024 calibration windows

## What this is

A format map for [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) at revision `c170c708c41dac9275d15a8fff4eca08d52bab71`, calibrated with FlipQuant (TM-OPT+TC) on **1,024 calibration windows**, four times the release's 256 (`edgeai-lab/Mistral-7B-Instruct-v0.3-FlipQuant`). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The map holds no weights: it is applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_16x64.pt` | 16 × 64 | 359,840 of 6,815,744 (5.28 %) | `8bb3d25d704d87ed1fb9f23b0958b5d87e23159769567dc08f3b22f4bf960960` |

The E0M3 share is 5.28 % (the 256-window map: 1.11 %). Only the 16x64 map is released here.

The FlipQuant code will be released with the paper.

## Calibration

`python -m calibration.train_map --model mistral-7b --unit 16x64 --epochs 5 --teacher-topk 1000 --fit-windows 1024` (FlipQuant commit `898d00e102cb5eb0f9ce7614d8d0db8a7cf1a6f7`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs: **640 steps** (the release: 160). Deterministic.
- Fit set: 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 0 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `15803812b91f9d8cea000d889bd6dfb8adc4fc10f3c5788d1fd367a90f1bdbe9`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item.
- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The map's `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120):

| setup | training | train_map end to end | GPU peak allocated | GPU peak reserved | trainer peak RSS | teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.7 min | 12.7 min | 14.6 min | 38.8 GiB | 39.7 GiB | 14.4 GiB | 2.93 GiB (cuda:0) | 40.4 GiB |

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
| FlipQuant 16x64, 256 windows (`Mistral-7B-Instruct-v0.3-FlipQuant`) | 75.77 | 44.81 ± 2.74 | 5.653 | 8.324 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 75.69 | 45.79 ± 2.74 | 5.636 | 8.304 |

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 83.51 | 83.32 | 74.59 | 81.73 | 59.90 | 73.53 |
| FourOverSix (reference) | 82.75 | 82.86 | 74.82 | 80.64 | 58.36 | 73.12 |
| FlipQuant 16x64, 256 windows (`Mistral-7B-Instruct-v0.3-FlipQuant`) | 82.86 | 82.94 | 74.66 | 80.64 | 59.56 | 73.96 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 83.24 | 82.82 | 74.98 | 80.85 | 58.45 | 73.80 |

**This repository against its references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this row gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| vs FourOverSix | +0.22 (491/443, p=0.12) | +1.29 (147/130, p=0.34) | -0.0128 ± 0.0012* | -0.0114 ± 0.0011* |
| vs its 256-window version (Mistral-7B-Instruct-v0.3-FlipQuant) | -0.08 (421/438, p=0.59) | +0.99 (148/135, p=0.48) | -0.0031 ± 0.0011* | -0.0024 ± 0.0009* |

## How to use

The FlipQuant code will be released with the paper. With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels:

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

This map is released under the same license as the base model, the Apache License 2.0.

The base model card declares Apache-2.0, and the base repository has no license file; `LICENSE` is the Apache License 2.0 text from https://www.apache.org/licenses/LICENSE-2.0.txt.

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `flipquant_16x64.pt` | 6,890,735 | `8bb3d25d704d87ed1fb9f23b0958b5d87e23159769567dc08f3b22f4bf960960` |
| `ppl_summary.json` | 7,662 | `a67971b9ffe295db1962316fcff67888a5d9832fd7d4b4754949fd6f26c60607` |

- `flipquant_16x64.pt`: the map (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads it).
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.

**Provenance of the map file.** It is the calibration output with only its `meta` changed: the run directory and calibration-record paths are relative instead of local absolute paths, the data-extension code reference names the file without its repository owner, and the internal branch and commit of the reference implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the calibration output → this repository's file: `b6adf69a31000018a5c5ac2e137cf0b66373f0d150f1a40c6470b6da6d0ad1f5` → `8bb3d25d704d87ed1fb9f23b0958b5d87e23159769567dc08f3b22f4bf960960`.
