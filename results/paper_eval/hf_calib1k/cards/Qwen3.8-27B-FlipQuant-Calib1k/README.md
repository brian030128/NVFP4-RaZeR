---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- quantization
- fp4
- nvfp4
- mixfp4
- flipquant
---

# Qwen3.8-27B FlipQuant MixFP4 map, 1,024 calibration windows

## What this is

A format map for [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, calibrated with FlipQuant (TM-OPT+TC) on **1,024 calibration windows**, four times the release's 256 (`edgeai-lab/Qwen3.8-27B-FlipQuant`). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The map holds no weights: it is applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_16x64.pt` | 16 × 64 | 602,694 of 23,779,840 (2.53 %) | `63117f22990a843960ebc72139f62f78b5567272dd89208e503ca9f5894f4398` |

The E0M3 share is 2.53 % (the 256-window map: 0.44 %). Only the 16x64 map is released here.

The FlipQuant code will be released with the paper.

## Calibration

`python -m calibration.train_map --model qwen3.8-27b --unit 16x64 --epochs 5 --teacher-topk 1000 --fit-windows 1024 --teacher-host` (FlipQuant commit `898d00e102cb5eb0f9ce7614d8d0db8a7cf1a6f7`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step (2 × 4 accumulated), 5 epochs: **640 steps** (the release: 160). Deterministic.
- Fit set: 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 233 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `4892a27ce5e95286212a4970ea5e390c5818c7bdb331ecad84bb7d00bf63e2c6`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item.
- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The map's `meta` records them as `non_default_settings`.

The BF16 teacher's top-1000 targets for 1,024 windows do not fit next to this model on the GPU, so they are kept in host memory (`--teacher-host`). On Llama-3.2-1B-Instruct, the same option gives the GPU-teacher map bit for bit.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120):

| setup | training | train_map end to end | GPU peak allocated | GPU peak reserved | trainer peak RSS | teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13.7 min | 144.0 min | 158.0 min | 90.5 GiB | 92.0 GiB | 51.9 GiB | 2.93 GiB (cpu) | 92.8 GiB |

## Results

From the FlipQuant paper's calibration-size study. The numbers are also in `ppl_summary.json`.

Native sm_120 kernels on one NVIDIA RTX PRO 6000 Blackwell. Every quantized row uses per-token FourOverSix activations, and every row is evaluated on the same data:

- PPL: WikiText-2 test (141 windows of 2,048 tokens) and C4 validation (256 windows of 2,048 tokens), FlipQuant's `--paper-convention`;
- zero-shot: PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge and LAMBADA (lm-eval 0.4.11, full evaluation splits; acc_norm for PIQA, HellaSwag and ARC, acc for WinoGrande and LAMBADA). Avg. is their mean;
- GSM8K: lm-eval's `gsm8k_llama`: 8-shot chain of thought with the chat template, greedy decoding, at most 1,024 new tokens, strict-match accuracy ± 2 SE.

| method | zero-shot Avg. (%) | GSM8K strict (%) | WikiText-2 PPL | C4 PPL |
|---|---:|---:|---:|---:|
| BF16 | 73.96 | 95.68 ± 1.12 ◊ | 7.052 | 9.893 |
| FourOverSix (reference) | 73.70 | 94.62 ± 1.24 | 7.302 | 10.190 |
| FlipQuant 16x64, 256 windows (`Qwen3.8-27B-FlipQuant`) | 73.91 | 95.15 ± 1.18 | 7.186 | 10.141 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 74.41 | 95.91 ± 1.09 | 7.078 | 10.122 |

◊ BF16's GSM8K ran at batch 48, the quantized rows at 64.

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 81.50 | 82.86 | 75.93 | 73.11 | 58.36 | 72.02 |
| FourOverSix (reference) | 81.07 | 82.35 | 75.14 | 74.12 | 58.96 | 70.56 |
| FlipQuant 16x64, 256 windows (`Qwen3.8-27B-FlipQuant`) | 81.77 | 82.50 | 74.66 | 75.08 | 58.36 | 71.07 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 81.72 | 82.22 | 75.85 | 75.97 | 59.30 | 71.41 |

**This repository against its references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this row gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| vs FourOverSix | +0.46* (523/423, p=0.0013) | +1.29* (32/15, p=0.019) | -0.0311 ± 0.0052* | -0.0067 ± 0.0011* |
| vs its 256-window version (Qwen3.8-27B-FlipQuant) | +0.16 (445/409, p=0.23) | +0.76 (26/16, p=0.16) | -0.0151 ± 0.0039* | -0.0018 ± 0.0009* |

## How to use

The FlipQuant code will be released with the paper. With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels:

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model qwen3.8-27b --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model qwen3.8-27b --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the PPL above.

## License

This map is released under the same license as the base model, the Apache License 2.0.

The base model card declares Apache-2.0, and the base repository has no license file; `LICENSE` is the Apache License 2.0 text from https://www.apache.org/licenses/LICENSE-2.0.txt.

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `flipquant_16x64.pt` | 23,945,423 | `63117f22990a843960ebc72139f62f78b5567272dd89208e503ca9f5894f4398` |
| `ppl_summary.json` | 7,610 | `936c31743f010bd9cff6c047e78f21bdf94fd0250b5275dd6b361035b28a981c` |

- `flipquant_16x64.pt`: the map (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads it).
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.

**Provenance of the map file.** It is the calibration output with only its `meta` changed: the run directory and calibration-record paths are relative instead of local absolute paths, the data-extension code reference names the file without its repository owner, and the internal branch and commit of the reference implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the calibration output → this repository's file: `3aaf4e4c31bccb22f86ea18974815d3b44c6ae9e0166fb5ee9910c2894a2bcfa` → `63117f22990a843960ebc72139f62f78b5567272dd89208e503ca9f5894f4398`.
