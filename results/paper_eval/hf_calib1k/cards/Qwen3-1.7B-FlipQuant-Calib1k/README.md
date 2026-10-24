---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
tags:
- quantization
- fp4
- nvfp4
- mixfp4
- flipquant
---

# Qwen3-1.7B FlipQuant MixFP4 map, 1,024 calibration windows

## What this is

A format map for [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, calibrated with FlipQuant (TM-OPT+TC) on **1,024 calibration windows**, four times the release's 256 (`edgeai-lab/Qwen3-1.7B-FlipQuant`). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The map holds no weights: it is applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_16x64.pt` | 16 × 64 | 112,189 of 1,376,256 (8.15 %) | `d72eb4d71274646b74fc320eeaa8e948a562c25ea567a0286d26b496c5deea4f` |

The E0M3 share is 8.15 % (the 256-window map: 1.19 %). Only the 16x64 map is released here.

The FlipQuant code will be released with the paper.

## Calibration

`python -m calibration.train_map --model qwen3-1.7b --unit 16x64 --epochs 5 --teacher-topk 1000 --fit-windows 1024` (FlipQuant commit `898d00e102cb5eb0f9ce7614d8d0db8a7cf1a6f7`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs: **640 steps** (the release: 160). Deterministic.
- Fit set: 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 0 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `bd5ef53d52cfa59badd26c93f4c70ed02fb6a3668ab17c568219195f0e6eff77`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item.
- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The map's `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120):

| setup | training | train_map end to end | GPU peak allocated | GPU peak reserved | trainer peak RSS | teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 78.1 s | 4.1 min | 5.6 min | 23.2 GiB | 23.5 GiB | 4.7 GiB | 2.93 GiB (cuda:0) | 24.3 GiB |

## Results

From the FlipQuant paper's calibration-size study. The numbers are also in `ppl_summary.json`.

Native sm_120 kernels on one NVIDIA RTX PRO 6000 Blackwell. Every quantized row uses per-token FourOverSix activations, and every row is evaluated on the same data:

- PPL: WikiText-2 test (141 windows of 2,048 tokens) and C4 validation (256 windows of 2,048 tokens), FlipQuant's `--paper-convention`;
- zero-shot: PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge and LAMBADA (lm-eval 0.4.11, full evaluation splits; acc_norm for PIQA, HellaSwag and ARC, acc for WinoGrande and LAMBADA). Avg. is their mean;
- GSM8K: lm-eval's `gsm8k_llama`: 8-shot chain of thought with the chat template, greedy decoding, at most 1,024 new tokens, strict-match accuracy ± 2 SE, with Qwen3's thinking mode off (`enable_thinking=False`).

| method | zero-shot Avg. (%) | GSM8K strict (%) | WikiText-2 PPL | C4 PPL |
|---|---:|---:|---:|---:|
| BF16 | 59.46 | 74.53 ± 2.40 | 16.716 | 19.246 |
| FourOverSix (reference) | 56.13 | 61.87 ± 2.67 | 19.475 | 20.907 |
| FlipQuant 16x64, 256 windows (`Qwen3-1.7B-FlipQuant`) | 58.08 | 61.33 ± 2.68 | 15.398 | 19.410 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 58.15 | 67.25 ± 2.58 | 15.762 | 19.571 |

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 72.25 | 60.40 | 61.01 | 69.70 | 42.92 | 50.48 |
| FourOverSix (reference) | 70.57 | 58.03 | 58.48 | 65.49 | 40.36 | 43.86 |
| FlipQuant 16x64, 256 windows (`Qwen3-1.7B-FlipQuant`) | 70.95 | 59.13 | 58.33 | 69.57 | 41.21 | 49.27 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 71.33 | 59.03 | 58.25 | 69.57 | 41.38 | 49.33 |

**This repository against its references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this row gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| vs FourOverSix | +2.30* (1553/1050, p=5.6e-23) | +5.38* (171/100, p=1.9e-05) | -0.2115 ± 0.0129* | -0.0660 ± 0.0038* |
| vs its 256-window version (Qwen3-1.7B-FlipQuant) | +0.00 (1082/1081, p=1) | +5.91* (171/93, p=1.8e-06) | +0.0234 ± 0.0030* | +0.0083 ± 0.0018* |

## How to use

The FlipQuant code will be released with the paper. With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels:

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model qwen3-1.7b --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model qwen3-1.7b --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the PPL above.

## License

This map is released under the same license as the base model, the Apache License 2.0.

`LICENSE` is a copy of the base model's license file (Qwen/Qwen3-1.7B).

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 11,343 | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |
| `flipquant_16x64.pt` | 1,442,311 | `d72eb4d71274646b74fc320eeaa8e948a562c25ea567a0286d26b496c5deea4f` |
| `ppl_summary.json` | 7,646 | `2809b2de04b315c6c7a1aa1396e1ab116931f2b15e982f1b80bbefc2f750a854` |

- `flipquant_16x64.pt`: the map (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads it).
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.

**Provenance of the map file.** It is the calibration output with only its `meta` changed: the run directory and calibration-record paths are relative instead of local absolute paths, the data-extension code reference names the file without its repository owner, and the internal branch and commit of the reference implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the calibration output → this repository's file: `38e5bc5ded32ee15ad1933a846414baf79ae5c0d9a3cdc60d14c4266bbf54f0a` → `d72eb4d71274646b74fc320eeaa8e948a562c25ea567a0286d26b496c5deea4f`.
