---
license: llama3.2
base_model: meta-llama/Llama-3.2-1B-Instruct
tags:
- quantization
- fp4
- nvfp4
- mixfp4
- flipquant
---

# Llama-3.2-1B-Instruct FlipQuant MixFP4 map, 1,024 calibration windows

**Built with Llama.**

Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. The agreement is in `LICENSE`, the Acceptable Use Policy it incorporates in `USE_POLICY.md`, and the notice in `NOTICE`.

## What this is

A format map for [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) at revision `9213176726f574b556790deb65791e0c5aa438b6`, calibrated with FlipQuant (TM-OPT+TC) on **1,024 calibration windows**, four times the release's 256 (`edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant`). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The map holds no weights: it is applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_16x64.pt` | 16 × 64 | 141,062 of 950,272 (14.84 %) | `8ba6e70703d93fdb01ef86864badd72d65caf69849fa11d2fd16080a09095d3f` |

The E0M3 share is 14.84 % (the 256-window map: 2.81 %). Only the 16x64 map is released here.

The FlipQuant code will be released with the paper.

## Calibration

`python -m calibration.train_map --model llama3.2-1b-instruct --unit 16x64 --epochs 5 --teacher-topk 1000 --fit-windows 1024` (FlipQuant commit `f912a30d9423a9813cd45dbb6998cc1dc816fd43`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs: **640 steps** (the release: 160). Deterministic.
- Fit set: 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 0 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `d73b2bb2f3179c9a6c1e614c092f10e20d74c8cf5135c3d84cdba551716b1bb4`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item.
- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The map's `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120):

| setup | training | train_map end to end | GPU peak allocated | GPU peak reserved | trainer peak RSS | teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.9 s | 2.6 min | 3.7 min | 16.6 GiB | 17.0 GiB | 3.2 GiB | 2.93 GiB (cuda:0) | 17.7 GiB |

## Results

From the FlipQuant paper's calibration-size study. The numbers are also in `ppl_summary.json`.

Native sm_120 kernels on one NVIDIA RTX PRO 6000 Blackwell. Every quantized row uses per-token FourOverSix activations, and every row is evaluated on the same data:

- PPL: WikiText-2 test (141 windows of 2,048 tokens) and C4 validation (256 windows of 2,048 tokens), FlipQuant's `--paper-convention`;
- zero-shot: PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge and LAMBADA (lm-eval 0.4.11, full evaluation splits; acc_norm for PIQA, HellaSwag and ARC, acc for WinoGrande and LAMBADA). Avg. is their mean;
- GSM8K: lm-eval's `gsm8k_llama`: 8-shot chain of thought with the chat template, greedy decoding, at most 1,024 new tokens, strict-match accuracy ± 2 SE. The chat template writes the current date into the system prompt; every row of this model renders 15 Oct 2026, so all rows see identical prompts.

| method | zero-shot Avg. (%) | GSM8K strict (%) | WikiText-2 PPL | C4 PPL |
|---|---:|---:|---:|---:|
| BF16 | 59.92 | 45.41 ± 2.74 | 13.163 | 18.519 |
| FourOverSix (reference) | 56.24 | 29.72 ± 2.52 | 15.315 | 21.449 |
| FlipQuant 16x64, 256 windows (`Llama-3.2-1B-Instruct-FlipQuant`) | 57.25 | 31.61 ± 2.56 | 14.454 | 19.684 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 57.93 | 35.18 ± 2.63 | 14.387 | 19.605 |

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 75.14 | 61.67 | 61.48 | 63.80 | 37.97 | 59.46 |
| FourOverSix (reference) | 71.38 | 57.74 | 58.17 | 60.19 | 36.35 | 53.64 |
| FlipQuant 16x64, 256 windows (`Llama-3.2-1B-Instruct-FlipQuant`) | 72.85 | 59.31 | 58.09 | 61.15 | 36.35 | 55.75 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 73.39 | 59.36 | 60.46 | 62.08 | 36.26 | 56.05 |

**This repository against its references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this row gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| vs FourOverSix | +1.82* (1336/939, p=8.6e-17) | +5.46* (175/103, p=1.9e-05) | -0.0625 ± 0.0029* | -0.0899 ± 0.0039* |
| vs its 256-window version (Llama-3.2-1B-Instruct-FlipQuant) | +0.37 (1046/965, p=0.074) | +3.56* (166/119, p=0.0063) | -0.0046 ± 0.0021* | -0.0040 ± 0.0016* |

## How to use

The FlipQuant code will be released with the paper. With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels:

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model llama3.2-1b-instruct --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model llama3.2-1b-instruct --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the PPL above.

## License

This map is released under the same license as the base model, the Llama 3.2 Community License.

The Llama 3.2 Community License (section 1.b) applies to derivative works: provide a copy of the agreement, prominently display “Built with Llama”, include “Llama” at the beginning of the name of an AI model created or improved with Llama materials or their outputs, and keep the notice “Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.” in a NOTICE file. This map is trained against Llama-3.2-1B-Instruct's outputs (KL to its BF16 logits).

- `LICENSE`: the Llama 3.2 Community License Agreement, the base repository's own license file (meta-llama/Llama-3.2-1B-Instruct at revision `9213176726f574b556790deb65791e0c5aa438b6`).
- `USE_POLICY.md`: the Llama 3.2 Acceptable Use Policy, which the agreement incorporates; the base repository's own `USE_POLICY.md`.
- `NOTICE`: “Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 7,712 | `0b4284c1f87029e67654c7953afa16279961632cf73dcfe33374c4c2f298fa35` |
| `NOTICE` | 116 | `3bc2d1a6b360a03d172058198631b931c9aadf01650892fb0d25b93c6b8f0025` |
| `README.md` | 9,558 | `cc70a83290cf9c9784e4a0eb1c48c15040bcf93be873ce42573e88e7dca78f37` |
| `USE_POLICY.md` | 6,021 | `40e2777d7faa6beaf98400654170f414d8ab29b921b5163ad4ea0a1d39894201` |
| `flipquant_16x64.pt` | 989,903 | `8ba6e70703d93fdb01ef86864badd72d65caf69849fa11d2fd16080a09095d3f` |
| `ppl_summary.json` | 7,719 | `9f977693f2bd27c67f7826c0a2fee90a2811add8bf467812f89c788f06db766b` |

- `flipquant_16x64.pt`: the map (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads it).
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.
- `USE_POLICY.md`: see License.
- `NOTICE`: see License.

**Provenance of the map file.** It is the calibration output with only its `meta` changed: the run directory and calibration-record paths are relative instead of local absolute paths, the data-extension code reference names the file without its repository owner, and the internal branch and commit of the reference implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the calibration output → this repository's file: `d51b7b1c0b0a08b642eba681b6c6ffa21c172ab320b38bc7643aa11198e76050` → `8ba6e70703d93fdb01ef86864badd72d65caf69849fa11d2fd16080a09095d3f`.
