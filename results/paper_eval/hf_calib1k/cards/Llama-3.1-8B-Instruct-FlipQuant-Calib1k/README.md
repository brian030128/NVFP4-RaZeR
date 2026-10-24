---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
tags:
- quantization
- fp4
- nvfp4
- mixfp4
- flipquant
---

# Llama-3.1-8B-Instruct FlipQuant MixFP4 maps, 1,024 calibration windows

**Built with Llama.**

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. The agreement is in `LICENSE`, the Acceptable Use Policy it incorporates in `USE_POLICY.md`, and the notice in `NOTICE`.

## What this is

Format maps for [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) at revision `0e9e39f249a16976918f6564b8830bc894c89659`, calibrated with FlipQuant (TM-OPT+TC) on **1,024 calibration windows**, four times the release's 256 (`edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant`). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 586,062 of 13,631,488 (4.30 %) | `989ef5ec3d992101c67770b4bc2726c28f966e1faeb65068a29b70145e0fc37e` |
| `flipquant_16x64.pt` | 16 × 64 | 377,393 of 6,815,744 (5.54 %) | `c973cb98c23ba211534bec29fcd3ae238447aab9fa911aa39035f87c8b6a654d` |
| `flipquant_256x64.pt` | 256 × 64 | 64,091 of 425,984 (15.05 %) | `8c2fb731fdf5e3fc8e5a5b2ed715efa81eb37a8b26c6c0a5782852f17765c421` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

With more windows every map elects more E0M3 tiles. The E0M3 shares of the release's 256-window maps are 8x64 0.61 %, 16x64 0.77 %, 256x64 2.22 %.

**The 256x64 map** has the highest E0M3 share of the three (15.05 %; 16x64: 5.54 %). Against the 16x64 map its PPL is higher, beyond 2 SE on both corpora (ΔNLL WikiText-2 +0.0077 ± 0.0016*, C4 +0.0074 ± 0.0013*); GSM8K +0.08 (82/81, p=1), zero-shot -0.48* (588/693, p=0.0036). Pooled over the study's eight models, 256x64 at 1,024 windows is the weakest of the three tiles (GSM8K −1.39*, zero-shot −0.24* against 16x64), and more windows did not improve it over its 256-window version (GSM8K −0.26 (n.s.)).

The FlipQuant code will be released with the paper.

## Calibration

`python -m calibration.train_map --model llama3.1-8b-instruct --unit <unit> --epochs 5 --teacher-topk 1000 --fit-windows 1024`, the same settings and the same 1,024 windows for the three tiles (FlipQuant revisions: 8x64 `394e6838825c23f234fe0a3ef4c252b0b6866be1`, 16x64 `898d00e102cb5eb0f9ce7614d8d0db8a7cf1a6f7`, 256x64 `394e6838825c23f234fe0a3ef4c252b0b6866be1`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs: **640 steps** for each tile (the release: 160). Deterministic.
- Fit set: 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 231 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `e4b509c5b0e7f90ccaf09296f56518923b4ed3b185935014d032946c8a689172`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item.
- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| unit | setup | training | train_map end to end | GPU peak allocated | GPU peak reserved | trainer peak RSS | teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x64 | 1.8 min | 13.2 min | 15.2 min | 44.0 GiB | 45.1 GiB | 15.9 GiB | 2.93 GiB (cuda:0) | 45.8 GiB |
| 16x64 | 1.8 min | 13.3 min | 15.4 min | 43.9 GiB | 44.8 GiB | 15.9 GiB | 2.93 GiB (cuda:0) | 45.5 GiB |
| 256x64 | 2.1 min | 13.2 min | 15.4 min | 43.8 GiB | 44.8 GiB | 15.9 GiB | 2.93 GiB (cuda:0) | 45.5 GiB |

## Results

From the FlipQuant paper's calibration-size study. The numbers are also in `ppl_summary.json`.

Native sm_120 kernels on one NVIDIA RTX PRO 6000 Blackwell. Every quantized row uses per-token FourOverSix activations, and every row is evaluated on the same data:

- PPL: WikiText-2 test (141 windows of 2,048 tokens) and C4 validation (256 windows of 2,048 tokens), FlipQuant's `--paper-convention`;
- zero-shot: PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge and LAMBADA (lm-eval 0.4.11, full evaluation splits; acc_norm for PIQA, HellaSwag and ARC, acc for WinoGrande and LAMBADA). Avg. is their mean;
- GSM8K: lm-eval's `gsm8k_llama`: 8-shot chain of thought with the chat template, greedy decoding, at most 1,024 new tokens, strict-match accuracy ± 2 SE.

| method | zero-shot Avg. (%) | GSM8K strict (%) | WikiText-2 PPL | C4 PPL |
|---|---:|---:|---:|---:|
| BF16 | 73.71 | 85.90 ± 1.92 | 7.217 | 10.385 |
| FourOverSix (reference) | 71.74 | 79.45 ± 2.22 | 7.819 | 11.267 |
| FlipQuant 8x64, 256 windows (`Llama-3.1-8B-Instruct-FlipQuant`) | 72.58 | 81.43 ± 2.14 | 7.695 | 11.138 |
| FlipQuant 8x64, 1,024 windows (**this repository**) | 72.39 | 81.20 ± 2.15 | 7.648 | 11.046 |
| FlipQuant 16x64, 256 windows (`Llama-3.1-8B-Instruct-FlipQuant`) | 72.58 | 80.06 ± 2.20 | 7.711 | 11.142 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 72.28 | 80.36 ± 2.19 | 7.652 | 11.040 |
| FlipQuant 256x64, 256 windows (`Llama-3.1-8B-Instruct-FlipQuant`) | 72.29 | 81.73 ± 2.13 | 7.750 | 11.217 |
| FlipQuant 256x64, 1,024 windows (**this repository**) | 71.64 | 80.44 ± 2.18 | 7.711 | 11.122 |

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 81.34 | 79.58 | 73.80 | 79.71 | 55.72 | 72.13 |
| FourOverSix (reference) | 80.14 | 78.04 | 71.74 | 77.02 | 52.05 | 71.45 |
| FlipQuant 8x64, 256 windows (`Llama-3.1-8B-Instruct-FlipQuant`) | 80.41 | 78.60 | 73.24 | 78.83 | 52.73 | 71.69 |
| FlipQuant 8x64, 1,024 windows (**this repository**) | 79.71 | 78.56 | 72.69 | 78.79 | 53.41 | 71.18 |
| FlipQuant 16x64, 256 windows (`Llama-3.1-8B-Instruct-FlipQuant`) | 79.60 | 78.56 | 73.64 | 78.96 | 53.24 | 71.49 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 79.82 | 78.52 | 73.48 | 78.28 | 52.56 | 71.03 |
| FlipQuant 256x64, 256 windows (`Llama-3.1-8B-Instruct-FlipQuant`) | 80.20 | 78.65 | 71.98 | 78.20 | 54.27 | 70.44 |
| FlipQuant 256x64, 1,024 windows (**this repository**) | 79.71 | 78.52 | 72.14 | 77.69 | 52.05 | 69.75 |

**The maps against their references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this map gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| 8x64 vs FourOverSix | +0.46* (691/591, p=0.0057) | +1.74 (103/80, p=0.1) | -0.0221 ± 0.0018* | -0.0198 ± 0.0041* |
| 8x64 vs its 256-window version | -0.20 (549/592, p=0.21) | -0.23 (78/81, p=0.87) | -0.0062 ± 0.0015* | -0.0083 ± 0.0022* |
| 8x64 vs the 16x64 map (this repository) | +0.10 (574/552, p=0.53) | +0.83 (89/78, p=0.44) | -0.0005 ± 0.0015 | +0.0005 ± 0.0011 |
| 16x64 vs FourOverSix | +0.36* (679/601, p=0.031) | +0.91 (100/88, p=0.42) | -0.0216 ± 0.0019* | -0.0203 ± 0.0038* |
| 16x64 vs its 256-window version | -0.23 (565/615, p=0.15) | +0.30 (84/80, p=0.81) | -0.0077 ± 0.0016* | -0.0092 ± 0.0022* |
| 256x64 vs FourOverSix | -0.12 (671/698, p=0.48) | +0.99 (111/98, p=0.41) | -0.0138 ± 0.0019* | -0.0129 ± 0.0036* |
| 256x64 vs its 256-window version | -0.43* (608/702, p=0.01) | -1.29 (84/101, p=0.24) | -0.0050 ± 0.0016* | -0.0085 ± 0.0031* |
| 256x64 vs the 16x64 map (this repository) | -0.48* (588/693, p=0.0036) | +0.08 (82/81, p=1) | +0.0077 ± 0.0016* | +0.0074 ± 0.0013* |

## How to use

The FlipQuant code will be released with the paper. With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels (any of the three maps):

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model llama3.1-8b-instruct --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model llama3.1-8b-instruct --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the PPL above.

## License

These maps are released under the same license as the base model, the Llama 3.1 Community License.

The Llama 3.1 Community License (section 1.b) applies to derivative works: provide a copy of the agreement, prominently display “Built with Llama”, include “Llama” at the beginning of the name of an AI model created or improved with Llama materials or their outputs, and keep the notice “Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.” in a NOTICE file. These maps are trained against Llama-3.1-8B-Instruct's outputs (KL to its BF16 logits).

- `LICENSE`: the Llama 3.1 Community License Agreement, the base repository's own license file (meta-llama/Llama-3.1-8B-Instruct at revision `0e9e39f249a16976918f6564b8830bc894c89659`).
- `USE_POLICY.md`: the Llama 3.1 Acceptable Use Policy, which the agreement incorporates; the base repository's own `USE_POLICY.md`.
- `NOTICE`: “Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 7,627 | `64e1b2889b7892e6bbe7a7ed5bfe6ff793c61f9d584345f8f41cf9f5cb30a369` |
| `NOTICE` | 116 | `9b4ce1898c6fdbefd6d0ca4d40a55cb994ed86ae3fd56166866b988bee7a6464` |
| `USE_POLICY.md` | 4,691 | `a568f2ebc73cec3fd74ba2afd992d4e945a8c7a9d851f9b66163aac834b7b859` |
| `flipquant_16x64.pt` | 6,890,735 | `c973cb98c23ba211534bec29fcd3ae238447aab9fa911aa39035f87c8b6a654d` |
| `flipquant_256x64.pt` | 501,077 | `8c2fb731fdf5e3fc8e5a5b2ed715efa81eb37a8b26c6c0a5782852f17765c421` |
| `flipquant_8x64.pt` | 13,706,441 | `989ef5ec3d992101c67770b4bc2726c28f966e1faeb65068a29b70145e0fc37e` |
| `ppl_summary.json` | 16,850 | `aafcd1b031fe06072cdaea68ad6bd5e2f3e22481e58ebe4fcbf4d27e8d82c2a7` |

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.
- `USE_POLICY.md`: see License.
- `NOTICE`: see License.

**Provenance of the map files.** They are the calibration outputs with only their `meta` changed: the run directory and calibration-record paths are relative instead of local absolute paths, the data-extension code reference names the file without its repository owner, and the internal branch and commit of the reference implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the calibration output → this repository's file:

- `flipquant_8x64.pt`: `e1b994d98df7dcefa4695f23be958acb0e2695a2586570212a22eb9fed8f8cdc` → `989ef5ec3d992101c67770b4bc2726c28f966e1faeb65068a29b70145e0fc37e`
- `flipquant_16x64.pt`: `aa6c013f515d8e3792f143f394b52c8e277e9eafdfc99d0bd2b2e14b00d0bd70` → `c973cb98c23ba211534bec29fcd3ae238447aab9fa911aa39035f87c8b6a654d`
- `flipquant_256x64.pt`: `04832fd1a9de0f6248120ece0f604657a1722e2e5478fe725bba7bf70f29d955` → `8c2fb731fdf5e3fc8e5a5b2ed715efa81eb37a8b26c6c0a5782852f17765c421`
