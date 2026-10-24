---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/NVIDIA-Nemotron-Nano-9B-v2
tags:
- quantization
- fp4
- nvfp4
- mixfp4
- flipquant
---

# NVIDIA-Nemotron-Nano-9B-v2 FlipQuant MixFP4 maps, 1,024 calibration windows

## What this is

Format maps for [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) at revision `6533e8de2c68e4536bf7c411d7a3ce5734111476`, calibrated with FlipQuant (TM-OPT+TC) on **1,024 calibration windows**, four times the release's 256 (`edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-FlipQuant`). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 645,487 of 15,061,760 (4.29 %) | `bbfba43713237770a032260a64cbf1e3f47031adc05b41b641c56b5ed39c341c` |
| `flipquant_16x64.pt` | 16 × 64 | 420,667 of 7,530,880 (5.59 %) | `7469f895ec4d7fe37024934d5ce648dce454194c8b34150eef0b198c1b70c44e` |
| `flipquant_256x64.pt` | 256 × 64 | 76,421 of 478,320 (15.98 %) | `8738a1c478ee39954e5740e32f82c61335ad99afb4c2bfcfaed4f635b74c5540` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

With more windows every map elects more E0M3 tiles. The E0M3 shares of the release's 256-window maps are 8x64 0.87 %, 16x64 1.06 %, 256x64 2.65 %.

**The 256x64 map** has the highest E0M3 share of the three (15.98 %; 16x64: 5.59 %). Against the 16x64 map its PPL is higher, beyond 2 SE on both corpora (ΔNLL WikiText-2 +0.0022 ± 0.0012*, C4 +0.0014 ± 0.0008*); GSM8K -2.12* (56/84, p=0.022), zero-shot +0.09 (488/469, p=0.56). Pooled over the study's eight models, 256x64 at 1,024 windows is the weakest of the three tiles (GSM8K −1.39*, zero-shot −0.24* against 16x64), and more windows did not improve it over its 256-window version (GSM8K −0.26 (n.s.)).

The FlipQuant code will be released with the paper.

## Calibration

`python -m calibration.train_map --model nemotron-nano-9b-v2 --unit <unit> --epochs 5 --teacher-topk 1000 --fit-windows 1024`, the same settings and the same 1,024 windows for the three tiles (FlipQuant revisions: 8x64 `394e6838825c23f234fe0a3ef4c252b0b6866be1`, 16x64 `898d00e102cb5eb0f9ce7614d8d0db8a7cf1a6f7`, 256x64 `394e6838825c23f234fe0a3ef4c252b0b6866be1`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step (2 × 4 accumulated), 5 epochs: **640 steps** for each tile (the release: 160). Deterministic.
- Fit set: 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 0 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `12038dae8cce1844a2633f5061ae1d0f93811aec43845bb941ca7023c8d39d76`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item.
- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| unit | setup | training | train_map end to end | GPU peak allocated | GPU peak reserved | trainer peak RSS | teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x64 | 11.2 min | 158.2 min | 169.6 min | 56.1 GiB | 57.5 GiB | 17.5 GiB | 2.93 GiB (cuda:0) | 58.2 GiB |
| 16x64 | 11.2 min | 159.0 min | 170.4 min | 55.9 GiB | 57.4 GiB | 17.5 GiB | 2.93 GiB (cuda:0) | 58.1 GiB |
| 256x64 | 11.2 min | 159.1 min | 170.5 min | 55.9 GiB | 57.3 GiB | 17.5 GiB | 2.93 GiB (cuda:0) | 58.0 GiB |

## Results

From the FlipQuant paper's calibration-size study. The numbers are also in `ppl_summary.json`.

Native sm_120 kernels on one NVIDIA RTX PRO 6000 Blackwell. Every quantized row uses per-token FourOverSix activations, and every row is evaluated on the same data:

- PPL: WikiText-2 test (141 windows of 2,048 tokens) and C4 validation (256 windows of 2,048 tokens), FlipQuant's `--paper-convention`;
- zero-shot: PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge and LAMBADA (lm-eval 0.4.11, full evaluation splits; acc_norm for PIQA, HellaSwag and ARC, acc for WinoGrande and LAMBADA). Avg. is their mean;
- GSM8K: lm-eval's `gsm8k_llama`: 8-shot chain of thought with the chat template, greedy decoding, at most 1,024 new tokens, strict-match accuracy ± 2 SE.

| method | zero-shot Avg. (%) | GSM8K strict (%) | WikiText-2 PPL | C4 PPL |
|---|---:|---:|---:|---:|
| BF16 | 72.35 | 87.41 ± 1.83 | 8.086 | 11.153 |
| FourOverSix (reference) | 70.89 | 88.93 ± 1.73 | 8.426 | 11.503 |
| FlipQuant 8x64, 256 windows (`NVIDIA-Nemotron-Nano-9B-v2-FlipQuant`) | 71.06 | 87.11 ± 1.85 | 8.354 | 11.444 |
| FlipQuant 8x64, 1,024 windows (**this repository**) | 71.29 | 88.02 ± 1.79 | 8.318 | 11.438 |
| FlipQuant 16x64, 256 windows (`NVIDIA-Nemotron-Nano-9B-v2-FlipQuant`) | 70.74 | 89.31 ± 1.70 | 8.353 | 11.445 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 71.09 | 88.63 ± 1.75 | 8.327 | 11.440 |
| FlipQuant 256x64, 256 windows (`NVIDIA-Nemotron-Nano-9B-v2-FlipQuant`) | 70.75 | 88.93 ± 1.73 | 8.380 | 11.468 |
| FlipQuant 256x64, 1,024 windows (**this repository**) | 71.24 | 86.50 ± 1.88 | 8.345 | 11.456 |

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 80.85 | 77.49 | 73.01 | 79.84 | 56.14 | 66.78 |
| FourOverSix (reference) | 79.98 | 76.55 | 71.51 | 79.04 | 53.92 | 64.31 |
| FlipQuant 8x64, 256 windows (`NVIDIA-Nemotron-Nano-9B-v2-FlipQuant`) | 80.47 | 76.52 | 72.77 | 78.83 | 52.99 | 64.76 |
| FlipQuant 8x64, 1,024 windows (**this repository**) | 80.36 | 76.85 | 73.24 | 78.91 | 53.58 | 64.80 |
| FlipQuant 16x64, 256 windows (`NVIDIA-Nemotron-Nano-9B-v2-FlipQuant`) | 80.79 | 76.69 | 71.74 | 78.58 | 52.90 | 63.73 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 80.52 | 76.60 | 72.77 | 78.58 | 53.33 | 64.76 |
| FlipQuant 256x64, 256 windows (`NVIDIA-Nemotron-Nano-9B-v2-FlipQuant`) | 80.36 | 76.70 | 72.53 | 78.20 | 53.41 | 63.28 |
| FlipQuant 256x64, 1,024 windows (**this repository**) | 80.36 | 76.72 | 73.80 | 78.49 | 53.33 | 64.74 |

**The maps against their references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this map gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| 8x64 vs FourOverSix | +0.35* (556/479, p=0.018) | -0.91 (54/66, p=0.32) | -0.0129 ± 0.0012* | -0.0057 ± 0.0008* |
| 8x64 vs its 256-window version | +0.22 (485/437, p=0.12) | +0.91 (68/56, p=0.32) | -0.0043 ± 0.0012* | -0.0005 ± 0.0008 |
| 8x64 vs the 16x64 map (this repository) | +0.19 (457/416, p=0.18) | -0.61 (55/63, p=0.52) | -0.0011 ± 0.0011 | -0.0001 ± 0.0007 |
| 16x64 vs FourOverSix | +0.16 (522/486, p=0.27) | -0.30 (52/56, p=0.77) | -0.0119 ± 0.0013* | -0.0056 ± 0.0008* |
| 16x64 vs its 256-window version | +0.26 (483/426, p=0.063) | -0.68 (46/55, p=0.43) | -0.0032 ± 0.0012* | -0.0005 ± 0.0008 |
| 256x64 vs FourOverSix | +0.25 (561/506, p=0.098) | -2.43* (45/77, p=0.0048) | -0.0096 ± 0.0014* | -0.0041 ± 0.0009* |
| 256x64 vs its 256-window version | +0.45* (582/483, p=0.0027) | -2.43* (52/84, p=0.0076) | -0.0041 ± 0.0012* | -0.0010 ± 0.0009* |
| 256x64 vs the 16x64 map (this repository) | +0.09 (488/469, p=0.56) | -2.12* (56/84, p=0.022) | +0.0022 ± 0.0012* | +0.0014 ± 0.0008* |

## How to use

The FlipQuant code will be released with the paper. With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels (any of the three maps):

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model nemotron-nano-9b-v2 --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model nemotron-nano-9b-v2 --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the PPL above.

## License

These maps are released under the same license as the base model, the NVIDIA Open Model License Agreement.

Governing terms: the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). `LICENSE` is the agreement (Last Modified: October 24, 2025), verbatim from the page the base model card links (the base repository has no license file; retrieved 2026-10-07 05:26 UTC); `NOTICE` is the attribution notice “Licensed by NVIDIA Corporation under the NVIDIA Open Model License” (section 3.1).

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 10,254 | `8cf5f9a673059fc727c03fae31c0d6e365bbf4ef1b7fd9b3099fefe1b49fdda1` |
| `NOTICE` | 67 | `5fc60716a9ba57f3792e71ed776b4065671c64d0c2db716698f4d646ad833eb1` |
| `flipquant_16x64.pt` | 7,571,967 | `7469f895ec4d7fe37024934d5ce648dce454194c8b34150eef0b198c1b70c44e` |
| `flipquant_256x64.pt` | 518,397 | `8738a1c478ee39954e5740e32f82c61335ad99afb4c2bfcfaed4f635b74c5540` |
| `flipquant_8x64.pt` | 15,100,673 | `bbfba43713237770a032260a64cbf1e3f47031adc05b41b641c56b5ed39c341c` |
| `ppl_summary.json` | 16,872 | `2dca80bf8f6fd78eecb8f04e77c6799ed796ffd85191dd9579d29866155ce07d` |

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.
- `NOTICE`: see License.

**Provenance of the map files.** They are the calibration outputs with only their `meta` changed: the run directory and calibration-record paths are relative instead of local absolute paths, the data-extension code reference names the file without its repository owner, and the internal branch and commit of the reference implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the calibration output → this repository's file:

- `flipquant_8x64.pt`: `e4e622ffafffd0e3a154cfaf800741abf0bfeccf2458b7605b7c8f6f87c8ebf5` → `bbfba43713237770a032260a64cbf1e3f47031adc05b41b641c56b5ed39c341c`
- `flipquant_16x64.pt`: `39d049e83e66c06456226f1fb3a5f7e408dba680e22f4e24b776748dde5d05b9` → `7469f895ec4d7fe37024934d5ce648dce454194c8b34150eef0b198c1b70c44e`
- `flipquant_256x64.pt`: `392c0f6659f039502e3bdba93753f66dced5af39322ad0b843997a3303cc6ab2` → `8738a1c478ee39954e5740e32f82c61335ad99afb4c2bfcfaed4f635b74c5540`
