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

# Llama-3.2-1B-Instruct FlipQuant MixFP4 maps, 1,024 calibration windows

**Built with Llama.**

Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. The agreement is in `LICENSE`, the Acceptable Use Policy it incorporates in `USE_POLICY.md`, and the notice in `NOTICE`.

## What this is

Format maps for [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) at revision `9213176726f574b556790deb65791e0c5aa438b6`, calibrated with FlipQuant (TM-OPT+TC) on **1,024 calibration windows**, four times the release's 256 (`edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant`). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 214,432 of 1,900,544 (11.28 %) | `c45684a0abb34d4832add625c17b99c72a840cfea33048c347376cc2dc100fc1` |
| `flipquant_16x64.pt` | 16 × 64 | 141,062 of 950,272 (14.84 %) | `8ba6e70703d93fdb01ef86864badd72d65caf69849fa11d2fd16080a09095d3f` |
| `flipquant_256x64.pt` | 256 × 64 | 23,572 of 59,392 (39.69 %) | `309f115a041cdcdce27d70500d3736776b2834a4b24921bce573066862d996b4` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

With more windows every map elects more E0M3 tiles. The E0M3 shares of the release's 256-window maps are 8x64 2.03 %, 16x64 2.81 %, 256x64 11.38 %.

**The 256x64 map** has the highest E0M3 share of the three (39.69 %; 16x64: 14.84 %). Against the 16x64 map its PPL is higher, beyond 2 SE on both corpora (ΔNLL WikiText-2 +0.0218 ± 0.0022*, C4 +0.0222 ± 0.0018*); GSM8K -2.27 (138/168, p=0.097), zero-shot -0.19 (1098/1139, p=0.4). Pooled over the study's eight models, 256x64 at 1,024 windows is the weakest of the three tiles (GSM8K −1.39*, zero-shot −0.24* against 16x64), and more windows did not improve it over its 256-window version (GSM8K −0.26 (n.s.)).

The FlipQuant code will be released with the paper.

## Calibration

`python -m calibration.train_map --model llama3.2-1b-instruct --unit <unit> --epochs 5 --teacher-topk 1000 --fit-windows 1024`, the same settings and the same 1,024 windows for the three tiles (FlipQuant revisions: 8x64 `394e6838825c23f234fe0a3ef4c252b0b6866be1`, 16x64 `f912a30d9423a9813cd45dbb6998cc1dc816fd43`, 256x64 `394e6838825c23f234fe0a3ef4c252b0b6866be1`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs: **640 steps** for each tile (the release: 160). Deterministic.
- Fit set: 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 0 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `d73b2bb2f3179c9a6c1e614c092f10e20d74c8cf5135c3d84cdba551716b1bb4`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item.
- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| unit | setup | training | train_map end to end | GPU peak allocated | GPU peak reserved | trainer peak RSS | teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x64 | 57.0 s | 2.6 min | 3.7 min | 16.6 GiB | 17.2 GiB | 3.2 GiB | 2.93 GiB (cuda:0) | 17.9 GiB |
| 16x64 | 56.9 s | 2.6 min | 3.7 min | 16.6 GiB | 17.0 GiB | 3.2 GiB | 2.93 GiB (cuda:0) | 17.7 GiB |
| 256x64 | 55.2 s | 2.6 min | 3.7 min | 16.6 GiB | 17.2 GiB | 3.2 GiB | 2.93 GiB (cuda:0) | 17.9 GiB |

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
| FlipQuant 8x64, 256 windows (`Llama-3.2-1B-Instruct-FlipQuant`) | 57.92 | 30.63 ± 2.54 | 14.453 | 19.679 |
| FlipQuant 8x64, 1,024 windows (**this repository**) | 57.82 | 31.99 ± 2.57 | 14.401 | 19.574 |
| FlipQuant 16x64, 256 windows (`Llama-3.2-1B-Instruct-FlipQuant`) | 57.25 | 31.61 ± 2.56 | 14.454 | 19.684 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 57.93 | 35.18 ± 2.63 | 14.387 | 19.605 |
| FlipQuant 256x64, 256 windows (`Llama-3.2-1B-Instruct-FlipQuant`) | 57.72 | 31.08 ± 2.55 | 14.657 | 20.059 |
| FlipQuant 256x64, 1,024 windows (**this repository**) | 58.12 | 32.90 ± 2.59 | 14.704 | 20.044 |

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 75.14 | 61.67 | 61.48 | 63.80 | 37.97 | 59.46 |
| FourOverSix (reference) | 71.38 | 57.74 | 58.17 | 60.19 | 36.35 | 53.64 |
| FlipQuant 8x64, 256 windows (`Llama-3.2-1B-Instruct-FlipQuant`) | 72.52 | 59.59 | 59.27 | 62.25 | 37.80 | 56.10 |
| FlipQuant 8x64, 1,024 windows (**this repository**) | 72.91 | 59.39 | 60.62 | 61.70 | 35.84 | 56.49 |
| FlipQuant 16x64, 256 windows (`Llama-3.2-1B-Instruct-FlipQuant`) | 72.85 | 59.31 | 58.09 | 61.15 | 36.35 | 55.75 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 73.39 | 59.36 | 60.46 | 62.08 | 36.26 | 56.05 |
| FlipQuant 256x64, 256 windows (`Llama-3.2-1B-Instruct-FlipQuant`) | 73.07 | 59.01 | 61.01 | 60.82 | 36.35 | 56.08 |
| FlipQuant 256x64, 1,024 windows (**this repository**) | 72.31 | 58.75 | 62.51 | 62.12 | 36.86 | 56.16 |

**The maps against their references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this map gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| 8x64 vs FourOverSix | +1.84* (1278/876, p=4.5e-18) | +2.27 (159/129, p=0.087) | -0.0616 ± 0.0030* | -0.0915 ± 0.0039* |
| 8x64 vs its 256-window version | -0.05 (917/929, p=0.8) | +1.36 (140/122, p=0.29) | -0.0036 ± 0.0021* | -0.0054 ± 0.0016* |
| 8x64 vs the 16x64 map (this repository) | +0.02 (967/962, p=0.93) | -3.18* (134/176, p=0.02) | +0.0010 ± 0.0021 | -0.0016 ± 0.0014* |
| 16x64 vs FourOverSix | +1.82* (1336/939, p=8.6e-17) | +5.46* (175/103, p=1.9e-05) | -0.0625 ± 0.0029* | -0.0899 ± 0.0039* |
| 16x64 vs its 256-window version | +0.37 (1046/965, p=0.074) | +3.56* (166/119, p=0.0063) | -0.0046 ± 0.0021* | -0.0040 ± 0.0016* |
| 256x64 vs FourOverSix | +1.63* (1363/1007, p=2.8e-13) | +3.18* (152/110, p=0.011) | -0.0407 ± 0.0029* | -0.0677 ± 0.0036* |
| 256x64 vs its 256-window version | +0.09 (1091/1071, p=0.68) | +1.82 (158/134, p=0.18) | +0.0032 ± 0.0023* | -0.0007 ± 0.0019 |
| 256x64 vs the 16x64 map (this repository) | -0.19 (1098/1139, p=0.4) | -2.27 (138/168, p=0.097) | +0.0218 ± 0.0022* | +0.0222 ± 0.0018* |

## How to use

The FlipQuant code will be released with the paper. With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels (any of the three maps):

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

These maps are released under the same license as the base model, the Llama 3.2 Community License.

The Llama 3.2 Community License (section 1.b) applies to derivative works: provide a copy of the agreement, prominently display “Built with Llama”, include “Llama” at the beginning of the name of an AI model created or improved with Llama materials or their outputs, and keep the notice “Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.” in a NOTICE file. These maps are trained against Llama-3.2-1B-Instruct's outputs (KL to its BF16 logits).

- `LICENSE`: the Llama 3.2 Community License Agreement, the base repository's own license file (meta-llama/Llama-3.2-1B-Instruct at revision `9213176726f574b556790deb65791e0c5aa438b6`).
- `USE_POLICY.md`: the Llama 3.2 Acceptable Use Policy, which the agreement incorporates; the base repository's own `USE_POLICY.md`.
- `NOTICE`: “Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 7,712 | `0b4284c1f87029e67654c7953afa16279961632cf73dcfe33374c4c2f298fa35` |
| `NOTICE` | 116 | `3bc2d1a6b360a03d172058198631b931c9aadf01650892fb0d25b93c6b8f0025` |
| `USE_POLICY.md` | 6,021 | `40e2777d7faa6beaf98400654170f414d8ab29b921b5163ad4ea0a1d39894201` |
| `flipquant_16x64.pt` | 989,903 | `8ba6e70703d93fdb01ef86864badd72d65caf69849fa11d2fd16080a09095d3f` |
| `flipquant_256x64.pt` | 99,077 | `309f115a041cdcdce27d70500d3736776b2834a4b24921bce573066862d996b4` |
| `flipquant_8x64.pt` | 1,940,121 | `c45684a0abb34d4832add625c17b99c72a840cfea33048c347376cc2dc100fc1` |
| `ppl_summary.json` | 16,951 | `440e4d11586c4641851e8d99e753abe20fd6a9cdd015a40417a2c8ee067f6f75` |

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.
- `USE_POLICY.md`: see License.
- `NOTICE`: see License.

**Provenance of the map files.** They are the calibration outputs with only their `meta` changed: the run directory and calibration-record paths are relative instead of local absolute paths, the data-extension code reference names the file without its repository owner, and the internal branch and commit of the reference implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the calibration output → this repository's file:

- `flipquant_8x64.pt`: `7e25acbff6b5068b612fc9fcb71f2bb912974570e2daa85851a5ef8938ca8107` → `c45684a0abb34d4832add625c17b99c72a840cfea33048c347376cc2dc100fc1`
- `flipquant_16x64.pt`: `d51b7b1c0b0a08b642eba681b6c6ffa21c172ab320b38bc7643aa11198e76050` → `8ba6e70703d93fdb01ef86864badd72d65caf69849fa11d2fd16080a09095d3f`
- `flipquant_256x64.pt`: `dd6c23de1c8dc42ce0b5e1e48ac5303a8e86f83e5da9dc928560314cc3b2bccc` → `309f115a041cdcdce27d70500d3736776b2834a4b24921bce573066862d996b4`
