---
license: apache-2.0
base_model: Qwen/Qwen3-8B
tags:
- quantization
- fp4
- nvfp4
- mixfp4
- flipquant
---

# Qwen3-8B FlipQuant MixFP4 maps, 1,024 calibration windows

## What this is

Format maps for [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) at revision `b968826d9c46dd6066d109eabc6255188de91218`, calibrated with FlipQuant (TM-OPT+TC) on **1,024 calibration windows**, four times the release's 256 (`edgeai-lab/Qwen3-8B-FlipQuant`). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 492,799 of 13,565,952 (3.63 %) | `262f1569777a66d743fa4fced48706206a5e10c08c14ef4427199d2924ecd093` |
| `flipquant_16x64.pt` | 16 × 64 | 300,918 of 6,782,976 (4.44 %) | `c1a7fbf7fe27678a09fc6c5b62e161504e71a133cbee460febe3570ef2406903` |
| `flipquant_256x64.pt` | 256 × 64 | 42,226 of 423,936 (9.96 %) | `ec48d0ecf9b16c91c3da4b9ad098d874e3f436bce02e38ced09fc7d6a703facc` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

With more windows every map elects more E0M3 tiles. The E0M3 shares of the release's 256-window maps are 8x64 0.49 %, 16x64 0.55 %, 256x64 1.27 %.

**The 256x64 map** has the highest E0M3 share of the three (9.96 %; 16x64: 4.44 %). Against the 16x64 map its PPL is higher, beyond 2 SE on both corpora (ΔNLL WikiText-2 +0.0048 ± 0.0019*, C4 +0.0059 ± 0.0013*); GSM8K -1.59* (41/62, p=0.048), zero-shot +0.14 (680/650, p=0.43). Pooled over the study's eight models, 256x64 at 1,024 windows is the weakest of the three tiles (GSM8K −1.39*, zero-shot −0.24* against 16x64), and more windows did not improve it over its 256-window version (GSM8K −0.26 (n.s.)).

The FlipQuant code will be released with the paper.

## Calibration

`python -m calibration.train_map --model qwen3-8b --unit <unit> --epochs 5 --teacher-topk 1000 --fit-windows 1024`, the same settings and the same 1,024 windows for the three tiles (FlipQuant revisions: 8x64 `394e6838825c23f234fe0a3ef4c252b0b6866be1`, 16x64 `898d00e102cb5eb0f9ce7614d8d0db8a7cf1a6f7`, 256x64 `394e6838825c23f234fe0a3ef4c252b0b6866be1`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs: **640 steps** for each tile (the release: 160). Deterministic.
- Fit set: 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 0 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `bd5ef53d52cfa59badd26c93f4c70ed02fb6a3668ab17c568219195f0e6eff77`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item.
- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| unit | setup | training | train_map end to end | GPU peak allocated | GPU peak reserved | trainer peak RSS | teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x64 | 2.0 min | 13.9 min | 16.1 min | 49.2 GiB | 49.8 GiB | 16.2 GiB | 2.93 GiB (cuda:0) | 50.6 GiB |
| 16x64 | 2.1 min | 13.9 min | 16.2 min | 49.0 GiB | 49.7 GiB | 16.2 GiB | 2.93 GiB (cuda:0) | 50.4 GiB |
| 256x64 | 2.1 min | 13.8 min | 16.1 min | 48.9 GiB | 49.5 GiB | 16.2 GiB | 2.93 GiB (cuda:0) | 50.3 GiB |

## Results

From the FlipQuant paper's calibration-size study. The numbers are also in `ppl_summary.json`.

Native sm_120 kernels on one NVIDIA RTX PRO 6000 Blackwell. Every quantized row uses per-token FourOverSix activations, and every row is evaluated on the same data:

- PPL: WikiText-2 test (141 windows of 2,048 tokens) and C4 validation (256 windows of 2,048 tokens), FlipQuant's `--paper-convention`;
- zero-shot: PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge and LAMBADA (lm-eval 0.4.11, full evaluation splits; acc_norm for PIQA, HellaSwag and ARC, acc for WinoGrande and LAMBADA). Avg. is their mean;
- GSM8K: lm-eval's `gsm8k_llama`: 8-shot chain of thought with the chat template, greedy decoding, at most 1,024 new tokens, strict-match accuracy ± 2 SE, with Qwen3's thinking mode off (`enable_thinking=False`).

| method | zero-shot Avg. (%) | GSM8K strict (%) | WikiText-2 PPL | C4 PPL |
|---|---:|---:|---:|---:|
| BF16 | 70.46 | 91.36 ± 1.55 | 9.725 | 13.300 |
| FourOverSix (reference) | 68.93 | 88.25 ± 1.77 | 10.007 | 13.757 |
| FlipQuant 8x64, 256 windows (`Qwen3-8B-FlipQuant`) | 70.03 | 89.46 ± 1.69 | 9.608 | 13.528 |
| FlipQuant 8x64, 1,024 windows (**this repository**) | 70.36 | 89.23 ± 1.71 | 9.613 | 13.543 |
| FlipQuant 16x64, 256 windows (`Qwen3-8B-FlipQuant`) | 69.73 | 89.31 ± 1.70 | 9.590 | 13.555 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 70.18 | 90.22 ± 1.64 | 9.584 | 13.487 |
| FlipQuant 256x64, 256 windows (`Qwen3-8B-FlipQuant`) | 69.62 | 89.76 ± 1.67 | 9.678 | 13.565 |
| FlipQuant 256x64, 1,024 windows (**this repository**) | 70.40 | 88.63 ± 1.75 | 9.631 | 13.567 |

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 77.80 | 74.93 | 68.03 | 80.85 | 56.91 | 64.25 |
| FourOverSix (reference) | 75.95 | 73.18 | 67.17 | 79.71 | 54.86 | 62.72 |
| FlipQuant 8x64, 256 windows (`Qwen3-8B-FlipQuant`) | 77.64 | 74.12 | 68.82 | 80.56 | 55.80 | 63.24 |
| FlipQuant 8x64, 1,024 windows (**this repository**) | 77.42 | 74.19 | 69.69 | 81.31 | 56.57 | 62.97 |
| FlipQuant 16x64, 256 windows (`Qwen3-8B-FlipQuant`) | 76.71 | 74.27 | 68.59 | 79.97 | 55.46 | 63.40 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 78.18 | 74.23 | 68.11 | 81.40 | 56.31 | 62.84 |
| FlipQuant 256x64, 256 windows (`Qwen3-8B-FlipQuant`) | 76.50 | 73.78 | 68.98 | 79.84 | 55.03 | 63.57 |
| FlipQuant 256x64, 1,024 windows (**this repository**) | 78.29 | 74.25 | 69.77 | 80.26 | 56.40 | 63.44 |

**The maps against their references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this map gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| 8x64 vs FourOverSix | +1.06* (843/612, p=1.5e-09) | +0.99 (60/47, p=0.25) | -0.0402 ± 0.0030* | -0.0157 ± 0.0020* |
| 8x64 vs its 256-window version | +0.12 (670/643, p=0.47) | -0.23 (37/40, p=0.82) | +0.0005 ± 0.0019 | +0.0011 ± 0.0012 |
| 8x64 vs the 16x64 map (this repository) | +0.05 (636/626, p=0.8) | -0.99 (35/48, p=0.19) | +0.0030 ± 0.0018* | +0.0041 ± 0.0011* |
| 16x64 vs FourOverSix | +1.01* (841/620, p=8.1e-09) | +1.97* (64/38, p=0.013) | -0.0432 ± 0.0032* | -0.0198 ± 0.0018* |
| 16x64 vs its 256-window version | +0.15 (675/643, p=0.39) | +0.91 (50/38, p=0.24) | -0.0006 ± 0.0020 | -0.0050 ± 0.0011* |
| 256x64 vs FourOverSix | +1.15* (881/630, p=1.1e-10) | +0.38 (63/58, p=0.72) | -0.0383 ± 0.0032* | -0.0140 ± 0.0019* |
| 256x64 vs its 256-window version | +0.50* (733/624, p=0.0034) | -1.14 (44/59, p=0.17) | -0.0049 ± 0.0021* | +0.0001 ± 0.0012 |
| 256x64 vs the 16x64 map (this repository) | +0.14 (680/650, p=0.43) | -1.59* (41/62, p=0.048) | +0.0048 ± 0.0019* | +0.0059 ± 0.0013* |

## How to use

The FlipQuant code will be released with the paper. With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels (any of the three maps):

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model qwen3-8b --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model qwen3-8b --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the PPL above.

## License

These maps are released under the same license as the base model, the Apache License 2.0.

`LICENSE` is a copy of the base model's license file (Qwen/Qwen3-8B).

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 11,343 | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |
| `flipquant_16x64.pt` | 6,866,839 | `c1a7fbf7fe27678a09fc6c5b62e161504e71a133cbee460febe3570ef2406903` |
| `flipquant_256x64.pt` | 507,865 | `ec48d0ecf9b16c91c3da4b9ad098d874e3f436bce02e38ced09fc7d6a703facc` |
| `flipquant_8x64.pt` | 13,649,749 | `262f1569777a66d743fa4fced48706206a5e10c08c14ef4427199d2924ecd093` |
| `ppl_summary.json` | 16,819 | `9bff0a074f4c45e1dccf8ca216db2a6af5e6c34d3cee78972465871913d62b02` |

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.

**Provenance of the map files.** They are the calibration outputs with only their `meta` changed: the run directory and calibration-record paths are relative instead of local absolute paths, the data-extension code reference names the file without its repository owner, and the internal branch and commit of the reference implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the calibration output → this repository's file:

- `flipquant_8x64.pt`: `1d63ef080428e6fc25f3449e7e66beac47c0ccf9d1f9b93d36fb76899da5b4be` → `262f1569777a66d743fa4fced48706206a5e10c08c14ef4427199d2924ecd093`
- `flipquant_16x64.pt`: `dc8681ff793dd6b1494719f93d4a4b7808f543effcaff1ad11255f40699918a3` → `c1a7fbf7fe27678a09fc6c5b62e161504e71a133cbee460febe3570ef2406903`
- `flipquant_256x64.pt`: `43bfeff0632cd2112c8c34636c84c0dbddbfddc78b65fb21a7e2ac9275023ac2` → `ec48d0ecf9b16c91c3da4b9ad098d874e3f436bce02e38ced09fc7d6a703facc`
