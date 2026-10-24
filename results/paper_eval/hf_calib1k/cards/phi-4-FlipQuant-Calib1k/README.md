---
license: mit
base_model: microsoft/phi-4
tags:
- quantization
- fp4
- nvfp4
- mixfp4
- flipquant
---

# phi-4 FlipQuant MixFP4 maps, 1,024 calibration windows

## What this is

Format maps for [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) at revision `2db69c1c3e91a05d2c64a3185acfbaf36f744e25`, calibrated with FlipQuant (TM-OPT+TC) on **1,024 calibration windows**, four times the release's 256 (`edgeai-lab/phi-4-FlipQuant`). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 758,874 of 26,624,000 (2.85 %) | `657e80d319daefda2abb3360bd7aca979111ea6866e9ef83eca13bdf521517e1` |
| `flipquant_16x64.pt` | 16 × 64 | 491,665 of 13,312,000 (3.69 %) | `265c1cb113b6b6cf5ddd186a906a896fb0e7157086121efd26515ab458507677` |
| `flipquant_256x64.pt` | 256 × 64 | 85,876 of 832,000 (10.32 %) | `d292e9e7f6601d94ceaa9da0dfe37dfa54ff8ac6499324ee7c4b10362e9640cb` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

With more windows every map elects more E0M3 tiles. The E0M3 shares of the release's 256-window maps are 8x64 0.49 %, 16x64 0.60 %, 256x64 1.33 %.

**The 256x64 map** has the highest E0M3 share of the three (10.32 %; 16x64: 3.69 %). Against the 16x64 map its PPL is higher, beyond 2 SE on both corpora (ΔNLL WikiText-2 +0.0028 ± 0.0013*, C4 +0.0012 ± 0.0008*); GSM8K -0.38 (14/19, p=0.49), zero-shot -0.11 (440/463, p=0.46). Pooled over the study's eight models, 256x64 at 1,024 windows is the weakest of the three tiles (GSM8K −1.39*, zero-shot −0.24* against 16x64), and more windows did not improve it over its 256-window version (GSM8K −0.26 (n.s.)).

The FlipQuant code will be released with the paper.

## Calibration

`python -m calibration.train_map --model phi4-14b --unit <unit> --epochs 5 --teacher-topk 1000 --fit-windows 1024`, the same settings and the same 1,024 windows for the three tiles (FlipQuant revisions: 8x64 `394e6838825c23f234fe0a3ef4c252b0b6866be1`, 16x64 `f912a30d9423a9813cd45dbb6998cc1dc816fd43`, 256x64 `394e6838825c23f234fe0a3ef4c252b0b6866be1`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs: **640 steps** for each tile (the release: 160). Deterministic.
- Fit set: 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 0 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `a78f561bb85616860e72d42c4c86a07b41a7ead127f52a03caf5f406e06a4d61`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item.
- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| unit | setup | training | train_map end to end | GPU peak allocated | GPU peak reserved | trainer peak RSS | teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x64 | 3.0 min | 24.7 min | 27.9 min | 62.4 GiB | 65.4 GiB | 28.3 GiB | 2.93 GiB (cuda:0) | 66.1 GiB |
| 16x64 | 3.0 min | 25.0 min | 28.1 min | 62.1 GiB | 65.3 GiB | 28.3 GiB | 2.93 GiB (cuda:0) | 66.0 GiB |
| 256x64 | 2.9 min | 24.7 min | 27.8 min | 61.9 GiB | 65.1 GiB | 28.3 GiB | 2.93 GiB (cuda:0) | 65.9 GiB |

## Results

From the FlipQuant paper's calibration-size study. The numbers are also in `ppl_summary.json`.

Native sm_120 kernels on one NVIDIA RTX PRO 6000 Blackwell. Every quantized row uses per-token FourOverSix activations, and every row is evaluated on the same data:

- PPL: WikiText-2 test (141 windows of 2,048 tokens) and C4 validation (256 windows of 2,048 tokens), FlipQuant's `--paper-convention`;
- zero-shot: PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge and LAMBADA (lm-eval 0.4.11, full evaluation splits; acc_norm for PIQA, HellaSwag and ARC, acc for WinoGrande and LAMBADA). Avg. is their mean;
- GSM8K: lm-eval's `gsm8k_llama`: 8-shot chain of thought with the chat template, greedy decoding, at most 1,024 new tokens, strict-match accuracy ± 2 SE.

| method | zero-shot Avg. (%) | GSM8K strict (%) | WikiText-2 PPL | C4 PPL |
|---|---:|---:|---:|---:|
| BF16 | 73.53 | 93.33 ± 1.37 | 6.462 | 10.310 |
| FourOverSix (reference) | 72.39 | 91.74 ± 1.52 | 6.662 | 10.544 |
| FlipQuant 8x64, 256 windows (`phi-4-FlipQuant`) | 72.59 | 92.27 ± 1.47 | 6.634 | 10.519 |
| FlipQuant 8x64, 1,024 windows (**this repository**) | 72.43 | 92.34 ± 1.46 | 6.608 | 10.492 |
| FlipQuant 16x64, 256 windows (`phi-4-FlipQuant`) | 72.58 | 92.12 ± 1.48 | 6.633 | 10.516 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 72.78 | 93.18 ± 1.39 | 6.618 | 10.503 |
| FlipQuant 256x64, 256 windows (`phi-4-FlipQuant`) | 72.55 | 92.42 ± 1.46 | 6.638 | 10.518 |
| FlipQuant 256x64, 1,024 windows (**this repository**) | 72.61 | 92.80 ± 1.42 | 6.636 | 10.515 |

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 81.12 | 81.95 | 76.95 | 72.77 | 56.14 | 72.27 |
| FourOverSix (reference) | 80.69 | 81.11 | 74.27 | 71.89 | 54.86 | 71.55 |
| FlipQuant 8x64, 256 windows (`phi-4-FlipQuant`) | 80.90 | 81.12 | 74.51 | 71.93 | 55.38 | 71.73 |
| FlipQuant 8x64, 1,024 windows (**this repository**) | 80.41 | 80.99 | 74.11 | 72.05 | 55.29 | 71.74 |
| FlipQuant 16x64, 256 windows (`phi-4-FlipQuant`) | 81.01 | 81.35 | 74.66 | 72.01 | 54.69 | 71.76 |
| FlipQuant 16x64, 1,024 windows (**this repository**) | 81.39 | 81.18 | 74.43 | 72.69 | 55.20 | 71.76 |
| FlipQuant 256x64, 256 windows (`phi-4-FlipQuant`) | 81.01 | 81.19 | 74.82 | 71.51 | 54.69 | 72.07 |
| FlipQuant 256x64, 1,024 windows (**this repository**) | 80.90 | 81.11 | 74.74 | 71.93 | 55.03 | 71.94 |

**The maps against their references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this map gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| 8x64 vs FourOverSix | +0.00 (458/458, p=1) | +0.61 (16/8, p=0.15) | -0.0081 ± 0.0015* | -0.0049 ± 0.0010* |
| 8x64 vs its 256-window version | -0.11 (446/470, p=0.45) | +0.08 (15/14, p=1) | -0.0040 ± 0.0012* | -0.0025 ± 0.0008* |
| 8x64 vs the 16x64 map (this repository) | -0.26 (400/456, p=0.06) | -0.83 (10/21, p=0.071) | -0.0015 ± 0.0013* | -0.0010 ± 0.0007* |
| 16x64 vs FourOverSix | +0.26 (477/421, p=0.066) | +1.44* (27/8, p=0.0019) | -0.0066 ± 0.0014* | -0.0039 ± 0.0009* |
| 16x64 vs its 256-window version | +0.04 (446/437, p=0.79) | +1.06* (25/11, p=0.029) | -0.0022 ± 0.0013* | -0.0012 ± 0.0007* |
| 256x64 vs FourOverSix | +0.15 (504/471, p=0.31) | +1.06* (22/8, p=0.016) | -0.0038 ± 0.0014* | -0.0027 ± 0.0009* |
| 256x64 vs its 256-window version | -0.02 (483/487, p=0.92) | +0.38 (23/18, p=0.53) | -0.0003 ± 0.0013 | -0.0003 ± 0.0008 |
| 256x64 vs the 16x64 map (this repository) | -0.11 (440/463, p=0.46) | -0.38 (14/19, p=0.49) | +0.0028 ± 0.0013* | +0.0012 ± 0.0008* |

## How to use

The FlipQuant code will be released with the paper. With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels (any of the three maps):

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model phi4-14b --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model phi4-14b --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the PPL above.

## License

These maps are released under the same license as the base model, the MIT License.

`LICENSE` is a copy of the base repository's license file (microsoft/phi-4).

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 1,084 | `fa8235e5b48faca34e3ca98cf4f694ef08bd216d28b58071a1f85b1d50cb814d` |
| `flipquant_16x64.pt` | 13,367,279 | `265c1cb113b6b6cf5ddd186a906a896fb0e7157086121efd26515ab458507677` |
| `flipquant_256x64.pt` | 884,565 | `d292e9e7f6601d94ceaa9da0dfe37dfa54ff8ac6499324ee7c4b10362e9640cb` |
| `flipquant_8x64.pt` | 26,679,177 | `657e80d319daefda2abb3360bd7aca979111ea6866e9ef83eca13bdf521517e1` |
| `ppl_summary.json` | 16,764 | `53a93cdd657151c014f83a2a3e10a7c4507361bcbbaba78a4d7166515781fbff` |

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.

**Provenance of the map files.** They are the calibration outputs with only their `meta` changed: the run directory and calibration-record paths are relative instead of local absolute paths, the data-extension code reference names the file without its repository owner, and the internal branch and commit of the reference implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the calibration output → this repository's file:

- `flipquant_8x64.pt`: `4cc61534f8756bdf1dd8fecd4a5936b05508924e1c882dfed45b84e1ff5e3e07` → `657e80d319daefda2abb3360bd7aca979111ea6866e9ef83eca13bdf521517e1`
- `flipquant_16x64.pt`: `370ddc6126d882c0eee3a38e9897b1d4349ae04342c69464fe33c53af3257c6f` → `265c1cb113b6b6cf5ddd186a906a896fb0e7157086121efd26515ab458507677`
- `flipquant_256x64.pt`: `87515afacca9cb291be4ae83fbb8e6d323570a768457f253d5109d1065923353` → `d292e9e7f6601d94ceaa9da0dfe37dfa54ff8ac6499324ee7c4b10362e9640cb`
