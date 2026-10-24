---
license: mit
base_model: microsoft/phi-4
tags:
- quantization
- fp4
- nvfp4
- gptq
---

# GPTQ NVFP4 weights for phi-4, calibrated with BF16 activations on 1,024 windows

These are **GPTQ‡** weights as in the FlipQuant paper's main perplexity table (GPTQ-quantized NVFP4 weights, calibrated with BF16 activations, evaluated with FourOverSix per-token activations), calibrated on **1,024 windows** instead of the main table's 256 (`edgeai-lab/phi-4-GPTQ-BF16Calib-NVFP4`). They were made for the FlipQuant paper's calibration-size study.

## What this is

`codes_nvfp4_act-fo6.pt` holds the NVFP4 codes of every quantized linear layer of [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) at revision `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` (160 layers: all text-model linear layers except the LM head). NVFP4 stores each weight as a 4-bit E2M1 code, with one E4M3 scale per 16 consecutive elements of a row and one FP32 scale per tensor.

**It is not a full checkpoint**: only the quantized linear layers' weights are here. Everything else (embeddings, norms, the LM head) comes from the BF16 base model at the pinned revision.

## How it was made

GPTQ (FlipQuant's `flipquant.gptq`), layer by layer in model order, on NVFP4's fixed round-to-nearest grid: each 16-element block keeps its round-to-nearest E4M3 scale and each tensor its FP32 scale, and GPTQ chooses the codes. The settings are the main table's; only the windows differ:

- Calibration data: 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 0 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `a78f561bb85616860e72d42c4c86a07b41a7ead127f52a03caf5f406e06a4d61`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item. sha256 of the windows' token ids, in order: `c72bd64b8170861f13188c013471e43785d06cde7e246ca848eb76bc190b93de`.
- BF16-activation calibration: during calibration every activation quantizer is off, so the layers already quantized run with quantized weights and BF16 activations, as in weight-only GPTQ, and the Hessian comes from the BF16 layer inputs. The codes therefore do not depend on the activation format they are evaluated with.
- Lazy-update block 128 columns, dampening 0.01 × mean(diag H), no act-order, the ModelOpt 0.47 handling of dead columns.
- Calibration batch 8; GPTQ took 20.5 min on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (the 256-window run: 15.0 min), peak allocated GPU memory 42.4 GiB.
- Code: FlipQuant revision `8dbe2e429fd656814868d1a4833c56aeac6d3732`, whose GPTQ code is the main table's, unchanged. On it, Llama-3.2-1B-Instruct's 256-window run reproduces the main table's codes and the published 256-window codes file bit for bit.
- Code digest (FlipQuant's `codes_sha256` over every module's packed codes, scales and formats): `d44640a0d878250865e680a15cc9f390220c58055f858a1aab0e241e2e5f2faa`.

The FlipQuant code (GPTQ, the native kernels and the evaluation) will be released with the paper.

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
| GPTQ‡, 256 windows (`phi-4-GPTQ-BF16Calib-NVFP4`) | 72.61 | 92.19 ± 1.48 | 6.688 | 10.553 |
| GPTQ‡, 1,024 windows (**this repository**) | 72.84 | 93.25 ± 1.38 | 6.669 | 10.540 |

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 81.12 | 81.95 | 76.95 | 72.77 | 56.14 | 72.27 |
| FourOverSix (reference) | 80.69 | 81.11 | 74.27 | 71.89 | 54.86 | 71.55 |
| GPTQ‡, 256 windows (`phi-4-GPTQ-BF16Calib-NVFP4`) | 80.63 | 80.91 | 73.09 | 72.98 | 56.06 | 72.02 |
| GPTQ‡, 1,024 windows (**this repository**) | 80.74 | 81.17 | 73.95 | 72.90 | 55.55 | 72.75 |

**This repository against its references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this row gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| vs FourOverSix | +0.44* (645/548, p=0.0054) | +1.52* (33/13, p=0.0045) | +0.0010 ± 0.0018 | -0.0004 ± 0.0011 |
| vs its 256-window version (phi-4-GPTQ-BF16Calib-NVFP4) | +0.32* (566/497, p=0.037) | +1.06 (31/17, p=0.059) | -0.0029 ± 0.0016* | -0.0013 ± 0.0010* |

## How to use

With FlipQuant (to be released with the paper), on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), the command that produced the PPL above:

```bash
python -m evaluation.ppl --model phi4-14b --revision 2db69c1c3e91a05d2c64a3185acfbaf36f744e25 --mode native \
    --weight nvfp4 --act fourover6 --any-act --paper-convention \
    --ptq gptq --gptq-propagate bf16 --gptq-hessian-input bf16 --gptq-calib release \
    --gptq-calib-record <calibration record> --gptq-calib-extension <the 1,024-window record> \
    --gptq-block 128 --gptq-damp 0.01 --gptq-batch 8 --gptq-cache codes_nvfp4_act-fo6.pt
```

FlipQuant reads the file instead of running GPTQ only if the file's stored settings, including the sha256 of the calibration windows, equal the run's; it rebuilds the windows from the calibration records to check this.

With PyTorch alone, one layer's weight (this reproduces FlipQuant's own dequantization bit for bit):

```python
import torch

obj = torch.load("codes_nvfp4_act-fo6.pt", map_location="cpu", weights_only=True)
obj["key"]                                    # the GPTQ settings
c = obj["codes"]["model.layers.0.mlp.down_proj"]                        # one linear layer, by its module name
nib = torch.stack((c["q"] & 15, c["q"] >> 4), -1).flatten(-2).long()    # [out, in] 4-bit codes
e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
values = torch.cat([e2m1, -e2m1])[nib]       # code = sign << 3 | magnitude index
scales = c["s"].view(torch.float8_e4m3fn).float()                       # [out, in / 16] E4M3 block scales
w = (values.unflatten(-1, (-1, 16)) * scales.unsqueeze(-1)).flatten(-2) * c["g"]   # the weight, FP32
```

In the file: `q` is uint8 `[5120, 8960]`-shaped (two 4-bit codes per byte, element 2j in the low nibble), `s` uint8 (the raw E4M3 bytes of the block scales), `g` the FP32 tensor scale, `fmt` 1 (every block E2M1).

## License

These weights are derived from the base model's weights and are released under the base model's license, the MIT License.

`LICENSE` is a copy of the base repository's license file (microsoft/phi-4).

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 1,084 | `fa8235e5b48faca34e3ca98cf4f694ef08bd216d28b58071a1f85b1d50cb814d` |
| `codes_nvfp4_act-fo6.pt` | 7,667,859,109 | `93af3d6527f2332045c47e57ce55b48085d141e14e97bd055b90eaf43465cfd9` |
| `ppl_summary.json` | 7,108 | `c996acdaf471149e1af1ffdcc5e20b9194abcfa7703ca689f982c1f01d28ef28` |

- `codes_nvfp4_act-fo6.pt`: the GPTQ codes (`flipquant-gptq/1`: `{format, key, codes}`; above), the calibration output, unchanged.
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.
