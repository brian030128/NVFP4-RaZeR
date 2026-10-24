---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
tags:
- quantization
- fp4
- nvfp4
- gptq
---

# GPTQ NVFP4 weights for Llama-3.1-8B-Instruct, calibrated with BF16 activations on 1,024 windows

**Built with Llama.**

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. The agreement is in `LICENSE`, the Acceptable Use Policy it incorporates in `USE_POLICY.md`, and the notice in `NOTICE`.

These are **GPTQ‡** weights as in the FlipQuant paper's main perplexity table (GPTQ-quantized NVFP4 weights, calibrated with BF16 activations, evaluated with FourOverSix per-token activations), calibrated on **1,024 windows** instead of the main table's 256 (`edgeai-lab/Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4`). They were made for the FlipQuant paper's calibration-size study.

## What this is

`codes_nvfp4_act-fo6.pt` holds the NVFP4 codes of every quantized linear layer of [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) at revision `0e9e39f249a16976918f6564b8830bc894c89659` (224 layers: all text-model linear layers except the LM head). NVFP4 stores each weight as a 4-bit E2M1 code, with one E4M3 scale per 16 consecutive elements of a row and one FP32 scale per tensor.

**It is not a full checkpoint**: only the quantized linear layers' weights are here. Everything else (embeddings, norms, the LM head) comes from the BF16 base model at the pinned revision.

## How it was made

GPTQ (FlipQuant's `flipquant.gptq`), layer by layer in model order, on NVFP4's fixed round-to-nearest grid: each 16-element block keeps its round-to-nearest E4M3 scale and each tensor its FP32 scale, and GPTQ chooses the codes. The settings are the main table's; only the windows differ:

- Calibration data: 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 231 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `e4b509c5b0e7f90ccaf09296f56518923b4ed3b185935014d032946c8a689172`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item. sha256 of the windows' token ids, in order: `6bce0119fbc5feca2ce74361bf60fd699307a4310beaa0f0ad59a3ade14892e6`.
- BF16-activation calibration: during calibration every activation quantizer is off, so the layers already quantized run with quantized weights and BF16 activations, as in weight-only GPTQ, and the Hessian comes from the BF16 layer inputs. The codes therefore do not depend on the activation format they are evaluated with.
- Lazy-update block 128 columns, dampening 0.01 × mean(diag H), no act-order, the ModelOpt 0.47 handling of dead columns.
- Calibration batch 8; GPTQ took 13.2 min on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (the 256-window run: 10.1 min), peak allocated GPU memory 25.4 GiB.
- Code: FlipQuant revision `8dbe2e429fd656814868d1a4833c56aeac6d3732`, whose GPTQ code is the main table's, unchanged. On it, Llama-3.2-1B-Instruct's 256-window run reproduces the main table's codes and the published 256-window codes file bit for bit.
- Code digest (FlipQuant's `codes_sha256` over every module's packed codes, scales and formats): `05b94c8ee1afafd139c785ad770e7c4016a30be10b5977e92c4c3b1ef2a8bec4`.

The FlipQuant code (GPTQ, the native kernels and the evaluation) will be released with the paper.

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
| GPTQ‡, 256 windows (`Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4`) | 71.85 | 81.80 ± 2.12 | 7.829 | 11.366 |
| GPTQ‡, 1,024 windows (**this repository**) | 71.65 | 81.43 ± 2.14 | 7.926 | 11.433 |

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 81.34 | 79.58 | 73.80 | 79.71 | 55.72 | 72.13 |
| FourOverSix (reference) | 80.14 | 78.04 | 71.74 | 77.02 | 52.05 | 71.45 |
| GPTQ‡, 256 windows (`Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4`) | 79.71 | 77.81 | 72.06 | 77.78 | 52.73 | 71.03 |
| GPTQ‡, 1,024 windows (**this repository**) | 79.05 | 77.91 | 71.27 | 76.60 | 53.84 | 71.22 |

**This repository against its references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this row gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| vs FourOverSix | -0.18 (787/827, p=0.33) | +1.97 (123/97, p=0.092) | +0.0136 ± 0.0125* | +0.0146 ± 0.0209 |
| vs its 256-window version (Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4) | -0.08 (707/724, p=0.67) | -0.38 (88/93, p=0.77) | +0.0123 ± 0.0086* | +0.0058 ± 0.0062 |

## How to use

With FlipQuant (to be released with the paper), on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), the command that produced the PPL above:

```bash
python -m evaluation.ppl --model llama3.1-8b-instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659 --mode native \
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

In the file: `q` is uint8 `[4096, 7168]`-shaped (two 4-bit codes per byte, element 2j in the low nibble), `s` uint8 (the raw E4M3 bytes of the block scales), `g` the FP32 tensor scale, `fmt` 1 (every block E2M1).

## License

These weights are derived from the base model's weights and are released under the base model's license, the Llama 3.1 Community License.

The Llama 3.1 Community License (section 1.b) applies to derivative works: provide a copy of the agreement, prominently display “Built with Llama”, include “Llama” at the beginning of the name of an AI model created or improved with Llama materials or their outputs, and keep the notice “Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.” in a NOTICE file.

- `LICENSE`: the Llama 3.1 Community License Agreement, the base repository's own license file (meta-llama/Llama-3.1-8B-Instruct at revision `0e9e39f249a16976918f6564b8830bc894c89659`).
- `USE_POLICY.md`: the Llama 3.1 Acceptable Use Policy, which the agreement incorporates; the base repository's own `USE_POLICY.md`.
- `NOTICE`: “Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 7,627 | `64e1b2889b7892e6bbe7a7ed5bfe6ff793c61f9d584345f8f41cf9f5cb30a369` |
| `NOTICE` | 116 | `9b4ce1898c6fdbefd6d0ca4d40a55cb994ed86ae3fd56166866b988bee7a6464` |
| `USE_POLICY.md` | 4,691 | `a568f2ebc73cec3fd74ba2afd992d4e945a8c7a9d851f9b66163aac834b7b859` |
| `codes_nvfp4_act-fo6.pt` | 3,926,070,201 | `692d721acbd50a37383cde92764f934f404c5794b3b0fbab732b428ba0bb2ffa` |
| `ppl_summary.json` | 7,183 | `9c2b1f03fbe7ae8b570b2be5401a2320326ef54bc71476121003325f55d773e8` |

- `codes_nvfp4_act-fo6.pt`: the GPTQ codes (`flipquant-gptq/1`: `{format, key, codes}`; above), the calibration output, unchanged.
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.
- `USE_POLICY.md`: see License.
- `NOTICE`: see License.
