---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/NVIDIA-Nemotron-Nano-9B-v2
tags:
- quantization
- fp4
- nvfp4
- gptq
---

# GPTQ NVFP4 weights for NVIDIA-Nemotron-Nano-9B-v2, calibrated with BF16 activations on 1,024 windows

These are **GPTQ‡** weights as in the FlipQuant paper's main perplexity table (GPTQ-quantized NVFP4 weights, calibrated with BF16 activations, evaluated with FourOverSix per-token activations), calibrated on **1,024 windows** instead of the main table's 256 (`edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4`). They were made for the FlipQuant paper's calibration-size study.

Not the same artifact as `edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-GPTQ-NVFP4`: this repository is GPTQ calibrated with BF16 activations; that one is calibrated with FourOverSix activations.

## What this is

`codes_nvfp4_act-fo6.pt` holds the NVFP4 codes of every quantized linear layer of [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) at revision `6533e8de2c68e4536bf7c411d7a3ce5734111476` (120 layers: all text-model linear layers except the LM head). NVFP4 stores each weight as a 4-bit E2M1 code, with one E4M3 scale per 16 consecutive elements of a row and one FP32 scale per tensor.

**It is not a full checkpoint**: only the quantized linear layers' weights are here. Everything else (embeddings, norms, the LM head, the Mamba / convolution parameters) comes from the BF16 base model at the pinned revision.

## How it was made

GPTQ (FlipQuant's `flipquant.gptq`), layer by layer in model order, on NVFP4's fixed round-to-nearest grid: each 16-element block keeps its round-to-nearest E4M3 scale and each tensor its FP32 scale, and GPTQ chooses the codes. The settings are the main table's; only the windows differ:

- Calibration data: 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 0 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `12038dae8cce1844a2633f5061ae1d0f93811aec43845bb941ca7023c8d39d76`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item. sha256 of the windows' token ids, in order: `eac51a94069272978887d3e0300c280ec121675511d53721869ac5e6ef8bc5f7`.
- BF16-activation calibration: during calibration every activation quantizer is off, so the layers already quantized run with quantized weights and BF16 activations, as in weight-only GPTQ, and the Hessian comes from the BF16 layer inputs. The codes therefore do not depend on the activation format they are evaluated with.
- Lazy-update block 128 columns, dampening 0.01 × mean(diag H), no act-order, the ModelOpt 0.47 handling of dead columns.
- Calibration batch 8; GPTQ took 12.5 min on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (the 256-window run: 9.2 min), peak allocated GPU memory 28.6 GiB.
- Code: FlipQuant revision `8dbe2e429fd656814868d1a4833c56aeac6d3732`, whose GPTQ code is the main table's, unchanged. On it, Llama-3.2-1B-Instruct's 256-window run reproduces the main table's codes and the published 256-window codes file bit for bit.
- Code digest (FlipQuant's `codes_sha256` over every module's packed codes, scales and formats): `81da7e818ed474cb96ad7dbe09f66d258657f68dcb4d2a43f8babf5a88b5399d`.

The FlipQuant code (GPTQ, the native kernels and the evaluation) will be released with the paper.

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
| GPTQ‡, 256 windows (`NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4`) | 70.86 | 87.19 ± 1.84 | 8.429 | 11.500 |
| GPTQ‡, 1,024 windows (**this repository**) | 71.15 | 87.04 ± 1.85 | 8.404 | 11.487 |

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 80.85 | 77.49 | 73.01 | 79.84 | 56.14 | 66.78 |
| FourOverSix (reference) | 79.98 | 76.55 | 71.51 | 79.04 | 53.92 | 64.31 |
| GPTQ‡, 256 windows (`NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4`) | 79.16 | 76.15 | 72.30 | 79.00 | 53.16 | 65.42 |
| GPTQ‡, 1,024 windows (**this repository**) | 80.79 | 76.56 | 71.67 | 78.66 | 53.24 | 66.00 |

**This repository against its references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this row gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| vs FourOverSix | +0.40* (637/549, p=0.011) | -1.90* (42/67, p=0.021) | -0.0026 ± 0.0016* | -0.0014 ± 0.0011* |
| vs its 256-window version (NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4) | +0.39* (611/525, p=0.012) | -0.15 (69/71, p=0.93) | -0.0030 ± 0.0014* | -0.0011 ± 0.0009* |

## How to use

With FlipQuant (to be released with the paper), on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), the command that produced the PPL above:

```bash
python -m evaluation.ppl --model nemotron-nano-9b-v2 --revision 6533e8de2c68e4536bf7c411d7a3ce5734111476 --mode native \
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
c = obj["codes"]["model.layers.0.mixer.in_proj"]                        # one linear layer, by its module name
nib = torch.stack((c["q"] & 15, c["q"] >> 4), -1).flatten(-2).long()    # [out, in] 4-bit codes
e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
values = torch.cat([e2m1, -e2m1])[nib]       # code = sign << 3 | magnitude index
scales = c["s"].view(torch.float8_e4m3fn).float()                       # [out, in / 16] E4M3 block scales
w = (values.unflatten(-1, (-1, 16)) * scales.unsqueeze(-1)).flatten(-2) * c["g"]   # the weight, FP32
```

In the file: `q` is uint8 `[22656, 2240]`-shaped (two 4-bit codes per byte, element 2j in the low nibble), `s` uint8 (the raw E4M3 bytes of the block scales), `g` the FP32 tensor scale, `fmt` 1 (every block E2M1).

## License

These weights are derived from the base model's weights and are released under the base model's license, the NVIDIA Open Model License Agreement.

Governing terms: the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). `LICENSE` is the agreement (Last Modified: October 24, 2025), verbatim from the page the base model card links (the base repository has no license file; retrieved 2026-10-07 05:26 UTC); `NOTICE` is the attribution notice “Licensed by NVIDIA Corporation under the NVIDIA Open Model License” (section 3.1).

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 10,254 | `8cf5f9a673059fc727c03fae31c0d6e365bbf4ef1b7fd9b3099fefe1b49fdda1` |
| `NOTICE` | 67 | `5fc60716a9ba57f3792e71ed776b4065671c64d0c2db716698f4d646ad833eb1` |
| `codes_nvfp4_act-fo6.pt` | 4,337,895,473 | `1f07143f30f6a9fd16cbe0ef796f10ff1936392173c954e4c6fd8013d50d422b` |
| `ppl_summary.json` | 7,203 | `422dfb289548d1d3de6d9bffbb8f60de4d335bc8aafbb36671581968d8b37a7d` |

- `codes_nvfp4_act-fo6.pt`: the GPTQ codes (`flipquant-gptq/1`: `{format, key, codes}`; above), the calibration output, unchanged.
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.
- `NOTICE`: see License.
