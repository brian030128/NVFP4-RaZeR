---
license: llama3.2
base_model: meta-llama/Llama-3.2-1B-Instruct
tags:
- quantization
- fp4
- nvfp4
- gptq
---

# GPTQ NVFP4 weights for Llama-3.2-1B-Instruct, calibrated with BF16 activations

**Built with Llama.**

Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. The agreement is in `LICENSE`, the Acceptable Use Policy it incorporates in `USE_POLICY.md`, and the notice in `NOTICE`.

These are the weights of the **GPTQ‡** baseline of the FlipQuant paper's main perplexity table: GPTQ-quantized NVFP4 weights, calibrated with BF16 activations and evaluated with FourOverSix per-token activations. They were made as a baseline for the FlipQuant paper.

## What this is

`codes_nvfp4_act-fo6.pt` holds the NVFP4 codes of every quantized linear layer of [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) at revision `9213176726f574b556790deb65791e0c5aa438b6` (112 layers: all text-model linear layers except the LM head). NVFP4 stores each weight as a 4-bit E2M1 code, with one E4M3 scale per 16 consecutive elements of a row and one FP32 scale per tensor.

**It is not a full checkpoint**: only the quantized linear layers' weights are here. Everything else (embeddings, norms, the LM head) comes from the BF16 base model at the pinned revision.

## How it was made

GPTQ (FlipQuant's `flipquant.gptq`), layer by layer in model order, on NVFP4's fixed round-to-nearest grid: each 16-element block keeps its round-to-nearest E4M3 scale and each tensor its FP32 scale, and GPTQ chooses the codes. The settings:

- Calibration data: 256 windows of 512 tokens, the FlipQuant release maps' fit set: the paper's 128 math/code windows (OpenWebMath and CodeParrot-clean) plus 128 more from the extension rule. sha256 of their token ids, in order: `1fde06eca80c4aeab249472410748ba84edca1186b964744630dbd1bdcffe3c2`.
- BF16-activation calibration: during calibration every activation quantizer is off, so the layers already quantized run with quantized weights and BF16 activations, as in weight-only GPTQ, and the Hessian comes from the BF16 layer inputs. The codes therefore do not depend on the activation format they are evaluated with.
- Lazy-update block 128 columns, dampening 0.01 × mean(diag H), no act-order, the ModelOpt 0.47 handling of dead columns.
- Calibration batch 8; GPTQ took 2.0 min on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition.
- Code: FlipQuant revision `3db0072`. This file was regenerated with it on 2026-10-14; its codes equal, digest for digest, the codes the main table was evaluated with.
- Code digest (FlipQuant's `codes_sha256` over every module's packed codes, scales and formats): `952850d7d8cf4d3d61b0eee6672151a6a2736f3fa6d49264b8dda09417f15e2c`.

The FlipQuant code (GPTQ, the native kernels and the evaluation) will be released with the paper.

## Perplexity

The FlipQuant paper's main table. Native sm_120 kernels on the NVIDIA RTX PRO 6000 Blackwell Workstation Edition; per-token FourOverSix activations for every quantized row except NVFP4 (per-token NVFP4 activations). WikiText-2 test (141 windows of 2048 tokens) and C4 validation (256 windows of 2048 tokens); every row on the same windows. ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; ± 2 SE over windows; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE. The numbers are also in `ppl_summary.json`.

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 13.1629 | 18.5189 | -0.1514 ± 0.0037 * | -0.1469 ± 0.0058 * | better / better |
| NVFP4 (round to nearest) | 15.5352 | 21.7669 | +0.0143 ± 0.0025 * | +0.0147 ± 0.0024 * | worse / worse |
| FourOverSix (reference) | 15.3152 | 21.4488 | — | — | reference |
| GPTQ‡ (these weights) | 15.5327 | 21.0190 | +0.0141 ± 0.0393 | -0.0202 ± 0.0034 * | inconclusive / better |

## How to use

With FlipQuant (to be released with the paper), on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), the command that produced the table:

```bash
python -m evaluation.ppl --model llama3.2-1b-instruct --revision 9213176726f574b556790deb65791e0c5aa438b6 --mode native \
    --weight nvfp4 --act fourover6 --any-act --paper-convention \
    --ptq gptq --gptq-propagate bf16 --gptq-hessian-input bf16 --gptq-calib release \
    --gptq-calib-record <calibration record> --gptq-calib-extension <fit-extension record> \
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

In the file: `q` is uint8 `[2048, 4096]`-shaped (two 4-bit codes per byte, element 2j in the low nibble), `s` uint8 (the raw E4M3 bytes of the block scales), `g` the FP32 tensor scale, `fmt` 1 (every block E2M1).

## License

These weights are derived from the base model's weights and are released under the base model's license, the Llama 3.2 Community License.

The Llama 3.2 Community License (section 1.b) applies to derivative works: provide a copy of the agreement, prominently display “Built with Llama”, include “Llama” at the beginning of the name of an AI model created or improved with Llama materials or their outputs, and keep the notice “Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.” in a NOTICE file.

- `LICENSE`: the Llama 3.2 Community License Agreement, the base repository's own license file (meta-llama/Llama-3.2-1B-Instruct at revision `9213176726f574b556790deb65791e0c5aa438b6`).
- `USE_POLICY.md`: the Llama 3.2 Acceptable Use Policy, which the agreement incorporates; the base repository's own `USE_POLICY.md`.
- `NOTICE`: “Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”

## Files

| file | bytes | sha256 |
|---|---:|---|
| `codes_nvfp4_act-fo6.pt` | 547,457,897 | `5f65382fe016aec8736c520051a32ee165ee80ee6572281b946055b78c3a4722` |
| `ppl_summary.json` | 3,990 | `18a2a509877e8b5834d10eaacf8430e8528d2a9422e53b2ae42be7a74f75a917` |
| `LICENSE` | 7,712 | `0b4284c1f87029e67654c7953afa16279961632cf73dcfe33374c4c2f298fa35` |
| `USE_POLICY.md` | 6,021 | `40e2777d7faa6beaf98400654170f414d8ab29b921b5163ad4ea0a1d39894201` |
| `NOTICE` | 116 | `3bc2d1a6b360a03d172058198631b931c9aadf01650892fb0d25b93c6b8f0025` |

- `codes_nvfp4_act-fo6.pt`: the GPTQ codes (`flipquant-gptq/1`: `{format, key, codes}`; above), the file behind the main table's GPTQ‡ cell, unchanged.
- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings and the GPTQ settings.
- `LICENSE`: see License.
- `USE_POLICY.md`: see License.
- `NOTICE`: see License.
