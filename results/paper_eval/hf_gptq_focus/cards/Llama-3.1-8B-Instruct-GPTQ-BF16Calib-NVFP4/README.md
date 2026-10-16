---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
tags:
- quantization
- fp4
- nvfp4
- gptq
---

# GPTQ NVFP4 weights for Llama-3.1-8B-Instruct, calibrated with BF16 activations

**Built with Llama.**

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. The agreement is in `LICENSE`, the Acceptable Use Policy it incorporates in `USE_POLICY.md`, and the notice in `NOTICE`.

These are the weights of the **GPTQ‡** baseline of the FlipQuant paper's main perplexity table: GPTQ-quantized NVFP4 weights, calibrated with BF16 activations and evaluated with FourOverSix per-token activations. They were made as a baseline for the FlipQuant paper.

## What this is

`codes_nvfp4_act-fo6.pt` holds the NVFP4 codes of every quantized linear layer of [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) at revision `0e9e39f249a16976918f6564b8830bc894c89659` (224 layers: all text-model linear layers except the LM head). NVFP4 stores each weight as a 4-bit E2M1 code, with one E4M3 scale per 16 consecutive elements of a row and one FP32 scale per tensor.

**It is not a full checkpoint**: only the quantized linear layers' weights are here. Everything else (embeddings, norms, the LM head) comes from the BF16 base model at the pinned revision.

## How it was made

GPTQ (FlipQuant's `flipquant.gptq`), layer by layer in model order, on NVFP4's fixed round-to-nearest grid: each 16-element block keeps its round-to-nearest E4M3 scale and each tensor its FP32 scale, and GPTQ chooses the codes. The settings:

- Calibration data: 256 windows of 512 tokens, the FlipQuant release maps' fit set: the paper's 128 math/code windows (OpenWebMath and CodeParrot-clean) plus 128 more from the extension rule. sha256 of their token ids, in order: `0a829ca2785566c33009da2717f95cdfbd9ebbbced27b311f864684d86a3e47d`.
- BF16-activation calibration: during calibration every activation quantizer is off, so the layers already quantized run with quantized weights and BF16 activations, as in weight-only GPTQ, and the Hessian comes from the BF16 layer inputs. The codes therefore do not depend on the activation format they are evaluated with.
- Lazy-update block 128 columns, dampening 0.01 × mean(diag H), no act-order, the ModelOpt 0.47 handling of dead columns.
- Calibration batch 8; GPTQ took 10.1 min on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition.
- Code: FlipQuant revision `3db0072`. This file was regenerated with it on 2026-10-14; its codes equal, digest for digest, the codes the main table was evaluated with.
- Code digest (FlipQuant's `codes_sha256` over every module's packed codes, scales and formats): `d3fe98167972cd1c11b184c134ba54c8455e14e04e5365707cd3812bbb8f84c9`.

The FlipQuant code (GPTQ, the native kernels and the evaluation) will be released with the paper.

## Perplexity

The FlipQuant paper's main table. Native sm_120 kernels on the NVIDIA RTX PRO 6000 Blackwell Workstation Edition; per-token FourOverSix activations for every quantized row except NVFP4 (per-token NVFP4 activations). WikiText-2 test (141 windows of 2048 tokens) and C4 validation (256 windows of 2048 tokens); every row on the same windows. ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; ± 2 SE over windows; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE. The numbers are also in `ppl_summary.json`.

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 7.2171 | 10.3853 | -0.0801 ± 0.0031 * | -0.0815 ± 0.0160 * | better / better |
| NVFP4 (round to nearest) | 7.8876 | 11.3409 | +0.0088 ± 0.0019 * | +0.0066 ± 0.0024 * | worse / worse |
| FourOverSix (reference) | 7.8186 | 11.2667 | — | — | reference |
| GPTQ‡ (these weights) | 7.8292 | 11.3664 | +0.0014 ± 0.0051 | +0.0088 ± 0.0192 | inconclusive / inconclusive |

## How to use

With FlipQuant (to be released with the paper), on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), the command that produced the table:

```bash
python -m evaluation.ppl --model llama3.1-8b-instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659 --mode native \
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
| `codes_nvfp4_act-fo6.pt` | 3,926,070,201 | `06ff8052565c6534f555813bfed862ab65ded2a4e0263bbda115bb4065b2f456` |
| `ppl_summary.json` | 3,998 | `9c3bab346da93624610bfa8af2d1f75f082c626d00b444007ed503a8f61f126d` |
| `LICENSE` | 7,627 | `64e1b2889b7892e6bbe7a7ed5bfe6ff793c61f9d584345f8f41cf9f5cb30a369` |
| `USE_POLICY.md` | 4,691 | `a568f2ebc73cec3fd74ba2afd992d4e945a8c7a9d851f9b66163aac834b7b859` |
| `NOTICE` | 116 | `9b4ce1898c6fdbefd6d0ca4d40a55cb994ed86ae3fd56166866b988bee7a6464` |

- `codes_nvfp4_act-fo6.pt`: the GPTQ codes (`flipquant-gptq/1`: `{format, key, codes}`; above), the file behind the main table's GPTQ‡ cell, unchanged.
- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings and the GPTQ settings.
- `LICENSE`: see License.
- `USE_POLICY.md`: see License.
- `NOTICE`: see License.
