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

# GPTQ NVFP4 weights for NVIDIA-Nemotron-Nano-9B-v2, calibrated with BF16 activations

These are the weights of the **GPTQ‡** baseline of the FlipQuant paper's main perplexity table: GPTQ-quantized NVFP4 weights, calibrated with BF16 activations and evaluated with FourOverSix per-token activations. They were made as a baseline for the FlipQuant paper.

Not the same artifact as `edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-GPTQ-NVFP4`: this repository is GPTQ calibrated with BF16 activations (the paper's main-table GPTQ‡); that one is calibrated with FourOverSix activations.

## What this is

`codes_nvfp4_act-fo6.pt` holds the NVFP4 codes of every quantized linear layer of [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) at revision `6533e8de2c68e4536bf7c411d7a3ce5734111476` (120 layers: all text-model linear layers except the LM head). NVFP4 stores each weight as a 4-bit E2M1 code, with one E4M3 scale per 16 consecutive elements of a row and one FP32 scale per tensor.

**It is not a full checkpoint**: only the quantized linear layers' weights are here. Everything else (embeddings, norms, the LM head, the Mamba / convolution parameters) comes from the BF16 base model at the pinned revision.

## How it was made

GPTQ (FlipQuant's `flipquant.gptq`), layer by layer in model order, on NVFP4's fixed round-to-nearest grid: each 16-element block keeps its round-to-nearest E4M3 scale and each tensor its FP32 scale, and GPTQ chooses the codes. The settings:

- Calibration data: 256 windows of 512 tokens, the FlipQuant release maps' fit set: the paper's 128 math/code windows (OpenWebMath and CodeParrot-clean) plus 128 more from the extension rule. sha256 of their token ids, in order: `024372b4513c7f3d28179f0fd46c27e49d3f86c0c18fc53da05886b007c90c22`.
- BF16-activation calibration: during calibration every activation quantizer is off, so the layers already quantized run with quantized weights and BF16 activations, as in weight-only GPTQ, and the Hessian comes from the BF16 layer inputs. The codes therefore do not depend on the activation format they are evaluated with.
- Lazy-update block 128 columns, dampening 0.01 × mean(diag H), no act-order, the ModelOpt 0.47 handling of dead columns.
- Calibration batch 8; GPTQ took 9.2 min on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition.
- Code: FlipQuant revision `1a2094e`, the run that produced the main table's numbers.
- Code digest (FlipQuant's `codes_sha256` over every module's packed codes, scales and formats): `22854fc2fac91247772fec16656384a0854eff405f2ff188e4153e0643c2f51b`.

The FlipQuant code (GPTQ, the native kernels and the evaluation) will be released with the paper.

## Perplexity

The FlipQuant paper's main table. Native sm_120 kernels on the NVIDIA RTX PRO 6000 Blackwell Workstation Edition; per-token FourOverSix activations for every quantized row except NVFP4 (per-token NVFP4 activations). WikiText-2 test (147 windows of 2048 tokens) and C4 validation (256 windows of 2048 tokens); every row on the same windows. ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; ± 2 SE over windows; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE. The numbers are also in `ppl_summary.json`.

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 8.0863 | 11.1528 | -0.0412 ± 0.0016 * | -0.0310 ± 0.0013 * | better / better |
| NVFP4 (round to nearest) | 8.4535 | 11.5437 | +0.0033 ± 0.0015 * | +0.0035 ± 0.0010 * | worse / worse |
| FourOverSix (reference) | 8.4260 | 11.5034 | — | — | reference |
| GPTQ‡ (these weights) | 8.4294 | 11.5005 | +0.0004 ± 0.0016 | -0.0003 ± 0.0011 | inconclusive / inconclusive |

## How to use

With FlipQuant (to be released with the paper), on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), the command that produced the table:

```bash
python -m evaluation.ppl --model nemotron-nano-9b-v2 --revision 6533e8de2c68e4536bf7c411d7a3ce5734111476 --mode native \
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
| `codes_nvfp4_act-fo6.pt` | 4,337,895,473 | `251884a98f6408df472f5ee61b3b4656a088ab1f63d2eb26307253b5b765a3a4` |
| `ppl_summary.json` | 4,005 | `160f57558b637a9d6fc7caa8535e95e149444a71f346154f4fe8d09840b3e5b0` |
| `LICENSE` | 10,254 | `8cf5f9a673059fc727c03fae31c0d6e365bbf4ef1b7fd9b3099fefe1b49fdda1` |
| `NOTICE` | 67 | `5fc60716a9ba57f3792e71ed776b4065671c64d0c2db716698f4d646ad833eb1` |

- `codes_nvfp4_act-fo6.pt`: the GPTQ codes (`flipquant-gptq/1`: `{format, key, codes}`; above), the file behind the main table's GPTQ‡ cell, unchanged.
- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings and the GPTQ settings.
- `LICENSE`: see License.
- `NOTICE`: see License.
