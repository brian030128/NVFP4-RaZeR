---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- quantization
- fp4
- nvfp4
- gptq
---

# GPTQ NVFP4 weights for Qwen3.8-27B, calibrated with BF16 activations

These are the weights of the **GPTQ‡** baseline of the FlipQuant paper's main perplexity table: GPTQ-quantized NVFP4 weights, calibrated with BF16 activations and evaluated with FourOverSix per-token activations. They were made as a baseline for the FlipQuant paper.

Not the same artifact as `edgeai-lab/Qwen3.8-27B-GPTQ-NVFP4`: this repository is GPTQ calibrated with BF16 activations (the paper's main-table GPTQ‡); that one is calibrated with FourOverSix activations.

## What this is

`codes_nvfp4_act-fo6.pt` holds the NVFP4 codes of every quantized linear layer of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (496 layers: all text-model linear layers except the LM head; the vision tower stays BF16). NVFP4 stores each weight as a 4-bit E2M1 code, with one E4M3 scale per 16 consecutive elements of a row and one FP32 scale per tensor.

**It is not a full checkpoint**: only the quantized linear layers' weights are here. Everything else (embeddings, norms, the LM head, the recurrent and convolution parameters) comes from the BF16 base model at the pinned revision.

## How it was made

GPTQ (FlipQuant's `flipquant.gptq`), layer by layer in model order, on NVFP4's fixed round-to-nearest grid: each 16-element block keeps its round-to-nearest E4M3 scale and each tensor its FP32 scale, and GPTQ chooses the codes. The settings:

- Calibration data: 256 windows of 512 tokens, the FlipQuant release maps' fit set: the paper's 128 math/code windows (OpenWebMath and CodeParrot-clean) plus 128 more from the extension rule. sha256 of their token ids, in order: `8a2b9e4aa6293091aed07f8540ab79da10ad3e08671a8da6337edf255bcc2066`.
- BF16-activation calibration: during calibration every activation quantizer is off, so the layers already quantized run with quantized weights and BF16 activations, as in weight-only GPTQ, and the Hessian comes from the BF16 layer inputs. The codes therefore do not depend on the activation format they are evaluated with.
- Lazy-update block 128 columns, dampening 0.01 × mean(diag H), no act-order, the ModelOpt 0.47 handling of dead columns.
- Calibration batch 1; GPTQ took 30.2 min on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition.
- Code: FlipQuant revision `c67a5bd`, the run that produced the main table's numbers.
- Code digest (FlipQuant's `codes_sha256` over every module's packed codes, scales and formats): `3c1bc3b2cfd80a11d9585c03f9c798f47fd4194813d47c659f9d6862e10e8f69`.

The FlipQuant code (GPTQ, the native kernels and the evaluation) will be released with the paper.

## Perplexity

The FlipQuant paper's main table. Native sm_120 kernels on the NVIDIA RTX PRO 6000 Blackwell Workstation Edition; per-token FourOverSix activations for every quantized row except NVFP4 (per-token NVFP4 activations). WikiText-2 test (145 windows of 2048 tokens) and C4 validation (256 windows of 2048 tokens); every row on the same windows. ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; ± 2 SE over windows; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE. The numbers are also in `ppl_summary.json`.

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 7.0523 | 9.8931 | -0.0348 ± 0.0047 * | -0.0296 ± 0.0026 * | better / better |
| NVFP4 (round to nearest) | 7.5740 | 10.2231 | +0.0365 ± 0.0081 * | +0.0032 ± 0.0011 * | worse / worse |
| FourOverSix (reference) | 7.3023 | 10.1903 | — | — | reference |
| GPTQ‡ (these weights) | 7.5278 | 10.1576 | +0.0304 ± 0.0070 * | -0.0032 ± 0.0013 * | worse / better |

## How to use

With FlipQuant (to be released with the paper), on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), the command that produced the table:

```bash
python -m evaluation.ppl --model qwen3.8-27b --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --mode native \
    --weight nvfp4 --act fourover6 --any-act --paper-convention \
    --ptq gptq --gptq-propagate bf16 --gptq-hessian-input bf16 --gptq-calib release \
    --gptq-calib-record <calibration record> --gptq-calib-extension <fit-extension record> \
    --gptq-block 128 --gptq-damp 0.01 --gptq-batch 1 --gptq-cache codes_nvfp4_act-fo6.pt
```

FlipQuant reads the file instead of running GPTQ only if the file's stored settings, including the sha256 of the calibration windows, equal the run's; it rebuilds the windows from the calibration records to check this.

With PyTorch alone, one layer's weight (this reproduces FlipQuant's own dequantization bit for bit):

```python
import torch

obj = torch.load("codes_nvfp4_act-fo6.pt", map_location="cpu", weights_only=True)
obj["key"]                                    # the GPTQ settings
c = obj["codes"]["model.language_model.layers.0.linear_attn.in_proj_a"]                        # one linear layer, by its module name
nib = torch.stack((c["q"] & 15, c["q"] >> 4), -1).flatten(-2).long()    # [out, in] 4-bit codes
e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
values = torch.cat([e2m1, -e2m1])[nib]       # code = sign << 3 | magnitude index
scales = c["s"].view(torch.float8_e4m3fn).float()                       # [out, in / 16] E4M3 block scales
w = (values.unflatten(-1, (-1, 16)) * scales.unsqueeze(-1)).flatten(-2) * c["g"]   # the weight, FP32
```

In the file: `q` is uint8 `[48, 2560]`-shaped (two 4-bit codes per byte, element 2j in the low nibble), `s` uint8 (the raw E4M3 bytes of the block scales), `g` the FP32 tensor scale, `fmt` 1 (every block E2M1).

## License

These weights are derived from the base model's weights and are released under the base model's license, the Apache License 2.0.

The base model card declares Apache-2.0, and the base repository has no license file; `LICENSE` is the Apache License 2.0 text from https://www.apache.org/licenses/LICENSE-2.0.txt.

## Files

| file | bytes | sha256 |
|---|---:|---|
| `codes_nvfp4_act-fo6.pt` | 13,697,654,681 | `f3d7de9ce2bd18a6d8f070613d8986ce00581dce5d6c8b2fd0cea94e189ad499` |
| `ppl_summary.json` | 3,946 | `2a3a6641cedf2cfd0f10f46e0836d247db63c2cf3b8ac2cd8cc374d8bcac2bb2` |
| `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |

- `codes_nvfp4_act-fo6.pt`: the GPTQ codes (`flipquant-gptq/1`: `{format, key, codes}`; above), the file behind the main table's GPTQ‡ cell, unchanged.
- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings and the GPTQ settings.
- `LICENSE`: see License.
