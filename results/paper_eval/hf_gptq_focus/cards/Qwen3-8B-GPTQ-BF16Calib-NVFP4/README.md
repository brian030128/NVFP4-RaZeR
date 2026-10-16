---
license: apache-2.0
base_model: Qwen/Qwen3-8B
tags:
- quantization
- fp4
- nvfp4
- gptq
---

# GPTQ NVFP4 weights for Qwen3-8B, calibrated with BF16 activations

These are the weights of the **GPTQ‡** baseline of the FlipQuant paper's main perplexity table: GPTQ-quantized NVFP4 weights, calibrated with BF16 activations and evaluated with FourOverSix per-token activations. They were made as a baseline for the FlipQuant paper.

## What this is

`codes_nvfp4_act-fo6.pt` holds the NVFP4 codes of every quantized linear layer of [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) at revision `b968826d9c46dd6066d109eabc6255188de91218` (252 layers: all text-model linear layers except the LM head). NVFP4 stores each weight as a 4-bit E2M1 code, with one E4M3 scale per 16 consecutive elements of a row and one FP32 scale per tensor.

**It is not a full checkpoint**: only the quantized linear layers' weights are here. Everything else (embeddings, norms, the LM head) comes from the BF16 base model at the pinned revision.

## How it was made

GPTQ (FlipQuant's `flipquant.gptq`), layer by layer in model order, on NVFP4's fixed round-to-nearest grid: each 16-element block keeps its round-to-nearest E4M3 scale and each tensor its FP32 scale, and GPTQ chooses the codes. The settings:

- Calibration data: 256 windows of 512 tokens, the FlipQuant release maps' fit set: the paper's 128 math/code windows (OpenWebMath and CodeParrot-clean) plus 128 more from the extension rule. sha256 of their token ids, in order: `1aeff38bfc63998b4933488daadaa7417bd642dd854104c569edf940e120edb1`.
- BF16-activation calibration: during calibration every activation quantizer is off, so the layers already quantized run with quantized weights and BF16 activations, as in weight-only GPTQ, and the Hessian comes from the BF16 layer inputs. The codes therefore do not depend on the activation format they are evaluated with.
- Lazy-update block 128 columns, dampening 0.01 × mean(diag H), no act-order, the ModelOpt 0.47 handling of dead columns.
- Calibration batch 8; GPTQ took 9.8 min on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition.
- Code: FlipQuant revision `c67a5bd`, the run that produced the main table's numbers.
- Code digest (FlipQuant's `codes_sha256` over every module's packed codes, scales and formats): `4cba489acee8be78238f8f491ec1633a00451f3c6c253ca247ca776fbdc149e2`.

The FlipQuant code (GPTQ, the native kernels and the evaluation) will be released with the paper.

## Perplexity

The FlipQuant paper's main table. Native sm_120 kernels on the NVIDIA RTX PRO 6000 Blackwell Workstation Edition; per-token FourOverSix activations for every quantized row except NVFP4 (per-token NVFP4 activations). WikiText-2 test (146 windows of 2048 tokens) and C4 validation (256 windows of 2048 tokens); every row on the same windows. ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; ± 2 SE over windows; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE. The numbers are also in `ppl_summary.json`.

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 9.7251 | 13.3004 | -0.0286 ± 0.0032 * | -0.0338 ± 0.0028 * | better / better |
| NVFP4 (round to nearest) | 10.0484 | 13.7705 | +0.0041 ± 0.0026 * | +0.0009 ± 0.0018 | worse / inconclusive |
| FourOverSix (reference) | 10.0071 | 13.7575 | — | — | reference |
| GPTQ‡ (these weights) | 10.1287 | 13.7039 | +0.0121 ± 0.0029 * | -0.0039 ± 0.0018 * | worse / better |

## How to use

With FlipQuant (to be released with the paper), on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), the command that produced the table:

```bash
python -m evaluation.ppl --model qwen3-8b --revision b968826d9c46dd6066d109eabc6255188de91218 --mode native \
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

In the file: `q` is uint8 `[4096, 6144]`-shaped (two 4-bit codes per byte, element 2j in the low nibble), `s` uint8 (the raw E4M3 bytes of the block scales), `g` the FP32 tensor scale, `fmt` 1 (every block E2M1).

## License

These weights are derived from the base model's weights and are released under the base model's license, the Apache License 2.0.

`LICENSE` is a copy of the base model's license file (Qwen/Qwen3-8B).

## Files

| file | bytes | sha256 |
|---|---:|---|
| `codes_nvfp4_act-fo6.pt` | 3,907,220,925 | `335a69b142c726a98b6879e6a47c8d7de4e295fd9efe889e7cab9a995606d269` |
| `ppl_summary.json` | 3,938 | `08f339af69cf008e6ec16f47dbc4c8607248340b7240fc9daa9e8882c5c043fb` |
| `LICENSE` | 11,343 | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |

- `codes_nvfp4_act-fo6.pt`: the GPTQ codes (`flipquant-gptq/1`: `{format, key, codes}`; above), the file behind the main table's GPTQ‡ cell, unchanged.
- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings and the GPTQ settings.
- `LICENSE`: see License.
