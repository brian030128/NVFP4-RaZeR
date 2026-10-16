---
license: mit
base_model: microsoft/phi-4
tags:
- quantization
- fp4
- nvfp4
- gptq
---

# GPTQ NVFP4 weights for phi-4, calibrated with BF16 activations

These are the weights of the **GPTQ‡** baseline of the FlipQuant paper's main perplexity table: GPTQ-quantized NVFP4 weights, calibrated with BF16 activations and evaluated with FourOverSix per-token activations. They were made as a baseline for the FlipQuant paper.

## What this is

`codes_nvfp4_act-fo6.pt` holds the NVFP4 codes of every quantized linear layer of [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) at revision `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` (160 layers: all text-model linear layers except the LM head). NVFP4 stores each weight as a 4-bit E2M1 code, with one E4M3 scale per 16 consecutive elements of a row and one FP32 scale per tensor.

**It is not a full checkpoint**: only the quantized linear layers' weights are here. Everything else (embeddings, norms, the LM head) comes from the BF16 base model at the pinned revision.

## How it was made

GPTQ (FlipQuant's `flipquant.gptq`), layer by layer in model order, on NVFP4's fixed round-to-nearest grid: each 16-element block keeps its round-to-nearest E4M3 scale and each tensor its FP32 scale, and GPTQ chooses the codes. The settings:

- Calibration data: 256 windows of 512 tokens, the FlipQuant release maps' fit set: the paper's 128 math/code windows (OpenWebMath and CodeParrot-clean) plus 128 more from the extension rule. sha256 of their token ids, in order: `ec67986df5da1bad58f1a31209495c2ebd41c91ff81ab3cc84a9264d4e76b329`.
- BF16-activation calibration: during calibration every activation quantizer is off, so the layers already quantized run with quantized weights and BF16 activations, as in weight-only GPTQ, and the Hessian comes from the BF16 layer inputs. The codes therefore do not depend on the activation format they are evaluated with.
- Lazy-update block 128 columns, dampening 0.01 × mean(diag H), no act-order, the ModelOpt 0.47 handling of dead columns.
- Calibration batch 8; GPTQ took 15.0 min on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition.
- Code: FlipQuant revision `c67a5bd`, the run that produced the main table's numbers.
- Code digest (FlipQuant's `codes_sha256` over every module's packed codes, scales and formats): `141a919d520c7e395bb700a33cffc447d9a05430636f787079daee7e70c7e273`.

The FlipQuant code (GPTQ, the native kernels and the evaluation) will be released with the paper.

## Perplexity

The FlipQuant paper's main table. Native sm_120 kernels on the NVIDIA RTX PRO 6000 Blackwell Workstation Edition; per-token FourOverSix activations for every quantized row except NVFP4 (per-token NVFP4 activations). WikiText-2 test (141 windows of 2048 tokens) and C4 validation (256 windows of 2048 tokens); every row on the same windows. ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; ± 2 SE over windows; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE. The numbers are also in `ppl_summary.json`.

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 6.4615 | 10.3098 | -0.0305 ± 0.0022 * | -0.0225 ± 0.0014 * | better / better |
| NVFP4 (round to nearest) | 6.6934 | 10.5795 | +0.0047 ± 0.0018 * | +0.0034 ± 0.0011 * | worse / worse |
| FourOverSix (reference) | 6.6617 | 10.5441 | — | — | reference |
| GPTQ‡ (these weights) | 6.6883 | 10.5534 | +0.0040 ± 0.0020 * | +0.0009 ± 0.0012 | worse / inconclusive |

## How to use

With FlipQuant (to be released with the paper), on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), the command that produced the table:

```bash
python -m evaluation.ppl --model phi4-14b --revision 2db69c1c3e91a05d2c64a3185acfbaf36f744e25 --mode native \
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

In the file: `q` is uint8 `[5120, 8960]`-shaped (two 4-bit codes per byte, element 2j in the low nibble), `s` uint8 (the raw E4M3 bytes of the block scales), `g` the FP32 tensor scale, `fmt` 1 (every block E2M1).

## License

These weights are derived from the base model's weights and are released under the base model's license, the MIT License.

`LICENSE` is a copy of the base repository's license file (microsoft/phi-4).

## Files

| file | bytes | sha256 |
|---|---:|---|
| `codes_nvfp4_act-fo6.pt` | 7,667,859,109 | `5c9522f802db3aa7ea01b07da6ade51e9aa18f2329b631e48ea564a0a85b94f1` |
| `ppl_summary.json` | 3,939 | `1320035800073d8aa4f817337bb72eb9a75cc59af95e0a37c4a442ac8e08ff4e` |
| `LICENSE` | 1,084 | `fa8235e5b48faca34e3ca98cf4f694ef08bd216d28b58071a1f85b1d50cb814d` |

- `codes_nvfp4_act-fo6.pt`: the GPTQ codes (`flipquant-gptq/1`: `{format, key, codes}`; above), the file behind the main table's GPTQ‡ cell, unchanged.
- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings and the GPTQ settings.
- `LICENSE`: see License.
