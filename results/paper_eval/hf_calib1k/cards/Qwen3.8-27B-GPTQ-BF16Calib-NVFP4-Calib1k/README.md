---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- quantization
- fp4
- nvfp4
- gptq
---

# GPTQ NVFP4 weights for Qwen3.8-27B, calibrated with BF16 activations on 1,024 windows

These are **GPTQ‡** weights as in the FlipQuant paper's main perplexity table (GPTQ-quantized NVFP4 weights, calibrated with BF16 activations, evaluated with FourOverSix per-token activations), calibrated on **1,024 windows** instead of the main table's 256 (`edgeai-lab/Qwen3.8-27B-GPTQ-BF16Calib-NVFP4`). They were made for the FlipQuant paper's calibration-size study.

Not the same artifact as `edgeai-lab/Qwen3.8-27B-GPTQ-NVFP4`: this repository is GPTQ calibrated with BF16 activations; that one is calibrated with FourOverSix activations.

## What this is

`codes_nvfp4_act-fo6.pt` holds the NVFP4 codes of every quantized linear layer of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (496 layers: all text-model linear layers except the LM head; the vision tower stays BF16). NVFP4 stores each weight as a 4-bit E2M1 code, with one E4M3 scale per 16 consecutive elements of a row and one FP32 scale per tensor.

**It is not a full checkpoint**: only the quantized linear layers' weights are here. Everything else (embeddings, norms, the LM head, the recurrent and convolution parameters) comes from the BF16 base model at the pinned revision.

## How it was made

GPTQ (FlipQuant's `flipquant.gptq`), layer by layer in model order, on NVFP4's fixed round-to-nearest grid: each 16-element block keeps its round-to-nearest E4M3 scale and each tensor its FP32 scale, and GPTQ chooses the codes. The settings are the main table's; only the windows differ:

- Calibration data: 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 233 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `4892a27ce5e95286212a4970ea5e390c5818c7bdb331ecad84bb7d00bf63e2c6`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item. sha256 of the windows' token ids, in order: `dae08ac01ba10b446c36bb7b7cec9454f224820ea3fc48c5022701741537ccc6`.
- BF16-activation calibration: during calibration every activation quantizer is off, so the layers already quantized run with quantized weights and BF16 activations, as in weight-only GPTQ, and the Hessian comes from the BF16 layer inputs. The codes therefore do not depend on the activation format they are evaluated with.
- Lazy-update block 128 columns, dampening 0.01 × mean(diag H), no act-order, the ModelOpt 0.47 handling of dead columns.
- Calibration batch 1; GPTQ took 43.0 min on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (the 256-window run: 30.2 min), peak allocated GPU memory 65.9 GiB.
- Code: FlipQuant revision `8dbe2e429fd656814868d1a4833c56aeac6d3732`, whose GPTQ code is the main table's, unchanged. On it, Llama-3.2-1B-Instruct's 256-window run reproduces the main table's codes and the published 256-window codes file bit for bit.
- Code digest (FlipQuant's `codes_sha256` over every module's packed codes, scales and formats): `7b1d9ac2fc15f00e3586a11f808ba39445e02bf3b70f80a858f3afa05e109bf8`.

The FlipQuant code (GPTQ, the native kernels and the evaluation) will be released with the paper.

## Results

From the FlipQuant paper's calibration-size study. The numbers are also in `ppl_summary.json`.

Native sm_120 kernels on one NVIDIA RTX PRO 6000 Blackwell. Every quantized row uses per-token FourOverSix activations, and every row is evaluated on the same data:

- PPL: WikiText-2 test (141 windows of 2,048 tokens) and C4 validation (256 windows of 2,048 tokens), FlipQuant's `--paper-convention`;
- zero-shot: PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge and LAMBADA (lm-eval 0.4.11, full evaluation splits; acc_norm for PIQA, HellaSwag and ARC, acc for WinoGrande and LAMBADA). Avg. is their mean;
- GSM8K: lm-eval's `gsm8k_llama`: 8-shot chain of thought with the chat template, greedy decoding, at most 1,024 new tokens, strict-match accuracy ± 2 SE.

| method | zero-shot Avg. (%) | GSM8K strict (%) | WikiText-2 PPL | C4 PPL |
|---|---:|---:|---:|---:|
| BF16 | 73.96 | 95.68 ± 1.12 ◊ | 7.052 | 9.893 |
| FourOverSix (reference) | 73.70 | 94.62 ± 1.24 | 7.302 | 10.190 |
| GPTQ‡, 256 windows (`Qwen3.8-27B-GPTQ-BF16Calib-NVFP4`) | 74.02 | 96.06 ± 1.07 | 7.528 | 10.158 |
| GPTQ‡, 1,024 windows (**this repository**) | 73.44 | 96.21 ± 1.05 | 7.444 | 10.143 |

◊ BF16's GSM8K ran at batch 48, the quantized rows at 64.

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 81.50 | 82.86 | 75.93 | 73.11 | 58.36 | 72.02 |
| FourOverSix (reference) | 81.07 | 82.35 | 75.14 | 74.12 | 58.96 | 70.56 |
| GPTQ‡, 256 windows (`Qwen3.8-27B-GPTQ-BF16Calib-NVFP4`) | 80.63 | 81.99 | 74.90 | 75.21 | 60.07 | 71.30 |
| GPTQ‡, 1,024 windows (**this repository**) | 81.01 | 82.21 | 74.82 | 74.66 | 56.91 | 70.99 |

**This repository against its references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this row gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| vs FourOverSix | -0.04 (578/586, p=0.84) | +1.59* (34/13, p=0.0031) | +0.0192 ± 0.0060* | -0.0047 ± 0.0011* |
| vs its 256-window version (Qwen3.8-27B-GPTQ-BF16Calib-NVFP4) | -0.17 (517/554, p=0.27) | +0.15 (16/14, p=0.86) | -0.0112 ± 0.0056* | -0.0014 ± 0.0011* |

## How to use

With FlipQuant (to be released with the paper), on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), the command that produced the PPL above:

```bash
python -m evaluation.ppl --model qwen3.8-27b --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --mode native \
    --weight nvfp4 --act fourover6 --any-act --paper-convention \
    --ptq gptq --gptq-propagate bf16 --gptq-hessian-input bf16 --gptq-calib release \
    --gptq-calib-record <calibration record> --gptq-calib-extension <the 1,024-window record> \
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
| `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `codes_nvfp4_act-fo6.pt` | 13,697,654,681 | `413373d07f37229f02ff7e30b3fcc940a3dbd8236f5eff82f0e4385f57bd6f9a` |
| `ppl_summary.json` | 7,088 | `95ec19c502cb7cadae942e1b3ce5a18eb2943d8d5bd64df63e00b05dc27c0e10` |

- `codes_nvfp4_act-fo6.pt`: the GPTQ codes (`flipquant-gptq/1`: `{format, key, codes}`; above), the calibration output, unchanged.
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.
