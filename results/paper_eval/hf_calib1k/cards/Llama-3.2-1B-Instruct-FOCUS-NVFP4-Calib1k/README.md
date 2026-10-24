---
license: llama3.2
base_model: meta-llama/Llama-3.2-1B-Instruct
tags:
- quantization
- fp4
- nvfp4
- focus
---

# FOCUS NVFP4 scales for Llama-3.2-1B-Instruct (our reproduction), 1,024 calibration windows

**Built with Llama.**

Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. The agreement is in `LICENSE`, the Acceptable Use Policy it incorporates in `USE_POLICY.md`, and the notice in `NOTICE`.

**This is our reproduction of FOCUS** ([“FOCUS: FP4 Optimization via Coupled-Relaxation and Dual-Granularity Scaling”](https://arxiv.org/abs/2608.01847), arXiv:2608.01847), **made as a baseline for the FlipQuant paper. It is not an official release by the FOCUS authors.** It is calibrated on 1,024 windows, four times the 256 of `edgeai-lab/Llama-3.2-1B-Instruct-FOCUS-NVFP4`.

## What this is

`focus.pt` holds the learned FOCUS scale parameters for [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) at revision `9213176726f574b556790deb65791e0c5aa438b6`, for every quantized linear layer (112 layers: all text-model linear layers except the LM head). Per layer, in FP32:

| field | count | role |
|---|---|---|
| `m` | one per 16-element block (60,817,408 in total) | the block-scale factor (Coupled-Relaxation Scaling); starts at 1 |
| `q` | one per 8-element sub-block, 2 per block (121,634,816 in total) | the sub-block relaxation logit (Dual-Granularity Scaling); coefficient `sigmoid(q)`, starts at `q = 6` |
| `s2` | one per tensor | the NVFP4 tensor scale `amax(abs(W)) / (6 · 448)`, frozen (it follows from the base weights) |

`m` and `q` are in row-major order of the weight `[out_features, in_features]`, 16-element blocks along `in_features`. Each layer also has `act_gscale = None`: activations use one scale per token, so there is no static activation scale. The file also stores the calibration settings (`config`) and the base model's id and revision (`source`).

**It is not a quantized checkpoint**: it holds no weights. Using it needs the BF16 base model at the pinned revision and the FOCUS implementation, which turns the base weights and these parameters into standard NVFP4 codes. For a block `b` of 16 weights with sub-blocks `k = 1, 2` of 8:

```
S_b   = e4m3( clamp( max|w_b| · m_b / (6 · s2), 2^-6, 448 ) )      # the block's E4M3 scale
c_i   = E2M1( clamp( w_i / (S_b · sigmoid(q_k) · s2), -6, 6 ) )      # element i in sub-block k
w'_i  = c_i · S_b · s2                                               # the deployed (dequantized) weight
```

`E2M1` rounds to the nearest FP4 value (ties to even); a block whose scaled maximum is 0 gets scale 1 before the clamp. `sigmoid(q)` only changes which code each element gets: the stored scales are `S_b` and `s2`, so the deployed model is plain NVFP4 (E2M1 elements, one E4M3 scale per 16 elements, one FP32 scale per tensor) and runs on standard NVFP4 kernels with no extra metadata or cost.

The code (our FOCUS implementation and the evaluation) will be released with the paper.

## Calibration

The hyperparameters follow the FOCUS paper's NVFP4 setting:

- learning rate 5e-3 for the block-scale factors `m`, 1e-3 for the relaxation logits `q`;
- 2 sub-blocks of 8 elements per 16-element block;
- KL-Top loss with k = 1000 (the BF16 model's top-1000 tokens per position, both distributions renormalized over them);
- AdamW (betas 0.9 / 0.999, weight decay 0) with a constant learning rate;
- 1 epoch, global batch 32;
- `q` initialized to 6 (and `m` to 1); seed 42 (the data order).

Only the scale parameters train; the model weights are frozen. The teacher is the unquantized BF16 model.

**Differences from the paper:**

- **Calibration data:** 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 0 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `d73b2bb2f3179c9a6c1e614c092f10e20d74c8cf5135c3d84cdba551716b1bb4`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item. That is 524,288 tokens and **32 optimizer steps** (the 256-window state: 8). The paper uses 1,248 WikiText-2 training samples of 2,048 tokens (39 steps).
- **Activations**, during calibration and evaluation: FourOverSix with one scale per token, as in the FlipQuant paper's protocol. The paper quantizes activations to NVFP4 with a static per-tensor scale.

Cost on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120), micro-batch 8 (4 forward/backward passes per step), activation checkpointing per decoder block:

| training (32 steps) | end to end (model load, data, training, save) | peak GPU memory |
|---:|---:|---:|
| 2.3 min | 2.8 min | 13.6 GiB |

The peak is PyTorch's peak allocated memory on the GPU (`torch.cuda.max_memory_allocated`). The training time includes the BF16 teacher's forward passes.

## Results

From the FlipQuant paper's calibration-size study. The numbers are also in `ppl_summary.json`.

Native sm_120 kernels on one NVIDIA RTX PRO 6000 Blackwell. Every quantized row uses per-token FourOverSix activations, and every row is evaluated on the same data:

- PPL: WikiText-2 test (141 windows of 2,048 tokens) and C4 validation (256 windows of 2,048 tokens), FlipQuant's `--paper-convention`;
- zero-shot: PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge and LAMBADA (lm-eval 0.4.11, full evaluation splits; acc_norm for PIQA, HellaSwag and ARC, acc for WinoGrande and LAMBADA). Avg. is their mean;
- GSM8K: lm-eval's `gsm8k_llama`: 8-shot chain of thought with the chat template, greedy decoding, at most 1,024 new tokens, strict-match accuracy ± 2 SE. The chat template writes the current date into the system prompt; every row of this model renders 15 Oct 2026, so all rows see identical prompts.

| method | zero-shot Avg. (%) | GSM8K strict (%) | WikiText-2 PPL | C4 PPL |
|---|---:|---:|---:|---:|
| BF16 | 59.92 | 45.41 ± 2.74 | 13.163 | 18.519 |
| FourOverSix (reference) | 56.24 | 29.72 ± 2.52 | 15.315 | 21.449 |
| FOCUS, 256 windows (`Llama-3.2-1B-Instruct-FOCUS-NVFP4`) | 57.40 | 32.15 ± 2.57 | 14.661 | 19.885 |
| FOCUS, 1,024 windows (**this repository**) | 57.90 | 32.22 ± 2.57 | 14.687 | 19.817 |

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 75.14 | 61.67 | 61.48 | 63.80 | 37.97 | 59.46 |
| FourOverSix (reference) | 71.38 | 57.74 | 58.17 | 60.19 | 36.35 | 53.64 |
| FOCUS, 256 windows (`Llama-3.2-1B-Instruct-FOCUS-NVFP4`) | 72.58 | 59.57 | 59.12 | 60.86 | 37.03 | 55.23 |
| FOCUS, 1,024 windows (**this repository**) | 72.74 | 59.27 | 59.67 | 61.91 | 36.86 | 56.92 |

**This repository against its references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this row gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| vs FourOverSix | +1.89* (1420/1006, p=4.3e-17) | +2.50 (161/128, p=0.06) | -0.0419 ± 0.0030* | -0.0791 ± 0.0043* |
| vs its 256-window version (Llama-3.2-1B-Instruct-FOCUS-NVFP4) | +0.41 (1085/995, p=0.051) | +0.08 (134/133, p=1) | +0.0018 ± 0.0023 | -0.0034 ± 0.0019* |

## How to read the file

With PyTorch alone:

```python
import torch

state = torch.load("focus.pt", map_location="cpu", weights_only=True)
state["config"]   # the calibration settings
state["source"]   # the base model id and revision
layer = state["layers"]["model.layers.0.self_attn.q_proj"]
layer["m"], layer["q"], layer["s2"]   # FP32 tensors (see above)
```

## License

This state is released under the same license as the base model, the Llama 3.2 Community License.

The Llama 3.2 Community License (section 1.b) applies to derivative works: provide a copy of the agreement, prominently display “Built with Llama”, include “Llama” at the beginning of the name of an AI model created or improved with Llama materials or their outputs, and keep the notice “Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.” in a NOTICE file. This state is trained against Llama-3.2-1B-Instruct's outputs (KL to its BF16 logits).

- `LICENSE`: the Llama 3.2 Community License Agreement, the base repository's own license file (meta-llama/Llama-3.2-1B-Instruct at revision `9213176726f574b556790deb65791e0c5aa438b6`).
- `USE_POLICY.md`: the Llama 3.2 Acceptable Use Policy, which the agreement incorporates; the base repository's own `USE_POLICY.md`.
- `NOTICE`: “Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 7,712 | `0b4284c1f87029e67654c7953afa16279961632cf73dcfe33374c4c2f298fa35` |
| `NOTICE` | 116 | `3bc2d1a6b360a03d172058198631b931c9aadf01650892fb0d25b93c6b8f0025` |
| `USE_POLICY.md` | 6,021 | `40e2777d7faa6beaf98400654170f414d8ab29b921b5163ad4ea0a1d39894201` |
| `focus.pt` | 729,889,075 | `079945eab571bc2784bbeb12cc4e0e43c4359187ca132a7989820ed3114524c6` |
| `ppl_summary.json` | 6,782 | `963f0ce80a7309325a25b46398fcba30cd697b1e707563c7624ccd89be455d9f` |

- `focus.pt`: the FOCUS state (`{config, layers, source}`; above). It is the calibration output, unchanged.
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.
- `USE_POLICY.md`: see License.
- `NOTICE`: see License.
