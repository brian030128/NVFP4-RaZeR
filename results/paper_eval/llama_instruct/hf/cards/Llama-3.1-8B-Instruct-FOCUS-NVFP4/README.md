---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
tags:
- quantization
- fp4
- nvfp4
- focus
---

# FOCUS NVFP4 scales for Llama-3.1-8B-Instruct (our reproduction)

**Built with Llama.**

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. The agreement is in `LICENSE`, the Acceptable Use Policy it incorporates in `USE_POLICY.md`, and the notice in `NOTICE`.

**This is our reproduction of FOCUS** ([“FOCUS: FP4 Optimization via Coupled-Relaxation and Dual-Granularity Scaling”](https://arxiv.org/abs/2608.01847), arXiv:2608.01847), **made as a baseline for the FlipQuant paper. It is not an official release by the FOCUS authors.**

## What this is

`focus.pt` holds the learned FOCUS scale parameters for [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) at revision `0e9e39f249a16976918f6564b8830bc894c89659`, for every quantized linear layer (224 layers: all text-model linear layers except the LM head). Per layer, in FP32:

| field | count | role |
|---|---|---|
| `m` | one per 16-element block (436,207,616 in total) | the block-scale factor (Coupled-Relaxation Scaling); starts at 1 |
| `q` | one per 8-element sub-block, 2 per block (872,415,232 in total) | the sub-block relaxation logit (Dual-Granularity Scaling); coefficient `sigmoid(q)`, starts at `q = 6` |
| `s2` | one per tensor | the NVFP4 tensor scale `amax(abs(W)) / (6 · 448)`, frozen (it follows from the base weights) |

`m` and `q` are in row-major order of the weight `[out_features, in_features]`, 16-element blocks along `in_features`. Each layer also has `act_gscale = None`: activations use one scale per token (below), so there is no static activation scale. The file also stores the calibration settings (`config`) and the base model's id and revision (`source`).

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

- **Calibration data.** The FlipQuant release set: 256 windows of 512 tokens of math and code text (131,072 tokens; OpenWebMath and CodeParrot-clean), the same fit set as the FlipQuant release maps, so 8 optimizer steps. The paper uses 1,248 WikiText-2 training samples of 2,048 tokens (39 steps).
- **Activations**, during calibration and evaluation: FourOverSix with one scale per token, as in the FlipQuant paper's protocol (NVFP4 activations in which each 16-element block's scale maps the block maximum to 6 or to 4, whichever has the lower error, and one FP32 scale per token). The paper quantizes activations to NVFP4 with a static per-tensor scale.

Cost on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120), micro-batch 8 (4 forward/backward passes per step), activation checkpointing per decoder block:

| training (8 steps) | end to end (model load, data, training, save) | peak GPU memory |
|---:|---:|---:|
| 3.9 min | 4.0 min | 53.0 GiB |

The peak is PyTorch's peak allocated memory on the GPU (`torch.cuda.max_memory_allocated`). The training time includes the BF16 teacher's forward passes.

## Perplexity

Native sm_120 kernels on the NVIDIA RTX PRO 6000 Blackwell Workstation Edition, with FOCUS deployed as standard NVFP4 codes and per-token FourOverSix activations. WikiText-2 test (141 windows of 2048 tokens) and C4 validation (256 windows of 2048 tokens). Every row is evaluated on the same windows:

- BF16: the unquantized model;
- NVFP4: round-to-nearest NVFP4 weights, NVFP4 activations with one scale per token;
- FourOverSix: round-to-nearest FourOverSix weights and per-token FourOverSix activations (the reference);
- FOCUS: this state, with per-token FourOverSix activations.

ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; ± 2 SE over windows; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE. The numbers are also in `ppl_summary.json`.

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 7.2171 | 10.3853 | -0.0801 ± 0.0031 * | -0.0815 ± 0.0160 * | better / better |
| NVFP4 | 7.8876 | 11.3409 | +0.0088 ± 0.0019 * | +0.0066 ± 0.0024 * | worse / worse |
| FourOverSix (reference) | 7.8186 | 11.2667 | — | — | reference |
| FOCUS (this state) | 7.7113 | 11.1618 | -0.0138 ± 0.0019 * | -0.0093 ± 0.0029 * | better / better |

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

This state is released under the base model's license, the Llama 3.1 Community License.

The Llama 3.1 Community License (section 1.b) applies to derivative works: provide a copy of the agreement, prominently display “Built with Llama”, include “Llama” at the beginning of the name of an AI model created or improved with Llama materials or their outputs, and keep the notice “Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.” in a NOTICE file. This state is trained against Llama-3.1-8B-Instruct's outputs (KL to its BF16 logits).

- `LICENSE`: the Llama 3.1 Community License Agreement, the base repository's own `LICENSE` file (meta-llama/Llama-3.1-8B-Instruct at revision `0e9e39f249a16976918f6564b8830bc894c89659`).
- `USE_POLICY.md`: the Llama 3.1 Acceptable Use Policy, which the agreement incorporates; the base repository's own `USE_POLICY.md`.
- `NOTICE`: “Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”

## Files

| file | bytes | sha256 |
|---|---:|---|
| `focus.pt` | 5,234,657,055 | `e311a189f43cf9161f837a855346683c1cd7225bb1984cd27eb796751c80c82a` |
| `ppl_summary.json` | 3,908 | `b1ba18ad0b511637cb6e3fb1d73278d64e20250ab170e364926c5927d583ceb1` |
| `LICENSE` | 7,627 | `64e1b2889b7892e6bbe7a7ed5bfe6ff793c61f9d584345f8f41cf9f5cb30a369` |
| `NOTICE` | 116 | `9b4ce1898c6fdbefd6d0ca4d40a55cb994ed86ae3fd56166866b988bee7a6464` |
| `USE_POLICY.md` | 4,691 | `a568f2ebc73cec3fd74ba2afd992d4e945a8c7a9d851f9b66163aac834b7b859` |

- `focus.pt`: the FOCUS state (`{config, layers, source}`; above). It is the calibration output, unchanged.
- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings and the calibration's settings and cost.
- `LICENSE`: see License.
- `NOTICE`: see License.
- `USE_POLICY.md`: see License.
