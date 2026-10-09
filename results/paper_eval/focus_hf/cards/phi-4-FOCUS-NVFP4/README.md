---
license: mit
base_model: microsoft/phi-4
tags:
- quantization
- fp4
- nvfp4
- focus
---

# FOCUS NVFP4 scales for phi-4 (our reproduction)

**This is our reproduction of FOCUS** ([“FOCUS: FP4 Optimization via Coupled-Relaxation and Dual-Granularity Scaling”](https://arxiv.org/abs/2608.01847), arXiv:2608.01847), **made as a baseline for the FlipQuant paper. It is not an official release by the FOCUS authors.**

## What this is

`focus.pt` holds the learned FOCUS scale parameters for [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) at revision `2db69c1c3e91a05d2c64a3185acfbaf36f744e25`, for every quantized linear layer (160 layers: all text-model linear layers except the LM head). Per layer, in FP32:

| field | count | role |
|---|---|---|
| `m` | one per 16-element block (851,968,000 in total) | the block-scale factor (Coupled-Relaxation Scaling); starts at 1 |
| `q` | one per 8-element sub-block, 2 per block (1,703,936,000 in total) | the sub-block relaxation logit (Dual-Granularity Scaling); coefficient `sigmoid(q)`, starts at `q = 6` |
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
| 6.9 min | 7.2 min | 91.0 GiB |

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
| BF16 | 6.4615 | 10.3098 | -0.0305 ± 0.0022 * | -0.0225 ± 0.0014 * | better / better |
| NVFP4 | 6.6934 | 10.5795 | +0.0047 ± 0.0018 * | +0.0034 ± 0.0011 * | worse / worse |
| FourOverSix (reference) | 6.6617 | 10.5441 | — | — | reference |
| FOCUS (this state) | 6.6279 | 10.5234 | -0.0051 ± 0.0018 * | -0.0020 ± 0.0012 * | better / better |

## How to read the file

With PyTorch alone:

```python
import torch

state = torch.load("focus.pt", map_location="cpu", weights_only=True)
state["config"]   # the calibration settings
state["source"]   # the base model id and revision
layer = state["layers"]["model.layers.0.self_attn.o_proj"]
layer["m"], layer["q"], layer["s2"]   # FP32 tensors (see above)
```

## License

This state is released under the base model's license, the MIT License.

`LICENSE` is a copy of the base repository's license file (microsoft/phi-4 at revision `2db69c1c3e91a05d2c64a3185acfbaf36f744e25`).

## Files

| file | bytes | sha256 |
|---|---:|---|
| `focus.pt` | 10,223,745,627 | `be66f52576587c0c71473e19c378bb71b2a48d89802ffa0d3e24c1c4690cd38f` |
| `ppl_summary.json` | 3,872 | `a6736001dbfc49b613c5b38d9caf1c9b0cefb94db3dbf33934a5230f9a6fba0a` |
| `LICENSE` | 1,084 | `fa8235e5b48faca34e3ca98cf4f694ef08bd216d28b58071a1f85b1d50cb814d` |

- `focus.pt`: the FOCUS state (`{config, layers, source}`; above). It is the calibration output, unchanged.
- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings and the calibration's settings and cost.
- `LICENSE`: see License.
