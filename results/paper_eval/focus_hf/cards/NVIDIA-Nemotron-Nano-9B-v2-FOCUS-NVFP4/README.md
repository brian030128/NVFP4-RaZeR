---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/NVIDIA-Nemotron-Nano-9B-v2
tags:
- quantization
- fp4
- nvfp4
- focus
---

# FOCUS NVFP4 scales for NVIDIA-Nemotron-Nano-9B-v2 (our reproduction)

**This is our reproduction of FOCUS** ([“FOCUS: FP4 Optimization via Coupled-Relaxation and Dual-Granularity Scaling”](https://arxiv.org/abs/2608.01847), arXiv:2608.01847), **made as a baseline for the FlipQuant paper. It is not an official release by the FOCUS authors.**

## What this is

`focus.pt` holds the learned FOCUS scale parameters for [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) at revision `6533e8de2c68e4536bf7c411d7a3ce5734111476`, for every quantized linear layer (120 layers: all text-model linear layers except the LM head). Per layer, in FP32:

| field | count | role |
|---|---|---|
| `m` | one per 16-element block (481,976,320 in total) | the block-scale factor (Coupled-Relaxation Scaling); starts at 1 |
| `q` | one per 8-element sub-block, 2 per block (963,952,640 in total) | the sub-block relaxation logit (Dual-Granularity Scaling); coefficient `sigmoid(q)`, starts at `q = 6` |
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
| 4.9 min | 5.1 min | 53.9 GiB |

The peak is PyTorch's peak allocated memory on the GPU (`torch.cuda.max_memory_allocated`). The training time includes the BF16 teacher's forward passes.

## Perplexity

Native sm_120 kernels on the NVIDIA RTX PRO 6000 Blackwell Workstation Edition, with FOCUS deployed as standard NVFP4 codes and per-token FourOverSix activations. WikiText-2 test (147 windows of 2048 tokens) and C4 validation (256 windows of 2048 tokens). Every row is evaluated on the same windows:

- BF16: the unquantized model;
- NVFP4: round-to-nearest NVFP4 weights, NVFP4 activations with one scale per token;
- FourOverSix: round-to-nearest FourOverSix weights and per-token FourOverSix activations (the reference);
- FOCUS: this state, with per-token FourOverSix activations.

ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; ± 2 SE over windows; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE. The numbers are also in `ppl_summary.json`.

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 8.0863 | 11.1528 | -0.0412 ± 0.0016 * | -0.0310 ± 0.0013 * | better / better |
| NVFP4 | 8.4535 | 11.5437 | +0.0033 ± 0.0015 * | +0.0035 ± 0.0010 * | worse / worse |
| FourOverSix (reference) | 8.4260 | 11.5034 | — | — | reference |
| FOCUS (this state) | 8.3685 | 11.4734 | -0.0068 ± 0.0016 * | -0.0026 ± 0.0011 * | better / better |

## How to read the file

With PyTorch alone:

```python
import torch

state = torch.load("focus.pt", map_location="cpu", weights_only=True)
state["config"]   # the calibration settings
state["source"]   # the base model id and revision
layer = state["layers"]["model.layers.0.mixer.in_proj"]
layer["m"], layer["q"], layer["s2"]   # FP32 tensors (see above)
```

## License

This state is released under the base model's license, the NVIDIA Open Model License Agreement.

Governing terms: the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

- `LICENSE`: the NVIDIA Open Model License Agreement (Last Modified: October 24, 2025), verbatim from https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/, the page the base model card links (the base repository has no license file); retrieved 2026-10-07 05:26 UTC. The section numbers are the ones the page displays.
- `NOTICE`: the attribution notice “Licensed by NVIDIA Corporation under the NVIDIA Open Model License” (section 3.1).

## Files

| file | bytes | sha256 |
|---|---:|---|
| `focus.pt` | 5,783,806,155 | `cb772e159dd8d6a5fb72f3ab63c14bb58a481db39bfa4aca48ca0600bd510c3f` |
| `ppl_summary.json` | 3,911 | `ff5702fc4950b6d26a2665be78f4c468a77a159a69509a87de05cf8c47b2fb26` |
| `LICENSE` | 10,254 | `8cf5f9a673059fc727c03fae31c0d6e365bbf4ef1b7fd9b3099fefe1b49fdda1` |
| `NOTICE` | 67 | `5fc60716a9ba57f3792e71ed776b4065671c64d0c2db716698f4d646ad833eb1` |

- `focus.pt`: the FOCUS state (`{config, layers, source}`; above). It is the calibration output, unchanged.
- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings and the calibration's settings and cost.
- `LICENSE`: see License.
- `NOTICE`: see License.
