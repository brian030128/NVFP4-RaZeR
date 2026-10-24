---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- quantization
- fp4
- nvfp4
- focus
---

# FOCUS NVFP4 scales for Qwen3.8-27B (our reproduction), 1,024 calibration windows

**This is our reproduction of FOCUS** ([“FOCUS: FP4 Optimization via Coupled-Relaxation and Dual-Granularity Scaling”](https://arxiv.org/abs/2608.01847), arXiv:2608.01847), **made as a baseline for the FlipQuant paper. It is not an official release by the FOCUS authors.** It is calibrated on 1,024 windows, four times the 256 of `edgeai-lab/Qwen3.8-27B-FOCUS-NVFP4`.

**Note on GSM8K.** This state stops its GSM8K answers early: GSM8K strict-match 88.40 %, with 115 of 1,319 strict-match answers missing (the 256-window state: 35). The behaviour is specific to this run. A 256-window state retrained locally exactly as this one (19 missing) and the study's 2,048-window state (7 missing) do not show it. FOCUS training is not reproducible run to run. The state is released as evaluated.

## What this is

`focus.pt` holds the learned FOCUS scale parameters for [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, for every quantized linear layer (496 layers: all text-model linear layers except the LM head). Per layer, in FP32:

| field | count | role |
|---|---|---|
| `m` | one per 16-element block (1,521,909,760 in total) | the block-scale factor (Coupled-Relaxation Scaling); starts at 1 |
| `q` | one per 8-element sub-block, 2 per block (3,043,819,520 in total) | the sub-block relaxation logit (Dual-Granularity Scaling); coefficient `sigmoid(q)`, starts at `q = 6` |
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

- **Calibration data:** 1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer. The other 896 come from the release extension rule (seed 20260930; one window per document, half from each source). It skips the record's fit documents, the development records and the published C4 evaluation documents (128 / 192 / 233 skipped). The set contains the release's 256 windows, window for window, so the 256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset and token sha256 (sha256 of the token-hash list: `4892a27ce5e95286212a4970ea5e390c5818c7bdb331ecad84bb7d00bf63e2c6`). A word 13-gram audit of the study's 2,048-window set, which contains these windows, against every evaluation text found no shared 13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into windows, 30 of 30 for each item. That is 524,288 tokens and **32 optimizer steps** (the 256-window state: 8). The paper uses 1,248 WikiText-2 training samples of 2,048 tokens (39 steps).
- **Activations**, during calibration and evaluation: FourOverSix with one scale per token, as in the FlipQuant paper's protocol. The paper quantizes activations to NVFP4 with a static per-tensor scale.
- **Training setup (a deviation):** trained locally on one RTX PRO 6000 with host-offloaded AdamW (`--offload`, micro-batch 1), not on 2 x H200 like its 256-window state. AdamW runs on the CPU in FP32 on gradients accumulated in the GPU path's order; only the summation order differs.

Cost on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120), micro-batch 1 (32 forward/backward passes per step), activation checkpointing per decoder block, AdamW offloaded to the host:

| training (32 steps) | end to end (model load, data, training, save) | peak GPU memory |
|---:|---:|---:|
| 296.8 min | 299.1 min | 91.8 GiB |

The peak is PyTorch's peak allocated memory on the GPU (`torch.cuda.max_memory_allocated`). The training time includes the BF16 teacher's forward passes.

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
| FOCUS, 256 windows, 2 x H200 (`Qwen3.8-27B-FOCUS-NVFP4`) | 74.30 | 93.56 ± 1.35 | 7.275 | 10.157 |
| FOCUS, 256 windows, retrained locally as this one (control; not released) | 74.39 | 95.60 ± 1.13 | 7.233 | 10.150 |
| FOCUS, 1,024 windows (**this repository**) | 74.72 | 88.40 ± 1.76 | 7.190 | 10.136 |

◊ BF16's GSM8K ran at batch 48, the quantized rows at 64.

Zero-shot accuracy (%) per task:

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 81.50 | 82.86 | 75.93 | 73.11 | 58.36 | 72.02 |
| FourOverSix (reference) | 81.07 | 82.35 | 75.14 | 74.12 | 58.96 | 70.56 |
| FOCUS, 256 windows, 2 x H200 (`Qwen3.8-27B-FOCUS-NVFP4`) | 81.56 | 82.19 | 75.37 | 76.68 | 59.56 | 70.42 |
| FOCUS, 256 windows, retrained locally as this one (control; not released) | 81.61 | 82.56 | 75.45 | 76.94 | 59.22 | 70.56 |
| FOCUS, 1,024 windows (**this repository**) | 81.28 | 82.45 | 76.24 | 77.15 | 60.15 | 71.07 |

**This repository against its references, paired.**

- Zero-shot: pooled over all 21,848 questions of the six tasks.
- GSM8K: strict-match over its 1,319 problems.
- Format: difference in points (questions only this row gets right / only the reference gets right, exact McNemar p).
- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.
- `*`: p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |
|---|---|---|---|---|
| vs FourOverSix | +0.64* (651/511, p=4.5e-05) | -6.22* (31/113, p=3.7e-12) | -0.0155 ± 0.0057* | -0.0053 ± 0.0012* |
| vs the 256-window state (2 x H200 cluster run; a different environment) | +0.38* (527/444, p=0.0085) | -5.16* (38/106, p=1.3e-08) | -0.0117 ± 0.0044* | -0.0021 ± 0.0010* |
| vs a 256-window state retrained locally as this one (the environment-matched control) | +0.16 (524/489, p=0.29) | -7.20* (22/117, p=7.6e-17) | -0.0060 ± 0.0037* | -0.0014 ± 0.0009* |

## How to read the file

With PyTorch alone:

```python
import torch

state = torch.load("focus.pt", map_location="cpu", weights_only=True)
state["config"]   # the calibration settings
state["source"]   # the base model id and revision
layer = state["layers"]["model.language_model.layers.0.linear_attn.out_proj"]
layer["m"], layer["q"], layer["s2"]   # FP32 tensors (see above)
```

## License

This state is released under the same license as the base model, the Apache License 2.0.

The base model card declares Apache-2.0, and the base repository has no license file; `LICENSE` is the Apache License 2.0 text from https://www.apache.org/licenses/LICENSE-2.0.txt.

## Files

| file | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `focus.pt` | 18,263,340,091 | `6272403d4f74f5c595651571802275047612753163111a32f465ca5db9b9f039` |
| `ppl_summary.json` | 8,317 | `df6f01a5640f66d3b2217bc97a8e4d02fbce4c983c20fc0a53cd495d44592282` |

- `focus.pt`: the FOCUS state (`{config, layers, source}`; above). It is the calibration output, unchanged.
- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration settings and cost.
- `LICENSE`: see License.
