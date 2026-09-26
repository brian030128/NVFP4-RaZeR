| phase | category | TM-OPT GPU s / epoch | TM-OPT+TC GPU s / epoch |
|---|---|---:|---:|
| forward | GEMM (model Linear/lm_head, fwd+bwd) | 3.03 | 2.97 |
| forward | other kernels (norms, rotary, elementwise, ...) | 0.79 | 0.79 |
| forward | act_quant_rows (training) | 1.49 | 1.48 |
| forward | memcpy (outside named regions) | 0.01 | 0.01 |
| forward | lean weight decode | 0.26 | 0.26 |
| forward | attention | 0.05 | 0.05 |
| forward | **total GPU / wall** | **5.63 / 4.73** | **5.56 / 4.62** |
| loss | other kernels (norms, rotary, elementwise, ...) | 0.68 | 0.68 |
| loss | memcpy (outside named regions) | 1.30 | 1.29 |
| loss | **total GPU / wall** | **1.99 / 5.29** | **1.98 / 5.15** |
| backward | hook: G = dy^T x (tile-gradient GEMM) | 14.70 | 2.97 |
| backward | GEMM (model Linear/lm_head, fwd+bwd) | 3.33 | 3.06 |
| backward | other kernels (norms, rotary, elementwise, ...) | 1.26 | 1.25 |
| backward | hook: candidate decode + D | 2.70 | 2.69 |
| backward | hook: G*(A-B), tile reduction, accumulate | 1.81 | 1.81 |
| backward | memcpy (outside named regions) | 0.08 | 0.08 |
| backward | lean weight decode | 0.22 | 0.21 |
| backward | attention | 0.21 | 0.21 |
| backward | **total GPU / wall** | **24.31 / 21.35** | **12.28 / 10.69** |
| optimizer | other kernels (norms, rotary, elementwise, ...) | 0.06 | 0.06 |
| optimizer | memcpy (outside named regions) | 0.01 | 0.01 |
| optimizer | **total GPU / wall** | **0.06 / 0.71** | **0.06 / 0.68** |

Epoch under the profiler: TM-OPT 35.3 s wall (32.0 s GPU); TM-OPT+TC 23.0 s wall (19.9 s GPU).

| case | FP32 error vs FP64 | TC error vs FP64 | threshold | result |
|---|---:|---:|---:|---|
| llama8b:model.layers.0.self_attn.q_proj:real:8x64 | 3.03e-07 | 5.36e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.self_attn.q_proj:real:16x64 | 2.92e-07 | 5.86e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.self_attn.q_proj:real:256x64 | 2.99e-07 | 5.30e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.self_attn.q_proj:random:8x64 | 2.91e-07 | 4.70e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.self_attn.q_proj:random:16x64 | 2.59e-07 | 4.66e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.self_attn.q_proj:random:256x64 | 3.09e-07 | 4.69e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.self_attn.k_proj:real:8x64 | 4.57e-07 | 5.33e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.self_attn.k_proj:real:16x64 | 5.61e-07 | 5.43e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.self_attn.k_proj:real:256x64 | 3.75e-07 | 5.58e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.self_attn.k_proj:random:8x64 | 2.54e-07 | 4.27e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.self_attn.k_proj:random:16x64 | 2.35e-07 | 4.21e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.self_attn.k_proj:random:256x64 | 3.26e-07 | 5.50e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.mlp.gate_proj:real:8x64 | 3.87e-07 | 5.29e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.mlp.gate_proj:real:16x64 | 3.29e-07 | 5.13e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.mlp.gate_proj:real:256x64 | 3.55e-07 | 4.03e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.mlp.gate_proj:random:8x64 | 3.64e-07 | 4.87e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.mlp.gate_proj:random:16x64 | 3.94e-07 | 4.82e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.mlp.gate_proj:random:256x64 | 3.59e-07 | 4.27e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.mlp.down_proj:real:8x64 | 1.45e-06 | 6.36e-06 | 2.91e-06 | FAIL |
| llama8b:model.layers.0.mlp.down_proj:real:16x64 | 1.16e-06 | 6.49e-06 | 2.32e-06 | FAIL |
| llama8b:model.layers.0.mlp.down_proj:real:256x64 | 6.63e-07 | 4.78e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.mlp.down_proj:random:8x64 | 5.97e-07 | 4.49e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.mlp.down_proj:random:16x64 | 6.60e-07 | 4.65e-06 | 2.00e-06 | FAIL |
| llama8b:model.layers.0.mlp.down_proj:random:256x64 | 8.73e-07 | 4.81e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.self_attn.q_proj:real:8x64 | 3.73e-07 | 5.12e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.self_attn.q_proj:real:16x64 | 3.54e-07 | 4.41e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.self_attn.q_proj:real:256x64 | 3.78e-07 | 5.31e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.self_attn.q_proj:random:8x64 | 2.95e-07 | 4.81e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.self_attn.q_proj:random:16x64 | 3.34e-07 | 4.58e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.self_attn.q_proj:random:256x64 | 2.47e-07 | 5.11e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.self_attn.k_proj:real:8x64 | 3.34e-07 | 4.88e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.self_attn.k_proj:real:16x64 | 3.52e-07 | 4.87e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.self_attn.k_proj:real:256x64 | 4.11e-07 | 6.32e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.self_attn.k_proj:random:8x64 | 2.36e-07 | 3.88e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.self_attn.k_proj:random:16x64 | 2.93e-07 | 4.87e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.self_attn.k_proj:random:256x64 | 2.99e-07 | 4.93e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.mlp.gate_proj:real:8x64 | 3.26e-07 | 4.30e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.mlp.gate_proj:real:16x64 | 3.38e-07 | 5.05e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.mlp.gate_proj:real:256x64 | 3.67e-07 | 4.61e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.mlp.gate_proj:random:8x64 | 3.37e-07 | 4.28e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.mlp.gate_proj:random:16x64 | 3.20e-07 | 4.75e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.mlp.gate_proj:random:256x64 | 3.94e-07 | 4.66e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.mlp.down_proj:real:8x64 | 6.92e-07 | 4.86e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.mlp.down_proj:real:16x64 | 7.88e-07 | 5.48e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.mlp.down_proj:real:256x64 | 7.00e-07 | 4.65e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.mlp.down_proj:random:8x64 | 6.33e-07 | 4.23e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.mlp.down_proj:random:16x64 | 8.18e-07 | 4.59e-06 | 2.00e-06 | FAIL |
| mistral7b:model.layers.0.mlp.down_proj:random:256x64 | 7.50e-07 | 4.35e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.self_attn.o_proj:real:8x64 | 3.31e-07 | 4.27e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.self_attn.o_proj:real:16x64 | 3.14e-07 | 4.88e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.self_attn.o_proj:real:256x64 | 5.04e-07 | 4.38e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.self_attn.o_proj:random:8x64 | 3.96e-07 | 4.89e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.self_attn.o_proj:random:16x64 | 3.36e-07 | 4.43e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.self_attn.o_proj:random:256x64 | 4.05e-07 | 4.86e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.self_attn.qkv_proj:real:8x64 | 7.59e-07 | 5.07e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.self_attn.qkv_proj:real:16x64 | 8.40e-07 | 4.99e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.self_attn.qkv_proj:real:256x64 | 5.72e-07 | 4.65e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.self_attn.qkv_proj:random:8x64 | 7.25e-07 | 5.40e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.self_attn.qkv_proj:random:16x64 | 8.58e-07 | 4.65e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.self_attn.qkv_proj:random:256x64 | 6.21e-07 | 4.67e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.mlp.gate_up_proj:real:8x64 | 3.13e-07 | 5.32e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.mlp.gate_up_proj:real:16x64 | 3.02e-07 | 4.30e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.mlp.gate_up_proj:real:256x64 | 3.06e-07 | 4.37e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.mlp.gate_up_proj:random:8x64 | 3.41e-07 | 4.60e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.mlp.gate_up_proj:random:16x64 | 4.15e-07 | 4.71e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.mlp.gate_up_proj:random:256x64 | 2.61e-07 | 4.01e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.mlp.down_proj:real:8x64 | 6.66e-07 | 4.58e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.mlp.down_proj:real:16x64 | 6.70e-07 | 4.74e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.mlp.down_proj:real:256x64 | 6.10e-07 | 4.41e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.mlp.down_proj:random:8x64 | 7.30e-07 | 4.84e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.mlp.down_proj:random:16x64 | 7.00e-07 | 4.67e-06 | 2.00e-06 | FAIL |
| phi4:model.layers.0.mlp.down_proj:random:256x64 | 8.12e-07 | 4.71e-06 | 2.00e-06 | FAIL |

Unit test: FAIL (72 of 72 cases fail); max error FP32 1.45e-06, TC 6.49e-06; GEMM time over the shapes FP32 123.3 ms, TC 28.8 ms.
