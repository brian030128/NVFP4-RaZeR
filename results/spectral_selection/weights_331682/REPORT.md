# Real-weight feasibility results

No datasets, activations, labels, tokenizer, or model forward passes were used.
Full-matrix spectral values are rank-16/128-iteration estimates, not certificates.
Crops preserve full-matrix candidate normalization. These are development models.

| Model / matrix | Spectral / baseline (estimate) | Frobenius / baseline | Tiles | Selection seconds | Stop |
|---|---:|---:|---:|---:|---|
| llama-3.2-1b-instruct / model.layers.0.self_attn.q_proj.weight | 1.00000000 | 1.00000000 | 0/8192 | 0.21 | proposal_rejected |
| llama-3.2-1b-instruct / model.layers.0.mlp.down_proj.weight | 0.99712325 | 0.99999803 | 1/32768 | 0.05 | proposal_rejected |
| llama-3.2-1b-instruct / model.layers.8.self_attn.q_proj.weight | 1.00000000 | 1.00000000 | 0/8192 | 0.02 | proposal_rejected |
| llama-3.2-1b-instruct / model.layers.8.mlp.down_proj.weight | 1.00000000 | 1.00000000 | 0/32768 | 0.03 | proposal_rejected |
| llama-3.2-1b-instruct / model.layers.15.self_attn.q_proj.weight | 1.00000000 | 1.00000000 | 0/8192 | 0.02 | proposal_rejected |
| llama-3.2-1b-instruct / model.layers.15.mlp.down_proj.weight | 1.00000000 | 1.00000000 | 0/32768 | 0.03 | proposal_rejected |
| llama-3.1-8b / model.layers.0.self_attn.q_proj.weight | 0.99650020 | 1.00023294 | 4/32768 | 7.73 | proposal_rejected |
| llama-3.1-8b / model.layers.0.mlp.down_proj.weight | 1.00000000 | 1.00000000 | 0/114688 | 0.06 | proposal_rejected |
| llama-3.1-8b / model.layers.16.self_attn.q_proj.weight | 1.00000000 | 1.00000000 | 0/32768 | 0.03 | proposal_rejected |
| llama-3.1-8b / model.layers.16.mlp.down_proj.weight | 1.00000000 | 1.00000000 | 0/114688 | 0.06 | proposal_rejected |
| llama-3.1-8b / model.layers.31.self_attn.q_proj.weight | 1.00000000 | 1.00000000 | 0/32768 | 0.06 | proposal_rejected |
| llama-3.1-8b / model.layers.31.mlp.down_proj.weight | 1.00000000 | 1.00000000 | 0/114688 | 0.06 | proposal_rejected |

Exact crop objectives, approximation errors, all candidate-policy audits, source hashes, and traces are in report.json.
Maps cover only sampled matrices and must not be treated as a whole-model export.
