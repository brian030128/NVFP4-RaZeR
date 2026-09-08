# Exact finite-switch diagnosis

Diagnostic only: no maps or selection parameters were changed. Candidate hashes match the original run.

For each original crop at the all-E2M1 baseline, evaluate every single-tile switch with float64 SVD.
A negative change along the original worst direction and a positive spectral change show that a new worst direction defeats the proposal.

```json
{
  "crops": 36,
  "best_directional_proposal_harmful": 20,
  "harmful_proposals_with_other_useful_toggle": 17,
  "any_useful_initial_toggle": 33,
  "harmful_proposal_max_abs_cosine": 0.5658206986822213
}
```

| Model / tensor / crop | Predicted directional change | Actual spectral change | Direction cosine | Any useful toggle exists |
|---|---:|---:|---:|---|
| llama-3.2-1b-instruct / model.layers.0.self_attn.q_proj.weight / first | -77.4736% | 3.9656% | 0.4486 | True |
| llama-3.2-1b-instruct / model.layers.0.self_attn.q_proj.weight / middle | -70.9185% | 15.1818% | 0.3422 | True |
| llama-3.2-1b-instruct / model.layers.0.self_attn.q_proj.weight / last | -75.0361% | 8.5300% | 0.1228 | True |
| llama-3.2-1b-instruct / model.layers.0.mlp.down_proj.weight / first | -46.3062% | -5.9073% | 0.1479 | True |
| llama-3.2-1b-instruct / model.layers.0.mlp.down_proj.weight / middle | -31.4379% | 14.0973% | 0.1496 | True |
| llama-3.2-1b-instruct / model.layers.0.mlp.down_proj.weight / last | -78.1382% | -6.6790% | 0.3832 | True |
| llama-3.2-1b-instruct / model.layers.8.self_attn.q_proj.weight / first | -29.4204% | 12.1772% | 0.5658 | False |
| llama-3.2-1b-instruct / model.layers.8.self_attn.q_proj.weight / middle | -74.7766% | 6.7924% | 0.0056 | True |
| llama-3.2-1b-instruct / model.layers.8.self_attn.q_proj.weight / last | -54.5520% | -8.9649% | 0.2207 | True |
| llama-3.2-1b-instruct / model.layers.8.mlp.down_proj.weight / first | -24.7865% | 7.8192% | 0.0663 | True |
| llama-3.2-1b-instruct / model.layers.8.mlp.down_proj.weight / middle | -45.8256% | -8.3085% | 0.3745 | True |
| llama-3.2-1b-instruct / model.layers.8.mlp.down_proj.weight / last | -74.0031% | -12.5477% | 0.5348 | True |
| llama-3.2-1b-instruct / model.layers.15.self_attn.q_proj.weight / first | -36.0810% | -9.2897% | 0.0664 | True |
| llama-3.2-1b-instruct / model.layers.15.self_attn.q_proj.weight / middle | -73.8588% | 10.1643% | 0.0972 | False |
| llama-3.2-1b-instruct / model.layers.15.self_attn.q_proj.weight / last | -38.0027% | -2.0346% | 0.5290 | True |
| llama-3.2-1b-instruct / model.layers.15.mlp.down_proj.weight / first | -43.8104% | -2.7945% | 0.3595 | True |
| llama-3.2-1b-instruct / model.layers.15.mlp.down_proj.weight / middle | -31.1609% | -3.3725% | 0.5833 | True |
| llama-3.2-1b-instruct / model.layers.15.mlp.down_proj.weight / last | -56.3358% | 1.3167% | 0.1161 | True |
| llama-3.1-8b / model.layers.0.self_attn.q_proj.weight / first | -62.4866% | 25.4890% | 0.0980 | True |
| llama-3.1-8b / model.layers.0.self_attn.q_proj.weight / middle | -72.7705% | 12.2671% | 0.3812 | True |
| llama-3.1-8b / model.layers.0.self_attn.q_proj.weight / last | -42.1807% | 6.2902% | 0.5410 | True |
| llama-3.1-8b / model.layers.0.mlp.down_proj.weight / first | -54.9998% | -1.6048% | 0.4557 | True |
| llama-3.1-8b / model.layers.0.mlp.down_proj.weight / middle | -53.5805% | 0.7099% | 0.4913 | True |
| llama-3.1-8b / model.layers.0.mlp.down_proj.weight / last | -60.8571% | -13.9773% | 0.3796 | True |
| llama-3.1-8b / model.layers.16.self_attn.q_proj.weight / first | -70.1367% | 22.8321% | 0.0212 | False |
| llama-3.1-8b / model.layers.16.self_attn.q_proj.weight / middle | -54.7128% | 12.6373% | 0.0901 | True |
| llama-3.1-8b / model.layers.16.self_attn.q_proj.weight / last | -70.7017% | 4.9796% | 0.3627 | True |
| llama-3.1-8b / model.layers.16.mlp.down_proj.weight / first | -23.9699% | 3.9646% | 0.3466 | True |
| llama-3.1-8b / model.layers.16.mlp.down_proj.weight / middle | -31.1335% | 0.2900% | 0.1986 | True |
| llama-3.1-8b / model.layers.16.mlp.down_proj.weight / last | -24.5971% | 0.0092% | 0.2682 | True |
| llama-3.1-8b / model.layers.31.self_attn.q_proj.weight / first | -35.3264% | -7.6905% | 0.0607 | True |
| llama-3.1-8b / model.layers.31.self_attn.q_proj.weight / middle | -52.1071% | -9.0700% | 0.4548 | True |
| llama-3.1-8b / model.layers.31.self_attn.q_proj.weight / last | -69.0200% | -5.4704% | 0.3640 | True |
| llama-3.1-8b / model.layers.31.mlp.down_proj.weight / first | -31.3539% | -4.9279% | 0.5204 | True |
| llama-3.1-8b / model.layers.31.mlp.down_proj.weight / middle | -27.8634% | -3.9813% | 0.5211 | True |
| llama-3.1-8b / model.layers.31.mlp.down_proj.weight / last | -23.0495% | 12.9468% | 0.4111 | True |
