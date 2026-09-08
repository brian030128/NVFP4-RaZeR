# Real-weight mechanism diagnostics

This report summarizes a fixed weight-only experiment. No maps were changed after audit.

```json
{
  "matrices": 12,
  "changed_matrices": 2,
  "selected_tiles": 5,
  "candidate_tiles": 565248,
  "crops": 36,
  "reference_crop_improvements": 33,
  "approximate_crop_improvements": 16,
  "approximate_crop_regressions": 0,
  "approximation_worse_than_reference": 31,
  "max_crop_estimation_relative_error": 8.626763858821107e-07,
  "max_full_audit_relative_residual": 1.065057062987762e-06,
  "max_selection_audit_relative_difference": 6.215449868118839e-07,
  "total_selection_seconds": 8.363423050381243,
  "total_candidate_seconds": 1.1342756380327046,
  "elapsed_seconds": 14.024948108009994,
  "max_peak_allocated_gib": 2.399902820587158
}
```

| Matrix / crop | Reference spectral / baseline | Approximate spectral / baseline | MSE spectral / baseline |
|---|---:|---:|---:|
| llama-3.2-1b-instruct / model.layers.0.self_attn.q_proj.weight / first | 0.96123327 | 1.00000000 | 1.01590089 |
| llama-3.2-1b-instruct / model.layers.0.self_attn.q_proj.weight / middle | 0.90632509 | 1.00000000 | 0.94558628 |
| llama-3.2-1b-instruct / model.layers.0.self_attn.q_proj.weight / last | 0.95455393 | 1.00000000 | 1.00000000 |
| llama-3.2-1b-instruct / model.layers.0.mlp.down_proj.weight / first | 0.90403448 | 0.90403448 | 0.98549288 |
| llama-3.2-1b-instruct / model.layers.0.mlp.down_proj.weight / middle | 0.97134308 | 1.00000000 | 0.97134308 |
| llama-3.2-1b-instruct / model.layers.0.mlp.down_proj.weight / last | 0.91547040 | 0.93321008 | 0.98515546 |
| llama-3.2-1b-instruct / model.layers.8.self_attn.q_proj.weight / first | 1.00000000 | 1.00000000 | 1.00539482 |
| llama-3.2-1b-instruct / model.layers.8.self_attn.q_proj.weight / middle | 0.90819880 | 1.00000000 | 1.03326315 |
| llama-3.2-1b-instruct / model.layers.8.self_attn.q_proj.weight / last | 0.87568695 | 0.89219310 | 0.91517157 |
| llama-3.2-1b-instruct / model.layers.8.mlp.down_proj.weight / first | 0.93147870 | 1.00000000 | 1.02112003 |
| llama-3.2-1b-instruct / model.layers.8.mlp.down_proj.weight / middle | 0.89642643 | 0.91691530 | 0.89642643 |
| llama-3.2-1b-instruct / model.layers.8.mlp.down_proj.weight / last | 0.86260329 | 0.87452281 | 0.86793749 |
| llama-3.2-1b-instruct / model.layers.15.self_attn.q_proj.weight / first | 0.90710288 | 0.90710288 | 0.91926419 |
| llama-3.2-1b-instruct / model.layers.15.self_attn.q_proj.weight / middle | 1.00000000 | 1.00000000 | 1.10361310 |
| llama-3.2-1b-instruct / model.layers.15.self_attn.q_proj.weight / last | 0.93545206 | 0.97965410 | 0.99567396 |
| llama-3.2-1b-instruct / model.layers.15.mlp.down_proj.weight / first | 0.89858666 | 0.97205534 | 1.00000000 |
| llama-3.2-1b-instruct / model.layers.15.mlp.down_proj.weight / middle | 0.96074087 | 0.96627499 | 0.96074087 |
| llama-3.2-1b-instruct / model.layers.15.mlp.down_proj.weight / last | 0.90328912 | 1.00000000 | 0.90328912 |
| llama-3.1-8b / model.layers.0.self_attn.q_proj.weight / first | 0.97298461 | 1.00000000 | 1.00000000 |
| llama-3.1-8b / model.layers.0.self_attn.q_proj.weight / middle | 0.97646953 | 1.00000000 | 0.98076826 |
| llama-3.1-8b / model.layers.0.self_attn.q_proj.weight / last | 0.90366528 | 1.00000000 | 1.00000000 |
| llama-3.1-8b / model.layers.0.mlp.down_proj.weight / first | 0.91717126 | 0.98395166 | 0.98019971 |
| llama-3.1-8b / model.layers.0.mlp.down_proj.weight / middle | 0.87149219 | 1.00000000 | 0.91074903 |
| llama-3.1-8b / model.layers.0.mlp.down_proj.weight / last | 0.83998210 | 0.86022652 | 0.91663453 |
| llama-3.1-8b / model.layers.16.self_attn.q_proj.weight / first | 1.00000000 | 1.00000000 | 1.26700427 |
| llama-3.1-8b / model.layers.16.self_attn.q_proj.weight / middle | 0.95333810 | 1.00000000 | 1.02331302 |
| llama-3.1-8b / model.layers.16.self_attn.q_proj.weight / last | 0.92320195 | 1.00000000 | 1.02903534 |
| llama-3.1-8b / model.layers.16.mlp.down_proj.weight / first | 0.92022513 | 1.00000000 | 1.07223976 |
| llama-3.1-8b / model.layers.16.mlp.down_proj.weight / middle | 0.90292157 | 1.00000000 | 1.03078125 |
| llama-3.1-8b / model.layers.16.mlp.down_proj.weight / last | 0.90495388 | 1.00000000 | 0.90495388 |
| llama-3.1-8b / model.layers.31.self_attn.q_proj.weight / first | 0.85244278 | 0.92309523 | 0.90228117 |
| llama-3.1-8b / model.layers.31.self_attn.q_proj.weight / middle | 0.85984201 | 0.90930020 | 0.89365543 |
| llama-3.1-8b / model.layers.31.self_attn.q_proj.weight / last | 0.87604759 | 0.94529607 | 0.96342503 |
| llama-3.1-8b / model.layers.31.mlp.down_proj.weight / first | 0.91616865 | 0.95072130 | 0.93831835 |
| llama-3.1-8b / model.layers.31.mlp.down_proj.weight / middle | 0.91557656 | 0.96018679 | 1.00000000 |
| llama-3.1-8b / model.layers.31.mlp.down_proj.weight / last | 0.96070759 | 1.00000000 | 0.96070759 |

Full matrices: lower-budget selector versus higher-budget audit. Neither is a certificate.

| Matrix | Selector / audit − 1 | Audit relative eigen residual | Baseline estimated stable rank | MSE audit spectral / baseline |
|---|---:|---:|---:|---:|
| llama-3.2-1b-instruct / model.layers.0.self_attn.q_proj.weight | 0.00000025 | 0.00000022 | 170.68 | 0.97346141 |
| llama-3.2-1b-instruct / model.layers.0.mlp.down_proj.weight | -0.00000038 | 0.00000021 | 381.39 | 0.97670595 |
| llama-3.2-1b-instruct / model.layers.8.self_attn.q_proj.weight | 0.00000045 | 0.00000052 | 297.59 | 1.00123324 |
| llama-3.2-1b-instruct / model.layers.8.mlp.down_proj.weight | 0.00000000 | 0.00000020 | 183.48 | 0.97778361 |
| llama-3.2-1b-instruct / model.layers.15.self_attn.q_proj.weight | -0.00000062 | 0.00000035 | 197.08 | 0.94278834 |
| llama-3.2-1b-instruct / model.layers.15.mlp.down_proj.weight | 0.00000000 | 0.00000043 | 475.04 | 0.99935812 |
| llama-3.1-8b / model.layers.0.self_attn.q_proj.weight | -0.00000022 | 0.00000032 | 148.58 | 1.02638081 |
| llama-3.1-8b / model.layers.0.mlp.down_proj.weight | 0.00000000 | 0.00000100 | 347.68 | 0.99466552 |
| llama-3.1-8b / model.layers.16.self_attn.q_proj.weight | -0.00000010 | 0.00000035 | 609.66 | 0.98067175 |
| llama-3.1-8b / model.layers.16.mlp.down_proj.weight | 0.00000033 | 0.00000025 | 463.98 | 0.97927541 |
| llama-3.1-8b / model.layers.31.self_attn.q_proj.weight | 0.00000018 | 0.00000043 | 466.85 | 0.93151718 |
| llama-3.1-8b / model.layers.31.mlp.down_proj.weight | 0.00000000 | 0.00000067 | 654.37 | 0.99854633 |
