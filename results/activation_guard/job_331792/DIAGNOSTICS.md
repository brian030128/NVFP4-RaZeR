# Activation guard: bound and cost diagnostics

Unfused float64 reference implementation; timing is not a production-kernel prediction.

```json
{
  "packets": 12,
  "accepted_packets": 0,
  "useful_proposals": 5,
  "harmful_proposals": 1,
  "useful_mse_maps": 0,
  "harmful_mse_maps": 1,
  "proposal_beats_mse": 6,
  "packets_with_useful_oracle_map": 4,
  "oracle_useful_maps": 16,
  "oracle_bound_accepted_maps": 0,
  "bound_violations": 0,
  "max_proposal_output_error_reduction_percent": 6.43589749097746,
  "metadata_total_bytes": 835632,
  "metadata_build_total_seconds": 0.2083958713337779,
  "metadata_percent_of_sampled_nvfp4_weight_bytes": 11.806233723958334,
  "mean_unfused_guard_milliseconds": 5.293636427571376,
  "min_uncertainty_to_proxy_gain_on_useful_proposals": 76.0106234701575,
  "max_uncertainty_to_proxy_gain_on_useful_proposals": 6103.625276222428
}
```

| Module | Residual epsilon | Metadata bytes | Weight-relative metadata | Build seconds |
|---|---:|---:|---:|---:|
| model.layers.0.self_attn.q_proj | 5.507848 | 278544 | 11.81% | 0.150 |
| model.layers.8.self_attn.q_proj | 6.369167 | 278544 | 11.81% | 0.029 |
| model.layers.15.self_attn.q_proj | 7.350127 | 278544 | 11.81% | 0.029 |

| Prompt / module | Proxy change / baseline exact loss | Bound penalty / baseline exact loss | Exact change / baseline exact loss | Guard ms |
|---|---:|---:|---:|---:|
| prose / model.layers.0.self_attn.q_proj | -0.004890 | 3.357353 | -0.064359 | 11.362 |
| prose / model.layers.8.self_attn.q_proj | 0.000000 | 0.000000 | 0.000000 | 4.757 |
| prose / model.layers.15.self_attn.q_proj | 0.000000 | 0.000000 | 0.000000 | 4.755 |
| code / model.layers.0.self_attn.q_proj | 0.000000 | 0.000000 | 0.000000 | 4.727 |
| code / model.layers.8.self_attn.q_proj | -0.007746 | 3.643786 | -0.007338 | 4.782 |
| code / model.layers.15.self_attn.q_proj | -0.000471 | 2.873889 | -0.006588 | 4.736 |
| math / model.layers.0.self_attn.q_proj | -0.007403 | 1.866107 | -0.010455 | 4.781 |
| math / model.layers.8.self_attn.q_proj | 0.000000 | 0.000000 | 0.000000 | 4.724 |
| math / model.layers.15.self_attn.q_proj | 0.000000 | 0.000000 | 0.000000 | 4.707 |
| multilingual / model.layers.0.self_attn.q_proj | -0.029610 | 2.250643 | -0.058768 | 4.754 |
| multilingual / model.layers.8.self_attn.q_proj | -0.000144 | 2.567877 | 0.001923 | 4.761 |
| multilingual / model.layers.15.self_attn.q_proj | 0.000000 | 0.000000 | 0.000000 | 4.677 |
