# Online activation-bound feasibility results

Three full attention projection matrices, four fixed prompts, one 16-token packet each.
This is a development mechanism test on W4A16 inputs, not an end-to-end W4A4 or domain benchmark.
The analytic bound controls aggregate packet output error; numerical tests are not interval proofs.

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
  "bound_violations": 0
}
```

| Prompt / layer | MSE error / baseline | Proposed error / baseline | Accepted | Guarded error / baseline | Four-tile exact optimum / baseline |
|---|---:|---:|---|---:|---:|
| prose / model.layers.0.self_attn.q_proj | 1.000000 | 0.935641 | False | 1.000000 | 0.972624 |
| prose / model.layers.8.self_attn.q_proj | 1.000000 | 1.000000 | False | 1.000000 | 1.000000 |
| prose / model.layers.15.self_attn.q_proj | 1.000000 | 1.000000 | False | 1.000000 | 1.000000 |
| code / model.layers.0.self_attn.q_proj | 1.000000 | 1.000000 | False | 1.000000 | 0.996762 |
| code / model.layers.8.self_attn.q_proj | 1.000000 | 0.992662 | False | 1.000000 | 0.993745 |
| code / model.layers.15.self_attn.q_proj | 1.000000 | 0.993412 | False | 1.000000 | 1.000000 |
| math / model.layers.0.self_attn.q_proj | 1.000000 | 0.989545 | False | 1.000000 | 0.945956 |
| math / model.layers.8.self_attn.q_proj | 1.000000 | 1.000000 | False | 1.000000 | 1.000000 |
| math / model.layers.15.self_attn.q_proj | 1.000000 | 1.000000 | False | 1.000000 | 1.000000 |
| multilingual / model.layers.0.self_attn.q_proj | 1.000000 | 0.941232 | False | 1.000000 | 1.000000 |
| multilingual / model.layers.8.self_attn.q_proj | 1.013325 | 1.001923 | False | 1.000000 | 1.000000 |
| multilingual / model.layers.15.self_attn.q_proj | 1.000000 | 1.000000 | False | 1.000000 | 1.000000 |

All maps, bounds, exact oracle values, prompt and metadata hashes, timings and metadata sizes are in report.json.
Timings are unfused float64 reference costs, not production kernel overhead.
