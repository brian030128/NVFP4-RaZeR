# Conditional format-selection mechanism panel

One Wiki calibration pass shared by all comparators. Nine matrices; three held-out input domains.
Layer reconstruction only, on pristine model trajectories. No downstream accuracy claim.

```json
{
  "summary": {
    "wiki": {
      "geometric_mean_conditional_over_dynamic": 0.9732042187969686,
      "matrices_improved": 9,
      "matrix_ratios": [
        0.9517964447106074,
        0.9511035892152083,
        0.9796439843538172,
        0.9801688974674304,
        0.9795066186113143,
        0.975090108503853,
        0.9757746310191031,
        0.9825584300444916,
        0.9838632057695916
      ]
    },
    "math": {
      "geometric_mean_conditional_over_dynamic": 0.973801150359579,
      "matrices_improved": 9,
      "matrix_ratios": [
        0.9442785784315346,
        0.9461049212084464,
        0.9970349405571254,
        0.984702943219799,
        0.9846964412099367,
        0.9739291829136927,
        0.9822203345732228,
        0.980414602013186,
        0.9721405448648565
      ]
    },
    "code": {
      "geometric_mean_conditional_over_dynamic": 0.9721717378157536,
      "matrices_improved": 9,
      "matrix_ratios": [
        0.9474015575771602,
        0.9648827609300693,
        0.9795265130625115,
        0.9845253480526639,
        0.9855755467618649,
        0.9744603660050862,
        0.9602529420297568,
        0.9805931074440991,
        0.9729860230299816
      ]
    }
  },
  "close_enough": true
}
```

| Matrix | Fit conditional / dynamic MSE | Wiki | Math | Code |
|---|---:|---:|---:|---:|
| model.layers.0.self_attn.q_proj | 0.948117 | 0.951796 | 0.944279 | 0.947402 |
| model.layers.0.self_attn.k_proj | 0.938558 | 0.951104 | 0.946105 | 0.964883 |
| model.layers.0.self_attn.v_proj | 0.978056 | 0.979644 | 0.997035 | 0.979527 |
| model.layers.8.self_attn.q_proj | 0.978374 | 0.980169 | 0.984703 | 0.984525 |
| model.layers.8.self_attn.k_proj | 0.981105 | 0.979507 | 0.984696 | 0.985576 |
| model.layers.8.self_attn.v_proj | 0.973988 | 0.975090 | 0.973929 | 0.974460 |
| model.layers.15.self_attn.q_proj | 0.968195 | 0.975775 | 0.982220 | 0.960253 |
| model.layers.15.self_attn.k_proj | 0.971246 | 0.982558 | 0.980415 | 0.980593 |
| model.layers.15.self_attn.v_proj | 0.973301 | 0.983863 | 0.972141 | 0.972986 |
