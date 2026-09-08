# Successive format-selection directions

Completed frozen studies: 6. Declared screens passed: 0.
This is an exploratory research record, not a claim of a validated universal selector.

Each stage froze a method and evaluated every named comparator. No evaluation loss selected a per-model or per-domain configuration.
Different stages use different evaluation examples; absolute PPL across stages must not be compared.

## conditional_format

```json
{
  "cells": 6,
  "conditional_beats_dynamic": 3,
  "supported_vs_dynamic": 2,
  "conditional_beats_e2m1": 3,
  "supported_harm_vs_e2m1": 2,
  "conditional_beats_four_over_six": 4,
  "supported_harm_vs_four_over_six": 2,
  "passes_declared_screen": false
}
```

| Model / domain | ΔPPL vs dynamic MSE | ΔPPL vs compensated E2M1 | ΔPPL vs FourOverSix |
|---|---:|---:|---:|
| llama1b / wiki | -0.842323 | -0.402497 | -1.167278 |
| llama1b / math | -0.256900 | -0.007210 | +0.184717 |
| llama1b / code | +0.026951 | +0.078341 | -0.044832 |
| opt350m / wiki | -0.919410 | +0.518191 | -0.990729 |
| opt350m / math | +0.109012 | +0.626952 | +0.504212 |
| opt350m / code | +0.659448 | -0.078411 | -0.701215 |

## asymmetric_format

```json
{
  "cells": 6,
  "conditional_beats_dynamic": 6,
  "supported_vs_dynamic": 2,
  "conditional_beats_e2m1": 6,
  "supported_harm_vs_e2m1": 0,
  "conditional_beats_four_over_six": 3,
  "supported_harm_vs_four_over_six": 2,
  "passes_declared_screen": false
}
```

| Model / domain | ΔPPL vs dynamic MSE | ΔPPL vs compensated E2M1 | ΔPPL vs FourOverSix |
|---|---:|---:|---:|
| llama1b / wiki | -0.530208 | -0.055390 | -0.094675 |
| llama1b / math | -0.093829 | -0.004958 | +0.400599 |
| llama1b / code | -0.245220 | -0.647954 | +0.735657 |
| opt350m / wiki | -0.619044 | -0.382401 | -2.397479 |
| opt350m / math | -0.293323 | -0.102537 | +0.268382 |
| opt350m / code | -0.723677 | -2.686231 | -1.657435 |

## consumer_activation

```json
{
  "cells": 9,
  "beats_four_over_six": 3,
  "ppl_gain_at_least_point01": 3,
  "supported_vs_four_over_six": 0,
  "supported_harm_vs_four_over_six": 0,
  "beats_mechanism_ablation": 5,
  "passes_declared_screen": false
}
```

| Model / domain | ΔPPL vs FourOverSix | ΔPPL vs activation_mse |
|---|---:|---:|
| llama1b / wiki | +0.079264 | +0.043913 |
| llama1b / math | +0.010962 | +0.036458 |
| llama1b / code | -0.072037 | -0.008684 |
| opt350m / wiki | +0.106635 | +0.191777 |
| opt350m / math | -0.124582 | -0.323774 |
| opt350m / code | +0.375406 | -0.152993 |
| qwen06b / wiki | +0.167501 | -0.086687 |
| qwen06b / math | -0.076082 | +0.067140 |
| qwen06b / code | +0.325067 | -0.165095 |

## interacting_format

```json
{
  "cells": 9,
  "beats_four_over_six": 6,
  "ppl_gain_at_least_point01": 6,
  "supported_vs_four_over_six": 3,
  "supported_harm_vs_four_over_six": 2,
  "beats_mechanism_ablation": 5,
  "passes_declared_screen": false
}
```

| Model / domain | ΔPPL vs FourOverSix | ΔPPL vs independent |
|---|---:|---:|
| llama1b / wiki | -0.190573 | -0.758555 |
| llama1b / math | +0.071253 | -0.050136 |
| llama1b / code | -0.368538 | -0.200144 |
| opt350m / wiki | -0.183522 | -0.278684 |
| opt350m / math | +0.584161 | +0.424586 |
| opt350m / code | -0.271334 | +1.072563 |
| qwen06b / wiki | -1.926150 | -1.156807 |
| qwen06b / math | -0.734849 | +0.019886 |
| qwen06b / code | +0.507817 | +0.496179 |

## fisher_format

```json
{
  "cells": 9,
  "beats_four_over_six": 4,
  "ppl_gain_at_least_point01": 4,
  "supported_vs_four_over_six": 3,
  "supported_harm_vs_four_over_six": 1,
  "beats_mechanism_ablation": 7,
  "passes_declared_screen": false
}
```

| Model / domain | ΔPPL vs FourOverSix | ΔPPL vs uniform |
|---|---:|---:|
| llama1b / wiki | -0.304401 | -0.089805 |
| llama1b / math | +0.040171 | -0.128838 |
| llama1b / code | -0.249125 | +0.009112 |
| opt350m / wiki | +0.364540 | -0.080845 |
| opt350m / math | +0.257020 | -0.216034 |
| opt350m / code | +0.000356 | -0.039571 |
| qwen06b / wiki | -1.446599 | -0.041397 |
| qwen06b / math | -0.413765 | +0.110894 |
| qwen06b / code | +0.259500 | -0.160471 |

## branched_format

```json
{
  "cells": 9,
  "beats_four_over_six": 7,
  "ppl_gain_at_least_point01": 7,
  "supported_vs_four_over_six": 5,
  "supported_harm_vs_four_over_six": 2,
  "beats_mechanism_ablation": 6,
  "passes_declared_screen": false,
  "supported_harm_vs_e2m1": 1
}
```

| Model / domain | ΔPPL vs FourOverSix | ΔPPL vs dynamic_mse |
|---|---:|---:|
| llama1b / wiki | -1.393416 | +0.227175 |
| llama1b / math | +0.287529 | -0.062932 |
| llama1b / code | -0.375121 | -0.466078 |
| opt350m / wiki | -2.235915 | -1.449141 |
| opt350m / math | +0.867600 | -0.516167 |
| opt350m / code | -0.603291 | +1.087544 |
| qwen06b / wiki | -3.262640 | +0.103660 |
| qwen06b / math | -0.614861 | -0.001127 |
| qwen06b / code | -0.078448 | -0.110281 |
