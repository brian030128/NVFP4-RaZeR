# Teacher-Fisher tile formats: llama1b

All candidates fixed at original weights. One shared fit pass; fresh evaluation.
Reference-text loss only; descriptive paired intervals.

| Domain | FourOverSix PPL | Uniform PPL | Diagonal-Fisher PPL | Block-Fisher PPL | Block-Fisher − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| wiki | 14.549403 | 14.334807 | 14.286334 | 14.245002 | -0.021144 ± 0.009760 |
| math | 5.310044 | 5.479053 | 5.430831 | 5.350216 | +0.007537 ± 0.016673 |
| code | 8.840964 | 8.582726 | 8.680057 | 8.591838 | -0.028583 ± 0.031969 |
