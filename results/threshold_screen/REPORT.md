# Search-free count rules screened against the measured curve

Any monotone threshold on the election score selects a prefix of the same ranking a
cap truncates, so each rule is screened here by the count it produces. Counts come
from the saved 192-sequence pooled score tables; no model is evaluated. The measured
ΔPPL column is the held-out C4 value at the nearest measured count in
[cap_sweep](../cap_sweep/REPORT.md) and is indicative only.

## olmo1b

2,097,152 type blocks; 30,485 eligible at the shipped 2SE threshold.

| Rule | Selected | Nearest measured count | ΔPPL there |
|---|---:|---:|---:|
| 2SE | 30,485 | — | — |
| 3SE | 4,964 | — | — |
| 4SE | 1,765 | — | — |
| 5SE | 831 | — | — |
| 6SE | 435 | — | — |
| 8SE | 29 | — | — |
| 10SE | 4 | — | — |
| 12SE | 0 | — | — |
| relmin0.5 | 1 | — | — |
| relmin0.3 | 2 | — | — |
| relmin0.2 | 6 | — | — |
| relmin0.1 | 17 | — | — |
| relmin0.05 | 36 | — | — |
| relmin0.02 | 141 | — | — |
| mass0.3 | 30,438 | — | — |
| mass0.5 | 30,267 | — | — |
| mass0.7 | 29,594 | — | — |
| mass0.9 | 24,174 | — | — |

## pythia14b

2,359,296 type blocks; 76,665 eligible at the shipped 2SE threshold.

| Rule | Selected | Nearest measured count | ΔPPL there |
|---|---:|---:|---:|
| 2SE | 76,665 | — | — |
| 3SE | 20,736 | — | — |
| 4SE | 7,442 | — | — |
| 5SE | 3,403 | — | — |
| 6SE | 1,732 | — | — |
| 8SE | 486 | — | — |
| 10SE | 143 | — | — |
| 12SE | 57 | — | — |
| relmin0.5 | 4 | — | — |
| relmin0.3 | 25 | — | — |
| relmin0.2 | 89 | — | — |
| relmin0.1 | 365 | — | — |
| relmin0.05 | 1,072 | — | — |
| relmin0.02 | 3,126 | — | — |
| mass0.3 | 75,575 | — | — |
| mass0.5 | 72,773 | — | — |
| mass0.7 | 64,572 | — | — |
| mass0.9 | 43,301 | — | — |

## qwen4b

7,096,320 type blocks; 80,978 eligible at the shipped 2SE threshold.

| Rule | Selected | Nearest measured count | ΔPPL there |
|---|---:|---:|---:|
| 2SE | 80,978 | — | — |
| 3SE | 14,994 | — | — |
| 4SE | 4,660 | — | — |
| 5SE | 1,757 | — | — |
| 6SE | 747 | — | — |
| 8SE | 179 | — | — |
| 10SE | 62 | — | — |
| 12SE | 26 | — | — |
| relmin0.5 | 1 | — | — |
| relmin0.3 | 4 | — | — |
| relmin0.2 | 6 | — | — |
| relmin0.1 | 6 | — | — |
| relmin0.05 | 23 | — | — |
| relmin0.02 | 108 | — | — |
| mass0.3 | 80,486 | — | — |
| mass0.5 | 78,643 | — | — |
| mass0.7 | 72,222 | — | — |
| mass0.9 | 51,780 | — | — |

## llama8b

13,631,488 type blocks; 97,908 eligible at the shipped 2SE threshold.

| Rule | Selected | Nearest measured count | ΔPPL there |
|---|---:|---:|---:|
| 2SE | 97,908 | — | — |
| 3SE | 2,820 | — | — |
| 4SE | 704 | — | — |
| 5SE | 408 | — | — |
| 6SE | 247 | — | — |
| 8SE | 105 | — | — |
| 10SE | 57 | — | — |
| 12SE | 30 | — | — |
| relmin0.5 | 2 | — | — |
| relmin0.3 | 5 | — | — |
| relmin0.2 | 7 | — | — |
| relmin0.1 | 18 | — | — |
| relmin0.05 | 49 | — | — |
| relmin0.02 | 131 | — | — |
| mass0.3 | 97,317 | — | — |
| mass0.5 | 93,739 | — | — |
| mass0.7 | 81,178 | — | — |
| mass0.9 | 52,741 | — | — |

## Reading

A usable search-free rule must land near each model's measured optimum without
being told the optimum. Counts far above it are the harmful regime; counts far
below it leave most of the gain unclaimed. Selecting a rule by looking at this
table would be selection on the evaluation set, so any rule chosen here must be
fixed in advance and validated on models and protocols not screened.
