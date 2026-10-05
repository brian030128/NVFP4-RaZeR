## Llama-3.1-8B 16x64, A (full vocabulary, 20 epochs)

| | NVFP4-RaZeR | flipquant | flipquant / RaZeR |
|---|---:|---:|---:|
| setup (s) | 58.2 | 58.4 | 1.0042 |
| per epoch (s) | 23.01 | 22.89 | 0.9948 |
| training (s) | 460.1 | 457.7 | 0.9948 |
| trainer's total (s) | 518.3 | 516.2 | 0.9959 |
| end to end (s) | — | 529.4 (wrapper overhead +13.2 s) | |
| peak GPU allocated (GiB) | 40.48 | 40.48 | +0.00 GiB |
| peak GPU reserved (GiB) | 41.45 | 41.47 | +0.02 GiB |
| peak host RSS, sampled (GiB) | 18.19 | 18.16 | -0.03 GiB |
| peak host RSS, ru_maxrss (GiB) | 18.13 | 18.10 | -0.03 GiB |
| host RSS after the model load (GiB) | 18.19 | 18.16 | -0.03 GiB |
| wrapper process peak RSS (GiB) | | 0.78 | |
| map sha256 | 54070819a774fed6… | 54070819a774fed6… | equal |

Criteria: same map sha256: pass; per-epoch time within 3 %: pass; trainer's total time within 3 %: pass; GPU peak allocated within 0.1 GiB: pass; GPU peak reserved within 0.1 GiB: pass; host RSS (sampled) within 0.5 GiB: pass; host RSS (ru_maxrss) within 0.5 GiB: pass.

## Llama-3.1-8B 16x64, C (top-256, 5 epochs)

| | NVFP4-RaZeR | flipquant | flipquant / RaZeR |
|---|---:|---:|---:|
| setup (s) | 56.5 | 57.0 | 1.0104 |
| per epoch (s) | 20.14 | 20.13 | 0.9994 |
| training (s) | 100.7 | 100.6 | 0.9994 |
| trainer's total (s) | 157.2 | 157.7 | 1.0033 |
| end to end (s) | — | 168.8 (wrapper overhead +11.1 s) | |
| peak GPU allocated (GiB) | 41.07 | 41.07 | +0.00 GiB |
| peak GPU reserved (GiB) | 42.09 | 42.13 | +0.04 GiB |
| peak host RSS, sampled (GiB) | 15.72 | 15.01 | -0.72 GiB |
| peak host RSS, ru_maxrss (GiB) | 15.88 | 15.88 | +0.00 GiB |
| host RSS after the model load (GiB) | 2.55 | 2.60 | +0.05 GiB |
| wrapper process peak RSS (GiB) | | 0.78 | |
| map sha256 | af6c751500cc59fa… | af6c751500cc59fa… | equal |

Criteria: same map sha256: pass; per-epoch time within 3 %: pass; trainer's total time within 3 %: pass; GPU peak allocated within 0.1 GiB: pass; GPU peak reserved within 0.1 GiB: pass; host RSS (sampled) within 0.5 GiB: FAIL; host RSS (ru_maxrss) within 0.5 GiB: pass.

## Phi-4 16x64 (full vocabulary, 20 epochs), back to back

| | NVFP4-RaZeR | flipquant | flipquant / RaZeR |
|---|---:|---:|---:|
| setup (s) | 96.1 | 97.5 | 1.0150 |
| per epoch (s) | 39.93 | 39.90 | 0.9993 |
| training (s) | 798.6 | 798.0 | 0.9993 |
| trainer's total (s) | 894.8 | 895.6 | 1.0009 |
| end to end (s) | 902.3 | 908.7 | 1.0071 |
| peak GPU allocated (GiB) | 59.10 | 59.10 | +0.00 GiB |
| peak GPU reserved (GiB) | 62.36 | 62.38 | +0.02 GiB |
| peak host RSS, sampled (GiB) | 27.51 | 28.09 | +0.57 GiB |
| peak host RSS, ru_maxrss (GiB) | 28.23 | 28.23 | -0.00 GiB |
| host RSS after the model load (GiB) | 14.90 | 14.91 | +0.00 GiB |
| wrapper process peak RSS (GiB) | | 0.90 | |
| map sha256 | 110243a64cd15ee0… | 110243a64cd15ee0… | equal |

Criteria: same map sha256: pass; per-epoch time within 3 %: pass; trainer's total time within 3 %: pass; GPU peak allocated within 0.1 GiB: pass; GPU peak reserved within 0.1 GiB: pass; host RSS (sampled) within 0.5 GiB: FAIL; host RSS (ru_maxrss) within 0.5 GiB: pass.

