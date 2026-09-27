| model | unit | method | per epoch | selection | setup (model / data / teacher / packing / initial dev) | peak GPU allocated / reserved | host RSS | micro-batch × accum |
|---|---|---|---:|---:|---|---:|---:|---|
| Llama-3.1-8B | 8x64 | TM-OPT | 35.2 s | 13.9 min | 2.0 min (1 / 52 / 18 / 36 / 13 s) | 40.8 / 42.0 GiB | 42.2 GiB | 8 × 1 |
| Llama-3.1-8B | 8x64 | TM-OPT+TC | 22.8 s | 9.8 min | 1.4 min (1 / 13 / 17 / 37 / 13 s) | 40.8 / 42.0 GiB | 42.3 GiB | 8 × 1 |
| Llama-3.1-8B | 8x64 | MR-OPT (reference) | 9 rounds, 34 s/pass | 28.0 min | 1.6 min (1 / 12 / 17 / 38 / 17 s) | 41.8 / 43.3 GiB | 42.2 GiB | 16 / 8 |
| Llama-3.1-8B | 16x64 | TM-OPT | 35.2 s | 14.0 min | 1.4 min (1 / 12 / 18 / 37 / 13 s) | 40.6 / 41.8 GiB | 42.3 GiB | 8 × 1 |
| Llama-3.1-8B | 16x64 | TM-OPT+TC | 22.9 s | 9.8 min | 1.3 min (1 / 12 / 17 / 36 / 13 s) | 40.6 / 41.6 GiB | 42.3 GiB | 8 × 1 |
| Llama-3.1-8B | 16x64 | MR-OPT (reference) | 15 rounds, 35 s/pass | 45.4 min | 1.7 min (1 / 12 / 18 / 39 / 18 s) | 41.2 / 42.4 GiB | 42.2 GiB | 16 / 8 |
| Llama-3.1-8B | 256x64 | TM-OPT | 35.2 s | 14.0 min | 1.4 min (1 / 12 / 18 / 37 / 13 s) | 40.5 / 41.7 GiB | 42.2 GiB | 8 × 1 |
| Llama-3.1-8B | 256x64 | TM-OPT+TC | 22.8 s | 9.8 min | 1.4 min (1 / 14 / 18 / 37 / 13 s) | 40.5 / 41.6 GiB | 42.2 GiB | 8 × 1 |
| Llama-3.1-8B | 256x64 | MR-OPT (reference) | 8 rounds, 36 s/pass | 17.5 min | 1.7 min (1 / 12 / 17 / 39 / 17 s) | 40.7 / 41.6 GiB | 42.3 GiB | 16 / 8 |
| Mistral-7B-v0.3 | 8x64 | TM-OPT | 32.0 s | 12.1 min | 1.2 min (1 / 11 / 13 / 37 / 9 s) | 36.2 / 37.0 GiB | 14.7 GiB | 8 × 1 |
| Mistral-7B-v0.3 | 8x64 | TM-OPT+TC | 19.6 s | 7.9 min | 1.2 min (1 / 11 / 13 / 37 / 8 s) | 36.2 / 37.0 GiB | 14.7 GiB | 8 × 1 |
| Mistral-7B-v0.3 | 8x64 | MR-OPT (reference) | 20 rounds, 32 s/pass | 40.5 min | 1.5 min (1 / 12 / 14 / 39 / 13 s) | 37.0 / 38.2 GiB | 14.7 GiB | 16 / 8 |
| Mistral-7B-v0.3 | 16x64 | TM-OPT | 31.9 s | 12.0 min | 1.2 min (1 / 12 / 14 / 37 / 9 s) | 36.0 / 36.9 GiB | 14.7 GiB | 8 × 1 |
| Mistral-7B-v0.3 | 16x64 | TM-OPT+TC | 19.7 s | 8.0 min | 1.2 min (1 / 11 / 13 / 37 / 9 s) | 36.0 / 37.0 GiB | 14.7 GiB | 8 × 1 |
| Mistral-7B-v0.3 | 16x64 | MR-OPT (reference) | 10 rounds, 32 s/pass | 18.5 min | 1.5 min (1 / 17 / 14 / 39 / 13 s) | 36.4 / 37.5 GiB | 14.7 GiB | 16 / 8 |
| Mistral-7B-v0.3 | 256x64 | TM-OPT | 31.9 s | 12.0 min | 1.2 min (1 / 12 / 14 / 37 / 9 s) | 36.0 / 36.7 GiB | 14.7 GiB | 8 × 1 |
| Mistral-7B-v0.3 | 256x64 | TM-OPT+TC | 19.6 s | 7.9 min | 1.2 min (1 / 11 / 14 / 36 / 9 s) | 36.0 / 36.7 GiB | 14.7 GiB | 8 × 1 |
| Mistral-7B-v0.3 | 256x64 | MR-OPT (reference) | 14 rounds, 33 s/pass | 22.4 min | 1.4 min (1 / 11 / 14 / 39 / 13 s) | 35.8 / 36.8 GiB | 14.7 GiB | 16 / 8 |
| Phi-4 | 8x64 | TM-OPT | 63.2 s | 23.6 min | 2.1 min (2 / 11 / 24 / 73 / 16 s) | 59.7 / 63.4 GiB | 33.7 GiB | 8 × 1 |
| Phi-4 | 8x64 | TM-OPT+TC | 39.8 s | 15.8 min | 2.1 min (2 / 11 / 23 / 72 / 16 s) | 59.7 / 62.8 GiB | 33.7 GiB | 8 × 1 |
| Phi-4 | 8x64 | MR-OPT (reference) | 10 rounds, 65 s/pass | 36.0 min | 2.6 min (2 / 12 / 23 / 77 / 23 s) | 61.3 / 64.2 GiB | 33.6 GiB | 16 / 8 |
| Phi-4 | 16x64 | TM-OPT | 63.1 s | 23.6 min | 2.1 min (2 / 12 / 23 / 73 / 16 s) | 59.4 / 63.4 GiB | 33.7 GiB | 8 × 1 |
| Phi-4 | 16x64 | TM-OPT+TC | 39.8 s | 15.8 min | 2.1 min (2 / 13 / 23 / 72 / 16 s) | 59.4 / 62.9 GiB | 33.7 GiB | 8 × 1 |
| Phi-4 | 16x64 | MR-OPT (reference) | 13 rounds, 66 s/pass | 43.5 min | 2.6 min (2 / 12 / 23 / 78 / 23 s) | 60.1 / 63.0 GiB | 33.7 GiB | 16 / 8 |
| Phi-4 | 256x64 | TM-OPT | 63.2 s | 23.6 min | 2.1 min (2 / 12 / 23 / 74 / 16 s) | 59.2 / 63.2 GiB | 33.7 GiB | 8 × 1 |
| Phi-4 | 256x64 | TM-OPT+TC | 39.8 s | 15.9 min | 2.1 min (2 / 14 / 23 / 73 / 16 s) | 59.2 / 62.4 GiB | 33.7 GiB | 8 × 1 |
| Phi-4 | 256x64 | MR-OPT (reference) | 14 rounds, 67 s/pass | 35.3 min | 2.6 min (2 / 12 / 23 / 78 / 23 s) | 59.0 / 61.7 GiB | 33.6 GiB | 16 / 8 |
| Qwen3.8-27B | 8x64 | TM-OPT | 248.5 s | 90.5 min | 6.5 min (4 / 11 / 199 / 132 / 43 s) | 90.4 / 92.0 GiB | 79.4 GiB | 2 × 4 |
| Qwen3.8-27B | 8x64 | TM-OPT+TC | 216.6 s | 79.8 min | 6.5 min (4 / 12 / 197 / 132 / 42 s) | 90.4 / 92.0 GiB | 79.5 GiB | 2 × 4 |
| Qwen3.8-27B | 16x64 | TM-OPT | 248.3 s | 90.4 min | 6.5 min (4 / 12 / 202 / 130 / 42 s) | 89.9 / 91.5 GiB | 79.3 GiB | 2 × 4 |
| Qwen3.8-27B | 16x64 | TM-OPT+TC | 217.0 s | 79.9 min | 6.6 min (4 / 12 / 203 / 131 / 43 s) | 89.9 / 91.5 GiB | 79.3 GiB | 2 × 4 |
| Qwen3.8-27B | 256x64 | TM-OPT | 247.5 s | 90.1 min | 6.5 min (4 / 13 / 198 / 130 / 43 s) | 89.6 / 91.0 GiB | 79.3 GiB | 2 × 4 |
| Qwen3.8-27B | 256x64 | TM-OPT+TC | 216.5 s | 79.7 min | 6.6 min (4 / 14 / 202 / 132 / 42 s) | 89.6 / 91.0 GiB | 79.4 GiB | 2 × 4 |
| Llama-3.1-8B | — | QAT C1 (reference: full-weight QAT, 1 epoch, non-deterministic) | 26.9 s | 0.4 min (1 epoch) | 2.4 min | 85.3 / 87.9 GiB | 71.5 GiB | 8 × 1 |
| Llama-3.1-8B | 8x64 | TM-OPT+TC, non-deterministic (3-epoch probe) | 20.9 s | — | — | — | — | 8 × 1 |
