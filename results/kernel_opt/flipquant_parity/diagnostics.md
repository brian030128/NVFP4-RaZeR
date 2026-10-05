# flipquant GEMM parity: diagnostics

## Bisection with the parity driver (flipquant worker against the RaZeR harness)

Values: the median GEMM time (µs) of 256x64 k_proj, flipquant / RaZeR, and the cells that differ by more than 2 % in every round.

| run | set-up and timed | per-cell check | T=1 | T=16 | cells beyond 2 % in every round |
|---|---|---|---|---|---|
| D_all_t1_16 | every policy and projection, T in {1, 16} | full check | 7.20 / 7.58 (0.949) | 7.74 / 7.55 (1.025) | 256x64 q_proj T=1 (0.955), 256x64 k_proj T=1 (0.949), 256x64 k_proj T=16 (1.025), 256x64 v_proj T=1 (0.945), 256x64 v_proj T=16 (1.034), 256x64 o_proj T=1 (0.962) |
| E_allpol_k | every policy, k_proj, T in {1, 16} | full check | 7.23 / 7.58 (0.954) | 7.74 / 7.52 (1.030) | 256x64 k_proj T=1 (0.954), 256x64 k_proj T=16 (1.030) |
| A_256_k | 256x64 only, k_proj, T in {1, 16} | full check | 7.62 / 7.55 (1.008) | 7.49 / 7.49 (1.000) | — |
| F_256_allproj | 256x64 only, every projection, T in {1, 16} | full check | 7.58 / 7.58 (1.000) | 7.49 / 7.52 (0.996) | — |
| E1_rehome | as E_allpol_k, every timed buffer moved to fresh segments first (--rehome) | full check | 7.23 / 7.65 (0.946) | 7.76 / 7.52 (1.032) | 256x64 k_proj T=1 (0.946), 256x64 k_proj T=16 (1.032) |
| E2_nochecks | as E_allpol_k | none (--no-checks) | 7.58 / 7.62 (0.996) | 7.55 / 7.55 (1.000) | stock_ko k_proj T=1 (0.950), 16x64 k_proj T=16 (1.029), stock_ko k_proj T=16 (1.030) |
| E3_noprof | as E_allpol_k | fused forward + isolated GEMM, no profiler | 7.58 / 7.58 (1.000) | 7.52 / 7.52 (1.000) | 16x64 k_proj T=1 (0.954), stock_ko k_proj T=1 (0.954), stock_ko k_proj T=16 (1.030) |
| E3_forward | as E_allpol_k | fused forward under the profiler only | 7.62 / 7.65 (0.996) | 7.55 / 7.55 (1.000) | stock_ko k_proj T=1 (0.954), stock_ko k_proj T=16 (1.030) |
| E3_isolated | as E_allpol_k | isolated GEMM under the profiler only | 7.58 / 7.63 (0.994) | 7.55 / 7.55 (1.000) | stock_ko k_proj T=1 (0.954), 16x64 k_proj T=16 (1.025), stock_ko k_proj T=16 (1.030) |
| E4_P_I | as E_allpol_k | profiled forward + isolated GEMM (the check's first two parts) | 7.60 / 7.63 (0.996) | 7.55 / 7.55 (1.000) | stock_ko k_proj T=1 (0.956), 16x64 k_proj T=16 (1.025), stock_ko k_proj T=16 (1.030) |
| E4_P_I_H | as E_allpol_k | ... + the output hash (D2H copy) | 7.61 / 7.63 (0.998) | 7.55 / 7.55 (1.000) | stock_ko k_proj T=1 (0.954), 16x64 k_proj T=16 (1.025), stock_ko k_proj T=16 (1.026) |
| E4_P_I_E | as E_allpol_k | ... + torch.equal of the two outputs | 7.20 / 7.63 (0.943) | 7.74 / 7.55 (1.025) | 256x64 k_proj T=1 (0.943) |
| E4_P_I_E_H | as E_allpol_k | ... + torch.equal + the output hash (= the full check) | 7.26 / 7.62 (0.954) | 7.78 / 7.55 (1.030) | 256x64 k_proj T=1 (0.954), 256x64 k_proj T=16 (1.030) |
| E5_full_addr | as E_allpol_k, buffer addresses recorded | full check | 7.25 / 7.65 (0.948) | 7.74 / 7.52 (1.030) | 256x64 k_proj T=1 (0.948), 256x64 k_proj T=16 (1.030) |
| E5_noeq_addr | as E_allpol_k, buffer addresses recorded | ... + the output hash, no torch.equal | 7.62 / 7.62 (1.000) | 7.54 / 7.55 (0.998) | stock_ko k_proj T=1 (0.950), 16x64 k_proj T=16 (1.025), stock_ko k_proj T=16 (1.024) |

## The timed buffers with and without the trigger (E5, flipquant worker, 256x64 k_proj)

Offsets of the first timed launch's buffers within their 16 MiB-aligned region (the high bits are the process's own address-space base).

| run | T | GEMM (µs) | x | packed | sf | gs | y |
|---|---:|---:|---|---|---|---|---|
| E5_full_addr | 1 | 7.25 | 0x684000 | 0x686800 | 0x7cb000 | 0x68a600 | 0x686000 |
| E5_full_addr | 16 | 7.74 | 0x5e0000 | 0x7eb000 | 0x7f3000 | 0x68a600 | 0x680000 |
| E5_noeq_addr | 1 | 7.62 | 0x684000 | 0x686800 | 0x7cb000 | 0x68a600 | 0x686000 |
| E5_noeq_addr | 16 | 7.54 | 0x1e0000 | 0x7eb000 | 0x7f3000 | 0x68a600 | 0x680000 |

## In one NVFP4-RaZeR process: what moves these cells

| diagnostic | what changes between measurements | cell | min–max (µs) | spread |
|---|---|---|---|---:|
| placement (9 trials) | the rotation copies cloned again after a pad, or the activation buffers reallocated after a pad | 256x64/k_proj/1 | 7.52–7.55 | 0.4 % |
| placement (9 trials) | the rotation copies cloned again after a pad, or the activation buffers reallocated after a pad | 256x64/k_proj/16 | 7.46–7.49 | 0.4 % |
| placement (9 trials) | the rotation copies cloned again after a pad, or the activation buffers reallocated after a pad | 256x64/q_proj/1 | 8.42–8.70 | 3.3 % |
| placement (9 trials) | the rotation copies cloned again after a pad, or the activation buffers reallocated after a pad | stock_wB_ko/gate_proj/16 | 31.42–31.87 | 1.4 % |
| code instances (6) | the same builds loaded again from byte-identical copies | 256x64/q_proj/1 | 8.40–8.48 | 1.0 % |
| code instances (6) | the same builds loaded again from byte-identical copies | 256x64/k_proj/1 | 7.52–7.58 | 0.8 % |
| code instances (6) | the same builds loaded again from byte-identical copies | 256x64/k_proj/16 | 7.49–7.52 | 0.4 % |
| code instances (6) | the same builds loaded again from byte-identical copies | stock_wB_ko/gate_proj/16 | 31.46–31.52 | 0.2 % |
| activation placement (24 offsets) | the input, quantized activation, scales and output carved at seeded 4 KiB-aligned offsets of a 512 MiB arena | 256x64/k_proj/1 | 7.42–7.56 | 1.8 % |
| activation placement (24 offsets) | the input, quantized activation, scales and output carved at seeded 4 KiB-aligned offsets of a 512 MiB arena | 256x64/q_proj/1 | 8.30–8.51 | 2.5 % |
| activation placement (24 offsets) | the input, quantized activation, scales and output carved at seeded 4 KiB-aligned offsets of a 512 MiB arena | 256x64/k_proj/16 | 7.45–7.52 | 1.0 % |
| activation placement (24 offsets) | the input, quantized activation, scales and output carved at seeded 4 KiB-aligned offsets of a 512 MiB arena | 256x64/gate_proj/1 | 22.41–22.58 | 0.8 % |
| activation placement (24 offsets) | the input, quantized activation, scales and output carved at seeded 4 KiB-aligned offsets of a 512 MiB arena | 16x64/k_proj/1 | 7.55–7.63 | 1.1 % |
| activation placement (24 offsets) | the input, quantized activation, scales and output carved at seeded 4 KiB-aligned offsets of a 512 MiB arena | stock_wB_ko/gate_proj/16 | 31.26–32.18 | 2.9 % |

## Paired allocation diagnostics (flipquant worker against the RaZeR harness, no per-cell checks)

| record | phase | segments holding the copies: fq / rz | 256x64 k_proj T=1 | T=16 | largest |dev| |
|---|---|---|---|---|---:|
| alloc_llama8b.json | as_is | 443 (largest 0.44 GiB) / 532 (largest 0.03 GiB) | 7.58 / 7.55 | 7.55 / 7.52 | 0.8 % |
| alloc_llama8b.json | swapped | 536 (largest 0.03 GiB) / 263 (largest 16.00 GiB) | 7.58 / 7.57 | 7.52 / 7.52 | 0.5 % |
| alloc_llama8b.json | both_fresh | 536 (largest 0.03 GiB) / 535 (largest 0.03 GiB) | 7.62 / 7.55 | 7.55 / 7.52 | 0.8 % |
| alloc_full_localize.json | as_is | 1124 (largest 0.44 GiB) / 1363 (largest 0.03 GiB) | 7.62 / 7.58 | 7.55 / 7.55 | 1.4 % |
| alloc_full_localize.json | fq_copies | 1365 (largest 0.03 GiB) / 1363 (largest 0.03 GiB) | 7.58 / 7.58 | 7.54 / 7.52 | 0.5 % |
| alloc_full_localize.json | fq_fresh | 1374 (largest 0.03 GiB) / 1363 (largest 0.03 GiB) | 7.58 / 7.55 | 7.52 / 7.52 | 0.4 % |
| alloc_full_localize.json | both_fresh | 1374 (largest 0.03 GiB) / 1374 (largest 0.03 GiB) | 7.62 / 7.55 | 7.54 / 7.52 | 0.8 % |
