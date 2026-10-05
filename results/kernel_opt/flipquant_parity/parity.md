# flipquant GEMM parity: tables

Per-forward GEMM time of a policy at T: the sum over the projections of (modules) x (the median GEMM time of the timed module), in µs. RaZeR and flipquant are the values of llama8b; then the flipquant / RaZeR ratio of every flipquant / RaZeR-harness run, and the ratio of every other pairing (first slot / second slot): the A/A control (two RaZeR harness processes), flipquant against RaZeR's own model path, RaZeR's model path against its harness.

## 8x64 map, mixed_wB_ko

| T | RaZeR | flipquant | llama8b: fq/rz | llama8b_r2: fq/rz | llama8b_r3: fq/rz | llama8b_nochk: fq/rz | llama8b_fqV: fq/rz | llama8b_rzK: fq/rz | llama8b_aa: RaZeR harness / RaZeR harness | llama8b_rzmrz: RaZeR model path / RaZeR harness | llama8b_fqrzm: flipquant / RaZeR model path |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,322.9 | 3,322.4 | 0.9998 | 0.9995 | 0.9991 | 0.9981 | 1.0001 | 1.0000 | 0.9986 | 1.0005 | 0.9962 |
| 16 | 3,313.1 | 3,319.3 | 1.0018 | 1.0023 | 1.0023 | 1.0009 | 1.0016 | 1.0012 | 1.0012 | 1.0014 | 0.9995 |
| 128 | 3,567.6 | 3,567.6 | 1.0000 | 1.0004 | 1.0011 | 0.9993 | 1.0003 | 1.0006 | 0.9996 | 1.0020 | 0.9997 |
| 512 | 7,693.3 | 7,694.3 | 1.0001 | 1.0028 | 1.0011 | 0.9996 | 0.9980 | 1.0019 | 1.0014 | 0.9998 | 1.0011 |
| 2048 | 25,143.6 | 25,142.6 | 1.0000 | 1.0028 | 0.9999 | 1.0007 | 1.0019 | 1.0027 | 1.0022 | 1.0010 | 0.9996 |
| 8192 | 94,793.2 | 94,688.1 | 0.9989 | 1.0006 | 0.9996 | 1.0008 | 1.0001 | 0.9995 | 1.0016 | 0.9997 | 0.9985 |

## 16x64 map, mixed_ko

| T | RaZeR | flipquant | llama8b: fq/rz | llama8b_r2: fq/rz | llama8b_r3: fq/rz | llama8b_nochk: fq/rz | llama8b_fqV: fq/rz | llama8b_rzK: fq/rz | llama8b_aa: RaZeR harness / RaZeR harness | llama8b_rzmrz: RaZeR model path / RaZeR harness | llama8b_fqrzm: flipquant / RaZeR model path |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,307.5 | 3,308.6 | 1.0003 | 0.9991 | 1.0005 | 0.9938 | 0.9998 | 0.9983 | 1.0002 | 1.0023 | 1.0014 |
| 16 | 3,298.2 | 3,296.3 | 0.9994 | 0.9988 | 1.0005 | 1.0003 | 0.9991 | 1.0008 | 0.9978 | 1.0029 | 1.0008 |
| 128 | 3,592.7 | 3,592.1 | 0.9998 | 0.9980 | 0.9979 | 0.9985 | 0.9997 | 0.9989 | 0.9997 | 1.0017 | 0.9997 |
| 512 | 7,224.3 | 7,202.3 | 0.9970 | 1.0013 | 0.9971 | 1.0009 | 0.9951 | 0.9967 | 1.0028 | 1.0014 | 0.9998 |
| 2048 | 23,806.3 | 23,778.2 | 0.9988 | 1.0046 | 1.0004 | 1.0001 | 1.0008 | 1.0025 | 1.0009 | 1.0010 | 1.0005 |
| 8192 | 89,201.6 | 89,052.1 | 0.9983 | 0.9997 | 0.9984 | 1.0016 | 0.9995 | 0.9988 | 1.0002 | 0.9993 | 0.9990 |

## 256x64 map, mixed256_ko

| T | RaZeR | flipquant | llama8b: fq/rz | llama8b_r2: fq/rz | llama8b_r3: fq/rz | llama8b_nochk: fq/rz | llama8b_fqV: fq/rz | llama8b_rzK: fq/rz | llama8b_aa: RaZeR harness / RaZeR harness | llama8b_rzmrz: RaZeR model path / RaZeR harness | llama8b_fqrzm: flipquant / RaZeR model path |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,285.0 | 3,216.9 | 0.9793 | 0.9788 | 0.9794 | 0.9989 | 0.9785 | 0.9811 | 1.0008 | 1.0013 | 0.9790 |
| 16 | 3,275.7 | 3,286.5 | 1.0033 | 1.0023 | 1.0027 | 1.0003 | 1.0046 | 1.0038 | 1.0011 | 1.0006 | 0.9991 |
| 128 | 3,531.8 | 3,528.6 | 0.9991 | 0.9993 | 0.9987 | 1.0007 | 0.9984 | 0.9983 | 1.0006 | 1.0009 | 0.9973 |
| 512 | 7,096.2 | 7,098.8 | 1.0004 | 1.0050 | 1.0050 | 1.0004 | 0.9996 | 1.0012 | 1.0019 | 1.0014 | 1.0049 |
| 2048 | 23,855.5 | 23,836.5 | 0.9992 | 1.0070 | 0.9999 | 0.9996 | 1.0033 | 1.0014 | 1.0018 | 1.0010 | 1.0018 |
| 8192 | 89,384.9 | 89,388.0 | 1.0000 | 0.9990 | 1.0008 | 1.0001 | 1.0005 | 0.9995 | 1.0000 | 1.0006 | 1.0001 |

## FourOverSix, stock_ko

| T | RaZeR | flipquant | llama8b: fq/rz | llama8b_r2: fq/rz | llama8b_r3: fq/rz | llama8b_nochk: fq/rz | llama8b_fqV: fq/rz | llama8b_rzK: fq/rz | llama8b_aa: RaZeR harness / RaZeR harness | llama8b_rzmrz: RaZeR model path / RaZeR harness | llama8b_fqrzm: flipquant / RaZeR model path |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,277.3 | 3,273.2 | 0.9988 | 0.9983 | 0.9992 | 0.9769 | 0.9998 | 0.9981 | 1.0011 | 1.0027 | 1.0005 |
| 16 | 3,259.9 | 3,260.9 | 1.0003 | 1.0008 | 1.0013 | 1.0047 | 1.0013 | 1.0011 | 1.0009 | 1.0035 | 1.0011 |
| 128 | 3,502.1 | 3,498.5 | 0.9990 | 1.0001 | 1.0003 | 0.9997 | 0.9996 | 0.9996 | 1.0009 | 1.0028 | 0.9993 |
| 512 | 6,952.4 | 6,938.0 | 0.9979 | 0.9996 | 0.9986 | 1.0009 | 0.9933 | 0.9988 | 1.0002 | 0.9980 | 1.0010 |
| 2048 | 23,453.0 | 23,462.8 | 1.0004 | 1.0057 | 1.0008 | 1.0000 | 1.0031 | 1.0016 | 1.0004 | 1.0005 | 1.0020 |
| 8192 | 87,822.2 | 87,727.6 | 0.9989 | 1.0010 | 0.9997 | 1.0005 | 0.9997 | 0.9989 | 1.0015 | 1.0015 | 0.9995 |

## FourOverSix, stock_wB_ko (weights on B)

| T | RaZeR | flipquant | llama8b: fq/rz | llama8b_r2: fq/rz | llama8b_r3: fq/rz | llama8b_nochk: fq/rz | llama8b_fqV: fq/rz | llama8b_rzK: fq/rz | llama8b_aa: RaZeR harness / RaZeR harness | llama8b_rzmrz: RaZeR model path / RaZeR harness | llama8b_fqrzm: flipquant / RaZeR model path |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5,614.0 | 5,619.7 | 1.0010 | 1.0014 | 1.0015 | 1.0005 | 1.0017 | 1.0018 | 1.0007 | 1.0005 | 1.0009 |
| 16 | 5,432.2 | 5,465.0 | 1.0060 | 1.0057 | 1.0057 | 1.0050 | 1.0053 | 1.0057 | 1.0006 | 1.0007 | 1.0050 |
| 128 | 4,961.8 | 4,960.2 | 0.9997 | 0.9997 | 0.9992 | 1.0008 | 0.9992 | 0.9992 | 0.9997 | 0.9999 | 0.9988 |
| 512 | 7,327.2 | 7,300.5 | 0.9964 | 0.9968 | 0.9964 | 0.9982 | 0.9936 | 0.9968 | 0.9987 | 0.9966 | 0.9979 |
| 2048 | 23,413.1 | 23,385.5 | 0.9988 | 1.0029 | 1.0002 | 0.9991 | 1.0021 | 1.0025 | 1.0026 | 1.0009 | 0.9992 |
| 8192 | 87,751.6 | 87,632.8 | 0.9986 | 0.9993 | 0.9996 | 1.0003 | 1.0004 | 1.0004 | 1.0015 | 0.9996 | 0.9992 |

## Criteria per run

| criterion | llama8b (flipquant / RaZeR harness) | llama8b_r2 (flipquant / RaZeR harness) | llama8b_r3 (flipquant / RaZeR harness) | llama8b_nochk (flipquant / RaZeR harness) | llama8b_aa (RaZeR harness / RaZeR harness) | llama8b_rzmrz (RaZeR model path / RaZeR harness) | llama8b_fqrzm (flipquant / RaZeR model path) | llama8b_fqV (flipquant / RaZeR harness) | llama8b_rzK (flipquant / RaZeR harness) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| operands equal bit for bit (every timed module) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| outputs equal bit for bit (fq vs rz; isolated path vs the fused forward) | pass | pass | pass | not checked | pass | pass | pass | pass | pass |
| same kernel selection: build, width, scheduler row, defines, SASS hash, GEMM kernel name | pass | pass | pass | not checked | pass | pass | pass | pass | pass |
| same launch in every round (build, width, scheduler row, GEMM and quantizer kernel names) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| the tile tables have the same content | pass | pass | pass | not checked | pass | pass | pass | pass | pass |
| same activation quantizer (kernel name, build, mode) | pass | pass | pass | not checked | pass | pass | pass | pass | pass |
| same binaries (SASS, device code, host code; loaded bytes up to the compilation id) | pass | pass | pass | not checked | pass | pass | pass | pass | pass |
| per-forward GEMM sum within +-1 % for every policy and T | **FAIL** | **FAIL** | **FAIL** | **FAIL** | pass | pass | **FAIL** | **FAIL** | **FAIL** |
| no cell slower in every round by more than 2 % | **FAIL** | **FAIL** | **FAIL** | **FAIL** | pass | pass | **FAIL** | **FAIL** | **FAIL** |

## Cell-level spread per run

| run | cells | max abs deviation | cells beyond 1 % | beyond 2 % | slower (fq) by > 2 % in every round | faster by > 2 % in every round | per-forward sums: max abs deviation |
|---|---:|---:|---:|---:|---|---|---:|
| llama8b (flipquant / RaZeR harness) | 210 | 5.3 % | 15 | 8 | 256x64 k_proj T=16 (1.026), 256x64 v_proj T=16 (1.032) | 256x64 q_proj T=1 (0.959), 256x64 k_proj T=1 (0.951), 256x64 v_proj T=1 (0.947), 256x64 o_proj T=1 (0.962) | 2.07 % |
| llama8b_r2 (flipquant / RaZeR harness) | 210 | 5.5 % | 18 | 6 | 256x64 k_proj T=16 (1.030), 256x64 v_proj T=16 (1.034) | 256x64 q_proj T=1 (0.959), 256x64 k_proj T=1 (0.953), 256x64 v_proj T=1 (0.945), 256x64 o_proj T=1 (0.961) | 2.12 % |
| llama8b_r3 (flipquant / RaZeR harness) | 210 | 4.9 % | 17 | 7 | 256x64 k_proj T=16 (1.028), 256x64 v_proj T=16 (1.030) | 256x64 q_proj T=1 (0.962), 256x64 k_proj T=1 (0.954), 256x64 v_proj T=1 (0.951), 256x64 o_proj T=1 (0.959) | 2.06 % |
| llama8b_nochk (flipquant / RaZeR harness) | 210 | 4.7 % | 25 | 8 | 16x64 k_proj T=16 (1.029), stock_ko k_proj T=16 (1.030), stock_ko v_proj T=16 (1.034) | stock_ko q_proj T=1 (0.968), stock_ko k_proj T=1 (0.958), 16x64 v_proj T=1 (0.955), stock_ko v_proj T=1 (0.953) | 2.31 % |
| llama8b_aa (RaZeR harness / RaZeR harness) | 210 | 2.7 % | 5 | 1 | — | — | 0.28 % |
| llama8b_rzmrz (RaZeR model path / RaZeR harness) | 210 | 1.0 % | 0 | 0 | — | — | 0.35 % |
| llama8b_fqrzm (flipquant / RaZeR model path) | 210 | 4.6 % | 16 | 8 | 256x64 k_proj T=16 (1.028), 256x64 v_proj T=16 (1.034) | 256x64 q_proj T=1 (0.970), 256x64 k_proj T=1 (0.954), 256x64 v_proj T=1 (0.954), 256x64 o_proj T=1 (0.972) | 2.10 % |
| llama8b_fqV (flipquant / RaZeR harness) | 210 | 5.3 % | 19 | 7 | 256x64 k_proj T=16 (1.030), 256x64 v_proj T=16 (1.032) | 256x64 q_proj T=1 (0.962), 256x64 k_proj T=1 (0.947), 256x64 v_proj T=1 (0.947), 256x64 o_proj T=1 (0.961) | 2.15 % |
| llama8b_rzK (flipquant / RaZeR harness) | 210 | 5.0 % | 16 | 7 | 256x64 k_proj T=16 (1.025), 256x64 v_proj T=16 (1.030) | 256x64 q_proj T=1 (0.966), 256x64 k_proj T=1 (0.950), 256x64 v_proj T=1 (0.954), 256x64 o_proj T=1 (0.962) | 1.89 % |

