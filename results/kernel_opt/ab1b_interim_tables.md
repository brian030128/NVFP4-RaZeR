## M1: per-forward GEMM time (isolated launches, cold weights; µs)

### Llama-3.1-8B

Checks: bitwise 1008 ok=True, launch counts ok=True, other processes 0.

| T | stock_wA | stock_wB | 8x64 before (typ.) | 8x64 after (typ.) | after vs before (typ.) [round range] | after vs before (worst) | vs stock_wA: before → after (typ.) | vs stock_wB: before → after (typ.) |
|---|---|---|---|---|---|---|---|---|

Against optimization 1's set (auto_wB1, typical tags):

| T | optimization 1 | after | after vs optimization 1 [round range] | vs stock_wA: optimization 1 → after |
|---|---|---|---|---|
| 1 | 3337.7 | 3344.8 | +0.2 % [+0.2, +0.3] | +1.7 % → +1.9 % |
| 4 | 3331.0 | 3336.2 | +0.2 % [+0.0, +0.4] | +1.6 % → +1.7 % |
| 16 | 3330.1 | 3339.2 | +0.3 % [+0.1, +0.4] | +1.8 % → +2.1 % |
| 32 | 3380.2 | 3378.7 | -0.0 % [-0.2, +0.0] | +2.9 % → +2.9 % |
| 64 | 3429.3 | 3436.5 | +0.2 % [-0.1, +0.4] | +2.6 % → +2.8 % |
| 128 | 3669.0 | 3668.4 | -0.0 % [-0.1, +0.0] | +3.9 % → +3.9 % |
| 256 | 5719.5 | 4540.4 | -20.6 % [-20.7, -20.5] | +26.2 % → +0.2 % |
| 512 | 7992.8 | 7967.7 | -0.3 % [-1.0, +0.1] | +14.5 % → +14.1 % |
| 1024 | 14693.2 | 14338.0 | -2.4 % [-2.6, -2.2] | +8.4 % → +5.7 % |
| 2048 | 25625.3 | 25607.0 | -0.1 % [-0.2, +0.0] | +9.6 % → +9.5 % |
| 4096 | 50223.2 | 50164.8 | -0.1 % [-0.5, -0.0] | +9.0 % → +8.8 % |
| 8192 | 96750.9 | 96635.7 | -0.1 % [-0.2, -0.0] | +10.3 % → +10.2 % |

Against the original n8k64_wB:

| T | stock_wA | stock_wB | 8x64 before (typ.) | 8x64 after (typ.) | after vs before (typ.) [round range] | after vs before (worst) | vs stock_wA: before → after (typ.) | vs stock_wB: before → after (typ.) |
|---|---|---|---|---|---|---|---|---|
| 1 | 3281.9 | 5703.6 | 6088.7 | 3344.8 | -45.1 % [-45.1, -45.0] | -45.3 % | +85.5 % → +1.9 % | +6.8 % → -41.4 % |
| 4 | 3279.3 | 5669.8 | 6058.5 | 3336.2 | -44.9 % [-45.0, -44.9] | -45.1 % | +84.7 % → +1.7 % | +6.9 % → -41.2 % |
| 16 | 3271.2 | 5493.7 | 5888.5 | 3339.2 | -43.3 % [-43.4, -43.2] | -43.5 % | +80.0 % → +2.1 % | +7.2 % → -39.2 % |
| 32 | 3285.0 | 5291.0 | 5690.3 | 3378.7 | -40.6 % [-40.7, -40.6] | -40.7 % | +73.2 % → +2.9 % | +7.5 % → -36.1 % |
| 64 | 3341.8 | 5000.7 | 5446.7 | 3436.5 | -36.9 % [-37.0, -36.8] | -37.5 % | +63.0 % → +2.8 % | +8.9 % → -31.3 % |
| 128 | 3531.2 | 4988.9 | 5462.5 | 3668.4 | -32.8 % [-32.9, -32.8] | -33.2 % | +54.7 % → +3.9 % | +9.5 % → -26.5 % |
| 256 | 4531.1 | 5745.1 | 6348.7 | 4540.4 | -28.5 % [-28.6, -28.4] | -29.1 % | +40.1 % → +0.2 % | +10.5 % → -21.0 % |
| 512 | 6981.6 | 7352.3 | 8517.0 | 7967.7 | -6.4 % [-6.7, -6.2] | -6.6 % | +22.0 % → +14.1 % | +15.8 % → +8.4 % |
| 1024 | 13560.2 | 13616.0 | 14772.6 | 14338.0 | -2.9 % [-3.1, -2.7] | -2.9 % | +8.9 % → +5.7 % | +8.5 % → +5.3 % |
| 2048 | 23379.8 | 23249.8 | 25604.9 | 25607.0 | +0.0 % [-0.2, +0.2] | +0.2 % | +9.5 % → +9.5 % | +10.1 % → +10.1 % |
| 4096 | 46093.0 | 45749.9 | 50093.2 | 50164.8 | +0.1 % [-0.1, +0.2] | +0.2 % | +8.7 % → +8.8 % | +9.5 % → +9.7 % |
| 8192 | 87721.2 | 87395.1 | 96694.5 | 96635.7 | -0.1 % [-0.1, +0.0] | -0.0 % | +10.2 % → +10.2 % | +10.6 % → +10.6 % |

Width chosen (build per projection, typical tags):

- T = 1: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 4: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 16: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 32: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 64: q_proj n8k64_wB_m32, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m32, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB_m32
- T = 128: q_proj n8k64_wB_m64, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB_m64, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB_m64
- T = 256: q_proj n8k64_wB_n64, k_proj n8k64_wB_m32, v_proj n8k64_wB_m32, o_proj n8k64_wB_n64, gate_proj n8k64_wB_n64, up_proj n8k64_wB_n64, down_proj n8k64_wB_n64
- T = 512: q_proj n8k64_wB, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 1024: q_proj n8k64_wB, k_proj n8k64_wB_n64, v_proj n8k64_wB_n64, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 2048: q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 4096: q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 8192: q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB

### Mistral-7B-v0.3

Checks: bitwise 1008 ok=True, launch counts ok=True, other processes 0.

| T | stock_wA | stock_wB | 8x64 before (typ.) | 8x64 after (typ.) | after vs before (typ.) [round range] | after vs before (worst) | vs stock_wA: before → after (typ.) | vs stock_wB: before → after (typ.) |
|---|---|---|---|---|---|---|---|---|

Against optimization 1's set (auto_wB1, typical tags):

| T | optimization 1 | after | after vs optimization 1 [round range] | vs stock_wA: optimization 1 → after |
|---|---|---|---|---|
| 1 | 3332.1 | 3343.8 | +0.4 % [+0.2, +0.5] | +1.4 % → +1.8 % |
| 4 | 3335.7 | 3340.2 | +0.1 % [+0.1, +0.2] | +1.7 % → +1.8 % |
| 16 | 3334.1 | 3341.3 | +0.2 % [+0.2, +0.3] | +1.9 % → +2.1 % |
| 32 | 3379.7 | 3381.2 | +0.0 % [-0.1, +0.2] | +2.7 % → +2.8 % |
| 64 | 3431.4 | 3433.4 | +0.1 % [-0.0, +0.1] | +2.6 % → +2.7 % |
| 128 | 3670.0 | 3673.0 | +0.1 % [-0.0, +0.2] | +3.9 % → +3.9 % |
| 256 | 5719.5 | 4539.3 | -20.6 % [-20.7, -20.6] | +26.2 % → +0.2 % |
| 512 | 8034.3 | 8038.3 | +0.1 % [-0.2, +0.4] | +14.7 % → +14.7 % |
| 1024 | 14747.5 | 14391.2 | -2.4 % [-2.5, -2.4] | +8.5 % → +5.9 % |
| 2048 | 25705.3 | 25725.2 | +0.1 % [+0.0, +0.4] | +9.5 % → +9.6 % |
| 4096 | 50204.7 | 50447.0 | +0.5 % [-1.0, +0.9] | +8.9 % → +9.4 % |
| 8192 | 96767.7 | 96834.8 | +0.1 % [-0.1, +0.2] | +10.0 % → +10.1 % |

Against the original n8k64_wB:

| T | stock_wA | stock_wB | 8x64 before (typ.) | 8x64 after (typ.) | after vs before (typ.) [round range] | after vs before (worst) | vs stock_wA: before → after (typ.) | vs stock_wB: before → after (typ.) |
|---|---|---|---|---|---|---|---|---|
| 1 | 3285.9 | 5707.8 | 6107.1 | 3343.8 | -45.2 % [-45.3, -45.2] | -45.3 % | +85.9 % → +1.8 % | +7.0 % → -41.4 % |
| 4 | 3280.9 | 5672.4 | 6066.7 | 3340.2 | -44.9 % [-45.0, -44.9] | -45.1 % | +84.9 % → +1.8 % | +7.0 % → -41.1 % |
| 16 | 3271.1 | 5499.9 | 5901.3 | 3341.3 | -43.4 % [-43.4, -43.4] | -43.5 % | +80.4 % → +2.1 % | +7.3 % → -39.2 % |
| 32 | 3290.6 | 5296.1 | 5693.4 | 3381.2 | -40.6 % [-40.6, -40.6] | -40.8 % | +73.0 % → +2.8 % | +7.5 % → -36.2 % |
| 64 | 3343.3 | 5007.3 | 5449.1 | 3433.4 | -37.0 % [-37.1, -36.9] | -37.1 % | +63.0 % → +2.7 % | +8.8 % → -31.4 % |
| 128 | 3533.8 | 4993.5 | 5467.6 | 3673.0 | -32.8 % [-32.9, -32.8] | -33.1 % | +54.7 % → +3.9 % | +9.5 % → -26.4 % |
| 256 | 4532.2 | 5746.7 | 6344.2 | 4539.3 | -28.4 % [-28.5, -28.4] | -28.9 % | +40.0 % → +0.2 % | +10.4 % → -21.0 % |
| 512 | 7007.2 | 7368.7 | 8561.6 | 8038.3 | -6.1 % [-6.2, -5.7] | -6.0 % | +22.2 % → +14.7 % | +16.2 % → +9.1 % |
| 1024 | 13588.4 | 13648.9 | 14805.9 | 14391.2 | -2.8 % [-3.0, -2.6] | -2.7 % | +9.0 % → +5.9 % | +8.5 % → +5.4 % |
| 2048 | 23481.6 | 23305.0 | 25687.9 | 25725.2 | +0.1 % [+0.0, +0.4] | +0.1 % | +9.4 % → +9.6 % | +10.2 % → +10.4 % |
| 4096 | 46110.4 | 45682.3 | 50166.9 | 50447.0 | +0.6 % [-0.1, +0.4] | -0.5 % | +8.8 % → +9.4 % | +9.8 % → +10.4 % |
| 8192 | 87931.7 | 87493.0 | 96740.6 | 96834.8 | +0.1 % [-0.1, +0.2] | +0.2 % | +10.0 % → +10.1 % | +10.6 % → +10.7 % |

Width chosen (build per projection, typical tags):

- T = 1: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 4: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 16: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 32: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 64: q_proj n8k64_wB_m32, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m32, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB_m32
- T = 128: q_proj n8k64_wB_m64, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB_m64, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB_m64
- T = 256: q_proj n8k64_wB_n64, k_proj n8k64_wB_m32, v_proj n8k64_wB_m32, o_proj n8k64_wB_n64, gate_proj n8k64_wB_n64, up_proj n8k64_wB_n64, down_proj n8k64_wB_n64
- T = 512: q_proj n8k64_wB, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 1024: q_proj n8k64_wB, k_proj n8k64_wB_n64, v_proj n8k64_wB_n64, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 2048: q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 4096: q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 8192: q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB

### Phi-4

Checks: bitwise 576 ok=True, launch counts ok=True, other processes 0.

| T | stock_wA | stock_wB | 8x64 before (typ.) | 8x64 after (typ.) | after vs before (typ.) [round range] | after vs before (worst) | vs stock_wA: before → after (typ.) | vs stock_wB: before → after (typ.) |
|---|---|---|---|---|---|---|---|---|

Against optimization 1's set (auto_wB1, typical tags):

| T | optimization 1 | after | after vs optimization 1 [round range] | vs stock_wA: optimization 1 → after |
|---|---|---|---|---|
| 1 | 5365.7 | 5359.3 | -0.1 % [-0.2, +0.0] | +0.5 % → +0.4 % |
| 4 | 5361.9 | 5358.7 | -0.1 % [-0.1, +0.1] | +0.6 % → +0.5 % |
| 16 | 5365.1 | 5359.4 | -0.1 % [-0.1, +0.0] | +1.0 % → +0.9 % |
| 32 | 5385.0 | 5379.2 | -0.1 % [-0.1, -0.1] | +0.6 % → +0.5 % |
| 64 | 5449.6 | 5448.3 | -0.0 % [-0.1, +0.1] | +1.1 % → +1.1 % |
| 128 | 5939.2 | 5671.1 | -4.5 % [-4.6, -4.5] | +5.3 % → +0.5 % |
| 256 | 8252.2 | 7548.2 | -8.5 % [-8.8, -8.4] | +17.7 % → +7.6 % |
| 512 | 13331.8 | 13221.8 | -0.8 % [-1.3, -0.5] | +11.2 % → +10.3 % |
| 1024 | 25161.6 | 25178.8 | +0.1 % [-0.2, +0.4] | +9.0 % → +9.0 % |
| 2048 | 49516.8 | 49577.6 | +0.1 % [-0.1, +0.4] | +8.4 % → +8.6 % |
| 4096 | 93539.8 | 93619.1 | +0.1 % [+0.0, +0.2] | +9.3 % → +9.4 % |
| 8192 | 184897.3 | 184872.9 | -0.0 % [-0.0, +0.1] | +9.7 % → +9.6 % |

Against the original n8k64_wB:

| T | stock_wA | stock_wB | 8x64 before (typ.) | 8x64 after (typ.) | after vs before (typ.) [round range] | after vs before (worst) | vs stock_wA: before → after (typ.) | vs stock_wB: before → after (typ.) |
|---|---|---|---|---|---|---|---|---|
| 1 | 5339.5 | 8145.2 | 8475.5 | 5359.3 | -36.8 % [-36.8, -36.6] | -36.6 % | +58.7 % → +0.4 % | +4.1 % → -34.2 % |
| 4 | 5331.2 | 7925.1 | 8267.4 | 5358.7 | -35.2 % [-35.3, -35.2] | -34.9 % | +55.1 % → +0.5 % | +4.3 % → -32.4 % |
| 16 | 5313.9 | 7787.5 | 8056.3 | 5359.4 | -33.5 % [-33.5, -33.4] | -33.6 % | +51.6 % → +0.9 % | +3.5 % → -31.2 % |
| 32 | 5354.8 | 7333.1 | 7584.6 | 5379.2 | -29.1 % [-29.1, -29.0] | -29.2 % | +41.6 % → +0.5 % | +3.4 % → -26.6 % |
| 64 | 5391.4 | 6669.4 | 7091.2 | 5448.3 | -23.2 % [-23.2, -23.2] | -23.2 % | +31.5 % → +1.1 % | +6.3 % → -18.3 % |
| 128 | 5641.6 | 6682.9 | 7147.5 | 5671.1 | -20.7 % [-20.7, -20.6] | -20.6 % | +26.7 % → +0.5 % | +7.0 % → -15.1 % |
| 256 | 7012.5 | 7629.5 | 8254.1 | 7548.2 | -8.6 % [-9.0, -8.3] | -9.0 % | +17.7 % → +7.6 % | +8.2 % → -1.1 % |
| 512 | 11989.1 | 12026.2 | 13330.5 | 13221.8 | -0.8 % [-1.3, -0.6] | -1.3 % | +11.2 % → +10.3 % | +10.8 % → +9.9 % |
| 1024 | 23093.8 | 23012.5 | 25166.1 | 25178.8 | +0.1 % [-0.2, +0.4] | -0.0 % | +9.0 % → +9.0 % | +9.4 % → +9.4 % |
| 2048 | 45660.7 | 45467.5 | 49538.0 | 49577.6 | +0.1 % [+0.0, +0.1] | +0.2 % | +8.5 % → +8.6 % | +9.0 % → +9.0 % |
| 4096 | 85586.5 | 85250.5 | 93538.5 | 93619.1 | +0.1 % [-0.1, +0.3] | +0.0 % | +9.3 % → +9.4 % | +9.7 % → +9.8 % |
| 8192 | 168610.6 | 167846.4 | 184750.0 | 184872.9 | +0.1 % [-0.0, +0.2] | +0.1 % | +9.6 % → +9.6 % | +10.1 % → +10.1 % |

Width chosen (build per projection, typical tags):

- T = 1: o_proj n8k64_wB_m16, qkv_proj n8k64_wB_m16, gate_up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 4: o_proj n8k64_wB_m16, qkv_proj n8k64_wB_m16, gate_up_proj n8k64_wB_m32, down_proj n8k64_wB_m16
- T = 16: o_proj n8k64_wB_m16, qkv_proj n8k64_wB_m32, gate_up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 32: o_proj n8k64_wB_m16, qkv_proj n8k64_wB_m32, gate_up_proj n8k64_wB_m32, down_proj n8k64_wB_m16
- T = 64: o_proj n8k64_wB_m32, qkv_proj n8k64_wB_m64, gate_up_proj n8k64_wB_m64, down_proj n8k64_wB_m32
- T = 128: o_proj n8k64_wB_m64, qkv_proj n8k64_wB_n64, gate_up_proj n8k64_wB_n64, down_proj n8k64_wB_m64
- T = 256: o_proj n8k64_wB_n64, qkv_proj n8k64_wB, gate_up_proj n8k64_wB, down_proj n8k64_wB_n64
- T = 512: o_proj n8k64_wB, qkv_proj n8k64_wB_n64, gate_up_proj n8k64_wB, down_proj n8k64_wB
- T = 1024: o_proj n8k64_wB, qkv_proj n8k64_wB, gate_up_proj n8k64_wB, down_proj n8k64_wB
- T = 2048: o_proj n8k64_wB, qkv_proj n8k64_wB, gate_up_proj n8k64_wB, down_proj n8k64_wB
- T = 4096: o_proj n8k64_wB, qkv_proj n8k64_wB, gate_up_proj n8k64_wB, down_proj n8k64_wB
- T = 8192: o_proj n8k64_wB, qkv_proj n8k64_wB, gate_up_proj n8k64_wB, down_proj n8k64_wB

### Qwen3.8-27B

Checks: bitwise 1728 ok=True, launch counts ok=True, other processes 0.

| T | stock_wA | stock_wB | 8x64 before (typ.) | 8x64 after (typ.) | after vs before (typ.) [round range] | after vs before (worst) | vs stock_wA: before → after (typ.) | vs stock_wB: before → after (typ.) |
|---|---|---|---|---|---|---|---|---|

Against optimization 1's set (auto_wB1, typical tags):

| T | optimization 1 | after | after vs optimization 1 [round range] | vs stock_wA: optimization 1 → after |
|---|---|---|---|---|
| 1 | 10730.7 | 10699.7 | -0.3 % [-0.3, -0.2] | +0.8 % → +0.5 % |
| 4 | 10680.3 | 10701.2 | +0.2 % [+0.1, +0.3] | +0.2 % → +0.4 % |
| 16 | 10652.7 | 10658.6 | +0.1 % [+0.0, +0.2] | +0.4 % → +0.5 % |
| 32 | 10725.4 | 10736.4 | +0.1 % [+0.1, +0.1] | +0.6 % → +0.7 % |
| 64 | 10851.4 | 10866.7 | +0.1 % [+0.1, +0.2] | +1.1 % → +1.2 % |
| 128 | 11654.4 | 11258.7 | -3.4 % [-3.4, -3.4] | +4.8 % → +1.3 % |
| 256 | 16477.2 | 15121.4 | -8.2 % [-8.3, -8.2] | +18.5 % → +8.7 % |
| 512 | 25747.9 | 25447.4 | -1.2 % [-1.6, -0.9] | +9.5 % → +8.2 % |
| 1024 | 48010.1 | 47728.9 | -0.6 % [-0.6, -0.4] | +8.6 % → +8.0 % |
| 2048 | 89615.7 | 89794.3 | +0.2 % [+0.1, +0.3] | +7.3 % → +7.5 % |
| 4096 | 169479.5 | 169479.0 | -0.0 % [-0.0, +0.1] | +8.4 % → +8.4 % |
| 8192 | 331461.2 | 331536.7 | +0.0 % [+0.0, +0.0] | +9.3 % → +9.3 % |

Against the original n8k64_wB:

| T | stock_wA | stock_wB | 8x64 before (typ.) | 8x64 after (typ.) | after vs before (typ.) [round range] | after vs before (worst) | vs stock_wA: before → after (typ.) | vs stock_wB: before → after (typ.) |
|---|---|---|---|---|---|---|---|---|
| 1 | 10650.6 | 16907.8 | 17782.6 | 10699.7 | -39.8 % [-39.9, -39.7] | -39.7 % | +67.0 % → +0.5 % | +5.2 % → -36.7 % |
| 4 | 10656.8 | 16916.5 | 17799.9 | 10701.2 | -39.9 % [-40.0, -39.8] | -39.9 % | +67.0 % → +0.4 % | +5.2 % → -36.7 % |
| 16 | 10606.4 | 16186.3 | 17121.5 | 10658.6 | -37.7 % [-37.8, -37.7] | -37.8 % | +61.4 % → +0.5 % | +5.8 % → -34.2 % |
| 32 | 10660.6 | 15464.2 | 16385.7 | 10736.4 | -34.5 % [-34.5, -34.4] | -34.6 % | +53.7 % → +0.7 % | +6.0 % → -30.6 % |
| 64 | 10734.8 | 14384.4 | 15355.6 | 10866.7 | -29.2 % [-29.3, -29.2] | -29.6 % | +43.0 % → +1.2 % | +6.8 % → -24.5 % |
| 128 | 11119.6 | 14393.0 | 15374.3 | 11258.7 | -26.8 % [-26.9, -26.7] | -27.2 % | +38.3 % → +1.3 % | +6.8 % → -21.8 % |
| 256 | 13907.3 | 16347.8 | 18043.9 | 15121.4 | -16.2 % [-16.5, -16.1] | -16.4 % | +29.7 % → +8.7 % | +10.4 % → -7.5 % |
| 512 | 23520.5 | 24807.4 | 27271.1 | 25447.4 | -6.7 % [-6.9, -6.5] | -6.7 % | +15.9 % → +8.2 % | +9.9 % → +2.6 % |
| 1024 | 44200.1 | 44988.2 | 49183.7 | 47728.9 | -3.0 % [-3.1, -2.8] | -2.8 % | +11.3 % → +8.0 % | +9.3 % → +6.1 % |
| 2048 | 83507.5 | 83722.4 | 90809.0 | 89794.3 | -1.1 % [-1.3, -0.9] | -1.1 % | +8.7 % → +7.5 % | +8.5 % → +7.3 % |
| 4096 | 156345.8 | 155567.0 | 170609.0 | 169479.0 | -0.7 % [-0.8, -0.5] | -0.7 % | +9.1 % → +8.4 % | +9.7 % → +8.9 % |
| 8192 | 303350.2 | 302301.9 | 331987.1 | 331536.7 | -0.1 % [-0.2, -0.1] | -0.1 % | +9.4 % → +9.3 % | +9.8 % → +9.7 % |

Width chosen (build per projection, typical tags):

- T = 1: out_proj n8k64_wB_m16, in_proj_qkv n8k64_wB_m16, in_proj_z n8k64_wB_m16, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16, q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16
- T = 4: out_proj n8k64_wB_m16, in_proj_qkv n8k64_wB_m16, in_proj_z n8k64_wB_m16, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16, q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16
- T = 16: out_proj n8k64_wB_m16, in_proj_qkv n8k64_wB_m16, in_proj_z n8k64_wB_m16, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16, q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16
- T = 32: out_proj n8k64_wB_m16, in_proj_qkv n8k64_wB_m32, in_proj_z n8k64_wB_m32, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB_m32, up_proj n8k64_wB_m32, down_proj n8k64_wB_m16, q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16
- T = 64: out_proj n8k64_wB_m32, in_proj_qkv n8k64_wB_m64, in_proj_z n8k64_wB_m64, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB_m64, up_proj n8k64_wB_m64, down_proj n8k64_wB_m32, q_proj n8k64_wB, k_proj n8k64_wB_m32, v_proj n8k64_wB_m32, o_proj n8k64_wB_m32
- T = 128: out_proj n8k64_wB_m64, in_proj_qkv n8k64_wB_n64, in_proj_z n8k64_wB_n64, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB_m64, q_proj n8k64_wB, k_proj n8k64_wB_m32, v_proj n8k64_wB_m32, o_proj n8k64_wB_m64
- T = 256: out_proj n8k64_wB_n64, in_proj_qkv n8k64_wB, in_proj_z n8k64_wB, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB_n64, q_proj n8k64_wB_n64, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB_n64
- T = 512: out_proj n8k64_wB, in_proj_qkv n8k64_wB, in_proj_z n8k64_wB_n64, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB, q_proj n8k64_wB, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB
- T = 1024: out_proj n8k64_wB, in_proj_qkv n8k64_wB, in_proj_z n8k64_wB, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB, q_proj n8k64_wB, k_proj n8k64_wB_n64, v_proj n8k64_wB_n64, o_proj n8k64_wB
- T = 2048: out_proj n8k64_wB, in_proj_qkv n8k64_wB, in_proj_z n8k64_wB, in_proj_b n8k64_wB_m32, in_proj_a n8k64_wB_m32, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB, q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB
- T = 4096: out_proj n8k64_wB, in_proj_qkv n8k64_wB, in_proj_z n8k64_wB, in_proj_b n8k64_wB_m64, in_proj_a n8k64_wB_m64, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB, q_proj n8k64_wB, k_proj n8k64_wB_n64, v_proj n8k64_wB_n64, o_proj n8k64_wB
- T = 8192: out_proj n8k64_wB, in_proj_qkv n8k64_wB, in_proj_z n8k64_wB, in_proj_b n8k64_wB_m64, in_proj_a n8k64_wB_m64, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB, q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB

## M2: prefill (ms per forward, CUDA graph; median over rounds)

### Llama-3.1-8B

| shape | fo6 | fo6-wB | ours-8x64 before | ours-8x64 after | after vs before [round range] | vs fo6: before → after | vs fo6-wB: before → after |
|---|---|---|---|---|---|---|---|
| 1x128 | 7.46 | 8.87 | 9.21 | 7.54 | -18.2 % [-18.2, -18.0] | +23.5 % → +1.1 % | +3.8 % → -15.0 % |
| 1x256 | 9.39 | 10.50 | 10.93 | 9.55 | -12.6 % [-12.7, -12.4] | +16.4 % → +1.7 % | +4.0 % → -9.1 % |
| 1x512 | 15.12 | 15.42 | 16.51 | 16.53 | +0.1 % [-2.7, +1.2] | +9.2 % → +9.4 % | +7.1 % → +7.2 % |
| 1x1024 | 31.35 | 31.08 | 32.77 | 32.45 | -1.0 % [-1.3, +0.9] | +4.5 % → +3.5 % | +5.4 % → +4.4 % |
| 1x2048 | 62.04 | 62.18 | 65.59 | 65.66 | +0.1 % [-0.4, +1.6] | +5.7 % → +5.8 % | +5.5 % → +5.6 % |
| 1x4096 | 134.20 | 134.22 | 141.18 | 141.48 | +0.2 % [-0.1, +1.5] | +5.2 % → +5.4 % | +5.2 % → +5.4 % |
| 1x8192 | 323.39 | 324.01 | 337.19 | 337.73 | +0.2 % [-0.3, +1.2] | +4.3 % → +4.4 % | +4.1 % → +4.2 % |
| 4x2048 | 275.66 | 276.67 | 289.92 | 290.19 | +0.1 % [-0.4, +1.2] | +5.2 % → +5.3 % | +4.8 % → +4.9 % |

Eager prefill, ms (supplementary, as in the paper; small T is host-bound):

| shape | fo6 | fo6-wB | before | after | after vs before |
|---|---|---|---|---|---|
| 1x128 | 35.16 | 35.21 | 35.03 | 36.19 | +3.3 % |
| 1x256 | 33.51 | 35.95 | 34.39 | 35.96 | +4.5 % |
| 1x512 | 36.64 | 36.36 | 31.73 | 35.25 | +11.1 % |
| 1x1024 | 35.82 | 34.92 | 33.55 | 34.14 | +1.8 % |
| 1x2048 | 61.62 | 61.73 | 65.09 | 65.11 | +0.0 % |
| 1x4096 | 133.26 | 133.23 | 140.04 | 140.16 | +0.1 % |
| 1x8192 | 320.44 | 321.09 | 334.20 | 334.12 | -0.0 % |
| 4x2048 | 273.66 | 274.55 | 287.62 | 287.68 | +0.0 % |

### Mistral-7B-v0.3

| shape | fo6 | fo6-wB | ours-8x64 before | ours-8x64 after | after vs before [round range] | vs fo6: before → after | vs fo6-wB: before → after |
|---|---|---|---|---|---|---|---|
| 1x128 | 6.99 | 8.41 | 8.75 | 7.07 | -19.2 % [-19.2, -19.2] | +25.2 % → +1.2 % | +4.1 % → -15.9 % |
| 1x256 | 8.80 | 9.92 | 10.36 | 8.95 | -13.5 % [-13.6, -13.4] | +17.6 % → +1.7 % | +4.4 % → -9.7 % |
| 1x512 | 13.42 | 13.79 | 14.73 | 14.55 | -1.2 % [-4.3, -0.2] | +9.7 % → +8.4 % | +6.8 % → +5.5 % |
| 1x1024 | 28.28 | 27.80 | 29.91 | 29.77 | -0.5 % [-2.8, -0.3] | +5.8 % → +5.3 % | +7.6 % → +7.1 % |
| 1x2048 | 56.25 | 56.49 | 59.71 | 59.80 | +0.2 % [-0.0, +0.3] | +6.2 % → +6.3 % | +5.7 % → +5.9 % |
| 1x4096 | 123.36 | 123.45 | 130.36 | 130.19 | -0.1 % [-0.2, -0.0] | +5.7 % → +5.5 % | +5.6 % → +5.5 % |
| 1x8192 | 300.99 | 301.74 | 315.16 | 315.01 | -0.0 % [-0.2, +0.1] | +4.7 % → +4.7 % | +4.4 % → +4.4 % |
| 4x2048 | 253.58 | 254.27 | 267.76 | 267.53 | -0.1 % [-0.1, +0.0] | +5.6 % → +5.5 % | +5.3 % → +5.2 % |

Eager prefill, ms (supplementary, as in the paper; small T is host-bound):

| shape | fo6 | fo6-wB | before | after | after vs before |
|---|---|---|---|---|---|
| 1x128 | 34.39 | 34.61 | 30.04 | 31.83 | +6.0 % |
| 1x256 | 31.01 | 30.50 | 30.59 | 31.13 | +1.8 % |
| 1x512 | 31.27 | 30.59 | 30.65 | 30.89 | +0.8 % |
| 1x1024 | 31.26 | 30.77 | 30.62 | 31.16 | +1.8 % |
| 1x2048 | 55.87 | 56.04 | 59.35 | 59.39 | +0.1 % |
| 1x4096 | 121.88 | 122.02 | 128.74 | 128.78 | +0.0 % |
| 1x8192 | 297.99 | 298.94 | 311.79 | 311.78 | -0.0 % |
| 4x2048 | 251.51 | 252.27 | 265.52 | 265.35 | -0.1 % |

### Phi-4

| shape | fo6 | fo6-wB | ours-8x64 before | ours-8x64 after | after vs before [round range] | vs fo6: before → after | vs fo6-wB: before → after |
|---|---|---|---|---|---|---|---|
| 1x128 | 11.37 | 12.34 | 12.69 | 11.45 | -9.8 % [-9.9, -9.8] | +11.6 % → +0.6 % | +2.8 % → -7.2 % |
| 1x256 | 14.88 | 15.18 | 15.96 | 15.82 | -0.9 % [-2.7, +1.5] | +7.3 % → +6.3 % | +5.2 % → +4.2 % |
| 1x512 | 27.52 | 27.75 | 29.30 | 29.92 | +2.1 % [+1.0, +2.7] | +6.4 % → +8.7 % | +5.6 % → +7.8 % |
| 1x1024 | 55.13 | 55.46 | 58.66 | 58.63 | -0.0 % [-0.3, +0.4] | +6.4 % → +6.3 % | +5.8 % → +5.7 % |
| 1x2048 | 111.22 | 111.68 | 118.22 | 118.25 | +0.0 % [-0.1, +0.4] | +6.3 % → +6.3 % | +5.9 % → +5.9 % |
| 1x4096 | 237.01 | 237.96 | 250.87 | 250.78 | -0.0 % [-0.1, +0.2] | +5.8 % → +5.8 % | +5.4 % → +5.4 % |
| 1x8192 | 559.62 | 562.37 | 588.08 | 588.02 | -0.0 % [-0.1, +0.2] | +5.1 % → +5.1 % | +4.6 % → +4.6 % |
| 4x2048 | 487.54 | 490.53 | 515.82 | 515.55 | -0.1 % [-0.1, +0.2] | +5.8 % → +5.7 % | +5.2 % → +5.1 % |

Eager prefill, ms (supplementary, as in the paper; small T is host-bound):

| shape | fo6 | fo6-wB | before | after | after vs before |
|---|---|---|---|---|---|
| 1x128 | 35.42 | 35.04 | 40.37 | 41.09 | +1.8 % |
| 1x256 | 35.93 | 37.39 | 40.95 | 36.24 | -11.5 % |
| 1x512 | 36.08 | 35.98 | 35.21 | 36.36 | +3.3 % |
| 1x1024 | 54.34 | 54.66 | 57.89 | 57.87 | -0.0 % |
| 1x2048 | 109.41 | 109.87 | 116.16 | 116.18 | +0.0 % |
| 1x4096 | 233.31 | 234.14 | 246.53 | 246.49 | -0.0 % |
| 1x8192 | 551.41 | 553.92 | 578.79 | 578.57 | -0.0 % |
| 4x2048 | 481.31 | 483.88 | 508.82 | 508.58 | -0.0 % |

### Qwen3.8-27B

| shape | fo6 | fo6-wB | ours-8x64 before | ours-8x64 after | after vs before [round range] | vs fo6: before → after | vs fo6-wB: before → after |
|---|---|---|---|---|---|---|---|
| 1x128 | 57.73 | 60.79 | 61.28 | 57.73 | -5.8 % [-6.0, -5.7] | +6.1 % → -0.0 % | +0.8 % → -5.0 % |
| 1x256 | 72.95 | 75.00 | 76.13 | 73.69 | -3.2 % [-3.3, -3.2] | +4.3 % → +1.0 % | +1.5 % → -1.7 % |
| 1x512 | 108.41 | 109.10 | 111.81 | 111.39 | -0.4 % [-0.5, -0.4] | +3.1 % → +2.8 % | +2.5 % → +2.1 % |
| 1x1024 | 193.58 | 194.68 | 200.23 | 199.52 | -0.4 % [-0.5, -0.1] | +3.4 % → +3.1 % | +2.9 % → +2.5 % |
| 1x2048 | 379.80 | 380.60 | 392.45 | 391.21 | -0.3 % [-0.4, -0.2] | +3.3 % → +3.0 % | +3.1 % → +2.8 % |
| 1x4096 | 804.88 | 806.06 | 828.29 | 828.11 | -0.0 % [-0.3, +0.2] | +2.9 % → +2.9 % | +2.8 % → +2.7 % |
| 1x8192 | 1805.70 | 1809.76 | 1853.13 | 1852.49 | -0.0 % [-0.1, +0.2] | +2.6 % → +2.6 % | +2.4 % → +2.4 % |
| 4x2048 | 1673.61 | 1676.04 | 1720.30 | 1720.92 | +0.0 % [-0.1, +0.1] | +2.8 % → +2.8 % | +2.6 % → +2.7 % |

Eager prefill, ms (supplementary, as in the paper; small T is host-bound):

| shape | fo6 | fo6-wB | before | after | after vs before |
|---|---|---|---|---|---|
| 1x128 | 406.21 | 394.70 | 394.15 | 400.96 | +1.7 % |
| 1x256 | 444.73 | 461.16 | 442.12 | 450.72 | +1.9 % |
| 1x512 | 547.45 | 547.07 | 581.34 | 550.14 | -5.4 % |
| 1x1024 | 763.53 | 747.55 | 741.30 | 755.13 | +1.9 % |
| 1x2048 | 1160.08 | 1143.83 | 1218.91 | 1145.32 | -6.0 % |
| 1x4096 | 2003.82 | 1959.41 | 1966.56 | 1948.19 | -0.9 % |
| 1x8192 | 3712.53 | 3630.83 | 3826.46 | 3673.18 | -4.0 % |
| 4x2048 | 1663.95 | 1667.22 | 1710.43 | 1710.55 | +0.0 % |

## M3: decode (ms per token; median over rounds)

