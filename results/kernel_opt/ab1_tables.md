## M1: per-forward GEMM time (isolated launches, cold weights; µs)

### Llama-3.1-8B

Checks: bitwise 672 ok=True, launch counts ok=True, other processes 0.

| T | stock_wA | stock_wB | 8x64 before (typ.) | 8x64 after (typ.) | after vs before (typ.) [round range] | after vs before (worst) | vs stock_wA: before → after (typ.) | vs stock_wB: before → after (typ.) |
|---|---|---|---|---|---|---|---|---|
| 1 | 3284.0 | 5681.1 | 6066.6 | 3340.3 | -44.9 % [-45.0, -44.9] | -45.1 % | +84.7 % → +1.7 % | +6.8 % → -41.2 % |
| 4 | 3274.8 | 5668.8 | 6056.9 | 3336.7 | -44.9 % [-45.0, -44.9] | -45.1 % | +85.0 % → +1.9 % | +6.8 % → -41.1 % |
| 16 | 3270.7 | 5494.8 | 5887.9 | 3342.8 | -43.2 % [-43.3, -43.3] | -43.5 % | +80.0 % → +2.2 % | +7.2 % → -39.2 % |
| 32 | 3292.6 | 5291.0 | 5689.8 | 3382.7 | -40.5 % [-40.6, -40.6] | -40.8 % | +72.8 % → +2.7 % | +7.5 % → -36.1 % |
| 64 | 3344.8 | 4995.5 | 5440.0 | 3432.9 | -36.9 % [-37.0, -36.9] | -37.3 % | +62.6 % → +2.6 % | +8.9 % → -31.3 % |
| 128 | 3527.7 | 4991.5 | 5464.0 | 3677.7 | -32.7 % [-32.7, -32.6] | -33.2 % | +54.9 % → +4.3 % | +9.5 % → -26.3 % |
| 256 | 4548.0 | 5744.1 | 6343.7 | 5735.9 | -9.6 % [-9.6, -9.6] | -9.9 % | +39.5 % → +26.1 % | +10.4 % → -0.1 % |
| 512 | 6980.1 | 7353.3 | 8520.1 | 8005.6 | -6.0 % [-6.0, -5.9] | -6.1 % | +22.1 % → +14.7 % | +15.9 % → +8.9 % |
| 1024 | 13532.6 | 13578.7 | 14728.6 | 14661.6 | -0.5 % [-0.7, -0.3] | -0.3 % | +8.8 % → +8.3 % | +8.5 % → +8.0 % |
| 2048 | 23330.7 | 23152.0 | 25536.4 | 25578.9 | +0.2 % [-0.1, +0.5] | +0.1 % | +9.5 % → +9.6 % | +10.3 % → +10.5 % |
| 4096 | 45796.1 | 45467.4 | 50334.5 | 50045.7 | -0.6 % [-1.0, +0.1] | -0.2 % | +9.9 % → +9.3 % | +10.7 % → +10.1 % |
| 8192 | 87433.7 | 87002.1 | 96287.2 | 96309.8 | +0.0 % [-0.1, +0.1] | -0.2 % | +10.1 % → +10.2 % | +10.7 % → +10.7 % |

Width chosen (build per projection, typical tags):

- T = 1: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 4: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 16: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 32: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 64: q_proj n8k64_wB_m32, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m32, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB_m32
- T = 128: q_proj n8k64_wB_m64, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB_m64, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB_m64
- T = 256: q_proj n8k64_wB_m64, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB_m64, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 512: q_proj n8k64_wB, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 1024: q_proj n8k64_wB, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 2048: q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 4096: q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 8192: q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB

### Mistral-7B-v0.3

Checks: bitwise 672 ok=True, launch counts ok=True, other processes 0.

| T | stock_wA | stock_wB | 8x64 before (typ.) | 8x64 after (typ.) | after vs before (typ.) [round range] | after vs before (worst) | vs stock_wA: before → after (typ.) | vs stock_wB: before → after (typ.) |
|---|---|---|---|---|---|---|---|---|
| 1 | 3285.0 | 5685.8 | 6085.1 | 3345.4 | -45.0 % [-45.1, -45.0] | -45.1 % | +85.2 % → +1.8 % | +7.0 % → -41.2 % |
| 4 | 3278.3 | 5672.9 | 6065.6 | 3343.3 | -44.9 % [-45.0, -44.8] | -45.1 % | +85.0 % → +2.0 % | +6.9 % → -41.1 % |
| 16 | 3275.7 | 5498.9 | 5897.2 | 3341.8 | -43.3 % [-43.4, -43.3] | -43.5 % | +80.0 % → +2.0 % | +7.2 % → -39.2 % |
| 32 | 3293.2 | 5297.6 | 5690.4 | 3385.3 | -40.5 % [-40.5, -40.5] | -40.6 % | +72.8 % → +2.8 % | +7.4 % → -36.1 % |
| 64 | 3343.4 | 4997.1 | 5434.3 | 3435.0 | -36.8 % [-36.9, -36.8] | -37.1 % | +62.5 % → +2.7 % | +8.8 % → -31.3 % |
| 128 | 3531.2 | 4992.0 | 5462.5 | 3677.1 | -32.7 % [-32.7, -32.7] | -33.0 % | +54.7 % → +4.1 % | +9.4 % → -26.3 % |
| 256 | 4552.2 | 5748.2 | 6342.1 | 5738.0 | -9.5 % [-9.6, -9.5] | -9.7 % | +39.3 % → +26.0 % | +10.3 % → -0.2 % |
| 512 | 6994.4 | 7376.3 | 8551.9 | 8038.9 | -6.0 % [-6.1, -5.7] | -5.7 % | +22.3 % → +14.9 % | +15.9 % → +9.0 % |
| 1024 | 13564.4 | 13629.9 | 14767.1 | 14700.0 | -0.5 % [-0.5, -0.4] | -0.2 % | +8.9 % → +8.4 % | +8.3 % → +7.9 % |
| 2048 | 23425.9 | 23271.8 | 25714.6 | 25712.0 | -0.0 % [-0.2, +0.3] | +0.1 % | +9.8 % → +9.8 % | +10.5 % → +10.5 % |
| 4096 | 46108.4 | 45693.2 | 50408.7 | 50181.4 | -0.5 % [-0.7, +0.0] | +0.3 % | +9.3 % → +8.8 % | +10.3 % → +9.8 % |
| 8192 | 87819.9 | 87316.6 | 96638.4 | 96630.8 | -0.0 % [-0.0, +0.0] | -0.1 % | +10.0 % → +10.0 % | +10.7 % → +10.7 % |

Width chosen (build per projection, typical tags):

- T = 1: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 4: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 16: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 32: q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 64: q_proj n8k64_wB_m32, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m32, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB_m32
- T = 128: q_proj n8k64_wB_m64, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB_m64, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB_m64
- T = 256: q_proj n8k64_wB_m64, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB_m64, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 512: q_proj n8k64_wB, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 1024: q_proj n8k64_wB, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 2048: q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 4096: q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB
- T = 8192: q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB

### Phi-4

Checks: bitwise 384 ok=True, launch counts ok=True, other processes 0.

| T | stock_wA | stock_wB | 8x64 before (typ.) | 8x64 after (typ.) | after vs before (typ.) [round range] | after vs before (worst) | vs stock_wA: before → after (typ.) | vs stock_wB: before → after (typ.) |
|---|---|---|---|---|---|---|---|---|
| 1 | 5317.1 | 7831.0 | 8152.9 | 5349.0 | -34.4 % [-34.4, -34.3] | -33.9 % | +53.3 % → +0.6 % | +4.1 % → -31.7 % |
| 4 | 5317.7 | 8184.3 | 8535.6 | 5350.4 | -37.3 % [-37.3, -37.2] | -37.0 % | +60.5 % → +0.6 % | +4.3 % → -34.6 % |
| 16 | 5308.1 | 7744.0 | 8007.0 | 5357.4 | -33.1 % [-33.2, -33.0] | -33.1 % | +50.8 % → +0.9 % | +3.4 % → -30.8 % |
| 32 | 5346.5 | 7326.0 | 7584.0 | 5378.5 | -29.1 % [-29.2, -29.0] | -29.2 % | +41.8 % → +0.6 % | +3.5 % → -26.6 % |
| 64 | 5372.7 | 6698.2 | 7075.2 | 5446.3 | -23.0 % [-23.1, -22.9] | -23.3 % | +31.7 % → +1.4 % | +5.6 % → -18.7 % |
| 128 | 5635.8 | 6706.5 | 7132.1 | 5949.4 | -16.6 % [-16.7, -16.6] | -16.9 % | +26.5 % → +5.6 % | +6.3 % → -11.3 % |
| 256 | 6981.1 | 7621.1 | 8252.7 | 8255.3 | +0.0 % [-0.1, +0.1] | -0.1 % | +18.2 % → +18.3 % | +8.3 % → +8.3 % |
| 512 | 11936.6 | 11941.7 | 13252.4 | 13233.9 | -0.1 % [-0.1, -0.0] | +0.0 % | +11.0 % → +10.9 % | +11.0 % → +10.8 % |
| 1024 | 22886.3 | 22795.4 | 24950.3 | 24932.4 | -0.1 % [-0.4, +0.1] | -0.1 % | +9.0 % → +8.9 % | +9.5 % → +9.4 % |
| 2048 | 45288.8 | 45094.9 | 49192.1 | 49174.1 | -0.0 % [-0.1, +0.1] | -0.1 % | +8.6 % → +8.6 % | +9.1 % → +9.0 % |
| 4096 | 85511.9 | 85090.8 | 93254.6 | 93264.2 | +0.0 % [-0.1, +0.0] | +0.1 % | +9.1 % → +9.1 % | +9.6 % → +9.6 % |
| 8192 | 167704.2 | 167398.8 | 184295.4 | 184278.2 | -0.0 % [+0.0, +0.1] | -0.0 % | +9.9 % → +9.9 % | +10.1 % → +10.1 % |

Width chosen (build per projection, typical tags):

- T = 1: o_proj n8k64_wB_m16, qkv_proj n8k64_wB_m16, gate_up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 4: o_proj n8k64_wB_m16, qkv_proj n8k64_wB_m16, gate_up_proj n8k64_wB_m16, down_proj n8k64_wB_m16
- T = 16: o_proj n8k64_wB_m16, qkv_proj n8k64_wB_m16, gate_up_proj n8k64_wB_m16, down_proj n8k64_wB_m32
- T = 32: o_proj n8k64_wB_m16, qkv_proj n8k64_wB_m32, gate_up_proj n8k64_wB_m32, down_proj n8k64_wB_m16
- T = 64: o_proj n8k64_wB_m32, qkv_proj n8k64_wB_m64, gate_up_proj n8k64_wB_m64, down_proj n8k64_wB_m32
- T = 128: o_proj n8k64_wB_m64, qkv_proj n8k64_wB, gate_up_proj n8k64_wB, down_proj n8k64_wB_m64
- T = 256: o_proj n8k64_wB, qkv_proj n8k64_wB, gate_up_proj n8k64_wB, down_proj n8k64_wB
- T = 512: o_proj n8k64_wB, qkv_proj n8k64_wB, gate_up_proj n8k64_wB, down_proj n8k64_wB
- T = 1024: o_proj n8k64_wB, qkv_proj n8k64_wB, gate_up_proj n8k64_wB, down_proj n8k64_wB
- T = 2048: o_proj n8k64_wB, qkv_proj n8k64_wB, gate_up_proj n8k64_wB, down_proj n8k64_wB
- T = 4096: o_proj n8k64_wB, qkv_proj n8k64_wB, gate_up_proj n8k64_wB, down_proj n8k64_wB
- T = 8192: o_proj n8k64_wB, qkv_proj n8k64_wB, gate_up_proj n8k64_wB, down_proj n8k64_wB

### Qwen3.8-27B

Checks: bitwise 1152 ok=True, launch counts ok=True, other processes 0.

| T | stock_wA | stock_wB | 8x64 before (typ.) | 8x64 after (typ.) | after vs before (typ.) [round range] | after vs before (worst) | vs stock_wA: before → after (typ.) | vs stock_wB: before → after (typ.) |
|---|---|---|---|---|---|---|---|---|
| 1 | 10657.0 | 16702.7 | 17577.8 | 10708.4 | -39.1 % [-39.2, -39.0] | -39.1 % | +64.9 % → +0.5 % | +5.2 % → -35.9 % |
| 4 | 10661.3 | 16739.6 | 17626.1 | 10673.9 | -39.4 % [-39.5, -39.4] | -39.3 % | +65.3 % → +0.1 % | +5.3 % → -36.2 % |
| 16 | 10618.1 | 16168.0 | 17093.6 | 10658.3 | -37.6 % [-37.7, -37.6] | -37.7 % | +61.0 % → +0.4 % | +5.7 % → -34.1 % |
| 32 | 10666.2 | 15526.5 | 16447.6 | 10738.1 | -34.7 % [-34.7, -34.7] | -34.8 % | +54.2 % → +0.7 % | +5.9 % → -30.8 % |
| 64 | 10747.9 | 14373.8 | 15345.4 | 10862.9 | -29.2 % [-29.2, -29.2] | -29.5 % | +42.8 % → +1.1 % | +6.8 % → -24.4 % |
| 128 | 11133.7 | 14390.0 | 15369.9 | 11682.7 | -24.0 % [-24.0, -23.9] | -24.5 % | +38.0 % → +4.9 % | +6.8 % → -18.8 % |
| 256 | 13894.9 | 16351.9 | 18039.9 | 16492.0 | -8.6 % [-8.6, -8.5] | -8.7 % | +29.8 % → +18.7 % | +10.3 % → +0.9 % |
| 512 | 23536.0 | 24757.4 | 27226.2 | 25787.0 | -5.3 % [-5.5, -5.1] | -5.7 % | +15.7 % → +9.6 % | +10.0 % → +4.2 % |
| 1024 | 43889.9 | 44738.6 | 49080.6 | 47792.4 | -2.6 % [-2.7, -2.5] | -2.5 % | +11.8 % → +8.9 % | +9.7 % → +6.8 % |
| 2048 | 83213.7 | 83361.7 | 90542.7 | 89290.5 | -1.4 % [-1.4, -1.2] | -1.3 % | +8.8 % → +7.3 % | +8.6 % → +7.1 % |
| 4096 | 156686.8 | 155908.6 | 170608.6 | 169650.3 | -0.6 % [-0.7, -0.5] | -0.7 % | +8.9 % → +8.3 % | +9.4 % → +8.8 % |
| 8192 | 302730.3 | 301888.3 | 331653.6 | 330959.5 | -0.2 % [-0.4, -0.1] | -0.2 % | +9.6 % → +9.3 % | +9.9 % → +9.6 % |

Width chosen (build per projection, typical tags):

- T = 1: out_proj n8k64_wB_m16, in_proj_qkv n8k64_wB_m32, in_proj_z n8k64_wB_m16, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16, q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16
- T = 4: out_proj n8k64_wB_m16, in_proj_qkv n8k64_wB_m16, in_proj_z n8k64_wB_m16, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16, q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16
- T = 16: out_proj n8k64_wB_m16, in_proj_qkv n8k64_wB_m16, in_proj_z n8k64_wB_m16, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB_m16, up_proj n8k64_wB_m16, down_proj n8k64_wB_m16, q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16
- T = 32: out_proj n8k64_wB_m16, in_proj_qkv n8k64_wB_m32, in_proj_z n8k64_wB_m32, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB_m32, up_proj n8k64_wB_m32, down_proj n8k64_wB_m16, q_proj n8k64_wB_m16, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m16
- T = 64: out_proj n8k64_wB_m32, in_proj_qkv n8k64_wB_m64, in_proj_z n8k64_wB_m64, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB_m64, up_proj n8k64_wB_m64, down_proj n8k64_wB_m32, q_proj n8k64_wB, k_proj n8k64_wB_m32, v_proj n8k64_wB_m32, o_proj n8k64_wB_m32
- T = 128: out_proj n8k64_wB_m64, in_proj_qkv n8k64_wB, in_proj_z n8k64_wB, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB_m64, q_proj n8k64_wB, k_proj n8k64_wB_m16, v_proj n8k64_wB_m16, o_proj n8k64_wB_m64
- T = 256: out_proj n8k64_wB, in_proj_qkv n8k64_wB, in_proj_z n8k64_wB, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB, q_proj n8k64_wB, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB
- T = 512: out_proj n8k64_wB, in_proj_qkv n8k64_wB, in_proj_z n8k64_wB, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB, q_proj n8k64_wB, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB
- T = 1024: out_proj n8k64_wB, in_proj_qkv n8k64_wB, in_proj_z n8k64_wB, in_proj_b n8k64_wB_m16, in_proj_a n8k64_wB_m16, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB, q_proj n8k64_wB, k_proj n8k64_wB_m64, v_proj n8k64_wB_m64, o_proj n8k64_wB
- T = 2048: out_proj n8k64_wB, in_proj_qkv n8k64_wB, in_proj_z n8k64_wB, in_proj_b n8k64_wB_m32, in_proj_a n8k64_wB_m32, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB, q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB
- T = 4096: out_proj n8k64_wB, in_proj_qkv n8k64_wB, in_proj_z n8k64_wB, in_proj_b n8k64_wB_m64, in_proj_a n8k64_wB_m64, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB, q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB
- T = 8192: out_proj n8k64_wB, in_proj_qkv n8k64_wB, in_proj_z n8k64_wB, in_proj_b n8k64_wB_m64, in_proj_a n8k64_wB_m64, gate_proj n8k64_wB, up_proj n8k64_wB, down_proj n8k64_wB, q_proj n8k64_wB, k_proj n8k64_wB, v_proj n8k64_wB, o_proj n8k64_wB

## M2: prefill (ms per forward, CUDA graph; median over rounds)

### Llama-3.1-8B

| shape | fo6 | fo6-wB | ours-8x64 before | ours-8x64 after | after vs before [round range] | vs fo6: before → after | vs fo6-wB: before → after |
|---|---|---|---|---|---|---|---|
| 1x128 | 7.46 | 8.87 | 9.21 | 7.54 | -18.2 % [-18.2, -18.0] | +23.5 % → +1.1 % | +3.8 % → -15.0 % |
| 1x256 | 9.38 | 10.50 | 10.93 | 10.38 | -5.1 % [-5.1, -4.8] | +16.5 % → +10.6 % | +4.1 % → -1.2 % |
| 1x512 | 15.13 | 15.51 | 16.80 | 16.47 | -2.0 % [-2.3, +0.8] | +11.1 % → +8.9 % | +8.3 % → +6.1 % |
| 1x1024 | 31.25 | 31.04 | 32.55 | 32.97 | +1.3 % [-0.9, +2.9] | +4.2 % → +5.5 % | +4.9 % → +6.2 % |
| 1x2048 | 61.94 | 62.10 | 65.48 | 65.58 | +0.2 % [-0.3, +1.5] | +5.7 % → +5.9 % | +5.4 % → +5.6 % |
| 1x4096 | 133.96 | 134.12 | 141.05 | 141.08 | +0.0 % [-0.4, +1.3] | +5.3 % → +5.3 % | +5.2 % → +5.2 % |
| 1x8192 | 322.95 | 324.01 | 336.93 | 337.22 | +0.1 % [-0.3, +1.2] | +4.3 % → +4.4 % | +4.0 % → +4.1 % |
| 4x2048 | 275.55 | 276.48 | 289.84 | 289.66 | -0.1 % [-0.4, +1.2] | +5.2 % → +5.1 % | +4.8 % → +4.8 % |

Eager prefill, ms (supplementary, as in the paper; small T is host-bound):

| shape | fo6 | fo6-wB | before | after | after vs before |
|---|---|---|---|---|---|
| 1x128 | 35.94 | 30.65 | 30.18 | 35.81 | +18.6 % |
| 1x256 | 36.95 | 30.91 | 30.82 | 36.51 | +18.5 % |
| 1x512 | 37.15 | 31.41 | 31.29 | 32.06 | +2.5 % |
| 1x1024 | 37.32 | 33.16 | 33.46 | 33.68 | +0.6 % |
| 1x2048 | 61.56 | 61.70 | 65.04 | 65.14 | +0.2 % |
| 1x4096 | 133.25 | 133.37 | 140.03 | 140.10 | +0.1 % |
| 1x8192 | 319.98 | 320.79 | 333.67 | 334.05 | +0.1 % |
| 4x2048 | 273.38 | 274.29 | 287.08 | 287.38 | +0.1 % |

### Mistral-7B-v0.3

| shape | fo6 | fo6-wB | ours-8x64 before | ours-8x64 after | after vs before [round range] | vs fo6: before → after | vs fo6-wB: before → after |
|---|---|---|---|---|---|---|---|
| 1x128 | 6.99 | 8.41 | 8.75 | 7.07 | -19.2 % [-19.2, -19.2] | +25.2 % → +1.2 % | +4.0 % → -15.9 % |
| 1x256 | 8.80 | 9.92 | 10.35 | 9.79 | -5.4 % [-5.4, -5.3] | +17.6 % → +11.3 % | +4.4 % → -1.3 % |
| 1x512 | 13.33 | 13.72 | 14.92 | 14.59 | -2.2 % [-2.7, -1.6] | +11.9 % → +9.4 % | +8.8 % → +6.3 % |
| 1x1024 | 28.01 | 27.67 | 29.81 | 29.95 | +0.4 % [-1.1, +0.8] | +6.4 % → +6.9 % | +7.8 % → +8.2 % |
| 1x2048 | 56.07 | 56.35 | 59.56 | 59.52 | -0.1 % [-0.2, +0.1] | +6.2 % → +6.2 % | +5.7 % → +5.6 % |
| 1x4096 | 122.97 | 123.18 | 129.97 | 129.99 | +0.0 % [-0.2, +0.1] | +5.7 % → +5.7 % | +5.5 % → +5.5 % |
| 1x8192 | 300.25 | 301.07 | 314.36 | 314.40 | +0.0 % [-0.1, +0.2] | +4.7 % → +4.7 % | +4.4 % → +4.4 % |
| 4x2048 | 252.75 | 253.68 | 266.96 | 266.87 | -0.0 % [-0.1, +0.1] | +5.6 % → +5.6 % | +5.2 % → +5.2 % |

Eager prefill, ms (supplementary, as in the paper; small T is host-bound):

| shape | fo6 | fo6-wB | before | after | after vs before |
|---|---|---|---|---|---|
| 1x128 | 30.84 | 34.71 | 34.93 | 30.97 | -11.3 % |
| 1x256 | 35.54 | 30.69 | 33.72 | 35.00 | +3.8 % |
| 1x512 | 31.85 | 30.70 | 29.93 | 31.27 | +4.5 % |
| 1x1024 | 31.56 | 31.04 | 32.12 | 31.53 | -1.8 % |
| 1x2048 | 55.64 | 55.85 | 59.16 | 59.23 | +0.1 % |
| 1x4096 | 121.59 | 121.76 | 128.35 | 128.52 | +0.1 % |
| 1x8192 | 297.27 | 298.28 | 310.78 | 310.97 | +0.1 % |
| 4x2048 | 250.91 | 251.88 | 264.68 | 264.70 | +0.0 % |

### Phi-4

| shape | fo6 | fo6-wB | ours-8x64 before | ours-8x64 after | after vs before [round range] | vs fo6: before → after | vs fo6-wB: before → after |
|---|---|---|---|---|---|---|---|
| 1x128 | 11.37 | 12.34 | 12.69 | 11.62 | -8.4 % [-8.5, -8.4] | +11.6 % → +2.2 % | +2.9 % → -5.8 % |
| 1x256 | 14.97 | 15.16 | 15.82 | 15.88 | +0.4 % [-0.3, +0.9] | +5.7 % → +6.1 % | +4.4 % → +4.7 % |
| 1x512 | 27.47 | 27.67 | 29.43 | 29.44 | +0.0 % [-1.0, +0.4] | +7.1 % → +7.2 % | +6.4 % → +6.4 % |
| 1x1024 | 54.99 | 55.30 | 58.43 | 58.49 | +0.1 % [-0.1, +0.4] | +6.3 % → +6.4 % | +5.7 % → +5.8 % |
| 1x2048 | 110.96 | 111.54 | 117.81 | 117.89 | +0.1 % [-0.0, +0.4] | +6.2 % → +6.2 % | +5.6 % → +5.7 % |
| 1x4096 | 236.57 | 237.56 | 249.95 | 250.12 | +0.1 % [+0.0, +0.3] | +5.7 % → +5.7 % | +5.2 % → +5.3 % |
| 1x8192 | 558.77 | 561.79 | 586.01 | 586.53 | +0.1 % [-0.0, +0.3] | +4.9 % → +5.0 % | +4.3 % → +4.4 % |
| 4x2048 | 486.58 | 489.71 | 514.13 | 514.41 | +0.1 % [-0.0, +0.2] | +5.7 % → +5.7 % | +5.0 % → +5.0 % |

Eager prefill, ms (supplementary, as in the paper; small T is host-bound):

| shape | fo6 | fo6-wB | before | after | after vs before |
|---|---|---|---|---|---|
| 1x128 | 35.32 | 35.01 | 40.85 | 35.96 | -12.0 % |
| 1x256 | 35.90 | 35.09 | 35.66 | 37.10 | +4.0 % |
| 1x512 | 35.88 | 35.60 | 36.02 | 36.28 | +0.7 % |
| 1x1024 | 54.21 | 54.50 | 57.72 | 57.72 | +0.0 % |
| 1x2048 | 109.26 | 109.73 | 115.89 | 115.86 | -0.0 % |
| 1x4096 | 233.02 | 233.82 | 245.94 | 246.07 | +0.1 % |
| 1x8192 | 550.47 | 552.76 | 576.99 | 577.07 | +0.0 % |
| 4x2048 | 480.16 | 482.95 | 506.81 | 507.29 | +0.1 % |

### Qwen3.8-27B

| shape | fo6 | fo6-wB | ours-8x64 before | ours-8x64 after | after vs before [round range] | vs fo6: before → after | vs fo6-wB: before → after |
|---|---|---|---|---|---|---|---|
| 1x128 | 57.72 | 60.79 | 61.29 | 57.99 | -5.4 % [-5.5, -5.3] | +6.2 % → +0.5 % | +0.8 % → -4.6 % |
| 1x256 | 72.92 | 75.00 | 76.14 | 74.68 | -1.9 % [-2.0, -1.7] | +4.4 % → +2.4 % | +1.5 % → -0.4 % |
| 1x512 | 108.43 | 109.04 | 111.83 | 111.34 | -0.4 % [-0.6, -0.2] | +3.1 % → +2.7 % | +2.6 % → +2.1 % |
| 1x1024 | 193.72 | 194.62 | 200.27 | 199.58 | -0.3 % [-0.4, -0.1] | +3.4 % → +3.0 % | +2.9 % → +2.6 % |
| 1x2048 | 379.79 | 380.75 | 392.36 | 391.67 | -0.2 % [-0.3, +0.1] | +3.3 % → +3.1 % | +3.0 % → +2.9 % |
| 1x4096 | 805.26 | 806.04 | 828.83 | 827.95 | -0.1 % [-0.3, +0.0] | +2.9 % → +2.8 % | +2.8 % → +2.7 % |
| 1x8192 | 1807.18 | 1810.41 | 1853.43 | 1852.12 | -0.1 % [-0.3, +0.1] | +2.6 % → +2.5 % | +2.4 % → +2.3 % |
| 4x2048 | 1673.73 | 1676.46 | 1721.03 | 1720.35 | -0.0 % [-0.2, +0.1] | +2.8 % → +2.8 % | +2.7 % → +2.6 % |

Eager prefill, ms (supplementary, as in the paper; small T is host-bound):

| shape | fo6 | fo6-wB | before | after | after vs before |
|---|---|---|---|---|---|
| 1x128 | 401.98 | 397.17 | 403.60 | 394.98 | -2.1 % |
| 1x256 | 442.30 | 474.02 | 505.17 | 450.26 | -10.9 % |
| 1x512 | 545.12 | 545.01 | 544.85 | 546.00 | +0.2 % |
| 1x1024 | 752.26 | 740.89 | 744.63 | 742.61 | -0.3 % |
| 1x2048 | 1142.22 | 1271.31 | 1152.73 | 1142.12 | -0.9 % |
| 1x4096 | 1964.12 | 1967.48 | 1954.36 | 1989.04 | +1.8 % |
| 1x8192 | 3632.90 | 3762.87 | 3701.57 | 3640.80 | -1.6 % |
| 4x2048 | 1664.96 | 1667.32 | 1711.84 | 1710.42 | -0.1 % |

## M3: decode (ms per token; median over rounds)

### Llama-3.1-8B

| shape | fo6 | fo6-wB | ours-8x64 before | ours-8x64 after | after vs before [round range] | vs fo6: before → after | vs fo6-wB: before → after |
|---|---|---|---|---|---|---|---|
| 1x512 | 7.23 | 9.57 | 9.86 | 7.28 | -26.2 % [-26.3, -26.2] | +36.4 % → +0.7 % | +3.0 % → -23.9 % |
| 4x512 | 8.57 | 10.62 | 10.83 | 8.60 | -20.7 % [-20.8, -20.6] | +26.4 % → +0.3 % | +2.0 % → -19.0 % |
| 16x512 | 14.59 | 16.45 | 16.79 | 14.78 | -12.0 % [-12.1, -12.0] | +15.1 % → +1.3 % | +2.1 % → -10.2 % |
| 1x2048 | 10.19 | 12.27 | 12.50 | 10.21 | -18.3 % [-18.5, -18.3] | +22.6 % → +0.2 % | +1.9 % → -16.8 % |
| 4x2048 | 13.82 | 15.72 | 16.04 | 13.97 | -12.9 % [-13.0, -12.9] | +16.1 % → +1.1 % | +2.0 % → -11.1 % |
| 16x2048 | 35.10 | 37.07 | 37.39 | 35.24 | -5.8 % [-6.1, -5.8] | +6.5 % → +0.4 % | +0.9 % → -4.9 % |

Tokens per second (Experiment D's convention; higher is faster):

| setting | fo6 | fo6-wB | before | after | after vs before | vs fo6: before → after |
|---|---|---|---|---|---|---|
| 1x512 | 138.3 | 104.5 | 101.4 | 137.3 | +35.4 % | -26.7 % → -0.7 % |
| 4x512 | 466.5 | 376.7 | 369.2 | 465.3 | +26.0 % | -20.9 % → -0.3 % |
| 16x512 | 1096.7 | 972.4 | 952.7 | 1082.4 | +13.6 % | -13.1 % → -1.3 % |
| 1x2048 | 98.1 | 81.5 | 80.0 | 97.9 | +22.4 % | -18.5 % → -0.2 % |
| 4x2048 | 289.5 | 254.5 | 249.4 | 286.2 | +14.8 % | -13.9 % → -1.1 % |
| 16x2048 | 455.8 | 431.6 | 427.9 | 454.0 | +6.1 % | -6.1 % → -0.4 % |

### Mistral-7B-v0.3

| shape | fo6 | fo6-wB | ours-8x64 before | ours-8x64 after | after vs before [round range] | vs fo6: before → after | vs fo6-wB: before → after |
|---|---|---|---|---|---|---|---|
| 1x512 | 6.75 | 9.08 | 9.38 | 6.80 | -27.5 % [-27.5, -27.5] | +38.9 % → +0.7 % | +3.2 % → -25.1 % |
| 4x512 | 8.08 | 10.13 | 10.35 | 8.10 | -21.7 % [-21.7, -21.7] | +28.0 % → +0.3 % | +2.1 % → -20.0 % |
| 16x512 | 14.09 | 15.96 | 16.30 | 14.28 | -12.4 % [-12.4, -12.4] | +15.7 % → +1.4 % | +2.2 % → -10.5 % |
| 1x2048 | 9.67 | 11.78 | 12.01 | 9.70 | -19.2 % [-19.3, -19.2] | +24.2 % → +0.3 % | +1.9 % → -17.7 % |
| 4x2048 | 13.32 | 15.24 | 15.57 | 13.48 | -13.4 % [-13.4, -13.4] | +16.9 % → +1.2 % | +2.1 % → -11.5 % |
| 16x2048 | 34.55 | 36.54 | 36.87 | 34.69 | -5.9 % [-6.0, -5.9] | +6.7 % → +0.4 % | +0.9 % → -5.1 % |

Tokens per second (Experiment D's convention; higher is faster):

| setting | fo6 | fo6-wB | before | after | after vs before | vs fo6: before → after |
|---|---|---|---|---|---|---|
| 1x512 | 148.1 | 110.1 | 106.7 | 147.0 | +37.9 % | -28.0 % → -0.7 % |
| 4x512 | 494.9 | 394.8 | 386.5 | 493.6 | +27.7 % | -21.9 % → -0.3 % |
| 16x512 | 1135.6 | 1002.6 | 981.5 | 1120.4 | +14.2 % | -13.6 % → -1.3 % |
| 1x2048 | 103.4 | 84.9 | 83.3 | 103.1 | +23.8 % | -19.5 % → -0.3 % |
| 4x2048 | 300.3 | 262.5 | 257.0 | 296.7 | +15.4 % | -14.4 % → -1.2 % |
| 16x2048 | 463.1 | 437.9 | 434.0 | 461.2 | +6.3 % | -6.3 % → -0.4 % |

### Phi-4

| shape | fo6 | fo6-wB | ours-8x64 before | ours-8x64 after | after vs before [round range] | vs fo6: before → after | vs fo6-wB: before → after |
|---|---|---|---|---|---|---|---|
| 1x512 | 10.46 | 12.87 | 13.13 | 10.46 | -20.3 % [-20.4, -20.3] | +25.5 % → +0.0 % | +2.0 % → -18.8 % |
| 4x512 | 12.23 | 14.93 | 15.09 | 12.23 | -19.0 % [-19.0, -18.9] | +23.4 % → -0.0 % | +1.1 % → -18.1 % |
| 16x512 | 22.42 | 24.31 | 24.61 | 22.49 | -8.6 % [-8.6, -8.6] | +9.8 % → +0.3 % | +1.2 % → -7.5 % |
| 1x2048 | 14.11 | 16.76 | 16.94 | 14.10 | -16.8 % [-16.8, -16.8] | +20.0 % → -0.1 % | +1.1 % → -15.9 % |
| 4x2048 | 19.91 | 22.15 | 22.42 | 19.91 | -11.2 % [-11.2, -11.2] | +12.6 % → +0.0 % | +1.2 % → -10.1 % |
| 16x2048 | 55.44 | 57.81 | 58.14 | 55.69 | -4.2 % [-4.3, -4.1] | +4.9 % → +0.5 % | +0.6 % → -3.7 % |

Tokens per second (Experiment D's convention; higher is faster):

| setting | fo6 | fo6-wB | before | after | after vs before | vs fo6: before → after |
|---|---|---|---|---|---|---|
| 1x512 | 95.6 | 77.7 | 76.2 | 95.6 | +25.5 % | -20.3 % → -0.0 % |
| 4x512 | 327.1 | 268.0 | 265.1 | 327.1 | +23.4 % | -18.9 % → +0.0 % |
| 16x512 | 713.8 | 658.2 | 650.1 | 711.4 | +9.4 % | -8.9 % → -0.3 % |
| 1x2048 | 70.9 | 59.7 | 59.0 | 70.9 | +20.2 % | -16.7 % → +0.1 % |
| 4x2048 | 200.9 | 180.6 | 178.4 | 200.9 | +12.6 % | -11.2 % → -0.0 % |
| 16x2048 | 288.6 | 276.8 | 275.2 | 287.3 | +4.4 % | -4.7 % → -0.4 % |

