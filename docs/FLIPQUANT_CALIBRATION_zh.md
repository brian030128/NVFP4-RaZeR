# FlipQuant 校準設定

FlipQuant 的程式路徑是 TM-OPT+TC（`run_train_map.py --tm-opt --tile-grad-tc`）。

整理日期為 2026-09-30，內容取自：
- 已 commit 的 run 紀錄：`results/nodev_cost/runs/nd_tc_*/report.json` 中的 `args` 和 `settings` 欄位；
- `run_train_map.py` 中的 `--tm-opt` preset。

## 1. 格式與候選

| 項目 | 設定 |
|---|---|
| 量化範圍 | 所有 text `nn.Linear`，不含 lm_head 和 embedding。Llama 224 個、Mistral 224 個、Phi-4 160 個、Qwen 496 個。Qwen 的 vision tower、recurrent/conv、norm 不量化 |
| Tensor scale | 每個 tensor 一個 FP32 scale：amax / (6·448) |
| Scale block | 每 16 個元素共用一個 UE4M3 scale |
| E2M1 候選 | FourOverSix：每個 block 依 MSE 在 max/6 和 max/4 兩種 scale 中選一個 |
| E0M3 候選 | alpha = 1：scale 為 e4m3(max/7)，整數格點 −7..7 |
| 格式旗標 | 放在 UE4M3 scale 的 bit 7，不額外占記憶體 |
| Type tile | 8x64（權重放 B）、16x64（權重放 A）；另有 256x64 放在附錄，部署時拆成 16x64 的 granule |

## 2. 參數化與最佳化

| 項目 | 設定 |
|---|---|
| 可訓練參數 | 每個 tile 一個 logit θ（例如 Llama 16x64 共 6,815,744 個）。權重和 scale 都凍結，不訓練 |
| 參數化 | STE：forward 用 m = 1[θ > 0]，也就是可直接部署的硬 map；backward 取 dm/dθ = 1 |
| 權重 | W(θ) = E2M1 候選 + m ⊙ (E0M3 候選 − E2M1 候選)，其中 m 展開到 tile 大小 |
| Tile 梯度 | dKL/dm_u = ⟨G, (E0M3 − E2M1) 在 tile u 上的部分⟩，其中 G = dyᵀx。由每層輸出的 backward hook 計算，不會產生權重梯度張量 |
| TC | G 的 GEMM 用 BF16 tensor core 計算，FP32 累加、FP32 輸出（`--tile-grad-tc`） |
| 初始值 | θ = −1，所以起點全部是 E2M1，也就是純 FourOverSix |
| Optimizer | Adam（不是 AdamW），β = (0.9, 0.999)，ε = 1e-12，沒有 weight decay |
| 學習率 | 0.02，常數，沒有 warmup 和衰減 |
| Batch | 每個 optimizer step 8 條序列，每個 epoch 16 步 |
| Epoch | 20，共 320 步 |
| 洗牌 | 每個 epoch 依 seed 0 重新打亂序列順序 |
| 輸出 | 最後一個 epoch 的 map（θ > 0 的 tile 設為 E0M3）；沒有 early stopping，也沒有挑選 checkpoint |

## 3. 資料與 loss

| 項目 | 設定 |
|---|---|
| Fit set | 128 條序列 × 512 token：64 條 math（open-web-math，固定 revision `fde8ef8`）、64 條 code（codeparrot-clean，固定 revision `35a59fb`）。兩者都不含 WikiText 和 C4 |
| Teacher | 原始 BF16 模型。事先算好完整 vocab 的 log-softmax，以 bf16 存在 CPU 記憶體（Llama 約 16.8 GB） |
| Loss | 對 teacher 的 KL：每條序列先對 511 個預測位置取 token 平均，再對同一步的所有序列取平均 |
| Dev set | 不使用。實際校準時開啟 `--no-dev --no-eval`，完全不讀取 dev 資料 |

## 4. 訓練時的 forward

| 項目 | 設定 |
|---|---|
| Activation 量化 | 每個 token 一個 scale 的 FourOverSix fake quant（`fourover6_rows`，和 `quantize_rows` 逐位元相同）；梯度以 STE 直通 |
| 其他 | 關閉 TF32；attention 等其餘部分維持 BF16 |

## 5. 工程設定（`--tm-opt` preset）

這些設定都不改變方法本身。每一項都已驗證和 legacy 路徑逐位元相同，或誤差在容許範圍內。

| Flag | 值 | 作用 |
|---|---|---|
| `memory_mode` | lean | 兩個候選以 4-bit 打包常駐（Llama 為 7.3 GiB），用到時才解碼，不保留 BF16 權重 |
| `fused_act_quant` | on | 使用融合的 activation 量化 kernel |
| `chunked_loss` | on | KL 和它的梯度每次處理 2 條序列，降低峰值記憶體 |
| `deterministic` | on | 開啟 `torch.use_deterministic_algorithms`，並設定 `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| `tile_grad_tc` | on | 即第 2 節的 TC |
| `tile_grad_kernel`（B1） | off | 不在 preset 裡 |
| `single_pass_epilogue`、dev/eval backend = native | on | 只影響監控和評估；實際校準時兩者都不會執行 |
| 環境變數 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | |

## 6. 各模型的差異（規則：只有 batch 的切法可以不同）

| 模型 | Batch 設定 |
|---|---|
| Llama-3.1-8B、Mistral-7B-v0.3、Phi-4 | batch 8，不做梯度累積 |
| Qwen3.8-27B | micro-batch 2 × 累積 4，optimizer batch 仍為 8 |

Llama 的校準紀錄是用 transformers 4.57.3 建立的，現在改用 5.16.1 執行，所以要加 `--transformers-deviation`。這個旗標只用於版本確認，不影響計算結果。

## 7. 部署與評估

- **匯出：** map 用 `export_map_artifact.py` 匯出成 packed FP4 artifact，並通過 ownership 檢查。
- **執行：** 在 SM120 上以 NativeLinear 執行。Activation 為每個 token 一個 scale 的 FourOverSix（convention (c)），epilogue 只做一次 rounding。
- **Kernel：**
  - 16x64 和 256x64 用 `auto`：權重放 A，依 tile 表選擇寬度。
  - 8x64 用 `n8k64_wB`：權重放 B。

## 8. 校準成本（不使用 dev，deterministic，RTX PRO 6000）

| 模型 | 時間 | GPU 峰值 | Host |
|---|---:|---:|---:|
| Llama-3.1-8B | 8.5–8.6 min | 40.5 GiB | 18.5 GiB |
| Mistral-7B-v0.3 | 7.5 min | 36 GiB | 14.7 GiB |
| Phi-4 | 14.8 min | 59 GiB | 28.5 GiB |
| Qwen3.8-27B（8x64） | 76 min | 90 GiB | 52.2 GiB |

三種 tile 大小的校準成本相同，差距在 0.1 分鐘以內。詳見 `results/nodev_cost/REPORT.md`。
