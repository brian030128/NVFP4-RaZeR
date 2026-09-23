# 下一步：外部 blocker，勿自動重試

本次三項追加嘗試均已結束，沒有活躍 GPU 工作。Qwen coarse evaluation 通過；
Llama/Mistral calibration attempt4 雖完成，但原始 scores/maps 身份不符，不能使用。

- T2 primary PPL：150/150；secondary accuracy：80/96。Llama 的 16 cells
  仍缺，原三次嘗試無效；本次授權不包含第四次 accuracy 嘗試。
- T3：20/36。Qwen 六粒度完整；Llama/Mistral 各缺 N32/64/128/256 × Wiki/C4。
- T0/T1/T4 與可完成的交付整理已完成，完整驗收仍不通過。

最小缺件：Llama/Mistral 原 seed0 全量逐 sequence scores 或完整 child covariance，
需通過既有原始 moments/map/hash gates。若只能重新 calibration，必須先解決
exact-reproduction 差異，再另行取得追加授權；不能盲目啟動 attempt5。
Llama accuracy 需要有效既有 paired predictions，或另外授權額外嘗試及獨占 A6000 時段。

取得資料後先依 REPRODUCE.md 檢查身份，執行 scripts/validate_parent_moments.py；
僅在通過後依 granularity_job_plan.csv 與 plans/ 中的 frozen plan 續跑缺項。
目前禁止直接執行已耗盡授權的 GPU commands。不要重跑已完成的 Qwen 或 T2 PPL。

CPU 檢查（使用 REPRODUCE.md 中的既有 Python 環境）：
`python scripts/granularity_table.py`、`python scripts/accuracy_table.py`。
不要把 CPU table checks 當成補足缺失模型評估。
