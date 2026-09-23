# T0 資料稽核

三模型五 draws 的 45 張 N16 natural maps 均重新讀取並與各自 128-sequence FP64 moments 逐 tile 比對一致。
maps、moments、manifest 對照既有紀錄 SHA；未重新載入模型或驗算完整 candidate tensors。模型/Tokenizer revision、候選 hash、資料 revision 與每 sequence token hash 見 FROZEN_PROTOCOL.yaml。

T1 與 T2 map/veto CPU 分析可做。seed0 完整逐 N8 sequence scores 目前僅 Qwen 存在；Llama/Mistral 僅 sampled scores，不能重建全域 parent SE。CE–KL cross moments 不是 children covariance。

使用者已明確取代 H200/Slurm 限制：本機 A6000/Ada 合計最多三卡、禁止 foreign co-tenancy，無卡則等待。新結果須保留執行身份並通過 anchor。不得以現有 summary 當作新的 150/36 cells。

seed0 protocol label 為 aligned-primary，draw1–4 為 aligned-robustness；此標籤差異保留。其模型/候選/activation/固定 protocol seal 等實質身份欄位逐項相符，不以標籤相等冒充比較身份。

來源內容保留唯讀。calibration overlap 已另表；有文件 hash 不等於證明 draws 獨立。evaluation doc/token identity 於 T2 重用稽核另驗。

補充搜尋見 [RAW_SCORE_SEARCH_EXPANDED.json](results/RAW_SCORE_SEARCH_EXPANDED.json)：明確包含 ignored files，並跟隨 mechanism、follow-up、boundary 等 campaign symlinks。114 個符合已記錄檔名條件的候選中，完整 `raw_scores_full` 仍僅已驗證的 Qwen seed0；其他 sampled、additional-calibration、aggregation 與已被拒絕的新 parent moments 不可當成缺少的 seed0 全域 covariance。此為範圍明列的檔名盤點，不假稱已辨識任意命名的張量內容。六個 nested raw ZIP 的目錄盤點另見 `results/RAW_ARCHIVE_SEARCH.json`。
