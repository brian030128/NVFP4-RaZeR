# FlipQuant release maps (DRAFT records)

Calibrated with flipquant's `calibration.train_map --fit-windows 256 --epochs 5 --teacher-topk 1000` (learning rate 0.02, initial logit −1, seed 0, the paper's per-model batch), one job at a time on one NVIDIA RTX PRO 6000 (sm_120). PPL: native kernels, `--paper-convention`, WikiText-2 and C4 (all windows). Nothing here is uploaded anywhere.

## Calibration: E0M3 tiles

| model | unit | flipquant commit | E0M3 tiles | extension check |
|---|---|---|---:|---|
| qwen3-1.7b | 8x64 | `ecf8a7f` | 24,675 / 2,752,512 (0.90 %) | pass |
| qwen3-1.7b | 16x64 | `ecf8a7f` | 16,351 / 1,376,256 (1.19 %) | pass |
| qwen3-1.7b | 256x64 | `ecf8a7f` | 3,340 / 86,016 (3.88 %) | pass |
| qwen3-8b | 8x64 | `120173a` | 67,105 / 13,565,952 (0.49 %) | pass |
| qwen3-8b | 16x64 | `120173a` | 37,511 / 6,782,976 (0.55 %) | pass |
| qwen3-8b | 256x64 | `120173a` | 5,386 / 423,936 (1.27 %) | pass |
| mistral-7b | 8x64 | `ecf8a7f` | 116,977 / 13,631,488 (0.86 %) | pass |
| mistral-7b | 16x64 | `ecf8a7f` | 75,906 / 6,815,744 (1.11 %) | pass |
| mistral-7b | 256x64 | `ecf8a7f` | 11,491 / 425,984 (2.70 %) | pass |
| mistral-7b-base | 8x64 | `120173a` | 106,186 / 13,631,488 (0.78 %) | pass |
| mistral-7b-base | 16x64 | `120173a` | 66,133 / 6,815,744 (0.97 %) | pass |
| mistral-7b-base | 256x64 | `120173a` | 11,004 / 425,984 (2.58 %) | pass |
| llama3.1-8b | 8x64 | `120173a` | 50,571 / 13,631,488 (0.37 %) | pass |
| llama3.1-8b | 16x64 | `120173a` | 32,125 / 6,815,744 (0.47 %) | pass |
| llama3.1-8b | 256x64 | `120173a` | 5,245 / 425,984 (1.23 %) | pass |
| nemotron-nano-9b-v2 | 8x64 | `120173a` | 130,579 / 15,061,760 (0.87 %) | pass |
| nemotron-nano-9b-v2 | 16x64 | `120173a` | 80,098 / 7,530,880 (1.06 %) | pass |
| nemotron-nano-9b-v2 | 256x64 | `120173a` | 12,679 / 478,320 (2.65 %) | pass |
| phi4-14b | 8x64 | `120173a` | 129,730 / 26,624,000 (0.49 %) | pass |
| phi4-14b | 16x64 | `120173a` | 79,591 / 13,312,000 (0.60 %) | pass |
| phi4-14b | 256x64 | `120173a` | 11,059 / 832,000 (1.33 %) | pass |
| qwen3.8-27b | 8x64 | `120173a` | 169,073 / 47,559,680 (0.36 %) | pass |
| qwen3.8-27b | 16x64 | `120173a` | 105,278 / 23,779,840 (0.44 %) | pass |
| qwen3.8-27b | 256x64 | `120173a` | 13,758 / 1,492,480 (0.92 %) | pass |

## Calibration: time

| model | unit | model load | data (+ extension) | teacher | lean packing | setup | per epoch (mean, min–max) | training | trainer total | train_map end-to-end | batch × accum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3-1.7b | 8x64 | 0.7 s | 20.4 s | 12.4 s | 5.8 s | 39.2 s | 12.0 s (11.6–12.3) | 60 s | 99 s | 111 s ᴿ | 8 × 1 |
| qwen3-1.7b | 16x64 | 0.8 s | 20.1 s | 12.2 s | 5.7 s | 38.8 s | 12.2 s (11.9–12.6) | 61 s | 100 s | 112 s ᴿ | 8 × 1 |
| qwen3-1.7b | 256x64 | 0.8 s | 20.3 s | 12.7 s | 5.8 s | 39.6 s | 12.2 s (12.0–12.7) | 61 s | 101 s | 110 s ᴿ | 8 × 1 |
| qwen3-8b | 8x64 | 2.4 s | 19.3 s | 14.8 s | 35.9 s | 72.3 s | 41.6 s (40.7–42.2) | 3.5 min | 4.7 min | 4.9 min ᴿ | 8 × 1 |
| qwen3-8b | 16x64 | 1.6 s | 21.2 s | 15.3 s | 36.0 s | 74.1 s | 42.0 s (41.7–42.2) | 3.5 min | 4.7 min | 4.9 min ᴿ | 8 × 1 |
| qwen3-8b | 256x64 | 1.6 s | 19.2 s | 15.1 s | 37.0 s | 72.9 s | 42.0 s (41.6–42.2) | 3.5 min | 4.7 min | 4.9 min ᴿ | 8 × 1 |
| mistral-7b | 8x64 | 1.7 s | 20.9 s | 10.6 s | 36.4 s | 69.5 s | 37.8 s (37.0–38.4) | 3.2 min | 4.3 min | 4.6 min ᴿ | 8 × 1 |
| mistral-7b | 16x64 | 1.5 s | 18.9 s | 10.6 s | 35.9 s | 66.9 s | 38.2 s (37.9–38.4) | 3.2 min | 4.3 min | 4.5 min ᴿ | 8 × 1 |
| mistral-7b | 256x64 | 1.5 s | 19.3 s | 10.6 s | 36.6 s | 68.0 s | 38.2 s (37.9–38.4) | 3.2 min | 4.3 min | 4.5 min ᴿ | 8 × 1 |
| mistral-7b-base | 8x64 | 1.6 s | 20.0 s | 10.8 s | 36.2 s | 68.6 s | 38.1 s (37.5–38.5) | 3.2 min | 4.3 min | 4.5 min ᴿ | 8 × 1 |
| mistral-7b-base | 16x64 | 1.7 s | 19.6 s | 11.4 s | 36.6 s | 69.2 s | 38.2 s (37.9–38.5) | 3.2 min | 4.3 min | 4.5 min ᴿ | 8 × 1 |
| mistral-7b-base | 256x64 | 1.6 s | 19.6 s | 11.0 s | 36.3 s | 68.5 s | 38.2 s (37.9–38.4) | 3.2 min | 4.3 min | 4.5 min ᴿ | 8 × 1 |
| llama3.1-8b | 8x64 | 1.7 s | 21.6 s | 11.8 s | 36.8 s | 71.9 s | 40.2 s (39.7–40.5) | 3.3 min | 4.5 min | 4.7 min ᴿ | 8 × 1 |
| llama3.1-8b | 16x64 | 1.7 s | 20.5 s | 11.7 s | 36.5 s | 70.4 s | 40.2 s (39.9–40.4) | 3.4 min | 4.5 min | 4.7 min ᴿ | 8 × 1 |
| llama3.1-8b | 256x64 | 1.7 s | 19.5 s | 11.8 s | 36.2 s | 69.1 s | 40.2 s (39.9–40.4) | 3.4 min | 4.5 min | 4.7 min ᴿ | 8 × 1 |
| nemotron-nano-9b-v2 | 8x64 | 2.3 s | 19.8 s | 149.9 s | 40.5 s | 212.5 s | 479.2 s (479.1–479.6) | 39.9 min | 43.5 min | 43.8 min ᴿ | 2 × 4 |
| nemotron-nano-9b-v2 | 16x64 | 1.8 s | 20.8 s | 151.7 s | 41.1 s | 215.4 s | 478.8 s (478.7–479.2) | 39.9 min | 43.5 min | 43.7 min | 2 × 4 |
| nemotron-nano-9b-v2 | 256x64 | 1.8 s | 19.9 s | 151.7 s | 40.9 s | 214.3 s | 481.2 s (481.0–481.4) | 40.1 min | 43.7 min | 43.9 min | 2 × 4 |
| phi4-14b | 8x64 | 3.6 s | 19.6 s | 17.5 s | 73.6 s | 114.2 s | 75.0 s (73.7–75.6) | 6.3 min | 8.2 min | 8.4 min ᴿ | 8 × 1 |
| phi4-14b | 16x64 | 3.6 s | 20.3 s | 18.3 s | 72.7 s | 114.9 s | 75.2 s (74.5–75.5) | 6.3 min | 8.2 min | 8.4 min ᴿ | 8 × 1 |
| phi4-14b | 256x64 | 3.5 s | 19.9 s | 18.3 s | 72.4 s | 114.0 s | 75.2 s (74.6–75.5) | 6.3 min | 8.2 min | 8.4 min ᴿ | 8 × 1 |
| qwen3.8-27b | 8x64 | 6.7 s | 19.2 s | 165.9 s | 129.7 s | 321.6 s | 432.8 s (429.7–435.1) | 36.1 min | 41.4 min | 41.6 min | 2 × 4 |
| qwen3.8-27b | 16x64 | 6.7 s | 19.4 s | 163.3 s | 130.6 s | 319.9 s | 430.9 s (428.2–434.2) | 35.9 min | 41.2 min | 41.5 min | 2 × 4 |
| qwen3.8-27b | 256x64 | 6.7 s | 19.4 s | 165.6 s | 127.7 s | 319.3 s | 431.1 s (428.1–432.8) | 35.9 min | 41.3 min | 41.5 min | 2 × 4 |

## Calibration: memory

| model | unit | GPU peak allocated | GPU peak reserved | trainer peak RSS (ru_maxrss) | RSS after model load | teacher (top-1000) | wrapper RSS | process-tree peak (10 Hz) | source of the last two |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3-1.7b | 8x64 | 21.0 GiB | 21.4 GiB | 4.7 GiB | 1.0 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 4.8 GiB | re-run |
| qwen3-1.7b | 16x64 | 21.0 GiB | 21.4 GiB | 4.7 GiB | 1.0 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 4.4 GiB | re-run |
| qwen3-1.7b | 256x64 | 21.0 GiB | 21.4 GiB | 4.7 GiB | 1.0 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 5.0 GiB | re-run |
| qwen3-8b | 8x64 | 47.0 GiB | 47.6 GiB | 16.2 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 16.4 GiB | re-run |
| qwen3-8b | 16x64 | 46.8 GiB | 47.5 GiB | 16.2 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 16.2 GiB | re-run |
| qwen3-8b | 256x64 | 46.7 GiB | 47.3 GiB | 16.2 GiB | 0.9 GiB | 0.73 GiB | 0.9 GiB (VmHWM) | 15.0 GiB | re-run |
| mistral-7b | 8x64 | 36.8 GiB | 37.6 GiB | 14.4 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 13.8 GiB | re-run |
| mistral-7b | 16x64 | 36.6 GiB | 37.5 GiB | 14.4 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 13.9 GiB | re-run |
| mistral-7b | 256x64 | 36.5 GiB | 37.3 GiB | 14.4 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 13.7 GiB | re-run |
| mistral-7b-base | 8x64 | 36.8 GiB | 37.6 GiB | 14.4 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 14.3 GiB | re-run |
| mistral-7b-base | 16x64 | 36.6 GiB | 37.5 GiB | 14.4 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 14.8 GiB | re-run |
| mistral-7b-base | 256x64 | 36.5 GiB | 37.4 GiB | 14.4 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 14.4 GiB | re-run |
| llama3.1-8b | 8x64 | 41.9 GiB | 42.9 GiB | 15.9 GiB | 0.9 GiB | 0.73 GiB | 0.9 GiB (VmHWM) | 15.4 GiB | re-run |
| llama3.1-8b | 16x64 | 41.7 GiB | 42.8 GiB | 15.9 GiB | 0.9 GiB | 0.73 GiB | 0.9 GiB (VmHWM) | 15.5 GiB | re-run |
| llama3.1-8b | 256x64 | 41.6 GiB | 42.6 GiB | 15.9 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 15.3 GiB | re-run |
| nemotron-nano-9b-v2 | 8x64 | 53.9 GiB | 55.3 GiB | 17.5 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 16.8 GiB | re-run |
| nemotron-nano-9b-v2 | 16x64 | 53.7 GiB | 55.2 GiB | 17.5 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 3.1 GiB (sampled after attaching; the processes' exact VmHWM peaks sum to 18.3 GiB) | live, attached late |
| nemotron-nano-9b-v2 | 256x64 | 53.7 GiB | 54.9 GiB | 17.5 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 16.1 GiB | live |
| phi4-14b | 8x64 | 60.2 GiB | 63.0 GiB | 28.2 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 27.8 GiB | re-run |
| phi4-14b | 16x64 | 59.9 GiB | 63.0 GiB | 28.2 GiB | 0.9 GiB | 0.73 GiB | 0.9 GiB (VmHWM) | 28.4 GiB | re-run |
| phi4-14b | 256x64 | 59.7 GiB | 63.0 GiB | 28.2 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 27.2 GiB | re-run |
| qwen3.8-27b | 8x64 | 91.7 GiB | 92.3 GiB | 51.9 GiB | 1.0 GiB | 0.73 GiB | 0.9 GiB (VmHWM) | 51.9 GiB | live |
| qwen3.8-27b | 16x64 | 91.3 GiB | 92.8 GiB | 51.9 GiB | 1.0 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 52.3 GiB | live |
| qwen3.8-27b | 256x64 | 90.9 GiB | 92.3 GiB | 51.9 GiB | 1.0 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 52.2 GiB | live |

## One-time data preparation

train_map's first run of a model on a new data root prepares its calibration record and development sets (`calibration.tmopt_common.prepare` and `prepare_development`; the builder rule's checks, re-tokenizing the fit windows, hashing the model's weights on the CPU unless an archived record has them, the development draws). Measured on 2026-10-06 in a fresh data root with measure.py's sampling plus each process's VmHWM; the files it wrote are byte-identical to the ones the release runs used. The dataset shards were already in the local Hugging Face cache; a first-ever preparation also downloads them.

| model | wall time | process-tree peak (10 Hz) | wrapper | largest process (VmHWM) | records identical to the release run's |
|---|---:|---:|---:|---:|---|
| qwen3-1.7b | 2.7 min | 6.2 GiB | 0.5 GiB | 5.7 GiB | yes (11 files) |
| qwen3-8b | 3.1 min | 16.1 GiB | 0.6 GiB | 15.6 GiB | yes (11 files) |
| mistral-7b | 3.0 min | 16.1 GiB | 0.5 GiB | 15.6 GiB | yes (11 files) |
| mistral-7b-base | 106 s | 14.5 GiB | 0.5 GiB | 14.1 GiB | yes (11 files) |
| llama3.1-8b | 20 s | 1.8 GiB | 0.5 GiB | 1.3 GiB | yes (5 files) |
| nemotron-nano-9b-v2 | 3.2 min | 17.5 GiB | 0.6 GiB | 17.0 GiB | yes (11 files) |
| phi4-14b | 3.8 min | 28.6 GiB | 0.5 GiB | 28.1 GiB | yes (11 files) |
| qwen3.8-27b | 92 s | 2.0 GiB | 0.5 GiB | 1.5 GiB | yes (11 files) |

Time and memory: the trainer's numbers (phases, epochs, GPU peaks, its own `ru_maxrss`, RSS after the model load, teacher storage) come from its PhaseMonitor in the release run. The wrapper's own RSS, the process-tree peak (wrapper + trainer, sampled at 10 Hz) and the end-to-end time come from topk-cal's `measure.py` method: for the first runs (ᴿ, source “re-run”) from a measured re-run of the same calibration that reproduced the same map, for the later ones from a watcher attached to the run itself (“live”; Nemotron 16x64 was attached 13 min after its start, so its tree peak covers only the rest, while its VmHWM peaks are exact). The end-to-end time excludes the one-time data preparation, which has its own row.

## Perplexity

### qwen3-1.7b (Qwen3-1.7B)

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 16.7162 | 19.2463 | -0.1527 ± 0.0112 * | -0.0828 ± 0.0042 * | better / better |
| FourOverSix (NVFP4, stock kernels) | 19.4745 | 20.9072 | — | — | reference |
| FlipQuant 8x64 (this release) | 15.4039 | 19.4491 | -0.2345 ± 0.0131 * | -0.0723 ± 0.0041 * | better / better |
| FlipQuant 16x64 (this release) | 15.3979 | 19.4097 | -0.2349 ± 0.0127 * | -0.0743 ± 0.0040 * | better / better |
| FlipQuant 256x64 (this release) | 15.9680 | 19.7617 | -0.1985 ± 0.0123 * | -0.0563 ± 0.0035 * | better / better |
| paper-setting map 8x64 (reference) | 15.6918 | 19.5472 | | | |
| paper-setting map 16x64 (reference) | 15.8021 | 19.6422 | | | |
| paper-setting map 256x64 (reference) | 16.4056 | 19.8559 | | | |

Paper-setting reference: `main_ppl` records (qwen3_1p7b); against this release, bf16: per-window NLLs identical; fo6: per-window NLLs identical.

### qwen3-8b (Qwen3-8B)

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 9.7251 | 13.3004 | -0.0286 ± 0.0032 * | -0.0338 ± 0.0028 * | better / better |
| FourOverSix (NVFP4, stock kernels) | 10.0071 | 13.7575 | — | — | reference |
| FlipQuant 8x64 (this release) | 9.6083 | 13.5284 | -0.0407 ± 0.0031 * | -0.0168 ± 0.0020 * | better / better |
| FlipQuant 16x64 (this release) | 9.5902 | 13.5554 | -0.0425 ± 0.0030 * | -0.0148 ± 0.0019 * | better / better |
| FlipQuant 256x64 (this release) | 9.6779 | 13.5653 | -0.0334 ± 0.0027 * | -0.0141 ± 0.0016 * | better / better |
| paper-setting map 8x64 (reference) | 9.5580 | 13.4969 | | | |
| paper-setting map 16x64 (reference) | 9.5871 | 13.5296 | | | |
| paper-setting map 256x64 (reference) | 9.7426 | 13.6293 | | | |

Paper-setting reference: `main_ppl` records (qwen3_8b); against this release, bf16: per-window NLLs identical; fo6: per-window NLLs identical.

### mistral-7b (Mistral-7B-Instruct-v0.3)

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 5.4961 | 8.1351 | -0.0379 ± 0.0015 * | -0.0319 ± 0.0017 * | better / better |
| FourOverSix (NVFP4, stock kernels) | 5.7083 | 8.3987 | — | — | reference |
| FlipQuant 8x64 (this release) | 5.6501 | 8.3188 | -0.0103 ± 0.0012 * | -0.0096 ± 0.0009 * | better / better |
| FlipQuant 16x64 (this release) | 5.6532 | 8.3236 | -0.0097 ± 0.0011 * | -0.0090 ± 0.0014 * | better / better |
| FlipQuant 256x64 (this release) | 5.6622 | 8.3397 | -0.0081 ± 0.0011 * | -0.0070 ± 0.0011 * | better / better |
| paper-setting map 8x64 (reference) | 5.6393 | 8.2939 | | | |
| paper-setting map 16x64 (reference) | 5.6384 | 8.2966 | | | |
| paper-setting map 256x64 (reference) | 5.6509 | 8.3145 | | | |

Paper-setting reference: `main_ppl` records (mistral7b_ins); against this release, bf16: per-window NLLs identical; fo6: per-window NLLs identical.

### mistral-7b-base (Mistral-7B-v0.3)

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 5.3182 | 7.8306 | -0.0377 ± 0.0014 * | -0.0292 ± 0.0017 * | better / better |
| FourOverSix (NVFP4, stock kernels) | 5.5224 | 8.0623 | — | — | reference |
| FlipQuant 8x64 (this release) | 5.4831 | 8.0250 | -0.0071 ± 0.0009 * | -0.0046 ± 0.0008 * | better / better |
| FlipQuant 16x64 (this release) | 5.4885 | 8.0321 | -0.0062 ± 0.0009 * | -0.0037 ± 0.0012 * | better / better |
| FlipQuant 256x64 (this release) | 5.4916 | 8.0336 | -0.0056 ± 0.0010 * | -0.0036 ± 0.0012 * | better / better |
| paper-setting map 8x64 (reference) | 5.4868 | 8.0215 | | | |
| paper-setting map 16x64 (reference) | 5.4899 | 8.0323 | | | |
| paper-setting map 256x64 (reference) | 5.4904 | 8.0279 | | | |

Paper-setting reference: `paper` records (mistral7b); against this release, bf16: per-window NLLs identical; fo6: per-window NLLs identical.

### llama3.1-8b (Llama-3.1-8B)

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 6.2403 | 8.9579 | -0.0962 ± 0.0069 * | -0.0925 ± 0.0157 * | better / better |
| FourOverSix (NVFP4, stock kernels) | 6.8706 | 9.8257 | — | — | reference |
| FlipQuant 8x64 (this release) | 6.7985 | 9.7013 | -0.0105 ± 0.0017 * | -0.0127 ± 0.0026 * | better / better |
| FlipQuant 16x64 (this release) | 6.8005 | 9.7087 | -0.0103 ± 0.0017 * | -0.0120 ± 0.0030 * | better / better |
| FlipQuant 256x64 (this release) | 6.8391 | 9.7529 | -0.0046 ± 0.0019 * | -0.0074 ± 0.0021 * | better / better |
| paper-setting map 8x64 (reference) | 6.7877 | 9.6774 | | | |
| paper-setting map 16x64 (reference) | 6.7827 | 9.6827 | | | |
| paper-setting map 256x64 (reference) | 6.8007 | 9.7252 | | | |

Paper-setting reference: `paper` records (llama8b); against this release, bf16: per-window NLLs identical; fo6: per-window NLLs identical.

### nemotron-nano-9b-v2 (NVIDIA-Nemotron-Nano-9B-v2)

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 8.0863 | 11.1534 | -0.0408 ± 0.0016 * | -0.0314 ± 0.0013 * | better / better |
| FourOverSix (NVFP4, stock kernels) | 8.4227 | 11.5093 | — | — | reference |
| FlipQuant 8x64 (this release) | 8.3477 | 11.4532 | -0.0089 ± 0.0012 * | -0.0049 ± 0.0008 * | better / better |
| FlipQuant 16x64 (this release) | 8.3570 | 11.4468 | -0.0078 ± 0.0012 * | -0.0054 ± 0.0008 * | better / better |
| FlipQuant 256x64 (this release) | 8.3706 | 11.4719 | -0.0062 ± 0.0012 * | -0.0033 ± 0.0008 * | better / better |
| paper-setting map 8x64 (reference) | 8.3335 | 11.4281 | | | |
| paper-setting map 16x64 (reference) | 8.3436 | 11.4358 | | | |
| paper-setting map 256x64 (reference) | 8.3519 | 11.4571 | | | |

Paper-setting reference: `main_ppl` records (nemotron9b); against this release, bf16: per-window NLLs identical; fo6: per-window NLLs identical.

### phi4-14b (Phi-4)

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 6.4615 | 10.3098 | -0.0305 ± 0.0022 * | -0.0225 ± 0.0014 * | better / better |
| FourOverSix (NVFP4, stock kernels) | 6.6617 | 10.5441 | — | — | reference |
| FlipQuant 8x64 (this release) | 6.6345 | 10.5187 | -0.0041 ± 0.0014 * | -0.0024 ± 0.0008 * | better / better |
| FlipQuant 16x64 (this release) | 6.6325 | 10.5157 | -0.0044 ± 0.0014 * | -0.0027 ± 0.0009 * | better / better |
| FlipQuant 256x64 (this release) | 6.6382 | 10.5181 | -0.0035 ± 0.0014 * | -0.0025 ± 0.0008 * | better / better |
| paper-setting map 8x64 (reference) | 6.6078 | 10.4979 | | | |
| paper-setting map 16x64 (reference) | 6.6133 | 10.5047 | | | |
| paper-setting map 256x64 (reference) | 6.6257 | 10.5146 | | | |

Paper-setting reference: `paper` records (phi4); against this release, bf16: per-window NLLs identical; fo6: per-window NLLs identical.

### qwen3.8-27b (Qwen3.8-27B)

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 7.0509 | 9.8935 | -0.0377 ± 0.0058 * | -0.0292 ± 0.0026 * | better / better |
| FourOverSix (NVFP4, stock kernels) | 7.3215 | 10.1869 | — | — | reference |
| FlipQuant 8x64 (this release) | 7.1430 | 10.1358 | -0.0247 ± 0.0059 * | -0.0050 ± 0.0009 * | better / better |
| FlipQuant 16x64 (this release) | 7.1705 | 10.1444 | -0.0208 ± 0.0046 * | -0.0042 ± 0.0010 * | better / better |
| FlipQuant 256x64 (this release) | 7.2369 | 10.1620 | -0.0116 ± 0.0037 * | -0.0024 ± 0.0008 * | better / better |
| paper-setting map 8x64 (reference) | 7.1074 | 10.1235 | | | |
| paper-setting map 16x64 (reference) | 7.1232 | 10.1276 | | | |
| paper-setting map 256x64 (reference) | 7.1773 | 10.1512 | | | |

Paper-setting reference: `paper` records (qwen27b); against this release, bf16: per-window NLLs identical; fo6: per-window NLLs identical.

## Provenance and checks

- **flipquant commit of each run** (branch optimized-defaults, pushed): `ecf8a7f`: 6 maps; `120173a`: 18 maps.
- **Fit extension** (every run passes): 128 new windows (64 math, 64 code), distinct documents and tokens, none of them a paper fit, development or published C4 document, nor a development window's tokens; identical across each model's three units. Skip sets (fit / development / published C4 documents): qwen3-1.7b 128/192/0; qwen3-8b 128/192/0; mistral-7b 128/192/0; mistral-7b-base 128/192/0; llama3.1-8b 128/192/231; nemotron-nano-9b-v2 128/192/0; phi4-14b 128/192/0; qwen3.8-27b 128/192/233. Only Llama-3.1-8B and Qwen3.8-27B have a published C4 record, so for the others the rule skips no C4 document. Checked directly instead: the 256 fit windows' documents against the C4 documents of this release's PPL evaluation (overlap / C4 documents): qwen3-1.7b 0/231; qwen3-8b 0/231; mistral-7b 0/235; mistral-7b-base 0/235; llama3.1-8b 0/231; nemotron-nano-9b-v2 0/233; phi4-14b 0/231; qwen3.8-27b 0/233.
- **Fit record** (the first 128 windows, sha256 of the record file): qwen3-1.7b: the same file as its committed paper-setting maps' record; qwen3-8b: the same file as its committed paper-setting maps' record; mistral-7b: the same file as its committed paper-setting maps' record; mistral-7b-base: the same file as the paper runs' record (multimodel/data); llama3.1-8b: flipquant's own record of the archived Llama fit set (`f3b37423e12c…`), not the paper runs' file; with it, `train_map --require-reproduction` reproduced the paper's Llama-3.1-8B 8x64 map bit for bit on 2026-10-05; nemotron-nano-9b-v2: the same file as its committed paper-setting maps' record; phi4-14b: the same file as the paper runs' record (multimodel/data); qwen3.8-27b: the same file as the paper runs' record (multimodel/data).
- **Qwen3.8-27B development sets:** redrawn by the vendored preparation and accepted by hash: fisher_subset_validate_v2_confirm `40d27f490937` matches (draw 5); preserved_row_confirm `a5f2480b09a7` matches (draw 1); ce_target_combinations_confirm `5484bd6bfe9d` matches (draw 4).
- **Repeat of an earlier run:** Llama-3.1-8B 16x64 at these settings was run before as topk-cal a53ce55's E256 (map `7bdc6e39cce82f66…`); this release's run map is `7bdc6e39cce82f66…`, the same.
- **mistral-7b-base routing:** `train_map --model mistral-7b-base --unit 8x64 --require-reproduction` at `120173a` reproduces the paper's Mistral-7B-v0.3 8x64 map (`8aacdd7706ca96d6…`).
- **Reference rows:** the paper-setting PPL records' BF16 and FourOverSix rows equal this release's per window for 16 of 16 rows, so those records use the same windows, convention and numerics.
- **Measurement re-runs** (topk-cal's measure.py): 19 calibrations re-run for the process-tree numbers; 19 of them reproduce their release map (run-map sha256 and tiles). Against the release runs, the re-runs' trainer total differs by at most 2.3 % and the mean epoch time by at most 1.0 %.
- **Simulated-quantization command** (model cards): smoke-tested on two windows per corpus for qwen3-1.7b, qwen3-8b, mistral-7b, mistral-7b-base, llama3.1-8b, nemotron-nano-9b-v2, phi4-14b; qwen3.8-27b: out of GPU memory (flipquant's fake path builds both candidates of every weight on the GPU next to the BF16 model; 96 GB is not enough for it), so its card gives the native command only.

## Map sha256

| model | file | sha256 |
|---|---|---|
| qwen3-1.7b | flipquant_8x64.pt | `4ddfc7361e60b1773f030add26e69a409620fbb0988de3130da0d3d3c90a5361` |
| qwen3-1.7b | flipquant_16x64.pt | `98ca73091a4e33d840d4762c75fefbaf7c13fad915448eda9f2e329bed39eda9` |
| qwen3-1.7b | flipquant_256x64.pt | `91df20cfb8ca39056d7166215afadba71f3c6b449adccc48deb0cf6f3a3e3abe` |
| qwen3-8b | flipquant_8x64.pt | `f0760449f1aba9665b042d2b44c710c6e3bf4a0a6d458983bd59723b2d82847e` |
| qwen3-8b | flipquant_16x64.pt | `3a363d75ea973a521e1fa4411c7ee85c1f1b8f88e3ed0360c0d14272c2151fc7` |
| qwen3-8b | flipquant_256x64.pt | `340fb09949a1f7980231c8a6b5bb35480d90a89634ae96c4ed777bd325e4e89d` |
| mistral-7b | flipquant_8x64.pt | `765729fb1586c02264b0c8907d7be5d19734a9d19d3e23dc687a878f1079ab14` |
| mistral-7b | flipquant_16x64.pt | `4ddbdd11c0fb79eb7051412a0879345acecb56d583cc98c02a2f9a6a388c071e` |
| mistral-7b | flipquant_256x64.pt | `a9e3732846fc82e7cdf23889387cc0a4e9627a431cb49252344b8caaeeda9aec` |
| mistral-7b-base | flipquant_8x64.pt | `9c2472331a1ec8bfd140f42f4ed430f5f25ac64d24dab53147a5dabb8d8d8153` |
| mistral-7b-base | flipquant_16x64.pt | `f8e4421784ed7985b9e4d8a1ba0650c5d02962cce0715a9b8b94c5e1447a19a1` |
| mistral-7b-base | flipquant_256x64.pt | `d8ea6a24aca0c1bb8544d54b0cbe0352cbbd73206e6f28f2c398f06e2cd71870` |
| llama3.1-8b | flipquant_8x64.pt | `946fd3c4b4017086a8d2cc8bfea1445ceda87a217e257ee923a2f516f8ba666c` |
| llama3.1-8b | flipquant_16x64.pt | `d4ccc18b61a6db781d7799723c47a905ca69407d72a7df009e04929ea877e150` |
| llama3.1-8b | flipquant_256x64.pt | `fcff3f58c18e786580b9c408dd666a0dc22cc2c547b43d17b9ac6051f38aa49c` |
| nemotron-nano-9b-v2 | flipquant_8x64.pt | `9ecc2ded8d5c1a8f7b677d556a245dd5515c5bea20b313889a11b0f32e20aad5` |
| nemotron-nano-9b-v2 | flipquant_16x64.pt | `6bae7360fc0083bb3e2bd57c1f1839640845b5188a58adb850ba33883cbd2d1e` |
| nemotron-nano-9b-v2 | flipquant_256x64.pt | `53c6d91fc6be656ad4196cbe7586d9ee75a5007b9f693ff0a0d43e13e0447e76` |
| phi4-14b | flipquant_8x64.pt | `fc45a7acffe95ac4bd3bbeb3b7eaa5fff513f1029b00965b51b991bec5b00dc6` |
| phi4-14b | flipquant_16x64.pt | `79c81a945a6e9e252e72581b4abb6f2d3c06938aa45a16e10edb831bd88c075d` |
| phi4-14b | flipquant_256x64.pt | `53d9f46fedeb579141a5a1575e69e92ba6a084f97247ee68328f65eefd871e89` |
| qwen3.8-27b | flipquant_8x64.pt | `17c8e5c79c9c3900d448765ba7324c3b8149cc5883f208e0a18e15ae4ef88ad0` |
| qwen3.8-27b | flipquant_16x64.pt | `9b051a0c76796d12236bbdd3c80bb0815db7c00f5f370110c5e7dc5d04cc710e` |
| qwen3.8-27b | flipquant_256x64.pt | `914e843b1ce4aa494ff0a093188a4435f45dac81f9cb56fe6147c13901188e5a` |
