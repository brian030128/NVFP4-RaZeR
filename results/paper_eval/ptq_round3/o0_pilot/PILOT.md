# O0 pilot: the BF16 GSM8K batch (not a result)

Part N's pilot documents per model (the 64 longest questions + 64 others), BF16, part N's command. A batch b fits if the pilot completes and nvidia-smi's peak plus the KV growth bound b x 1024 tokens x KV per token (64 KiB Qwen3.8-27B, 16 KiB Nemotron-Nano-9B-v2) is at most 97,280 MiB (95 GiB).

| model | batch | rc | nvidia-smi peak (MiB) | + KV bound (MiB) | total (MiB) | torch peak allocated (GiB) | seconds | strict / flexible (pilot docs) | fits |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Nemotron-Nano-9B-v2 | 64 | 0 | 55,408 | 1,024 | 56,432 | 45.64 | 58 | 0.7734375 / 0.8046875 | yes |
| Qwen3.8-27B | 64 | 0 | 94,320 | 4,096 | 98,416 | 83.15 | 372 | 0.953125 / 0.96875 | no |
| Qwen3.8-27B | 48 | 0 | 83,974 | 3,072 | 87,046 | 75.10 | 373 | 0.953125 / 0.96875 | yes |

Chosen (the largest batch that fits): Nemotron-Nano-9B-v2 64, Qwen3.8-27B 48

