---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- flipquant
- nvfp4
- mixfp4
- fp4
- quantization
---

# FlipQuant MixFP4 maps for Qwen3.8-27B (DRAFT)

> Draft model card, for review before any upload. The repository name, the flipquant link and the license wording are the uploader's decision.

## What this is

Format maps for [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, calibrated with FlipQuant (TM-OPT+TC). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 169,073 of 47,559,680 (0.36 %) | `17c8e5c79c9c3900d448765ba7324c3b8149cc5883f208e0a18e15ae4ef88ad0` |
| `flipquant_16x64.pt` | 16 × 64 | 105,278 of 23,779,840 (0.44 %) | `9b051a0c76796d12236bbdd3c80bb0815db7c00f5f370110c5e7dc5d04cc710e` |
| `flipquant_256x64.pt` | 256 × 64 | 13,758 of 1,492,480 (0.92 %) | `914e843b1ce4aa494ff0a093188a4435f45dac81f9cb56fe6147c13901188e5a` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

## Calibration

`python -m calibration.train_map --model qwen3.8-27b --unit <unit> --fit-windows 256 --epochs 5 --teacher-topk 1000` (flipquant commit `120173a4f67c67f5e67f93ea02fa7a9eaff3ef14`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, micro-batch 2 × accumulation 4 (8 sequences per step), 5 epochs, deterministic.
- Fit set: 256 windows of 512 tokens. They are the FlipQuant paper's 128 math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), plus 128 more by the extension rule, which skips the paper's fit, development and published C4 evaluation documents. Every window is recorded by document hash, offset and token hash (`records/<unit>/trainer_report.json`).
- These are not the paper's settings (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| model | unit | model load | data (+ extension) | teacher | lean packing | setup | per epoch (mean, min–max) | training | trainer total | train_map end-to-end | batch × accum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3.8-27b | 8x64 | 6.7 s | 19.2 s | 165.9 s | 129.7 s | 321.6 s | 432.8 s (429.7–435.1) | 36.1 min | 41.4 min | 41.6 min | 2 × 4 |
| qwen3.8-27b | 16x64 | 6.7 s | 19.4 s | 163.3 s | 130.6 s | 319.9 s | 430.9 s (428.2–434.2) | 35.9 min | 41.2 min | 41.5 min | 2 × 4 |
| qwen3.8-27b | 256x64 | 6.7 s | 19.4 s | 165.6 s | 127.7 s | 319.3 s | 431.1 s (428.1–432.8) | 35.9 min | 41.3 min | 41.5 min | 2 × 4 |

| model | unit | GPU peak allocated | GPU peak reserved | trainer peak RSS (ru_maxrss) | RSS after model load | teacher (top-1000) | wrapper RSS | process-tree peak (10 Hz) | source of the last two |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3.8-27b | 8x64 | 91.7 GiB | 92.3 GiB | 51.9 GiB | 1.0 GiB | 0.73 GiB | 0.9 GiB (VmHWM) | 51.9 GiB | live |
| qwen3.8-27b | 16x64 | 91.3 GiB | 92.8 GiB | 51.9 GiB | 1.0 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 52.3 GiB | live |
| qwen3.8-27b | 256x64 | 90.9 GiB | 92.3 GiB | 51.9 GiB | 1.0 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 52.2 GiB | live |

The one-time data preparation (fit and development records, before the first unit):

train_map's first run of a model on a new data root prepares its calibration record and development sets (`calibration.tmopt_common.prepare` and `prepare_development`; the builder rule's checks, re-tokenizing the fit windows, hashing the model's weights on the CPU unless an archived record has them, the development draws). Measured on 2026-10-06 in a fresh data root with measure.py's sampling plus each process's VmHWM; the files it wrote are byte-identical to the ones the release runs used. The dataset shards were already in the local Hugging Face cache; a first-ever preparation also downloads them.

| model | wall time | process-tree peak (10 Hz) | wrapper | largest process (VmHWM) | records identical to the release run's |
|---|---:|---:|---:|---:|---|
| qwen3.8-27b | 92 s | 2.0 GiB | 0.5 GiB | 1.5 GiB | yes (11 files) |

Time and memory: the trainer's numbers (phases, epochs, GPU peaks, its own `ru_maxrss`, RSS after the model load, teacher storage) come from its PhaseMonitor in the release run. The wrapper's own RSS, the process-tree peak (wrapper + trainer, sampled at 10 Hz) and the end-to-end time come from topk-cal's `measure.py` method: for the first runs (ᴿ, source “re-run”) from a measured re-run of the same calibration that reproduced the same map, for the later ones from a watcher attached to the run itself (“live”; Nemotron 16x64 was attached 13 min after its start, so its tree peak covers only the rest, while its VmHWM peaks are exact). The end-to-end time excludes the one-time data preparation, which has its own row.

## Perplexity

Native sm_120 kernels, flipquant's `--paper-convention` (per-token FourOverSix activations), WikiText-2 (146 windows of 2048 tokens) and C4 (256 windows of 2048 tokens). ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE.

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

The paper-setting rows are maps calibrated at the paper's settings (20 epochs, 128 windows, the full teacher), evaluated under the same convention; they are listed for comparison only.

## How to use (flipquant)

On an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels:

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model qwen3.8-27b --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

Measured here with the 16x64 map: 70 GiB of GPU memory at peak (2048-token windows, batch 1), 6.5 min for both corpora, model load included.

The native command produced the table above (with `--out`). flipquant's simulated (fake) path does not fit this model on one 96 GB GPU: it builds both candidate encodings of every weight on the GPU, next to the BF16 model, and ran out of memory while doing so (smoke test, 2026-10-06). Use the native kernels, or a GPU with more memory.

flipquant: <repository link to be added by the uploader>.

## License

The base model is under the Apache License 2.0. These maps are derived from it; this draft assumes the same license (the uploader's decision).

## Files

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `records/<unit>/`: `reproduction.json` and `run.json` (settings, commit, time and memory), the trainer's report (`trainer_report.json`: phases, the fit extension's windows) and its log.
- `ppl/`: the evaluation reports (per-window NLLs) and `significance.json`.
