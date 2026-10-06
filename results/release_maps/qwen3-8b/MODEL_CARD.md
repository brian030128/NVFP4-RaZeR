---
license: apache-2.0
base_model: Qwen/Qwen3-8B
tags:
- flipquant
- nvfp4
- mixfp4
- fp4
- quantization
---

# FlipQuant MixFP4 maps for Qwen3-8B (DRAFT)

> Draft model card, for review before any upload. The repository name, the flipquant link and the license wording are the uploader's decision.

## What this is

Format maps for [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) at revision `b968826d9c46dd6066d109eabc6255188de91218`, calibrated with FlipQuant (TM-OPT+TC). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 67,105 of 13,565,952 (0.49 %) | `f0760449f1aba9665b042d2b44c710c6e3bf4a0a6d458983bd59723b2d82847e` |
| `flipquant_16x64.pt` | 16 × 64 | 37,511 of 6,782,976 (0.55 %) | `3a363d75ea973a521e1fa4411c7ee85c1f1b8f88e3ed0360c0d14272c2151fc7` |
| `flipquant_256x64.pt` | 256 × 64 | 5,386 of 423,936 (1.27 %) | `340fb09949a1f7980231c8a6b5bb35480d90a89634ae96c4ed777bd325e4e89d` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

## Calibration

`python -m calibration.train_map --model qwen3-8b --unit <unit> --fit-windows 256 --epochs 5 --teacher-topk 1000` (flipquant commit `120173a4f67c67f5e67f93ea02fa7a9eaff3ef14`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs, deterministic.
- Fit set: 256 windows of 512 tokens. They are the FlipQuant paper's 128 math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), plus 128 more by the extension rule, which skips the paper's fit, development and published C4 evaluation documents. Every window is recorded by document hash, offset and token hash (`records/<unit>/trainer_report.json`).
- These are not the paper's settings (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| model | unit | model load | data (+ extension) | teacher | lean packing | setup | per epoch (mean, min–max) | training | trainer total | train_map end-to-end | batch × accum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3-8b | 8x64 | 2.4 s | 19.3 s | 14.8 s | 35.9 s | 72.3 s | 41.6 s (40.7–42.2) | 3.5 min | 4.7 min | 4.9 min ᴿ | 8 × 1 |
| qwen3-8b | 16x64 | 1.6 s | 21.2 s | 15.3 s | 36.0 s | 74.1 s | 42.0 s (41.7–42.2) | 3.5 min | 4.7 min | 4.9 min ᴿ | 8 × 1 |
| qwen3-8b | 256x64 | 1.6 s | 19.2 s | 15.1 s | 37.0 s | 72.9 s | 42.0 s (41.6–42.2) | 3.5 min | 4.7 min | 4.9 min ᴿ | 8 × 1 |

| model | unit | GPU peak allocated | GPU peak reserved | trainer peak RSS (ru_maxrss) | RSS after model load | teacher (top-1000) | wrapper RSS | process-tree peak (10 Hz) | source of the last two |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3-8b | 8x64 | 47.0 GiB | 47.6 GiB | 16.2 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 16.4 GiB | re-run |
| qwen3-8b | 16x64 | 46.8 GiB | 47.5 GiB | 16.2 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 16.2 GiB | re-run |
| qwen3-8b | 256x64 | 46.7 GiB | 47.3 GiB | 16.2 GiB | 0.9 GiB | 0.73 GiB | 0.9 GiB (VmHWM) | 15.0 GiB | re-run |

The one-time data preparation (fit and development records, before the first unit):

train_map's first run of a model on a new data root prepares its calibration record and development sets (`calibration.tmopt_common.prepare` and `prepare_development`; the builder rule's checks, re-tokenizing the fit windows, hashing the model's weights on the CPU unless an archived record has them, the development draws). Measured on 2026-10-06 in a fresh data root with measure.py's sampling plus each process's VmHWM; the files it wrote are byte-identical to the ones the release runs used. The dataset shards were already in the local Hugging Face cache; a first-ever preparation also downloads them.

| model | wall time | process-tree peak (10 Hz) | wrapper | largest process (VmHWM) | records identical to the release run's |
|---|---:|---:|---:|---:|---|
| qwen3-8b | 3.1 min | 16.1 GiB | 0.6 GiB | 15.6 GiB | yes (11 files) |

Time and memory: the trainer's numbers (phases, epochs, GPU peaks, its own `ru_maxrss`, RSS after the model load, teacher storage) come from its PhaseMonitor in the release run. The wrapper's own RSS, the process-tree peak (wrapper + trainer, sampled at 10 Hz) and the end-to-end time come from topk-cal's `measure.py` method: for the first runs (ᴿ, source “re-run”) from a measured re-run of the same calibration that reproduced the same map, for the later ones from a watcher attached to the run itself (“live”; Nemotron 16x64 was attached 13 min after its start, so its tree peak covers only the rest, while its VmHWM peaks are exact). The end-to-end time excludes the one-time data preparation, which has its own row.

## Perplexity

Native sm_120 kernels, flipquant's `--paper-convention` (per-token FourOverSix activations), WikiText-2 (146 windows of 2048 tokens) and C4 (256 windows of 2048 tokens). ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE.

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

The paper-setting rows are maps calibrated at the paper's settings (20 epochs, 128 windows, the full teacher), evaluated under the same convention; they are listed for comparison only.

## How to use (flipquant)

On an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels:

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model qwen3-8b --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

Measured here with the 16x64 map: 22 GiB of GPU memory at peak (2048-token windows, batch 1), 48 s for both corpora, model load included.

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model qwen3-8b --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the table above (with `--out`); the simulated one was smoke-tested on two windows per corpus.

flipquant: <repository link to be added by the uploader>.

## License

The base model is under the Apache License 2.0. These maps are derived from it; this draft assumes the same license (the uploader's decision).

## Files

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `records/<unit>/`: `reproduction.json` and `run.json` (settings, commit, time and memory), the trainer's report (`trainer_report.json`: phases, the fit extension's windows) and its log.
- `ppl/`: the evaluation reports (per-window NLLs) and `significance.json`.
