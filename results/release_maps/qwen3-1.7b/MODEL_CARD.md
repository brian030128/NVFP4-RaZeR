---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
tags:
- flipquant
- nvfp4
- mixfp4
- fp4
- quantization
---

# FlipQuant MixFP4 maps for Qwen3-1.7B (DRAFT)

> Draft model card, for review before any upload. The repository name, the flipquant link and the license wording are the uploader's decision.

## What this is

Format maps for [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, calibrated with FlipQuant (TM-OPT+TC). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 24,675 of 2,752,512 (0.90 %) | `4ddfc7361e60b1773f030add26e69a409620fbb0988de3130da0d3d3c90a5361` |
| `flipquant_16x64.pt` | 16 × 64 | 16,351 of 1,376,256 (1.19 %) | `98ca73091a4e33d840d4762c75fefbaf7c13fad915448eda9f2e329bed39eda9` |
| `flipquant_256x64.pt` | 256 × 64 | 3,340 of 86,016 (3.88 %) | `91df20cfb8ca39056d7166215afadba71f3c6b449adccc48deb0cf6f3a3e3abe` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

## Calibration

`python -m calibration.train_map --model qwen3-1.7b --unit <unit> --fit-windows 256 --epochs 5 --teacher-topk 1000` (flipquant commit `ecf8a7f6cb4c48379d960d6d32a7818890fe4541`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs, deterministic.
- Fit set: 256 windows of 512 tokens. They are the FlipQuant paper's 128 math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), plus 128 more by the extension rule, which skips the paper's fit, development and published C4 evaluation documents. Every window is recorded by document hash, offset and token hash (`records/<unit>/trainer_report.json`).
- These are not the paper's settings (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| model | unit | model load | data (+ extension) | teacher | lean packing | setup | per epoch (mean, min–max) | training | trainer total | train_map end-to-end | batch × accum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3-1.7b | 8x64 | 0.7 s | 20.4 s | 12.4 s | 5.8 s | 39.2 s | 12.0 s (11.6–12.3) | 60 s | 99 s | 111 s ᴿ | 8 × 1 |
| qwen3-1.7b | 16x64 | 0.8 s | 20.1 s | 12.2 s | 5.7 s | 38.8 s | 12.2 s (11.9–12.6) | 61 s | 100 s | 112 s ᴿ | 8 × 1 |
| qwen3-1.7b | 256x64 | 0.8 s | 20.3 s | 12.7 s | 5.8 s | 39.6 s | 12.2 s (12.0–12.7) | 61 s | 101 s | 110 s ᴿ | 8 × 1 |

| model | unit | GPU peak allocated | GPU peak reserved | trainer peak RSS (ru_maxrss) | RSS after model load | teacher (top-1000) | wrapper RSS | process-tree peak (10 Hz) | source of the last two |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3-1.7b | 8x64 | 21.0 GiB | 21.4 GiB | 4.7 GiB | 1.0 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 4.8 GiB | re-run |
| qwen3-1.7b | 16x64 | 21.0 GiB | 21.4 GiB | 4.7 GiB | 1.0 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 4.4 GiB | re-run |
| qwen3-1.7b | 256x64 | 21.0 GiB | 21.4 GiB | 4.7 GiB | 1.0 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 5.0 GiB | re-run |

The one-time data preparation (fit and development records, before the first unit):

train_map's first run of a model on a new data root prepares its calibration record and development sets (`calibration.tmopt_common.prepare` and `prepare_development`; the builder rule's checks, re-tokenizing the fit windows, hashing the model's weights on the CPU unless an archived record has them, the development draws). Measured on 2026-10-06 in a fresh data root with measure.py's sampling plus each process's VmHWM; the files it wrote are byte-identical to the ones the release runs used. The dataset shards were already in the local Hugging Face cache; a first-ever preparation also downloads them.

| model | wall time | process-tree peak (10 Hz) | wrapper | largest process (VmHWM) | records identical to the release run's |
|---|---:|---:|---:|---:|---|
| qwen3-1.7b | 2.7 min | 6.2 GiB | 0.5 GiB | 5.7 GiB | yes (11 files) |

Time and memory: the trainer's numbers (phases, epochs, GPU peaks, its own `ru_maxrss`, RSS after the model load, teacher storage) come from its PhaseMonitor in the release run. The wrapper's own RSS, the process-tree peak (wrapper + trainer, sampled at 10 Hz) and the end-to-end time come from topk-cal's `measure.py` method: for the first runs (ᴿ, source “re-run”) from a measured re-run of the same calibration that reproduced the same map, for the later ones from a watcher attached to the run itself (“live”; Nemotron 16x64 was attached 13 min after its start, so its tree peak covers only the rest, while its VmHWM peaks are exact). The end-to-end time excludes the one-time data preparation, which has its own row.

## Perplexity

Native sm_120 kernels, flipquant's `--paper-convention` (per-token FourOverSix activations), WikiText-2 (146 windows of 2048 tokens) and C4 (256 windows of 2048 tokens). ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE.

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

The paper-setting rows are maps calibrated at the paper's settings (20 epochs, 128 windows, the full teacher), evaluated under the same convention; they are listed for comparison only.

## How to use (flipquant)

On an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels:

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model qwen3-1.7b --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

Measured here with the 16x64 map: 8 GiB of GPU memory at peak (2048-token windows, batch 1), 28 s for both corpora, model load included.

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model qwen3-1.7b --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the table above (with `--out`); the simulated one was smoke-tested on two windows per corpus.

flipquant: <repository link to be added by the uploader>.

## License

The base model is under the Apache License 2.0. These maps are derived from it; this draft assumes the same license (the uploader's decision).

## Files

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `records/<unit>/`: `reproduction.json` and `run.json` (settings, commit, time and memory), the trainer's report (`trainer_report.json`: phases, the fit extension's windows) and its log.
- `ppl/`: the evaluation reports (per-window NLLs) and `significance.json`.
