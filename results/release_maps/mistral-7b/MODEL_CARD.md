---
license: apache-2.0
base_model: mistralai/Mistral-7B-Instruct-v0.3
tags:
- flipquant
- nvfp4
- mixfp4
- fp4
- quantization
---

# FlipQuant MixFP4 maps for Mistral-7B-Instruct-v0.3 (DRAFT)

> Draft model card, for review before any upload. The repository name, the flipquant link and the license wording are the uploader's decision.

## What this is

Format maps for [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) at revision `c170c708c41dac9275d15a8fff4eca08d52bab71`, calibrated with FlipQuant (TM-OPT+TC). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 116,977 of 13,631,488 (0.86 %) | `765729fb1586c02264b0c8907d7be5d19734a9d19d3e23dc687a878f1079ab14` |
| `flipquant_16x64.pt` | 16 × 64 | 75,906 of 6,815,744 (1.11 %) | `4ddbdd11c0fb79eb7051412a0879345acecb56d583cc98c02a2f9a6a388c071e` |
| `flipquant_256x64.pt` | 256 × 64 | 11,491 of 425,984 (2.70 %) | `a9e3732846fc82e7cdf23889387cc0a4e9627a431cb49252344b8caaeeda9aec` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

## Calibration

`python -m calibration.train_map --model mistral-7b --unit <unit> --fit-windows 256 --epochs 5 --teacher-topk 1000` (flipquant commit `ecf8a7f6cb4c48379d960d6d32a7818890fe4541`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs, deterministic.
- Fit set: 256 windows of 512 tokens. They are the FlipQuant paper's 128 math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), plus 128 more by the extension rule, which skips the paper's fit, development and published C4 evaluation documents. Every window is recorded by document hash, offset and token hash (`records/<unit>/trainer_report.json`).
- These are not the paper's settings (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| model | unit | model load | data (+ extension) | teacher | lean packing | setup | per epoch (mean, min–max) | training | trainer total | train_map end-to-end | batch × accum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mistral-7b | 8x64 | 1.7 s | 20.9 s | 10.6 s | 36.4 s | 69.5 s | 37.8 s (37.0–38.4) | 3.2 min | 4.3 min | 4.6 min ᴿ | 8 × 1 |
| mistral-7b | 16x64 | 1.5 s | 18.9 s | 10.6 s | 35.9 s | 66.9 s | 38.2 s (37.9–38.4) | 3.2 min | 4.3 min | 4.5 min ᴿ | 8 × 1 |
| mistral-7b | 256x64 | 1.5 s | 19.3 s | 10.6 s | 36.6 s | 68.0 s | 38.2 s (37.9–38.4) | 3.2 min | 4.3 min | 4.5 min ᴿ | 8 × 1 |

| model | unit | GPU peak allocated | GPU peak reserved | trainer peak RSS (ru_maxrss) | RSS after model load | teacher (top-1000) | wrapper RSS | process-tree peak (10 Hz) | source of the last two |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mistral-7b | 8x64 | 36.8 GiB | 37.6 GiB | 14.4 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 13.8 GiB | re-run |
| mistral-7b | 16x64 | 36.6 GiB | 37.5 GiB | 14.4 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 13.9 GiB | re-run |
| mistral-7b | 256x64 | 36.5 GiB | 37.3 GiB | 14.4 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 13.7 GiB | re-run |

The one-time data preparation (fit and development records, before the first unit):

train_map's first run of a model on a new data root prepares its calibration record and development sets (`calibration.tmopt_common.prepare` and `prepare_development`; the builder rule's checks, re-tokenizing the fit windows, hashing the model's weights on the CPU unless an archived record has them, the development draws). Measured on 2026-10-06 in a fresh data root with measure.py's sampling plus each process's VmHWM; the files it wrote are byte-identical to the ones the release runs used. The dataset shards were already in the local Hugging Face cache; a first-ever preparation also downloads them.

| model | wall time | process-tree peak (10 Hz) | wrapper | largest process (VmHWM) | records identical to the release run's |
|---|---:|---:|---:|---:|---|
| mistral-7b | 3.0 min | 16.1 GiB | 0.5 GiB | 15.6 GiB | yes (11 files) |

Time and memory: the trainer's numbers (phases, epochs, GPU peaks, its own `ru_maxrss`, RSS after the model load, teacher storage) come from its PhaseMonitor in the release run. The wrapper's own RSS, the process-tree peak (wrapper + trainer, sampled at 10 Hz) and the end-to-end time come from topk-cal's `measure.py` method: for the first runs (ᴿ, source “re-run”) from a measured re-run of the same calibration that reproduced the same map, for the later ones from a watcher attached to the run itself (“live”; Nemotron 16x64 was attached 13 min after its start, so its tree peak covers only the rest, while its VmHWM peaks are exact). The end-to-end time excludes the one-time data preparation, which has its own row.

## Perplexity

Native sm_120 kernels, flipquant's `--paper-convention` (per-token FourOverSix activations), WikiText-2 (146 windows of 2048 tokens) and C4 (256 windows of 2048 tokens). ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE.

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

The paper-setting rows are maps calibrated at the paper's settings (20 epochs, 128 windows, the full teacher), evaluated under the same convention; they are listed for comparison only.

## How to use (flipquant)

On an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels:

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model mistral-7b --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

Measured here with the 16x64 map: 21 GiB of GPU memory at peak (2048-token windows, batch 1), 42 s for both corpora, model load included.

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model mistral-7b --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the table above (with `--out`); the simulated one was smoke-tested on two windows per corpus.

flipquant: <repository link to be added by the uploader>.

## License

The base model is under the Apache License 2.0. These maps are derived from it; this draft assumes the same license (the uploader's decision).

## Files

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `records/<unit>/`: `reproduction.json` and `run.json` (settings, commit, time and memory), the trainer's report (`trainer_report.json`: phases, the fit extension's windows) and its log.
- `ppl/`: the evaluation reports (per-window NLLs) and `significance.json`.
