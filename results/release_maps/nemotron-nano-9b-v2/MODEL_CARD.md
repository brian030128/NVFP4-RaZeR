---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/NVIDIA-Nemotron-Nano-9B-v2
tags:
- flipquant
- nvfp4
- mixfp4
- fp4
- quantization
---

# FlipQuant MixFP4 maps for NVIDIA-Nemotron-Nano-9B-v2 (DRAFT)

> Draft model card, for review before any upload. The repository name, the flipquant link and the license wording are the uploader's decision.

## What this is

Format maps for [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) at revision `6533e8de2c68e4536bf7c411d7a3ce5734111476`, calibrated with FlipQuant (TM-OPT+TC). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 130,579 of 15,061,760 (0.87 %) | `9ecc2ded8d5c1a8f7b677d556a245dd5515c5bea20b313889a11b0f32e20aad5` |
| `flipquant_16x64.pt` | 16 × 64 | 80,098 of 7,530,880 (1.06 %) | `6bae7360fc0083bb3e2bd57c1f1839640845b5188a58adb850ba33883cbd2d1e` |
| `flipquant_256x64.pt` | 256 × 64 | 12,679 of 478,320 (2.65 %) | `53c6d91fc6be656ad4196cbe7586d9ee75a5007b9f693ff0a0d43e13e0447e76` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

## Calibration

`python -m calibration.train_map --model nemotron-nano-9b-v2 --unit <unit> --fit-windows 256 --epochs 5 --teacher-topk 1000` (flipquant commit `120173a4f67c67f5e67f93ea02fa7a9eaff3ef14`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, micro-batch 2 × accumulation 4 (8 sequences per step), 5 epochs, deterministic.
- Fit set: 256 windows of 512 tokens. They are the FlipQuant paper's 128 math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), plus 128 more by the extension rule, which skips the paper's fit, development and published C4 evaluation documents. Every window is recorded by document hash, offset and token hash (`records/<unit>/trainer_report.json`).
- These are not the paper's settings (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| model | unit | model load | data (+ extension) | teacher | lean packing | setup | per epoch (mean, min–max) | training | trainer total | train_map end-to-end | batch × accum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| nemotron-nano-9b-v2 | 8x64 | 2.3 s | 19.8 s | 149.9 s | 40.5 s | 212.5 s | 479.2 s (479.1–479.6) | 39.9 min | 43.5 min | 43.8 min ᴿ | 2 × 4 |
| nemotron-nano-9b-v2 | 16x64 | 1.8 s | 20.8 s | 151.7 s | 41.1 s | 215.4 s | 478.8 s (478.7–479.2) | 39.9 min | 43.5 min | 43.7 min | 2 × 4 |
| nemotron-nano-9b-v2 | 256x64 | 1.8 s | 19.9 s | 151.7 s | 40.9 s | 214.3 s | 481.2 s (481.0–481.4) | 40.1 min | 43.7 min | 43.9 min | 2 × 4 |

| model | unit | GPU peak allocated | GPU peak reserved | trainer peak RSS (ru_maxrss) | RSS after model load | teacher (top-1000) | wrapper RSS | process-tree peak (10 Hz) | source of the last two |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| nemotron-nano-9b-v2 | 8x64 | 53.9 GiB | 55.3 GiB | 17.5 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 16.8 GiB | re-run |
| nemotron-nano-9b-v2 | 16x64 | 53.7 GiB | 55.2 GiB | 17.5 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 3.1 GiB (sampled after attaching; the processes' exact VmHWM peaks sum to 18.3 GiB) | live, attached late |
| nemotron-nano-9b-v2 | 256x64 | 53.7 GiB | 54.9 GiB | 17.5 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 16.1 GiB | live |

The one-time data preparation (fit and development records, before the first unit):

train_map's first run of a model on a new data root prepares its calibration record and development sets (`calibration.tmopt_common.prepare` and `prepare_development`; the builder rule's checks, re-tokenizing the fit windows, hashing the model's weights on the CPU unless an archived record has them, the development draws). Measured on 2026-10-06 in a fresh data root with measure.py's sampling plus each process's VmHWM; the files it wrote are byte-identical to the ones the release runs used. The dataset shards were already in the local Hugging Face cache; a first-ever preparation also downloads them.

| model | wall time | process-tree peak (10 Hz) | wrapper | largest process (VmHWM) | records identical to the release run's |
|---|---:|---:|---:|---:|---|
| nemotron-nano-9b-v2 | 3.2 min | 17.5 GiB | 0.6 GiB | 17.0 GiB | yes (11 files) |

Time and memory: the trainer's numbers (phases, epochs, GPU peaks, its own `ru_maxrss`, RSS after the model load, teacher storage) come from its PhaseMonitor in the release run. The wrapper's own RSS, the process-tree peak (wrapper + trainer, sampled at 10 Hz) and the end-to-end time come from topk-cal's `measure.py` method: for the first runs (ᴿ, source “re-run”) from a measured re-run of the same calibration that reproduced the same map, for the later ones from a watcher attached to the run itself (“live”; Nemotron 16x64 was attached 13 min after its start, so its tree peak covers only the rest, while its VmHWM peaks are exact). The end-to-end time excludes the one-time data preparation, which has its own row.

## Perplexity

Native sm_120 kernels, flipquant's `--paper-convention` (per-token FourOverSix activations), WikiText-2 (146 windows of 2048 tokens) and C4 (256 windows of 2048 tokens). ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE.

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

The paper-setting rows are maps calibrated at the paper's settings (20 epochs, 128 windows, the full teacher), evaluated under the same convention; they are listed for comparison only.

## How to use (flipquant)

On an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels:

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model nemotron-nano-9b-v2 --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

Measured here with the 16x64 map: 39 GiB of GPU memory at peak (2048-token windows, batch 1), 16.0 min for both corpora, model load included.

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model nemotron-nano-9b-v2 --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the table above (with `--out`); the simulated one was smoke-tested on two windows per corpus.

flipquant: <repository link to be added by the uploader>.

## License

The base model is under the NVIDIA Open Model License. These maps are derived from it; this draft assumes the same license (the uploader's decision).

## Files

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `records/<unit>/`: `reproduction.json` and `run.json` (settings, commit, time and memory), the trainer's report (`trainer_report.json`: phases, the fit extension's windows) and its log.
- `ppl/`: the evaluation reports (per-window NLLs) and `significance.json`.
