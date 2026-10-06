---
license: mit
base_model: microsoft/phi-4
tags:
- flipquant
- nvfp4
- mixfp4
- fp4
- quantization
---

# FlipQuant MixFP4 maps for Phi-4 (DRAFT)

> Draft model card, for review before any upload. The repository name, the flipquant link and the license wording are the uploader's decision.

## What this is

Format maps for [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) at revision `2db69c1c3e91a05d2c64a3185acfbaf36f744e25`, calibrated with FlipQuant (TM-OPT+TC). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 129,730 of 26,624,000 (0.49 %) | `fc45a7acffe95ac4bd3bbeb3b7eaa5fff513f1029b00965b51b991bec5b00dc6` |
| `flipquant_16x64.pt` | 16 × 64 | 79,591 of 13,312,000 (0.60 %) | `79c81a945a6e9e252e72581b4abb6f2d3c06938aa45a16e10edb831bd88c075d` |
| `flipquant_256x64.pt` | 256 × 64 | 11,059 of 832,000 (1.33 %) | `53d9f46fedeb579141a5a1575e69e92ba6a084f97247ee68328f65eefd871e89` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

## Calibration

`python -m calibration.train_map --model phi4-14b --unit <unit> --fit-windows 256 --epochs 5 --teacher-topk 1000` (flipquant commit `120173a4f67c67f5e67f93ea02fa7a9eaff3ef14`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs, deterministic.
- Fit set: 256 windows of 512 tokens. They are the FlipQuant paper's 128 math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), plus 128 more by the extension rule, which skips the paper's fit, development and published C4 evaluation documents. Every window is recorded by document hash, offset and token hash (`records/<unit>/trainer_report.json`).
- These are not the paper's settings (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| model | unit | model load | data (+ extension) | teacher | lean packing | setup | per epoch (mean, min–max) | training | trainer total | train_map end-to-end | batch × accum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| phi4-14b | 8x64 | 3.6 s | 19.6 s | 17.5 s | 73.6 s | 114.2 s | 75.0 s (73.7–75.6) | 6.3 min | 8.2 min | 8.4 min ᴿ | 8 × 1 |
| phi4-14b | 16x64 | 3.6 s | 20.3 s | 18.3 s | 72.7 s | 114.9 s | 75.2 s (74.5–75.5) | 6.3 min | 8.2 min | 8.4 min ᴿ | 8 × 1 |
| phi4-14b | 256x64 | 3.5 s | 19.9 s | 18.3 s | 72.4 s | 114.0 s | 75.2 s (74.6–75.5) | 6.3 min | 8.2 min | 8.4 min ᴿ | 8 × 1 |

| model | unit | GPU peak allocated | GPU peak reserved | trainer peak RSS (ru_maxrss) | RSS after model load | teacher (top-1000) | wrapper RSS | process-tree peak (10 Hz) | source of the last two |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| phi4-14b | 8x64 | 60.2 GiB | 63.0 GiB | 28.2 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 27.8 GiB | re-run |
| phi4-14b | 16x64 | 59.9 GiB | 63.0 GiB | 28.2 GiB | 0.9 GiB | 0.73 GiB | 0.9 GiB (VmHWM) | 28.4 GiB | re-run |
| phi4-14b | 256x64 | 59.7 GiB | 63.0 GiB | 28.2 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 27.2 GiB | re-run |

The one-time data preparation (fit and development records, before the first unit):

train_map's first run of a model on a new data root prepares its calibration record and development sets (`calibration.tmopt_common.prepare` and `prepare_development`; the builder rule's checks, re-tokenizing the fit windows, hashing the model's weights on the CPU unless an archived record has them, the development draws). Measured on 2026-10-06 in a fresh data root with measure.py's sampling plus each process's VmHWM; the files it wrote are byte-identical to the ones the release runs used. The dataset shards were already in the local Hugging Face cache; a first-ever preparation also downloads them.

| model | wall time | process-tree peak (10 Hz) | wrapper | largest process (VmHWM) | records identical to the release run's |
|---|---:|---:|---:|---:|---|
| phi4-14b | 3.8 min | 28.6 GiB | 0.5 GiB | 28.1 GiB | yes (11 files) |

Time and memory: the trainer's numbers (phases, epochs, GPU peaks, its own `ru_maxrss`, RSS after the model load, teacher storage) come from its PhaseMonitor in the release run. The wrapper's own RSS, the process-tree peak (wrapper + trainer, sampled at 10 Hz) and the end-to-end time come from topk-cal's `measure.py` method: for the first runs (ᴿ, source “re-run”) from a measured re-run of the same calibration that reproduced the same map, for the later ones from a watcher attached to the run itself (“live”; Nemotron 16x64 was attached 13 min after its start, so its tree peak covers only the rest, while its VmHWM peaks are exact). The end-to-end time excludes the one-time data preparation, which has its own row.

## Perplexity

Native sm_120 kernels, flipquant's `--paper-convention` (per-token FourOverSix activations), WikiText-2 (146 windows of 2048 tokens) and C4 (256 windows of 2048 tokens). ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE.

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

The paper-setting rows are maps calibrated at the paper's settings (20 epochs, 128 windows, the full teacher), evaluated under the same convention; they are listed for comparison only.

## How to use (flipquant)

On an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels:

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model phi4-14b --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

Measured here with the 16x64 map: 46 GiB of GPU memory at peak (2048-token windows, batch 1), 74 s for both corpora, model load included.

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model phi4-14b --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the table above (with `--out`); the simulated one was smoke-tested on two windows per corpus.

flipquant: <repository link to be added by the uploader>.

## License

The base model is under the MIT License. These maps are derived from it; this draft assumes the same license (the uploader's decision).

## Files

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `records/<unit>/`: `reproduction.json` and `run.json` (settings, commit, time and memory), the trainer's report (`trainer_report.json`: phases, the fit extension's windows) and its log.
- `ppl/`: the evaluation reports (per-window NLLs) and `significance.json`.
