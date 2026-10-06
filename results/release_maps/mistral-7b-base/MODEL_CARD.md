---
license: apache-2.0
base_model: mistralai/Mistral-7B-v0.3
tags:
- flipquant
- nvfp4
- mixfp4
- fp4
- quantization
---

# FlipQuant MixFP4 maps for Mistral-7B-v0.3 (DRAFT)

> Draft model card, for review before any upload. The repository name, the flipquant link and the license wording are the uploader's decision.

## What this is

Format maps for [mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) at revision `caa1feb0e54d415e2df31207e5f4e273e33509b1`, calibrated with FlipQuant (TM-OPT+TC). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 106,186 of 13,631,488 (0.78 %) | `9c2472331a1ec8bfd140f42f4ed430f5f25ac64d24dab53147a5dabb8d8d8153` |
| `flipquant_16x64.pt` | 16 × 64 | 66,133 of 6,815,744 (0.97 %) | `f8e4421784ed7985b9e4d8a1ba0650c5d02962cce0715a9b8b94c5e1447a19a1` |
| `flipquant_256x64.pt` | 256 × 64 | 11,004 of 425,984 (2.58 %) | `d8ea6a24aca0c1bb8544d54b0cbe0352cbbd73206e6f28f2c398f06e2cd71870` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

## Calibration

`python -m calibration.train_map --model mistral-7b-base --unit <unit> --fit-windows 256 --epochs 5 --teacher-topk 1000` (flipquant commit `120173a4f67c67f5e67f93ea02fa7a9eaff3ef14`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs, deterministic.
- Fit set: 256 windows of 512 tokens. They are the FlipQuant paper's 128 math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), plus 128 more by the extension rule, which skips the paper's fit, development and published C4 evaluation documents. Every window is recorded by document hash, offset and token hash (`records/<unit>/trainer_report.json`).
- These are not the paper's settings (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| model | unit | model load | data (+ extension) | teacher | lean packing | setup | per epoch (mean, min–max) | training | trainer total | train_map end-to-end | batch × accum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mistral-7b-base | 8x64 | 1.6 s | 20.0 s | 10.8 s | 36.2 s | 68.6 s | 38.1 s (37.5–38.5) | 3.2 min | 4.3 min | 4.5 min ᴿ | 8 × 1 |
| mistral-7b-base | 16x64 | 1.7 s | 19.6 s | 11.4 s | 36.6 s | 69.2 s | 38.2 s (37.9–38.5) | 3.2 min | 4.3 min | 4.5 min ᴿ | 8 × 1 |
| mistral-7b-base | 256x64 | 1.6 s | 19.6 s | 11.0 s | 36.3 s | 68.5 s | 38.2 s (37.9–38.4) | 3.2 min | 4.3 min | 4.5 min ᴿ | 8 × 1 |

| model | unit | GPU peak allocated | GPU peak reserved | trainer peak RSS (ru_maxrss) | RSS after model load | teacher (top-1000) | wrapper RSS | process-tree peak (10 Hz) | source of the last two |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mistral-7b-base | 8x64 | 36.8 GiB | 37.6 GiB | 14.4 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 14.3 GiB | re-run |
| mistral-7b-base | 16x64 | 36.6 GiB | 37.5 GiB | 14.4 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 14.8 GiB | re-run |
| mistral-7b-base | 256x64 | 36.5 GiB | 37.4 GiB | 14.4 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 14.4 GiB | re-run |

The one-time data preparation (fit and development records, before the first unit):

train_map's first run of a model on a new data root prepares its calibration record and development sets (`calibration.tmopt_common.prepare` and `prepare_development`; the builder rule's checks, re-tokenizing the fit windows, hashing the model's weights on the CPU unless an archived record has them, the development draws). Measured on 2026-10-06 in a fresh data root with measure.py's sampling plus each process's VmHWM; the files it wrote are byte-identical to the ones the release runs used. The dataset shards were already in the local Hugging Face cache; a first-ever preparation also downloads them.

| model | wall time | process-tree peak (10 Hz) | wrapper | largest process (VmHWM) | records identical to the release run's |
|---|---:|---:|---:|---:|---|
| mistral-7b-base | 106 s | 14.5 GiB | 0.5 GiB | 14.1 GiB | yes (11 files) |

Time and memory: the trainer's numbers (phases, epochs, GPU peaks, its own `ru_maxrss`, RSS after the model load, teacher storage) come from its PhaseMonitor in the release run. The wrapper's own RSS, the process-tree peak (wrapper + trainer, sampled at 10 Hz) and the end-to-end time come from topk-cal's `measure.py` method: for the first runs (ᴿ, source “re-run”) from a measured re-run of the same calibration that reproduced the same map, for the later ones from a watcher attached to the run itself (“live”; Nemotron 16x64 was attached 13 min after its start, so its tree peak covers only the rest, while its VmHWM peaks are exact). The end-to-end time excludes the one-time data preparation, which has its own row.

## Perplexity

Native sm_120 kernels, flipquant's `--paper-convention` (per-token FourOverSix activations), WikiText-2 (146 windows of 2048 tokens) and C4 (256 windows of 2048 tokens). ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE.

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

The paper-setting rows are maps calibrated at the paper's settings (20 epochs, 128 windows, the full teacher), evaluated under the same convention; they are listed for comparison only.

## How to use (flipquant)

On an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels:

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model mistral-7b-base --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

Measured here with the 16x64 map: 21 GiB of GPU memory at peak (2048-token windows, batch 1), 41 s for both corpora, model load included.

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model mistral-7b-base --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the table above (with `--out`); the simulated one was smoke-tested on two windows per corpus.

flipquant: <repository link to be added by the uploader>.

## License

The base model is under the Apache License 2.0. These maps are derived from it; this draft assumes the same license (the uploader's decision).

## Files

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `records/<unit>/`: `reproduction.json` and `run.json` (settings, commit, time and memory), the trainer's report (`trainer_report.json`: phases, the fit extension's windows) and its log.
- `ppl/`: the evaluation reports (per-window NLLs) and `significance.json`.
