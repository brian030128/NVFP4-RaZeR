---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B
tags:
- flipquant
- nvfp4
- mixfp4
- fp4
- quantization
---

# Llama-3.1-8B FlipQuant MixFP4 maps (DRAFT)

> Draft model card, for review before any upload. The repository name, the flipquant link and the license wording are the uploader's decision.

**Built with Llama.** The Llama 3.1 Community License (section 1.b) applies to derivative works: provide a copy of the agreement, prominently display “Built with Llama”, include “Llama” at the beginning of the name of an AI model created or improved with Llama materials or their outputs, and keep the notice “Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.” in a NOTICE file. These maps are trained against Llama-3.1-8B's outputs (KL to its BF16 logits). This directory's `NOTICE` holds that notice; a copy of the agreement (the base model's LICENSE) still has to be added by the uploader.

## What this is

Format maps for [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) at revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`, calibrated with FlipQuant (TM-OPT+TC). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 50,571 of 13,631,488 (0.37 %) | `946fd3c4b4017086a8d2cc8bfea1445ceda87a217e257ee923a2f516f8ba666c` |
| `flipquant_16x64.pt` | 16 × 64 | 32,125 of 6,815,744 (0.47 %) | `d4ccc18b61a6db781d7799723c47a905ca69407d72a7df009e04929ea877e150` |
| `flipquant_256x64.pt` | 256 × 64 | 5,245 of 425,984 (1.23 %) | `fcff3f58c18e786580b9c408dd666a0dc22cc2c547b43d17b9ac6051f38aa49c` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

## Calibration

`python -m calibration.train_map --model llama3.1-8b --unit <unit> --fit-windows 256 --epochs 5 --teacher-topk 1000` (flipquant commit `120173a4f67c67f5e67f93ea02fa7a9eaff3ef14`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs, deterministic.
- Fit set: 256 windows of 512 tokens. They are the FlipQuant paper's 128 math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), plus 128 more by the extension rule, which skips the paper's fit, development and published C4 evaluation documents. Every window is recorded by document hash, offset and token hash (`records/<unit>/trainer_report.json`).
- These are not the paper's settings (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| model | unit | model load | data (+ extension) | teacher | lean packing | setup | per epoch (mean, min–max) | training | trainer total | train_map end-to-end | batch × accum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| llama3.1-8b | 8x64 | 1.7 s | 21.6 s | 11.8 s | 36.8 s | 71.9 s | 40.2 s (39.7–40.5) | 3.3 min | 4.5 min | 4.7 min ᴿ | 8 × 1 |
| llama3.1-8b | 16x64 | 1.7 s | 20.5 s | 11.7 s | 36.5 s | 70.4 s | 40.2 s (39.9–40.4) | 3.4 min | 4.5 min | 4.7 min ᴿ | 8 × 1 |
| llama3.1-8b | 256x64 | 1.7 s | 19.5 s | 11.8 s | 36.2 s | 69.1 s | 40.2 s (39.9–40.4) | 3.4 min | 4.5 min | 4.7 min ᴿ | 8 × 1 |

| model | unit | GPU peak allocated | GPU peak reserved | trainer peak RSS (ru_maxrss) | RSS after model load | teacher (top-1000) | wrapper RSS | process-tree peak (10 Hz) | source of the last two |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| llama3.1-8b | 8x64 | 41.9 GiB | 42.9 GiB | 15.9 GiB | 0.9 GiB | 0.73 GiB | 0.9 GiB (VmHWM) | 15.4 GiB | re-run |
| llama3.1-8b | 16x64 | 41.7 GiB | 42.8 GiB | 15.9 GiB | 0.9 GiB | 0.73 GiB | 0.9 GiB (VmHWM) | 15.5 GiB | re-run |
| llama3.1-8b | 256x64 | 41.6 GiB | 42.6 GiB | 15.9 GiB | 0.9 GiB | 0.73 GiB | 0.8 GiB (VmHWM) | 15.3 GiB | re-run |

The one-time data preparation (fit and development records, before the first unit):

train_map's first run of a model on a new data root prepares its calibration record and development sets (`calibration.tmopt_common.prepare` and `prepare_development`; the builder rule's checks, re-tokenizing the fit windows, hashing the model's weights on the CPU unless an archived record has them, the development draws). Measured on 2026-10-06 in a fresh data root with measure.py's sampling plus each process's VmHWM; the files it wrote are byte-identical to the ones the release runs used. The dataset shards were already in the local Hugging Face cache; a first-ever preparation also downloads them.

| model | wall time | process-tree peak (10 Hz) | wrapper | largest process (VmHWM) | records identical to the release run's |
|---|---:|---:|---:|---:|---|
| llama3.1-8b | 20 s | 1.8 GiB | 0.5 GiB | 1.3 GiB | yes (5 files) |

Time and memory: the trainer's numbers (phases, epochs, GPU peaks, its own `ru_maxrss`, RSS after the model load, teacher storage) come from its PhaseMonitor in the release run. The wrapper's own RSS, the process-tree peak (wrapper + trainer, sampled at 10 Hz) and the end-to-end time come from topk-cal's `measure.py` method: for the first runs (ᴿ, source “re-run”) from a measured re-run of the same calibration that reproduced the same map, for the later ones from a watcher attached to the run itself (“live”; Nemotron 16x64 was attached 13 min after its start, so its tree peak covers only the rest, while its VmHWM peaks are exact). The end-to-end time excludes the one-time data preparation, which has its own row.

## Perplexity

Native sm_120 kernels, flipquant's `--paper-convention` (per-token FourOverSix activations), WikiText-2 (146 windows of 2048 tokens) and C4 (256 windows of 2048 tokens). ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE.

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

The paper-setting rows are maps calibrated at the paper's settings (20 epochs, 128 windows, the full teacher), evaluated under the same convention; they are listed for comparison only.

## How to use (flipquant)

On an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels:

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model llama3.1-8b --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

Measured here with the 16x64 map: 23 GiB of GPU memory at peak (2048-token windows, batch 1), 45 s for both corpora, model load included.

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model llama3.1-8b --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the table above (with `--out`); the simulated one was smoke-tested on two windows per corpus.

flipquant: <repository link to be added by the uploader>.

## License

The base model is under the Llama 3.1 Community License. These maps are derived from it; this draft assumes the same license (the uploader's decision). The Llama 3.1 Community License (section 1.b) applies to derivative works: provide a copy of the agreement, prominently display “Built with Llama”, include “Llama” at the beginning of the name of an AI model created or improved with Llama materials or their outputs, and keep the notice “Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.” in a NOTICE file. These maps are trained against Llama-3.1-8B's outputs (KL to its BF16 logits).

## Files

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `records/<unit>/`: `reproduction.json` and `run.json` (settings, commit, time and memory), the trainer's report (`trainer_report.json`: phases, the fit extension's windows) and its log.
- `ppl/`: the evaluation reports (per-window NLLs) and `significance.json`.
- `NOTICE`: the Llama 3.1 attribution notice.
