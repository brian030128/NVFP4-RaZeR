---
license: llama3.2
base_model: meta-llama/Llama-3.2-1B-Instruct
tags:
- quantization
- fp4
- nvfp4
- mixfp4
- flipquant
---

# Llama-3.2-1B-Instruct FlipQuant MixFP4 maps

**Built with Llama.**

Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. The agreement is in `LICENSE`, the Acceptable Use Policy it incorporates in `USE_POLICY.md`, and the notice in `NOTICE`.

## What this is

Format maps for [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) at revision `9213176726f574b556790deb65791e0c5aa438b6`, calibrated with FlipQuant (TM-OPT+TC). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 38,587 of 1,900,544 (2.03 %) | `36bf7d743cfa4a78c7db348e8ddbb50857cc17258bd1f21e9aafc94630ab6b66` |
| `flipquant_16x64.pt` | 16 × 64 | 26,656 of 950,272 (2.81 %) | `1c0f1746eb63a43180d6f8eb9ee506ce080a2c339939a70d07b431069eb6bc6a` |
| `flipquant_256x64.pt` | 256 × 64 | 6,757 of 59,392 (11.38 %) | `4d98536e354fe70cb679d2827158c08e6f1ee43f3d8c7f420093554500b6c92b` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

The FlipQuant code will be released with the paper.

## Calibration

`python -m calibration.train_map --model llama3.2-1b-instruct --unit <unit> --fit-windows 256 --epochs 5 --teacher-topk 1000` (FlipQuant commit `aa1240830e39d390397c6198ee1f2c37c9ae18c6`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs, deterministic.
- Fit set: 256 windows of 512 tokens. They are the FlipQuant paper's 128 math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer, plus 128 more by the extension rule, which skips the paper's fit, development and published C4 evaluation documents. The calibration records every window by document hash, offset and token hash. Its first 128 windows are identical, token for token, to the fit windows of the Llama-3.1-8B release (Llama-3.1-8B-FlipQuant): the paper rule (`builder_seed0` with this model's tokenizer) gives the same 128 windows. Its 128 extension windows share none with that release's: this model's own development draws contain all 128 of that release's extension documents, which the extension rule skips.
- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| unit | model load | data (+ extension) | teacher | lean packing | setup | per epoch (mean, min–max) | training | trainer total | train_map end-to-end | batch × accum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8x64 | 0.4 s | 19.7 s | 6.6 s | 5.1 s | 31.8 s | 7.6 s (7.5–7.8) | 38.2 s | 70.0 s | 78.5 s | 8 × 1 |
| 16x64 | 0.3 s | 20.8 s | 6.2 s | 5.0 s | 32.4 s | 7.7 s (7.6–8.0) | 38.7 s | 71.1 s | 79.8 s | 8 × 1 |
| 256x64 | 0.4 s | 21.3 s | 6.5 s | 5.3 s | 33.5 s | 7.7 s (7.6–7.9) | 38.7 s | 72.2 s | 81.0 s | 8 × 1 |

| unit | GPU peak allocated | GPU peak reserved | trainer peak RSS (ru_maxrss) | RSS peak during model load | teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |
|---|---:|---:|---:|---:|---:|---:|
| 8x64 | 14.4 GiB | 14.8 GiB | 3.3 GiB | 3.3 GiB | 0.73 GiB | 15.5 GiB |
| 16x64 | 14.4 GiB | 14.7 GiB | 3.3 GiB | 2.0 GiB | 0.73 GiB | 15.5 GiB |
| 256x64 | 14.4 GiB | 14.7 GiB | 3.3 GiB | 3.2 GiB | 0.73 GiB | 15.5 GiB |

Time and memory: the trainer's own phase timer and peak counters give the phases, the epochs, the GPU peaks, its `ru_maxrss` and the teacher storage; the end-to-end time and the device memory peak come from the run's wrapper (nvidia-smi sampled every 2 s).

The one-time data preparation (the calibration record: the paper rule's 128 windows re-tokenized and checked, the weights hashed on the CPU, the development records) took 5 min end to end, including the first download of the model's weights.

## Perplexity

Native sm_120 kernels, FlipQuant's `--paper-convention` (per-token FourOverSix activations; NVFP4 with per-token NVFP4 activations), WikiText-2 (141 windows of 2048 tokens) and C4 (256 windows of 2048 tokens). ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE. The numbers are also in `ppl_summary.json`.

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 13.1629 | 18.5189 | -0.1514 ± 0.0037 * | -0.1469 ± 0.0058 * | better / better |
| NVFP4 (per-token NVFP4 activations) | 15.5352 | 21.7669 | +0.0143 ± 0.0025 * | +0.0147 ± 0.0024 * | worse / worse |
| FourOverSix (NVFP4, stock kernels) | 15.3152 | 21.4488 | — | — | reference |
| FlipQuant 8x64 (this release) | 14.4527 | 19.6791 | -0.0580 ± 0.0029 * | -0.0861 ± 0.0039 * | better / better |
| FlipQuant 16x64 (this release) | 14.4536 | 19.6842 | -0.0579 ± 0.0028 * | -0.0859 ± 0.0038 * | better / better |
| FlipQuant 256x64 (this release) | 14.6573 | 20.0588 | -0.0439 ± 0.0028 * | -0.0670 ± 0.0033 * | better / better |

## How to use

The FlipQuant code will be released with the paper. With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels:

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model llama3.2-1b-instruct --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

Measured here with the 16x64 map: 4 GiB of GPU memory at peak (2048-token windows, batch 1), 16 s for both corpora, model load included.

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model llama3.2-1b-instruct --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the table above.

## License

These maps are released under the same license as the base model, the Llama 3.2 Community License.

The Llama 3.2 Community License (section 1.b) applies to derivative works: provide a copy of the agreement, prominently display “Built with Llama”, include “Llama” at the beginning of the name of an AI model created or improved with Llama materials or their outputs, and keep the notice “Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.” in a NOTICE file. These maps are trained against Llama-3.2-1B-Instruct's outputs (KL to its BF16 logits).

- `LICENSE`: the Llama 3.2 Community License Agreement, the base repository's own license file (meta-llama/Llama-3.2-1B-Instruct at revision `9213176726f574b556790deb65791e0c5aa438b6`).
- `USE_POLICY.md`: the Llama 3.2 Acceptable Use Policy, which the agreement incorporates; the base repository's own `USE_POLICY.md`.
- `NOTICE`: “Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”

## Files

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings and the calibration settings and cost.
- `LICENSE`: see License.
- `USE_POLICY.md`: see License.
- `NOTICE`: see License.

**Provenance of the map files.** They are the calibration outputs with only their `meta` changed: the run directory and calibration-record paths are relative instead of local absolute paths, the data-extension code reference names the file without its repository owner, and the internal branch and commit of the reference implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the calibration output → this repository's file:

- `flipquant_8x64.pt`: `4f693ed87bcd4912d48b86da7343e617b9619ca25b96ce7531300cbb1ca82970` → `36bf7d743cfa4a78c7db348e8ddbb50857cc17258bd1f21e9aafc94630ab6b66`
- `flipquant_16x64.pt`: `d9dfa5a7d2045af4bfa9f81ff204d04a7e72436c45826a38bdeab2c2291ad5c1` → `1c0f1746eb63a43180d6f8eb9ee506ce080a2c339939a70d07b431069eb6bc6a`
- `flipquant_256x64.pt`: `54effb555ff0527830482ee0a794c47ea52b5c0609fb1ec9f05cc7dc2fe937fa` → `4d98536e354fe70cb679d2827158c08e6f1ee43f3d8c7f420093554500b6c92b`
