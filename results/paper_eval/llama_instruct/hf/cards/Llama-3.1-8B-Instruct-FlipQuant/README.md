---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
tags:
- quantization
- fp4
- nvfp4
- mixfp4
- flipquant
---

# Llama-3.1-8B-Instruct FlipQuant MixFP4 maps

**Built with Llama.**

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. The agreement is in `LICENSE`, the Acceptable Use Policy it incorporates in `USE_POLICY.md`, and the notice in `NOTICE`.

## What this is

Format maps for [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) at revision `0e9e39f249a16976918f6564b8830bc894c89659`, calibrated with FlipQuant (TM-OPT+TC). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.

| file | tile (output rows × input columns) | E0M3 tiles | sha256 |
|---|---|---:|---|
| `flipquant_8x64.pt` | 8 × 64 | 83,771 of 13,631,488 (0.61 %) | `db896cece0fdef908ba1622b23a3c21b8f867e474ceb41439f94821c476f33ce` |
| `flipquant_16x64.pt` | 16 × 64 | 52,719 of 6,815,744 (0.77 %) | `3d527e1191f6e8bef3010016875c12419a81cec75e2ba231edf6943f8cd6f131` |
| `flipquant_256x64.pt` | 256 × 64 | 9,447 of 425,984 (2.22 %) | `84b009d76e27f883ab53d7c5ca2ed7bf71b7930affacc3de1d13da4aa56732c4` |

8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); 16x64 and 256x64 are coarser.

The FlipQuant code will be released with the paper.

## Calibration

`python -m calibration.train_map --model llama3.1-8b-instruct --unit <unit> --fit-windows 256 --epochs 5 --teacher-topk 1000` (FlipQuant commit `7b472b0f3797d0b3e16882e7e935cbcd2d207faf`):

- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per position plus one tail bucket, per-token FourOverSix activations.
- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, 8 sequences per step, 5 epochs, deterministic.
- Fit set: 256 windows of 512 tokens. They are the FlipQuant paper's 128 math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer, plus 128 more by the extension rule, which skips the paper's fit, development and published C4 evaluation documents. The calibration records every window by document hash, offset and token hash. Its 256 windows are identical, token for token, to the base model's release windows (Llama-3.1-8B-FlipQuant): the Instruct model shares the base tokenizer, the paper rule (`builder_seed0` with this model's tokenizer) reproduces the base model's 128 fit windows exactly, and the extension rule gives the same 128 more.
- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The maps' `meta` records them as `non_default_settings`.

Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:

| unit | model load | data (+ extension) | teacher | lean packing | setup | per epoch (mean, min–max) | training | trainer total | train_map end-to-end | batch × accum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8x64 | 5.9 s | 20.6 s | 12.3 s | 37.2 s | 76.1 s | 40.2 s (39.9–40.4) | 3.4 min | 4.6 min | 4.8 min | 8 × 1 |
| 16x64 | 1.6 s | 19.9 s | 11.8 s | 36.7 s | 70.0 s | 40.3 s (39.9–40.4) | 3.4 min | 4.5 min | 4.7 min | 8 × 1 |
| 256x64 | 1.7 s | 20.9 s | 11.7 s | 36.3 s | 70.5 s | 40.3 s (39.9–40.5) | 3.4 min | 4.5 min | 4.7 min | 8 × 1 |

| unit | GPU peak allocated | GPU peak reserved | trainer peak RSS (ru_maxrss) | RSS peak during model load | teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |
|---|---:|---:|---:|---:|---:|---:|
| 8x64 | 41.9 GiB | 42.9 GiB | 15.9 GiB | 15.9 GiB | 0.73 GiB | 43.6 GiB |
| 16x64 | 41.7 GiB | 42.7 GiB | 15.9 GiB | 15.7 GiB | 0.73 GiB | 43.4 GiB |
| 256x64 | 41.6 GiB | 42.6 GiB | 15.9 GiB | 14.8 GiB | 0.73 GiB | 43.3 GiB |

Time and memory: the trainer's own phase timer and peak counters give the phases, the epochs, the GPU peaks, its `ru_maxrss` and the teacher storage; the end-to-end time and the device memory peak come from the run's wrapper (nvidia-smi sampled every 2 s).

The one-time data preparation (the calibration record: the paper rule's 128 windows re-tokenized and checked, the weights hashed on the CPU, the development records) took 19 min end to end, including the first download of the model's weights.

## Perplexity

Native sm_120 kernels, FlipQuant's `--paper-convention` (per-token FourOverSix activations; NVFP4 with per-token NVFP4 activations), WikiText-2 (141 windows of 2048 tokens) and C4 (256 windows of 2048 tokens). ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE. The numbers are also in `ppl_summary.json`.

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 7.2171 | 10.3853 | -0.0801 ± 0.0031 * | -0.0815 ± 0.0160 * | better / better |
| NVFP4 (per-token NVFP4 activations) | 7.8876 | 11.3409 | +0.0088 ± 0.0019 * | +0.0066 ± 0.0024 * | worse / worse |
| FourOverSix (NVFP4, stock kernels) | 7.8186 | 11.2667 | — | — | reference |
| FlipQuant 8x64 (this release) | 7.6951 | 11.1384 | -0.0159 ± 0.0017 * | -0.0115 ± 0.0026 * | better / better |
| FlipQuant 16x64 (this release) | 7.7106 | 11.1420 | -0.0139 ± 0.0018 * | -0.0111 ± 0.0029 * | better / better |
| FlipQuant 256x64 (this release) | 7.7501 | 11.2171 | -0.0088 ± 0.0017 * | -0.0044 ± 0.0016 * | better / better |

## How to use

The FlipQuant code will be released with the paper. With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with the native kernels:

```bash
bash kernels/sm120/build_deploy.sh
python -m evaluation.ppl --model llama3.1-8b-instruct --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

Measured here with the 16x64 map: 21 GiB of GPU memory at peak (2048-token windows, batch 1), 42 s for both corpora, model load included.

On other GPUs, the same map with simulated (fake) quantization:

```bash
python -m evaluation.ppl --model llama3.1-8b-instruct --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention
```

The native command produced the table above.

## License

These maps are released under the same license as the base model, the Llama 3.1 Community License.

The Llama 3.1 Community License (section 1.b) applies to derivative works: provide a copy of the agreement, prominently display “Built with Llama”, include “Llama” at the beginning of the name of an AI model created or improved with Llama materials or their outputs, and keep the notice “Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.” in a NOTICE file. These maps are trained against Llama-3.1-8B-Instruct's outputs (KL to its BF16 logits).

- `LICENSE`: the Llama 3.1 Community License Agreement, the base repository's own `LICENSE` file (meta-llama/Llama-3.1-8B-Instruct at revision `0e9e39f249a16976918f6564b8830bc894c89659`).
- `USE_POLICY.md`: the Llama 3.1 Acceptable Use Policy, which the agreement incorporates; the base repository's own `USE_POLICY.md`.
- `NOTICE`: “Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”

## Files

- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads them).
- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings and the calibration settings and cost.
- `LICENSE`: see License.
- `USE_POLICY.md`: see License.
- `NOTICE`: see License.

**Provenance of the map files.** They are the calibration outputs with only their `meta` changed: the run directory and calibration-record paths are relative instead of local absolute paths, the data-extension code reference names the file without its repository owner, and the internal branch and commit of the reference implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the calibration output → this repository's file:

- `flipquant_8x64.pt`: `3bf00ea63e0a07371b85fed1d6ffba813d3c23540a494010626ceeb46832b9bf` → `db896cece0fdef908ba1622b23a3c21b8f867e474ceb41439f94821c476f33ce`
- `flipquant_16x64.pt`: `502ec6cfee834782192ed10f8a0d79a8bfddac9fe6cc4b5a8cb3d3fe0dd3ca86` → `3d527e1191f6e8bef3010016875c12419a81cec75e2ba231edf6943f8cd6f131`
- `flipquant_256x64.pt`: `7582c4dad38c93b6ba004ad1c47502144576386e4dc43ae34715d758920a5b08` → `84b009d76e27f883ab53d7c5ca2ed7bf71b7930affacc3de1d13da4aa56732c4`
