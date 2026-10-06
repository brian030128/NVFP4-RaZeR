# Development-set KL during calibration (Llama-3.1-8B, Phi-4 and Qwen3-1.7B, 16x64)

The release maps were calibrated with `--no-dev`, so their only progress signal was the per-epoch training KL. That is
the top-1000 KL on the 256 fit windows, which keep changing, and at epoch 5 it was still falling. This note measures the
held-out quantity: the full-vocabulary KL of the hard map against the BF16 model on 192 development windows, along the
calibration. It then checks the WikiText-2 and C4 perplexity of the maps at epoch 10.

## Setup

- **Trainer:** the vendored `calibration/tmopt/run_train_map.py` of flipquant 120173a (main = optimized-defaults), run
  directly. The command is the one `calibration.train_map` builds (`train_map.command`), with `--no-dev` dropped so
  the trainer's own development evaluation runs (monitor only). `--no-eval` stays. Commands: `experiments/release_maps/devkl_run.py`.
- **N, the release setting:** `--fit-windows 256 --teacher-topk 1000`, lr 0.02, init −1, seed 0, 8 sequences per step,
  32 steps per epoch.
  - It runs for **10 epochs instead of the release's 5**. The schedule is constant, so epochs 1–5 are the release run
    and epochs 6–10 show whether 5 epochs stopped early.
  - `--eval-every 1`.
- **O, the paper setting:** 128 windows, the full-vocabulary teacher, 20 epochs of 16 steps, `--eval-every 2`. The
  evaluations fall at the same steps as N's: 0, 32, …, 320.
- **Development set:** 192 windows of 512 tokens per model, three draws of 64. All 192 are distinct documents, none of
  them among the 128 paper fit documents or the 128 extension documents (also checked by window token hash;
  `dev_kl.json` → `disjointness`).
  - Llama-3.1-8B: the paper's three development sets (confirmation, gate_up_confirm, tile_refine_confirm). The release
    data root holds only their manifests, enough for the extension rule. The windows come from the paper's data root
    (`cost_comparison/data`); its manifests are byte-identical to the vendored ones, and `fresh.pt` matches its
    recorded sha256. The calibration record is the release root's.
  - Phi-4 and Qwen3-1.7B: fresh_dev1–3 of the release data root.
- **Metric:** the trainer's development evaluation in its TM-OPT configuration (`dev_backend native`). It is the mean
  over windows of the per-token KL(BF16 teacher ‖ model with the current hard map). The teacher's full-vocabulary
  log-probabilities are kept on the host in BF16. The model runs on the native b8x64 kernel (realquant
  `libb8x64.so`, NVFP4-RaZeR `repro_local/realquant/build`) with per-document FourOverSix activation scales (that
  evaluator's convention).
  - Step 0 is the FourOverSix start, before any flip. One evaluation takes 12.3 s (Llama), 14.2 s (Phi-4) or 9.1 s
    (Qwen3-1.7B).
- **Monitor only, checked** (`dev_kl.json` → `checks`): the development evaluation never touches training.
  - N's `map_epoch005.pt` is the release map: the tiles are equal and every tensor record is byte-identical. The
    file sha256 differs only through `torch.save`'s archive name, `map_epoch005/` against `map/`.
  - O's final `map.pt` is the paper-setting map. For Llama and Phi-4 that is the paper map (`54070819…`, `110243a6…`,
    the same file sha256). For Qwen3-1.7B it is the committed `maps/qwen3-1.7b/flipquant_16x64.pt` (its recorded run
    map `6aecdaef…`, the same file sha256, and the tiles are equal).
- **Perplexity of the N maps** (WikiText-2 and C4, the release evaluation's windows): flipquant's `evaluation.ppl` with the
  release evaluation's flags (`--mode native --weight mixfp4 --paper-convention`, all windows; `--unit 16x64` given
  explicitly, since the epoch maps carry no meta) and the same kernel builds. The records differ from the release's
  only in the map. Maps: N at epoch 10 for the three models, and Phi-4's lowest
  development-KL point, epoch 7. Qwen3-1.7B's lowest point is epoch 10, and so is Llama's.
  - The references are the existing per-window records on the same windows: FourOverSix and the release map
    (`flipquant_release/<model>/ppl/`), and the paper-setting map. That is the paper's step-03 record for Llama and
    Phi-4, and main-ppl's for Qwen3-1.7B. Each of those records' artifact names its source map, and that map is O's
    final map byte for byte (`checks` → `paper_setting_ppl_source_map_is_O_final`).
  - Paired over windows, ΔNLL (nats per token) ± 2 SE, over windows of 2048 tokens: 141 WikiText-2 windows (146 for
    Qwen3-1.7B) and 256 C4 windows. `ppl.json`.

## Results

Development KL (192 windows, full vocabulary), training KL as logged, E0M3 tiles; aligned by optimizer step.
N's training KL is the top-1000 + tail-bucket KL on 256 windows; O's is the full KL on 128 windows.

### Llama-3.1-8B 16x64

| step | N epoch | N dev KL | N train KL | N E0M3 | O epoch | O dev KL | O train KL | O E0M3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.10811 | — | 0 | 0 | 0.10811 | — | 0 |
| 32 | 1 | 0.10811 | 0.10106 | 0 | 2 | 0.10811 | 0.09695 | 0 |
| 64 | 2 | 0.10127 | 0.09955 | 712 | 4 | 0.10117 | 0.09278 | 1,157 |
| 96 | 3 | 0.09500 | 0.09302 | 3,531 | 6 | 0.09254 | 0.08228 | 13,400 |
| 128 | 4 | 0.09064 | 0.08636 | 14,037 | 8 | 0.08884 | 0.06903 | 46,127 |
| **160** | **5 (release)** | **0.08876** | 0.07883 | 32,125 | 10 | 0.08705 | 0.05986 | 79,006 |
| 192 | 6 | 0.08793 | 0.07135 | 56,133 | 12 | 0.08672 | 0.05423 | 108,060 |
| 224 | 7 | 0.08504 | 0.06398 | 81,550 | 14 | 0.08653 | 0.05020 | 134,091 |
| 256 | 8 | 0.08528 | 0.05925 | 108,393 | 16 | 0.08466 | 0.04659 | 157,770 |
| 288 | 9 | 0.08394 | 0.05518 | 131,596 | 18 | 0.08529 | 0.04432 | 179,966 |
| **320** | **10** | **0.08389** | 0.05207 | 157,497 | **20 (paper)** | **0.08464** | 0.04239 | 201,648 |

### Phi-4 16x64

| step | N epoch | N dev KL | N train KL | N E0M3 | O epoch | O dev KL | O train KL | O E0M3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.03765 | — | 0 | 0 | 0.03765 | — | 0 |
| 32 | 1 | 0.03765 | 0.03567 | 0 | 2 | 0.03765 | 0.03670 | 0 |
| 64 | 2 | 0.03683 | 0.03581 | 142 | 4 | 0.03724 | 0.03649 | 399 |
| 96 | 3 | 0.03608 | 0.03517 | 5,261 | 6 | 0.03558 | 0.03285 | 34,263 |
| 128 | 4 | 0.03559 | 0.03232 | 33,031 | 8 | 0.03534 | 0.02839 | 99,420 |
| **160** | **5 (release)** | **0.03493** | 0.02933 | 79,591 | 10 | 0.03515 | 0.02558 | 135,406 |
| 192 | 6 | 0.03501 | 0.02715 | 120,133 | 12 | 0.03483 | 0.02404 | 163,948 |
| 224 | 7 | 0.03424 | 0.02577 | 155,869 | 14 | 0.03501 | 0.02283 | 201,607 |
| 256 | 8 | 0.03461 | 0.02460 | 191,039 | 16 | 0.03523 | 0.02196 | 228,888 |
| 288 | 9 | 0.03456 | 0.02361 | 219,086 | 18 | 0.03519 | 0.02145 | 259,308 |
| **320** | **10** | **0.03485** | 0.02267 | 251,621 | **20 (paper)** | **0.03515** | 0.02026 | 287,062 |

### Qwen3-1.7B 16x64

| step | N epoch | N dev KL | N train KL | N E0M3 | O epoch | O dev KL | O train KL | O E0M3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.17266 | — | 0 | 0 | 0.17266 | — | 0 |
| 32 | 1 | 0.17266 | 0.16552 | 0 | 2 | 0.17266 | 0.16892 | 0 |
| 64 | 2 | 0.15535 | 0.16315 | 926 | 4 | 0.15479 | 0.16150 | 1,614 |
| 96 | 3 | 0.14407 | 0.13893 | 4,759 | 6 | 0.14388 | 0.12979 | 8,679 |
| 128 | 4 | 0.13771 | 0.12769 | 9,470 | 8 | 0.13765 | 0.11298 | 19,200 |
| **160** | **5 (release)** | **0.13567** | 0.11890 | 16,351 | 10 | 0.13444 | 0.09939 | 31,948 |
| 192 | 6 | 0.13304 | 0.10901 | 25,307 | 12 | 0.13491 | 0.08860 | 44,026 |
| 224 | 7 | 0.13107 | 0.10061 | 35,389 | 14 | 0.13331 | 0.08164 | 55,073 |
| 256 | 8 | 0.12993 | 0.09451 | 46,150 | 16 | 0.13436 | 0.07687 | 65,468 |
| 288 | 9 | 0.12973 | 0.08848 | 56,641 | 18 | 0.13093 | 0.07130 | 75,501 |
| **320** | **10** | **0.12771** | 0.08350 | 66,498 | **20 (paper)** | **0.13313** | 0.06807 | 85,085 |

![dev KL](dev_kl.png)

### Paired comparisons (per window, mean ± 2 SE over the 192 windows; `*` = |Δ| > 2 SE; `paired.json`)

| | Llama-3.1-8B | Phi-4 | Qwen3-1.7B |
|---|---:|---:|---:|
| release map (N epoch 5) − paper map (O end) | **+0.00412 ± 0.00184 \*** | −0.00022 ± 0.00058 | **+0.00254 ± 0.00224 \*** |
| N epoch 10 − paper map | −0.00075 ± 0.00178 | −0.00030 ± 0.00047 | **−0.00542 ± 0.00217 \*** |
| N epoch 10 − N epoch 5 | **−0.00487 ± 0.00189 \*** | −0.00008 ± 0.00064 | **−0.00796 ± 0.00239 \*** |
| release map − start (FourOverSix) | −0.01935 ± 0.00395 \* | −0.00272 ± 0.00083 \* | −0.03699 ± 0.00386 \* |
| paper map − start | −0.02347 ± 0.00460 \* | −0.00250 ± 0.00095 \* | −0.03953 ± 0.00393 \* |
| N epoch 10 − N's lowest point | 0 (lowest at step 320) | +0.00061 ± 0.00066 (lowest at step 224) | 0 (lowest at step 320) |
| O end − O's lowest point | 0 (lowest at step 320) | +0.00032 ± 0.00075 (lowest at step 192) | +0.00221 ± 0.00243 (lowest at step 288) |
| O step 320 − O step 160 | −0.00241 ± 0.00172 \* | +0.00000 ± 0.00053 | −0.00130 ± 0.00234 |

### Perplexity of the epoch-10 maps (`ppl.json`)

| model | corpus | FourOverSix | release map (N epoch 5) | N epoch 7 | N epoch 10 | paper-setting map |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | WikiText-2 | 6.8706 | 6.8005 | | 6.7890 | 6.7827 |
| Llama-3.1-8B | C4 | 9.8257 | 9.7087 | | 9.6786 | 9.6827 |
| Phi-4 | WikiText-2 | 6.6617 | 6.6325 | 6.6254 | 6.6206 | 6.6133 |
| Phi-4 | C4 | 10.5441 | 10.5157 | 10.5153 | 10.5117 | 10.5047 |
| Qwen3-1.7B | WikiText-2 | 19.4745 | 15.3979 | | 15.7424 | 15.8021 |
| Qwen3-1.7B | C4 | 20.9072 | 19.4097 | | 19.5632 | 19.6422 |

Paired ΔNLL (nats per token) ± 2 SE, WikiText-2 / C4; `*` = |Δ| > 2 SE:

| | Llama-3.1-8B | Phi-4 | Qwen3-1.7B |
|---|---|---|---|
| N epoch 10 − FourOverSix | −0.0119 ± 0.0017 \* / −0.0151 ± 0.0031 \* | −0.0062 ± 0.0014 \* / −0.0031 ± 0.0009 \* | −0.2127 ± 0.0128 \* / −0.0664 ± 0.0037 \* |
| N epoch 10 − release map | **−0.0017 ± 0.0014 \* / −0.0031 ± 0.0012 \*** | **−0.0018 ± 0.0012 \*** / −0.0004 ± 0.0007 | **+0.0221 ± 0.0029 \* / +0.0079 ± 0.0016 \*** |
| N epoch 10 − paper-setting map | +0.0009 ± 0.0015 / −0.0004 ± 0.0011 | +0.0011 ± 0.0012 / +0.0007 ± 0.0007 | **−0.0038 ± 0.0027 \* / −0.0040 ± 0.0016 \*** |
| N epoch 7 − release map | | −0.0011 ± 0.0012 / −0.0000 ± 0.0007 | |
| N epoch 7 − paper-setting map | | +0.0018 ± 0.0013 \* / +0.0010 ± 0.0007 \* | |
| release map − paper-setting map (for reference) | +0.0026 ± 0.0015 \* / +0.0027 ± 0.0012 \* | +0.0029 ± 0.0012 \* / +0.0010 ± 0.0007 \* | −0.0259 ± 0.0029 \* / −0.0119 ± 0.0016 \* |

## Findings

1. **Llama-3.1-8B: 5 epochs stop early, on both measures.**
   - Development KL: the release map is 0.00412 ± 0.00184 nats/token worse than the paper map. Five more epochs at
     the release setting remove the whole gap (−0.00075 ± 0.00178), and both runs are still at their lowest point at
     step 320.
   - Perplexity agrees. N at epoch 10 beats the release map on both corpora (−0.0017 / −0.0031) and equals the paper
     map within noise (+0.0009 ± 0.0015 / −0.0004 ± 0.0011).
2. **Phi-4: no lasting development-KL change after step 160; a small perplexity gain from epochs 6–10.**
   - Development KL: N dips significantly at epoch 7 (step 224 − step 160: −0.00068 ± 0.00047) and comes back. N at
     epoch 10 equals N at epoch 5 (−0.00008 ± 0.00064). The release map, the paper map and N at epoch 10 are equal
     within noise. The rises after the lowest points are within 2 SE, so they are not overfitting.
   - Perplexity: N at epoch 10 beats the release map on WikiText-2 (−0.0018 ± 0.0012), is level on C4, and equals the
     paper map on both.
   - Choosing the epoch by the lowest development KL buys nothing. N at epoch 7 is level with the release map and
     slightly behind the paper map (+0.0018 / +0.0010).
3. **Qwen3-1.7B: development KL and perplexity rank the release map in opposite directions.**
   - Development KL: 5 epochs stop early. The release map is 0.00254 ± 0.00224 worse than the paper map, and N at
     epoch 10 is better than both (−0.00796 against epoch 5, −0.00542 against the paper map). N is still at its lowest
     point at step 320, while the paper setting levels off after step 160 (step 320 − step 160: −0.00130 ± 0.00234).
   - Perplexity: the release map is the best of the three, ahead of the paper map by −0.0259 / −0.0119. N at epoch 10
     loses part of that lead (+0.0221 / +0.0079 against the release map) but stays ahead of the paper map
     (−0.0038 / −0.0040).
   - So on this model the development set (the fit domain) cannot predict WikiText-2 / C4 perplexity, even in sign.
4. **Across the three models, N at epoch 10 is never behind the paper-setting map** in perplexity: level on Llama and
   Phi-4, ahead on Qwen3-1.7B, on both corpora. The release map (epoch 5) is behind the paper map on Llama and Phi-4
   and ahead on Qwen3-1.7B.
5. **Development KL tracks the training KL only early.**
   - Up to about step 128 both fall together.
   - After that the training KL keeps falling steeply while the development KL flattens. Steps 160→320 of N:
     - Llama: training −34 %, development −5.5 %;
     - Phi-4: training −23 %, development flat;
     - Qwen3-1.7B: training −30 %, development −5.9 %.
   - The training KL keeps improving on a fit set the map increasingly specializes to. Held-out KL rises nowhere
     significantly.
6. **At equal steps, the new setting flips fewer tiles.**
   - At step 160: Llama 32,125 against 79,006; Phi-4 79,591 against 135,406; Qwen3-1.7B 16,351 against 31,948.
   - It reaches a similar development KL only later on Llama. On Phi-4 the extra flips buy nothing held-out, and on
     Qwen3-1.7B the new setting has the lower development KL from step 192 on.
   - A possible reason, not tested here: the top-1000 teacher, and a gradient spread over twice the windows per epoch.

**Caveats.**
- The development windows are math/code text, the fit set's domain. WikiText-2 and C4 PPL measure other text, and on
  Qwen3-1.7B the two disagree even in sign (finding 3).
- The development evaluator uses per-document activation scales, while the PPL convention is per token.
- One seed, one unit (16x64), three models. The epoch-10 maps are measurements only: nothing in the release directory
  or in flipquant changes.

## Cost

| run | trainer total | setup | epochs × mean time | dev evaluations | GPU peak allocated | host ru_maxrss |
|---|---:|---:|---:|---:|---:|---:|
| Llama N | 10.6 min | 97 s | 10 × 40.2 s | 10 × 12.3 s (+ initial 13.0 s, final 12.1 s) | 41.9 GiB | 26.5 GiB |
| Llama O | 11.3 min | 83 s | 20 × 23.0 s | 10 × 12.3 s (+ initial 13.0 s, final 12.2 s) | 40.6 GiB | 41.8 GiB |
| Phi-4 N | 17.6 min | 147 s | 10 × 75.2 s | 10 × 14.2 s (+ initial 15.7 s, final 14.1 s) | 60.2 GiB | 28.2 GiB |
| Phi-4 O | 18.0 min | 128 s | 20 × 39.9 s | 10 × 14.2 s (+ initial 15.9 s, final 14.2 s) | 59.4 GiB | 33.2 GiB |
| Qwen3-1.7B N | 4.7 min | 61 s | 10 × 12.0 s | 10 × 9.1 s (+ initial 9.4 s, final 9.1 s) | 21.1 GiB | 31.2 GiB |
| Qwen3-1.7B O | 5.8 min | 48 s | 20 × 9.9 s | 10 × 9.1 s (+ initial 9.3 s, final 9.3 s) | 19.7 GiB | 49.5 GiB |

The development teacher costs one BF16 forward over 192 windows at setup, plus host memory for its
log-probabilities: 192 × 511 × the vocabulary × 2 bytes, 25 GB for Llama, 20 GB for Phi-4 and 30 GB for Qwen3-1.7B.

Perplexity runs (load, install and both corpora; one RTX PRO 6000): Llama epoch 10 42 s (GPU peak 21.1 GiB), Phi-4
epochs 7 and 10 71 s each (42.0 GiB), Qwen3-1.7B epoch 10 25 s (4.5 GiB).
