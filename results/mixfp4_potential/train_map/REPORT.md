# Training the tile map with an optimizer (no backtracking)

Llama-3.1-8B, W4A4, same candidates, calibration data, teacher, activation
conventions and evaluation windows as the multi-round KL election of the root
`MIXFP4_REPORT.md` §3. Jobs 441206 (256×64 STE), 441207 (8×64 STE),
441208 (8×64 sigmoid), `run_train_map.py`. Maps and logits:
`/work/u4320956/mixfp4_potential/train_map/<run>/{map.pt,theta.pt,map_epochNNN.pt}`.

## Method

Each type tile `u` gets a latent logit `θ_u`, initialised so every tile starts
as FourOverSix E2M1. Weights are `W = B + expand(m(θ)) ⊙ (A − B)`.

- **STE**: `m = 1[θ > 0]` in the forward pass (the deployable hard map);
  backward treats `dm/dθ = 1`. Init `θ = −1`, lr 0.02.
- **Sigmoid**: `m = σ(θ/τ)`, τ annealed 1 → 0.1 geometrically; rounded at
  `θ > 0` for every evaluation. Init `θ = −3`, lr 0.05.

Loss: mean token KL to the BF16 teacher on the 128 math/code calibration
sequences, forward exactly as the multi-round scoring pass (per-token
FourOverSix activations, straight-through). The gradient a backward hook hands
the optimizer is `dKL/dm_u = Σ_u (δᵀx̃) ⊙ (A − B)` — the multi-round tile score,
on a minibatch. Adam (β 0.9/0.999, ε 1e-12, constant lr, no weight decay),
batch 8, 16 steps/epoch, 20 epochs (320 steps). **No candidate filter, no
backtracking, no acceptance test.** The 192 development documents were
evaluated every 2 epochs as a monitor only; the reported map is the last epoch.

## Results

Paired per window over the released WikiText-2 (2048-token) / C4 windows.
FourOverSix per-window NLLs from `results/kse_paper/job_336566`; multi-round
rows are the optimized maps of `MIXFP4_REPORT.md` §5 and reproduce its paired
numbers. Produced by `summarize_train_map.py`.

| Run | E0M3 tiles | final dev KL | WikiText | C4 | ΔPPL vs FourOverSix (wiki / c4) | ΔNLL vs FourOverSix ±2SE (wiki / c4) | ΔNLL vs multi-round ±2SE (wiki / c4) |
|---|---:|---:|---:|---:|---|---|---|
| multi-round 256x64 | 8,393 | 0.09572 | 6.835411 | 9.771621 | -0.0401 / -0.0521 | -0.00585±0.00187 / -0.00532±0.00220 | — |
| multi-round 8x64 | 3,645 | 0.09374 | 6.817620 | 9.759931 | -0.0579 / -0.0638 | -0.00846±0.00171 / -0.00652±0.00203 | — |
| **train STE 256x64** | 38,176 | 0.08868 | **6.805528** | **9.723484** | **-0.0700 / -0.1002** | -0.01023±0.00174 / -0.01026±0.00241 | **-0.00438±0.00148 / -0.00494±0.00193** |
| **train STE 8x64** | 329,837 | 0.08349 | **6.784682** | **9.675402** | **-0.0908 / -0.1483** | -0.01330±0.00184 / -0.01521±0.00329 | **-0.00484±0.00142 / -0.00870±0.00240** |
| **train sigmoid 8x64** | 212,778 | 0.08418 | **6.781821** | **9.681478** | **-0.0937 / -0.1423** | -0.01372±0.00184 / -0.01459±0.00325 | **-0.00526±0.00152 / -0.00807±0.00226** |

(For scale: FourOverSix 6.875525 / 9.823733, BF16 6.240087 / 8.958212.)

Every trained map beats multi-round on both datasets, significantly
(|mean| > 2 SE), at both tile sizes. At 8×64 the trained maps roughly
double the C4 gain over FourOverSix.

## Training curves (dev KL, monitor only; start 0.10845)

| epoch | STE 256×64 (E0M3) | STE 8×64 (E0M3) | sigmoid 8×64 (E0M3) |
|---:|---|---|---|
| 4 | 0.10109 (182) | 0.09951 (2,072) | 0.10283 (1,045) |
| 8 | 0.09489 (7,518) | 0.08907 (83,110) | 0.08939 (61,569) |
| 12 | 0.09050 (19,410) | 0.08598 (183,041) | 0.08563 (170,392) |
| 16 | 0.08953 (29,325) | 0.08318 (260,629) | 0.08360 (204,449) |
| 20 | 0.08868 (38,176) | 0.08349 (329,837) | 0.08418 (212,778) |

- Nothing flips for 3 epochs: Adam moves each logit ≈ lr per step, so the init
  margin −1 acts as a soft significance threshold.
- **STE 256×64 has not converged** (dev KL still falling, ~2k flips/epoch
  net). More epochs may help.
- **STE 8×64** keeps flipping ~50k tiles/epoch at the end; dev KL flattened
  after epoch 16 (0.08318 → 0.08349), a mild sign of fitting the 128
  calibration sequences. Train KL (0.039) is far below dev KL.
- **Sigmoid** settles: flips fall from 42k/epoch to 3.5k as τ anneals; the
  rounded map's dev KL tracks STE's.

## Caveats

- One seed and one hyperparameter setting per arm; no selection-variance
  estimate. The ±2 SE above excludes selection variance.
- Trained maps elect 4.5× (256×64) and 90× (8×64) more E0M3 tiles than
  multi-round (9.0% and 2.4% of tiles). E0M3 is free on SM100 for uniform
  maps; heterogeneous-map GEMM timing is still unmeasured.
- Llama only; Qwen (path-sensitive under multi-round) is not yet run.
- Zero-shot accuracy not yet evaluated on these maps.
