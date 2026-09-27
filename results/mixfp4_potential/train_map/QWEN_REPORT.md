# Trained MixFP4 maps on Qwen3.8-27B, and on nvidia/Qwen3.8-27B-NVFP4

Method: MIXFP4_REPORT.md §3 unchanged (STE, Adam lr 0.02 eps 1e-12, init −1, 20
epochs, 128 math/code calibration sequences, KL to the BF16 teacher). Qwen runs at
batch 1 × accumulation 8, i.e. the same 16 steps per epoch. Evaluation: released
protocol, WikiText-2 2,048-token windows and 256 C4 crops. ΔNLL is paired per window
± 2 SE. Summaries: `summarize_train_map.py`. Outputs: `/work/u4320956/mixfp4_potential/train_map/`.

## 1. Qwen3.8-27B, W4A4 (FourOverSix start)

Jobs 445055 (256×64) and 445056 (8×64), 3 h 20 min each on one H200. FourOverSix
per-window NLLs are from job 336969. The multi-round maps are the deprecated optimized runs.

| Policy | E0M3 tiles | dev KL | WikiText-2 | C4 | ΔPPL vs FourOverSix | ΔNLL vs FourOverSix (wiki / c4) | ΔNLL vs multi-round (wiki / c4) |
|---|---:|---:|---:|---:|---|---|---|
| BF16 | — | — | 7.050375 | 9.893323 | | | |
| NVFP4 FourOverSix | 0 | 0.04625 | 7.287076 | 10.188365 | — | — | |
| multi-round 256×64 (deprecated) | 39,095 | 0.04316 | 7.223045 | 10.152189 | −0.0640 / −0.0362 | −0.00883±0.00361 / −0.00356±0.00089 | |
| multi-round 8×64 (deprecated) | 17,571 | 0.04207 | 7.205417 | 10.148049 | −0.0817 / −0.0403 | −0.01127±0.00442 / −0.00396±0.00089 | |
| **Trained 256×64** | 70,267 (4.7%) | 0.04262 | **7.131810** | **10.145408** | **−0.1553 / −0.0430** | −0.02154±0.00389 / −0.00423±0.00101 | −0.01271±0.00331 / −0.00067±0.00084 |
| **Trained 8×64** | 610,431 (1.3%) | 0.04164 | **7.096120** | **10.125233** | **−0.1910 / −0.0631** | −0.02655±0.00438 / −0.00622±0.00101 | −0.01528±0.00449 / −0.00225±0.00082 |

- Both trained maps beat FourOverSix significantly on both datasets.
- They beat multi-round significantly on WikiText. On C4, 8×64 is significantly
  better and 256×64 ties.
- 8×64 closes 81% of the WikiText gap from FourOverSix to BF16 (7.287 → 7.096 vs
  7.050) and 21% of the C4 gap.
- One seed per arm. Multi-round Qwen varied by ±0.04 WikiText between runs, so
  selection variance is not covered by the ± 2 SE.

## 2. nvidia/Qwen3.8-27B-NVFP4 as the starting point

**What the checkpoint is** (revision `482ca0f`, modelopt 0.47):
- NVFP4 covers only the MLP (gate/up/down × 64 layers) and `lm_head`.
- Every attention projection, full and linear (Gated DeltaNet), is FP8 W8A8 per-tensor.
- The NVFP4 scales were calibrated with Model Optimizer's **Local-Hessian** algorithm
  (a Hessian-weighted block-scale search on 2,048 samples), per its model card, not
  GPTQ. The codes are round-to-nearest under those scales.
- NVFP4 activations use a static calibrated global scale and dynamic E4M3 scales per
  16 elements.

**Emulation** (`quantize/modelopt_ckpt.py`, `run_train_map.py --modelopt`):
- Every tensor is loaded from the checkpoint. All 783 unquantized tensors are
  bit-identical to our pinned BF16 Qwen3.8-27B.
- NVFP4 weights are decoded from nvidia's codes and scales. The decode is bitwise
  against an independent decoder and sanity-checked against the BF16 weights
  (NMSE < 5%). FP8 weights have NMSE ≤ 7.4e-4.
- Activations follow nvidia's recipe (static scales), in training and evaluation alike.

**MixFP4 on top:**
- Tiles exist only on the 192 NVFP4 MLP matrices: 1,044,480 at 256×64 and
  33,423,360 at 8×64.
- E2M1 candidate: nvidia's own codes and scales, bitwise.
- E0M3 candidate: `block_max/7` on the BF16 weight under nvidia's global scale.
- Attention stays FP8 and `lm_head` stays nvidia's NVFP4.
- The baseline is a zero-epoch run, i.e. the checkpoint as shipped, with matched
  per-window NLLs.

Jobs 445063 (baseline), 445064 (256×64) and 445065 (8×64), 2 h 16–19 min each.

| Policy | E0M3 tiles | dev KL | WikiText-2 | C4 | ΔPPL vs nvidia | ΔNLL vs nvidia (wiki / c4) |
|---|---:|---:|---:|---:|---|---|
| nvidia NVFP4 (as shipped) | 0 | 0.03540 | 7.185603 | 10.179512 | — | — |
| **+ MixFP4 256×64** | 47,724 (4.6%) | 0.03448 | 7.159577 | 10.173418 | −0.0260 / −0.0061 | −0.00363±0.00263 / −0.00060±0.00079 |
| **+ MixFP4 8×64** | 438,750 (1.3%) | 0.03388 | **7.147312** | **10.161573** | **−0.0383 / −0.0179** | −0.00534±0.00262 / −0.00176±0.00075 |

- **8×64 improves nvidia's model significantly on both WikiText and C4.**
  **256×64 improves WikiText significantly; C4 is within noise.**
- The gain is about a quarter of the gain on our own FourOverSix Qwen model
  (WikiText ΔNLL −0.0053 vs −0.0266 at 8×64). Two reasons are visible:
  - Only the MLP is NVFP4 here: 192 of 496 text matrices can take E0M3.
  - nvidia's model already sits closer to BF16: dev KL 0.0354 vs 0.0462, and
    WikiText 7.186 vs 7.287.
- Each arm is one seed and one hyperparameter setting.
