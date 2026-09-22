# Map-conditioned GPTQ at the frozen 256×64 MixFP4 map (Llama-3.1-8B)

**Question.** The 256×64 type tile forces misfit scale blocks into the tile's
format. Columnwise error feedback (GPTQ) can compensate those errors in later
columns without moving any weight or changing the map, at zero runtime cost.
Does it widen MixFP4's gain over FourOverSix?

**Answer: no.** GPTQ improves every arm by about 0.007 NLL, but once both sides
are compensated MixFP4's advantage over FourOverSix is no larger. It vanishes on
C4, the only contrast that was significant under RTN. The fine 8×64 map gains
essentially nothing over FourOverSix+GPTQ either.

## Setup

- Maps: the frozen CE+KL k=3 maps from
  `/work/u4320956/task_reorder/transfer_20260920/llama8b/calibration/compact_masks.pt`
  (raw 256×64: 187 tiles; fine 8×64: 3,345 tiles). GPTQ never changes a map. It
  re-rounds codes and E4M3 block scales only within each tile's fixed format.
- Quantizer: `quantize/mapped_gptq.py`. Columnwise GPTQ with 64-column lazy
  batches, 1% damping, and per-16 block scales fixed at scale-group start (the
  `quantize/branched_format.py` convention). Its RTN limit reproduces the evaluated
  RTN weights **bitwise** on 17 checked matrices, as well as in `tests/test_mapped_gptq.py`.
- Hessians: `E[xq xqᵀ]` of FourOverSix-quantized inputs on the same 128
  OpenWebMath/CodeParrot × 512-token calibration sequences used for tile
  election. One pass through the BF16-weight model (not layer-sequential).
- Evaluation: the released 2,048-token WikiText-2 / seed-0 C4 windows, W4A4 with
  FourOverSix tensor-wide activations. The RTN arms reproduce the report's
  6.875525/9.823733 and 6.866879/9.801361 exactly.
- Jobs 422929–422933, gov113008, H200 `dev`.

## Results

| Arm | E0M3 tiles | WikiText-2 | C4 |
|---|---:|---:|---:|
| FourOverSix, RTN | 0 | 6.875525 | 9.823733 |
| Raw 256×64, RTN | 187 | 6.866879 | 9.801361 |
| FourOverSix + GPTQ | 0 | 6.826416 | 9.757629 |
| **Raw 256×64 + GPTQ** | 187 | **6.818543** | 9.761208 |
| Fine 8×64 + GPTQ | 3,345 | 6.823711 | **9.751053** |

Paired per-window ΔNLL ± 2SE (141 WikiText, 256 C4 windows):

| Contrast | WikiText | C4 |
|---|---|---|
| GPTQ − RTN, FourOverSix | −0.007168 ± 0.002071 | −0.006751 ± 0.002217 |
| raw256 − FourOverSix, RTN | −0.001258 ± 0.001547 | −0.002280 ± 0.001455 |
| raw256 − FourOverSix, GPTQ | −0.001154 ± 0.001498 | +0.000366 ± 0.001297 |
| fine8×64 − FourOverSix, GPTQ | −0.000396 ± 0.001484 | −0.000675 ± 0.001327 |
| raw256 − fine8×64, GPTQ | −0.000758 ± 0.001510 | +0.001041 ± 0.001269 |

## Reading

- Error feedback and E0M3 election repair largely the same errors. The E0M3
  benefit measured at the RTN model does not add on top of GPTQ, even at 8×64.
- The maps were elected with gradients at the RTN model. Re-electing at the GPTQ
  model is the untested follow-up, but the collapse of the 3,345-tile 8×64 map
  suggests the available headroom is small.
- GPTQ is therefore not a MixFP4-specific improvement. Report it, if at all, as
  a baseline-strengthening control that both formats receive.
- Scope: one model, one calibration set, one non-sequential Hessian pass.

Reproduce: `sbatch slurm/mapped_gptq.sbatch <policy> <outroot>` for each policy
in `run_mapped_gptq.POLICIES`, then `python summarize_mapped_gptq.py <outroot>`.
