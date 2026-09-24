# Native shadow verification of the development evaluation — protocol

Written 2026-09-24, before the pre-check and the shadow run. The criteria below are fixed.
Anything changed after results are seen is appended in the last section with its reason.

**Question.** `run_multiround.py` accepts or rejects every backtracking step on the mean
development KL of a fake-quantized forward (BF16 GEMM on dequantized weights). Would any of
those decisions change if the development evaluation ran on the native mixfp4 kernel
instead?

Native cannot be bitwise equal to fake: it accumulates FP4 products in FP32 and rounds D to
bf16 before the activation's global scale is applied. So this protocol tests decisions, not
bits.

## Setup

- Same machine, environment, data root and flags as the cost study's B-256-opt: 1× RTX PRO
  6000, env `n16k64`, `--data-root /home/dev/n16k64_campaign/cost_comparison/data
  --transformers-deviation`.
- Native kernel: `/home/dev/n16k64_campaign/realquant/bin/libb8x64.so`, sha256 0e237ada…
  - It is the mixfp4 SM120 kernel at /home/dev/mixfp4 @7b3ab34, SASS-patched by
    `repro_local/realquant/build.sh`.
  - Weights are operand B. The format granule is 8 output rows × 64 K, so a 256x64 map unit is
    a union of 32 granules.
- Code: `run_multiround.py --shadow-native` with `repro_local/realquant/native_dev.py`.
  - Scoring is unchanged: fake quantization, BF16 backward.
  - Weights: both candidates, E2M1 FourOverSix and E0M3 α = 1, are packed once from the source
    weight. For every evaluation the packed weight is built by per-tile selection from the
    current map. Only modules whose map changed are repacked.
  - Activations: per-document tensor-wide FourOverSix (the rule of `quant_per_document`). Each
    document gets one FP32 global scale. Codes and UE4M3 block scales are computed against it
    by the fused Triton kernel with the scales given, and every token of the document gets that
    scale in the epilogue.
  - Eval batch 16 (8,192 tokens per forward). The teacher and the per-document KL/CE code are
    `run_multiround.py`'s. `lm_head` stays BF16, as in fake.
- Required checks. Each run aborts if any fails.
  - For all 224 matrices, decode(packed candidate) equals `decode_base` / `decode_alt`
    bitwise, signed zeros included.
  - decode(the packed weight for the map) equals the weight `apply()` installs, bitwise. This is
    checked for the start map, for one random mixed map (every unit flipped with probability
    1/2, seed 0), and for the start map again after restoring it.
  - The native activation values of the first 64 Linear calls equal `quant_per_document`
    bitwise. After those calls the checks stop, so they do not distort the timing.
  - The unit gate `repro_local/realquant/test_native_dev.py` passed before this protocol was
    written. It covers 14 matrices, random 256x64 and 8x64 maps, and 12 activation calls, with
    native vs fake F.linear relative error 3.4–3.7e-3 (BF16 output rounding).

## Shadow logic

- The fake evaluation makes every decision, exactly as without the flag. At every evaluation
  (the initial one and every try) the native evaluation runs on the same model state and is
  only logged.
- Native "current" is initialised with the native evaluation of the start state. When fake
  accepts a try, it is set to that try's native value.
- For every try, the log records:
  - fake new and current KL and CE, and fake's decision;
  - native new and current KL and CE, and native's decision: `improves(native_new,
    native_current)`, the same rule as fake (for `--objective kl`, `native_new.kl <
    native_current.kl`);
  - Δfake = fake_new.kl − fake_current.kl and Δnative = native_new.kl − native_current.kl;
  - discrepancy = |Δnative − Δfake| and margin = |Δfake|;
  - per-document KL and CE for both;
  - evaluation seconds, fake vs native, plus the native repacking seconds.

## Criteria

- **PASS ("no effect on the selection path"):** native's decision equals fake's on every try.
  With native deciding, the trajectory would then be identical by construction.
- **Otherwise, FAIL.** Report every disagreement with its round, step size, margin and
  discrepancy, and do not claim "no effect".
- Also reported:
  - the distribution of discrepancy against margin;
  - the smallest margin among accepted tries;
  - the smallest margin among all tries.

## Runs

1. **Pre-check** (`--check-start --shadow-native`, eval batch 16, score batch 8): native vs fake
   per-document development KL and CE, reported as mean and maximum absolute difference and as
   the difference of the means.
   - (a) at the FourOverSix start map;
   - (b) at the B-256-opt final map: `--init-map
     /home/dev/n16k64_campaign/cost_comparison/runs/B_256x64_opt/map.pt`, 7,625 E0M3 tiles,
     sha256 7e93ab4e….
   - **Bug threshold, fixed now.** Either of the following counts as an implementation bug,
     which is fixed before run 2:
     - the mean development KL of native and fake differs by more than 10% relative;
     - the mean development CE differs by more than 0.01 nats.

     For scale: in the cost study (`batch_check.json`), changing only the batch size moved
     per-document KL by up to about 11% and the mean over 8 documents by about 4%.
2. **Shadow run:** `run_multiround.py --unit 256x64 --objective kl --eval-batch 16 --score-batch 8
   --budget-hours 12 --shadow-native`, same data and settings as B-256-opt.
   - The scoring pass is not run-to-run deterministic on this GPU (cost-study PROTOCOL.md
     §10, deviation 4), so this run's fake trajectory is not expected to repeat B-256-opt's.
   - The test compares fake and native on the same states within this run.

## Output

`results/native_dev_shadow/REPORT.md` gives:
- the PASS/FAIL verdict;
- the table of disagreements, if any;
- discrepancy vs margin;
- the smallest accepted margin;
- the pre-check differences;
- the timing.

It is committed on `repro/n16k64-rtx-pro-6000` after REPORT.md is complete. Nothing is pushed.

## Deviations (append-only)

1. **2026-09-24 08:53 UTC: first pre-check attempt aborted by a required check. No measurement
   was taken.**
   - The native E0M3 alternative encoded a negative value rounded to zero as −0.
     `decode_alt` (quantize/packed_candidates.py), whose output `apply()` installs, stores E0M3
     codes in offset binary, so its zeros are +0.
   - The values were equal ("0 elements differ"), but the bits were not.
   - Fix: E0M3 nibbles take their sign from `code < 0` (rq.py's `e0m3_nibbles`); E2M1 keeps the
     sign-bit convention of `decode_base`. The unit gate now compares against
     `decode_base` / `decode_alt` and passes.
   - The aborted run is kept as `precheck_start_map_aborted1`.
