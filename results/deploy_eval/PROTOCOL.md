# B2/B3 and the deployment-path verification — protocol

Written 2026-09-27 on branch `tm-opt`, after the SM120-kernel merge and before any Part 2–3 run. The hash and
registration time are in `registration.json`. Deviations are appended in the last section, never edited in place.
Nothing is selected or tuned on WikiText-2, C4 or zero-shot. Zero-shot is not part of this task.

**Task** (user-approved, relayed by nvfp4-razer-c9). The final method stays TM-OPT+TC. This task:
- implements B2: evaluation through the SM120 NativeLinear deployment path;
- runs B3: the ownership check on every exported artifact;
- confirms that the deployment path changes the compressed models' perplexity only negligibly.

## Tools (B2, B3)

- **`export_map_artifact.py`:** a TM-OPT+TC `map.pt` becomes a MIXFP4MAP/1 map and then an SM120 artifact, through
  `mixfp4_sm120.artifact.export`, which checks every packed weight bit for bit against the fake-quant weight.
  - **Kernels:** 8x64 maps use `n8k64_wB`; 16x64 maps use `n16k64_wA`. **256x64 maps run as 16x64 granules on
    `n16k64_wA`**: each 256-row tile becomes its 16x64 granules, and the element mask is checked unchanged.
  - **Candidates:** FourOverSix E2M1 and E0M3 alpha=1, the calibration's (`run_train_map.py`). The exporter
    checks, per module and bitwise, that they equal `mixfp4_sm120.artifact.fake_quant_weight`'s candidates.
  - **Weights:** it also checks that the model's weights equal the calibration record's (sha256 per matrix).
  - **Baselines:** FourOverSix (`--kind four_over_six`) and pure NVFP4 (`--kind nvfp4`) artifacts.
- **B3 (`--ownership`):** `sm120/eval/ownership_check.py` runs on every map artifact. The kernel executes every
  weight element through an identity GEMM, and the result must equal the stored decode, bitwise, in the map's format.
  - **Baseline artifacts:** the same per-module check with an all-E2M1 map on `stock_wA`.
  - **Reported:** the elements checked (informative, i.e. nonzero, nibbles) and the mismatches; 0 expected.
- **`run_ppl_deploy.py`, convention (c):**
  - **Windows:** the released protocol windows (`run_baseline_protocol_audit.data(tok, prior, 2048)` +
    `validate_evaluation_data`), checked by token hash against `sm120/eval/reference`; batch 1 per window.
  - **Per-window value:** as `run_multiround.py`'s (mean cross-entropy over the window's 2047 predicted tokens,
    float32 logits), so the numbers pair window by window with the (a) evaluations.
  - **NativeLinear (c):** the kernel's own per-token activation scales and one-rounding epilogue. Kernel by the
    artifact's type block:
    - `n8k64_wB` for 8x64;
    - `auto` (the width-selecting weights-on-A mixed set) for 16x64 and 256x64-as-16x64;
    - `auto_stock` for FourOverSix and NVFP4.
  - **Fake (c):** the same weights in BF16. Every quantized Linear input is quantized per token (FourOverSix rows
    for the maps and FourOverSix, NVFP4 rows for NVFP4, as `sm120/eval/common.FakeQuant`), then a BF16 GEMM.
  - **BF16:** the reference.
  - **Checks recorded:** native coverage (every quantized Linear runs natively once per forward, in every
    forward), the dense GEMMs of one forward, and the wall time per policy.
- **Kept, for the appendix:** the existing evaluators, native (a) and fake (a) (`run_multiround.py
  --eval-backend native|fake`).

## Scope

- **Models:** Llama-3.1-8B, Mistral-7B-v0.3, Phi-4, Qwen3.8-27B.
- **Maps:** the committed TM-OPT+TC maps (seed 0; `results/tm_opt/REPORT_TC.md` and Part Q) at 8x64 and 16x64
  (main) and 256x64 (appendix).
- **Policies:** the TM-OPT+TC map, FourOverSix and pure NVFP4, each under NativeLinear (c) and fake (c); BF16 as the
  reference.
- **Runs:**
  - one exporter process per artifact: 5 artifacts per model, 20 in all, each with `--ownership`;
  - one evaluation process per model, fake and BF16 policies first, then the native policies.
- **Settings as for the committed evaluations:** data root, and `--transformers-deviation` for Llama. Idle GPU;
  kernels built in `sm120/build` from this repository.

## Analyses (registered)

**(i) Kernel numerics, like for like: NativeLinear (c) − fake (c).**
- **Measure:** paired per-window ΔNLL, mean ± 2 SE (ddof = 1), per model × policy × corpus. Policies: the three
  maps, FourOverSix and NVFP4.
- **Criterion (registered):** "negligible" means **not significant (|mean| ≤ 2 SE) in each cell**.
- **Also reported:** the maximum |ΔNLL| over windows, and the PPL difference.
- **If any cell is significantly different,** it is reported with diagnostics (e.g. `sm120/eval/layerwise.py`),
  and the criterion is not changed.

**(ii) Effect preservation.**
- **Measure:** ΔNLL(TM-OPT+TC − FourOverSix) and ΔNLL(TM-OPT+TC − NVFP4), paired per window, under NativeLinear (c)
  and under fake (c). Both are reported.
- **Also reported:** whether every map stays significantly better (mean + 2 SE < 0) than FourOverSix and NVFP4
  under NativeLinear (c).

**(iii) Convention effect (appendix, descriptive): NativeLinear (c) vs our native (a) evaluator.**
- **Measure:** the same maps and baselines, paired per window. The (a) values are the committed evaluations
  (`results/tm_opt`, Part Q), not re-run.

**Wall time per model:** NativeLinear (c) vs native (a).
- **NativeLinear (c):** the evaluation seconds of each policy, as measured.
- **Native (a):** the committed evaluation processes' `final_evaluation` phase divided by their number of maps.
  That phase also contains the window preparation, once; it is stated as an approximation.
- **Also reported:** the one-time costs, i.e. the artifact export (c) vs the candidate packing (a).

## Rules

- **Commit and push:** commit on `tm-opt` as chenjiaj109550158 <chenjiaj.cs13@nycu.edu.tw> after Part 1 and after
  Parts 2–3, and push each time (never main, no force). The CUTLASS submodule is a gitlink; its contents are not
  committed. If a push is blocked, report it and do not retry.
- **Reporting:** report to the user after Part 1 and after Parts 2–3, then stop.

**Before registration** the tools were smoke-tested in a scratch directory; no result of these tests is used:
- the Llama TM-OPT+TC 8x64 export, with the ownership check;
- a 3-window evaluation of BF16, fake (c) and NativeLinear.

## Deviations (append-only)

1. **2026-09-27 18:10 UTC: `analyze.py` extended after the runs.** The registered analyses and the criterion are
   unchanged. Added:
   - a check that the evaluator's BF16 window NLLs equal the committed BF16 evaluations bitwise;
   - for every significant cell of (i), as the protocol requires, the window-level diagnostics (z, median, the
     largest windows, the mean without the largest) and the `sm120/eval/layerwise.py` summaries. The layerwise runs
     are listed in `diagnostics/run_layerwise.sh`: the window with the largest |ΔNLL| and window 0, for the two map
     cells. The NVFP4 cell has no map, so `layerwise.py` does not apply to it.

   The records were copied into `results/deploy_eval/` (`runs/`, `artifacts/` without the weights, `diagnostics/`),
   which the script now reads by default.
