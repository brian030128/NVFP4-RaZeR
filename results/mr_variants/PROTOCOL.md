# MR-OPT variants: significant steps and warm start — protocol

Written 2026-09-25, before any run of this study. The hash and registration time are in
`registration.json`. Deviations are appended in the last section, never edited in place.

**Context.** The final model set is Llama-3.1-8B, Mistral-7B-v0.3, Phi-4 and Qwen3.8-27B. Part C,
the QAT study and zero-shot stay paused. Nothing is selected or tuned on WikiText-2, C4 or
zero-shot.

## Configurations

**MR-OPT** is the current optimized multi-round configuration. Its search settings:
- `run_multiround.py --objective kl --dev-backend native --skip-ce-backward --deterministic`;
- strict-decrease acceptance (the default);
- `--budget-hours 12`, `--record-dev-values`.

Its speed and memory settings:
- `--memory-mode lean`;
- `--fused-act-quant --single-pass-epilogue --tile-score-kernel --chunked-loss`;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- development-evaluation batch 16 and scoring batch 8; Qwen3.8-27B's settings are set at its gate.

| name | added flags |
|---|---|
| MR-OPT | (none) |
| MR-OPT+SIG | `--significant-steps`: a step is accepted only if the paired per-document change in development KL has mean + 2 SE < 0 |
| MR-OPT+WS | `--warm-start`: each round's backtracking starts at twice the previous round's accepted step instead of all candidates |
| MR-OPT+SIG+WS | both |

## Scope and order

- **Models and units:** Llama-3.1-8B, then Mistral-7B-v0.3, then Phi-4; units 256x64 and 8x64,
  all four configurations each.
- **Data:** Llama uses the Part B / Llama data (`/home/dev/n16k64_campaign/cost_comparison/data`).
  Mistral and Phi-4 use Part C's registered data (`/home/dev/n16k64_campaign/multimodel/data`).
- **Order:** within a model, MR-OPT runs first, at both units.
- **Gate before Qwen3.8-27B:** after Phi-4 is done, committed and reported, the study stops. The
  user must confirm Qwen3.8-27B and its settings first.
- **Llama sanity gate:** Llama's MR-OPT maps must reproduce the committed DET-NATIVE maps exactly
  (`map.pt` sha256 6e9704f5… at 256x64, 471aa56a… at 8x64). If they do not, the study stops and
  this is reported.
- **Part C candidates:** the MR-OPT runs of Mistral and Phi-4 are candidates for Part C's final
  results. Their maps and evaluations are kept with their run records.

## Reported per model × unit × configuration

- **Calibration:**
  - rounds (scoring passes), development evaluations and final E0M3 tiles;
  - development KL, start → end;
  - setup and optimization time;
  - peak GPU allocated/reserved and host RSS;
  - the stop reason.
- **Tile overlap** with the same model and unit's MR-OPT map: shared, only MR-OPT, only the
  variant.
- **Quantized model on the native kernel (primary).** The evaluation is WikiText-2 and C4 with
  2048-token windows, one per forward, and per-window tensor-wide FourOverSix activations
  (convention (a)).
  - **Process:** one native process per model, `--memory-mode lean --unit 8x64 --fused-act-quant
    --single-pass-epilogue`.
  - **Maps in it:** FourOverSix, NVFP4, all eight calibrated maps. The 256x64 maps are expanded
    exactly to 8x64 tiles, with the element masks checked.
  - **Statistics:** PPL, and paired per-window ΔNLL, mean ± 2 SE (SD with ddof = 1, over √n
    windows). Each map is compared with FourOverSix and with the same unit's MR-OPT map.
- **Fake evaluation (secondary):** FourOverSix, NVFP4 and the two MR-OPT maps in one fake process,
  plus BF16 in its own fake process. All evaluation processes of a model must use identical
  windows (token hashes).

## Decision rule (fixed before any result)

- **ACCEPTABLE:** a variant is acceptable for a model and unit if its map is not significantly
  worse than MR-OPT on either corpus. On native evaluation, the paired ΔNLL (variant − MR-OPT)
  plus 2 SE must not be above 0 on WikiText-2, and likewise on C4.
- **Recommendation:** the fastest configuration (total optimization time over the tested models
  and units) that is acceptable at both units on every tested model. If none is, MR-OPT stays.
- **Adoption:** the recommendation is reported only. The user decides.

## Rules

- **Commit and push:** commit as chenjiaj109550158 <chenjiaj.cs13@nycu.edu.tw> after each model.
  Try once to push `repro/n16k64-rtx-pro-6000` (never main, no force). If the push is blocked,
  report it and do not retry.
- **Reporting:** report to the user after each model, and give an overall time estimate after
  Llama.

## Deviations (append-only)

1. **2026-09-25, before any run: clarification of the decision rule.**
   - **The conflict:** the task defines ACCEPTABLE as "not significantly worse than MR-OPT on
     either corpus". It then gives the formula "paired ΔNLL + 2 SE is not above 0", which this
     protocol copied. The two disagree: the formula requires mean + 2 SE ≤ 0, that is, the variant
     significantly better than MR-OPT, or exactly equal to it.
   - **Both readings are reported:**
     - **(i) primary, as worded — not significantly worse:** mean − 2 SE ≤ 0 on WikiText-2 and on
       C4 (the Task 1/1b SAFE rule);
     - **(ii) the formula as given:** mean + 2 SE ≤ 0 on both corpora.
   - **The recommendation** is given under each reading, and the discrepancy is sent to the user.
2. **2026-09-25 17:46 UTC, the user's confirmation of the decision rule** (relayed by
   nvfp4-razer-c9). At this time only the first MR-OPT run (Llama, 256x64) was in progress, and no
   variant run had started.
   - **The official ACCEPTABLE rule is reading (i), not significantly worse than MR-OPT.** For each
     corpus, the paired ΔNLL (variant − MR-OPT) must satisfy mean − 2 SE ≤ 0, on both WikiText-2
     and C4, at both units. This is the Task 1/1b SAFE rule. The formula in the task text
     ("ΔNLL + 2 SE not above 0") was an error.
   - **Reading (ii)** (mean + 2 SE ≤ 0) is still reported, as secondary.
   - **The recommendation** is the fastest configuration that is acceptable under reading (i) at
     both units on every tested model. If there is none, MR-OPT stays.
3. **2026-09-26, after the queue stopped at the Qwen3.8-27B gate: a third unit, 16x64** (user
   request, relayed by nvfp4-razer-c9).
   - **Scope:** Llama-3.1-8B, Mistral-7B-v0.3 and Phi-4, all four configurations.
   - **Registration:** registered separately, before any 16x64 test or run, in
     [ADDENDUM_16x64.md](ADDENDUM_16x64.md) (hash and time in `registration_16x64.json`).
   - **Unchanged:** the recommendation over 8x64 and 256x64 (MR-OPT). The 16x64 results are
     additional information.
   - **The Qwen3.8-27B gate stays closed.**
