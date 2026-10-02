# The main W4A4 perplexity table (tab:main-ppl): protocol

Written 2026-10-02 on branch `main-ppl`, cut from `flipquant-maps` c89ee2c (tm-opt based), before any GPU run of this
study. The hashes and time are in `registration.json`. Deviations are appended at the end, never edited in place.

**The request.** The user's top-priority request, relayed by nvfp4-razer-c9: run the full W4A4 perplexity main table
and fill every number. The 8x64 kernel plan stays on hold. The plan was approved by nvfp4-razer-c9 on 2026-10-02.

## 1. The table

- **Columns:** WikiText-2 and C4 for six models: Qwen3-1.7B, Qwen3-8B, Mistral-7B, Nemotron-Nano-9B-v2, Phi-4 and
  Qwen3.8-27B. These are ~/flipquant's registry models, at the revisions the delivered FlipQuant maps were trained on.
- **Rows:**
  - BF16;
  - native (RTX PRO 6000): NVFP4, FourOverSix, and FlipQuant (ours) 8x64 / 16x64 / 256x64, the delivered
    `flipquant_<unit>` maps;
  - simulated: IF4 (Cook et al.) 1x16 and MixFP4 (Zou et al.) 1x16, with weights and activations both under each
    method's own per-block rule (§4).
- **Appendix only:** fake (c) FourOverSix per model (`fo6-fake`). It lets the simulated rows be compared with
  FourOverSix like for like, in the same simulator (approved with the plan).
- **Mistral:**
  - **Primary column:** Mistral-7B-Instruct-v0.3, the registry model and the delivered maps.
  - **Secondary column:** the base Mistral-7B-v0.3, so that the user can choose. It reuses the paper's native rows,
    base maps and BF16 (§6), and adds its simulated rows and the fake FourOverSix row.

| column | NVFP4-RaZeR key | model @ revision | maps |
|---|---|---|---|
| Qwen3-1.7B | qwen3_1p7b | Qwen/Qwen3-1.7B @ 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | delivered, trained 2026-10-01 |
| Qwen3-8B | qwen3_8b | Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218 | delivered, trained 2026-10-01 |
| Mistral-7B (primary) | mistral7b_ins | mistralai/Mistral-7B-Instruct-v0.3 @ c170c708c41dac9275d15a8fff4eca08d52bab71 | delivered, trained 2026-10-01 |
| Nemotron-Nano-9B-v2 | nemotron9b | nvidia/NVIDIA-Nemotron-Nano-9B-v2 @ 6533e8de2c68e4536bf7c411d7a3ce5734111476 | delivered, trained 2026-10-01 |
| Phi-4 | phi4 | microsoft/phi-4 @ 2db69c1c3e91a05d2c64a3185acfbaf36f744e25 | the paper's (delivered as copies) |
| Qwen3.8-27B | qwen27b | Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 | the paper's (delivered as copies) |
| Mistral-7B (secondary) | mistral7b | mistralai/Mistral-7B-v0.3 @ caa1feb0e54d415e2df31207e5f4e273e33509b1 | the paper's base-model maps |

## 2. One harness for every row: NVFP4-RaZeR `run_ppl_deploy.py` (the paper's step 03)

- **Windows:** the released protocol windows, `run_baseline_protocol_audit.data(tok, prior, 2048)`.
  - WikiText-2 test, joined and cut into 2048-token windows;
  - 256 C4 validation crops of 2048 tokens.
  - Token hashes are recorded. Within a model, every row must have the same windows (`analyze.py` asserts it).
- **NLL and PPL:**
  - batch 1, one window per forward;
  - the per-window NLL is the mean cross-entropy over the 2047 predicted tokens, from float32 logits;
  - use_cache is True for WikiText-2 and False for C4 (the paper's convention);
  - PPL = exp(mean NLL).
- **Processes:** one per (model, row), as in step 03 (`experiments/main_ppl/run.py`).
- **Environment:** the paper runner's (`experiments/paper/paper_common.env` + `ACCURACY_ENV`):
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, TF32 off.
- **Kernels:** `SM120_BUILD_DIR=/home/dev/NVFP4-RaZeR/sm120/build`, the paper's builds. They were built at tm-opt
  4b85cec on 2026-09-27, an ancestor of this branch, and step 03 ran on them. No kernel is changed.
- **Why this harness and not ~/flipquant razer-port:**
  - every paper PPL number came from it, so reuse can be proven by an exact re-run;
  - the simulated rows need NVFP4-RaZeR in any case (the user excluded the Cook/Zou rules from ~/flipquant);
  - one tool keeps the windows, the reduction and the pairing identical.
  - razer-port's kernels were shown bitwise equal to these in its stage 2.

## 3. Native rows (convention (c): per-token activations, NativeLinear)

- **Kernels:** step 03's defaults by type block.
  - NVFP4 and FourOverSix: the stock set (`auto_stock`), with their own per-token activation quantizers (nvfp4_rows /
    four_over_six_rows);
  - FlipQuant 8x64: `n8k64_wB` (weights on B);
  - 16x64 and 256x64: the mixed set (`auto`, 256x64 exported as 16x64 granules), with FourOverSix activations.
  - Every set computes the same outputs bit for bit.
- **Coverage:** checked in every forward by run_ppl_deploy. Every quantized Linear runs natively, with no fallback.
- **Artifacts of the four new models:** exported as step 02 does (`export_map_artifact.py --ownership`).
  - The maps are the delivered ones. Each map's source map.pt, whose sha256 the delivered file records, is checked
    against the delivered `~/flipquant/maps/<key>/flipquant_<unit>.pt` (sha256 and tiles) before it is exported.
  - The exporter checks the weights against the calibration record, and the packed weights against the fake-quant
    weights bitwise. Its ownership check decodes every weight through the kernel.
- **New-model plumbing:** the four models are added to `sm120/eval/common.MODELS` at the pinned revisions.
  - Their data roots are the map delivery's `/home/dev/n16k64_campaign/fqmaps/data`: fit-only calibration records with
    the weight hashes, written with transformers 5.16.1, so no transformers deviation.

## 4. Simulated rows: IF4 (Cook et al.) and MixFP4 (Zou et al.), W4A4, 1x16

**Simulator:** fake (c). The quantized weights are installed in BF16, every quantized Linear input is quantized and
dequantized per token, and the GEMM runs in BF16 (`run_ppl_deploy.py` policy `fake:w4a4:<rule>`).

**Weights:** Experiment A's 1x16 rules exactly (`quantize/adaptive_formats.quantize(w, rule)`), verified there
(`results/paper_extra/A/A1_CHECK.md`).

- **IF4 (Cook et al., arXiv 2603.28765; official code mit-han-lab/fouroversix @ dadfad69):**
  - α = amax / (6·448) over the tensor;
  - one block scale e4m3(bmax / 6), relative to α;
  - FP4 candidate: E2M1, round to nearest even;
  - INT4 candidate: round(clamp(x_b · 1.16666666, −7, 7)), half to even, dequantized × 0.8571428571;
  - per 16-block the lower sum of squared errors; ties keep FP4.
- **MixFP4 (Zou et al., arXiv 2605.31035, Algorithm 1):**
  - s32 = amax / 2688;
  - block scales: E2M1 e4m3(bmax / 6); E1M2 e4m3(bmax / 7), i.e. the integers −7..7;
  - round to nearest even;
  - per block the lower squared error; ties go to E1M2;
  - a zero (underflowed) scale gives a zero block, and such blocks are counted.

**Activations:** the same rule per 16-element block along K, at runtime, on every quantized Linear input, with the tensor
scale taken per token (α_t = amax_t / (6·448); s32_t = amax_t / 2688). Implemented in
`quantize/adaptive_formats.quantize_rows`.

- **The papers apply their rules to activations:**
  - Cook et al.'s official code: `ModuleQuantizationConfig.activation_dtype` defaults to the module dtype, so
    `--quantization-scheme if4` quantizes activations with IF4. For IF4 the scale rule is forced to static_6, so only
    FP4 vs INT4 is selected. amax = x.abs().max() over the tensor.
  - Zou et al.: "we quantize weights, activations, and gradients to simulate MixFP4 at the GEMM boundaries"
    (Algorithm 1, s32 over the tensor).
- **Deviation from both papers:** the per-token tensor scale. It was **decided by the coordinator (nvfp4-razer-c9)**,
  not by the papers or the user, for parity with the per-token activation scales of every native row (convention (c)).
  The coordinator is telling the user and will relay if per-tensor is wanted instead. Everything else is the per-tensor
  rule, operation for operation, with the row's amax.
- **Check before registration** (`check_act_rules.py`, CPU; `act_rules_check.json`):
  - 12 real Qwen3-1.7B activations: inputs of q_proj, o_proj, gate_proj and down_proj in layers 0, 14 and 27, on the
    first WikiText-2 window, 256 tokens each;
  - plus 4 edge rows: all zero, an E4M3-underflow row, a tie row, and an outlier;
  - 3,076 rows in all. For every row:
    - IF4's per-token result equals Cook et al.'s official reference run on that row alone, bitwise in BF16 values and
      per-block choices;
    - it also equals Experiment A's per-tensor rule on that row;
    - Zou's per-token result equals Experiment A's rule on that row.
  - **All equal.** On the real activations, 37.9 % (IF4) and 38.9 % (Zou) of the blocks take the uniform format.
- **Recorded per run:** the activation blocks' uniform share and the zero-scale blocks (`activation_format`), next to
  the weights' (`format`).

## 5. Derived results (`experiments/main_ppl/analyze.py`)

- **Paired ΔNLL(row − reference)** per model and corpus: the mean over windows of the per-window differences, with
  SE = sd / √n. "Significantly better" means ΔNLL + 2 SE < 0.
  - **References:** every row against native FourOverSix (and NVFP4). The simulated rows are also compared with the
    simulated FourOverSix (fo6-fake).
- **Loss recovered(row):** 1 − Σ_p [logPPL(row,p) − logPPL(BF16,p)] / Σ_p [logPPL(NVFP4,p) − logPPL(BF16,p)].
  - It is summed over the table's 12 model–corpus pairs. NVFP4 scores 0 % and BF16 100 %.
  - It is computed for both Mistral variants.
- **†:** a FlipQuant cell that is not significantly better than native FourOverSix.
- **Bold:** the best (lowest PPL) natively executable result per column, among NVFP4, FourOverSix and FlipQuant
  8x64 / 16x64 / 256x64.
- **Outputs:** `main_ppl.json` and `main_ppl.csv` (every PPL and paired ΔNLL with its SE); `table_main_instruct.tex`
  and `table_main_base.tex`; `table_dnll_appendix.tex` (Appendix D); `REPORT.md`.

## 6. Reuse of the paper's rows

- **What is reused:** Phi-4's, Qwen3.8-27B's and base Mistral-7B-v0.3's native and BF16 rows, from the paper's step 03
  records (`/home/dev/n16k64_campaign/paper/ppl/<model>/<row>/report.json`): the same harness, artifacts, kernels,
  windows and builds.
- **The condition:** re-runs of these rows must equal the records per window, bitwise (every NLL and the PPL):
  - Phi-4: bf16 and fo6;
  - Qwen3.8-27B: ours-8x64;
  - Mistral-7B-v0.3: bf16 and fo6.
- **On any mismatch:** that model's rows are not reused. The run stops and the coordinator is told.
- **New runs for these models:** their simulated rows and fake FourOverSix.

## 7. Order and feasibility

- **Feasibility first:** a Nemotron smoke run.
  - It exports the Nemotron FourOverSix and 8x64 artifacts.
  - Then BF16, FourOverSix, FlipQuant 8x64 and IF4 run on 2 windows per corpus. This is not a result.
  - NemotronH runs transformers' torch Mamba-2 path (mamba_ssm is not installed). In eval mode it calls in_proj and
    out_proj as modules.
  - If it fails, the run stops and the coordinator is told before anything else is tried.
- **Then the models,** smallest first: Qwen3-1.7B, Qwen3-8B, Mistral-7B-Instruct, Nemotron, Phi-4, Qwen3.8-27B, then base
  Mistral (the simulated and fake FourOverSix rows only).
- **Within a model:** BF16, the native rows, fake FourOverSix, then IF4 and Zou.
- **Not tuned:** nothing is selected or tuned on WikiText-2 or C4. The maps come from calibration data, and the rules
  are fixed.

## Deviations (append-only)

1. **2026-10-02 06:55 UTC, during the first model's runs, before any analysis output: formatting-only changes to
   `experiments/main_ppl/analyze.py`.** The definitions in §5 (the ΔNLL and its SE, loss recovered, the † and bold
   rules) are unchanged.
   - The PPL decimals of the main LaTeX tables are a parameter (`--decimals`, default 2).
   - The Appendix-D ΔNLL cells are in nats per token with 4 decimals (± 2 SE), as the paper's step 07 tables, instead of
     milli-nats.
   - A new output, `tables.md`, has every PPL (4 decimals), the ΔNLL table with * for |Δ| > 2 SE, and loss recovered.
   - The registered sha256 was in registration.json; the new one is 8c253aea5ccbd94e69cb173d84dc2086f8d2c610f466b9a48e52b96886c3234b.

2. **2026-10-02 07:00 UTC: a diagnostic run outside the table, plus robustness in the analysis.** Neither changes a
   number.
   - **What prompted it.** On Qwen3-1.7B, all three FlipQuant maps give a lower WikiText-2 PPL than BF16: 8x64 is
     15.69 vs 16.72, better than BF16 on 141 of 146 windows. NVFP4 and FourOverSix are worse than BF16 on every
     window.
   - **The diagnostic.** To separate the map from the native path, the same 8x64 map was evaluated in the fake (c)
     simulator (`fake:map:` of the exported .mixfp4map; `diagnostics/qwen3_1p7b/ours-8x64-fake`).
     - It gives WikiText-2 15.6974 and C4 19.5565 (native: 15.6918 and 19.5472).
     - It is better than BF16 on the same 141 of 146 WikiText-2 windows.
     - So the effect belongs to the map, not to the kernels. It is reported as found.
   - **The analysis change.** `analyze.py` now skips a record that is not complete, with a warning, and lists it in the
     outputs (`incomplete_records_skipped`) instead of stopping. That allows dry runs while rows are still running.
     The final analysis requires an empty list.

## Amendment 1 (2026-10-02 ~07:30 UTC, before any of its runs): the rules on the weights only, with FourOverSix activations

- **The request.** The user, relayed by nvfp4-razer-c9, asked for two more simulated rows per column, for all 7 columns
  (the 6 models and the base-Mistral column):
  - **IF4 (Cook et al.) 1x16 (W) + FO6 act** (label `if4w`);
  - **MixFP4 (Zou et al.) 1x16 (W) + FO6 act** (label `zouw`).
- **The setting.** The weights follow the rule's own 1x16 per-block selection, unchanged from the W4A4 rows. The
  activations are quantized by plain per-token FourOverSix, FlipQuant's activation quantizer (four_over_six_rows,
  convention (c)), with no per-block format selection.
  - This is Experiment A's setting (`fake:format:<rule>:1x16`), now run for the table models.
  - It is simulated (fake (c), BF16 GEMM), with the same windows and harness.
- **Placement.** The user will decide whether these rows go in the main table or the appendix. The outputs include them
  in both LaTeX tables, in a block separate from the native rows.
- **Labels.** The existing W4A4 rows are now titled "(W+A)" to keep the two pairs apart. Their records and labels (`if4`,
  `zou`) are unchanged.
- **Comparisons.** Like the other simulated rows, the new rows are compared with the simulated FourOverSix (fo6-fake),
  like for like, and with native FourOverSix. They are included in the ΔNLL ± 2 SE table, loss recovered, the CSV/JSON
  and the report.
- **Cross-check with Experiment A.** Phi-4 and the base Mistral-7B-v0.3 have Experiment A records of the identical
  setting: the same model, revision, data root, windows and harness. They are `if4-1x16`, `zou-1x16` and A's fake
  FourOverSix `fo6`, which is the same setting as our `fo6-fake`.
  - The rows are run anyway, and each run's per-window NLLs are compared with A's record bitwise. The result is logged
    in commands.log as `CROSSCHECK-A`.
  - A mismatch is reported, never hidden. A's records are not substituted for this study's runs.
- **Order.** After the registered queue. Rows of a model the queue has already finished may run concurrently with the
  queue on the same GPU. Accuracy runs are deterministic, so concurrency changes only the timing, and no timing is
  reported.
- **Code.** In `experiments/main_ppl/run.py`: rows `if4w`/`zouw` (`WEIGHT_ONLY`), and `crosscheck_a` for if4w, zouw and
  fo6-fake. In `experiments/main_ppl/analyze.py`: the two rows, and the "(W+A)" titles. Hashes are in
  `registration_amendment1.json`.

3. **2026-10-02 11:41 UTC: the scope is cut to BF16, NVFP4 and FourOverSix (the user's decision, relayed by
   nvfp4-razer-c9).** The FlipQuant calibration settings are going to change, so the table now holds only NVFP4 and
   FourOverSix, with BF16 as the reference.
   - **The queue.** The driver was stopped at 11:41 UTC. Its running process, Nemotron `zou` (W+A), was left to finish,
     so its record is complete, not partial. The automatic launch of Qwen3.8-27B's `if4w`/`zouw` was cancelled.
   - **No further rows of these kinds run:** FlipQuant, IF4 / Zou (W+A or W-only), or fake FourOverSix.
   - **Still run: the registered reuse rechecks.**
     - Phi-4: bf16 and fo6. Base Mistral-7B-v0.3: bf16 and fo6.
     - **Qwen3.8-27B: switched from ours-8x64 to fo6,** since the FlipQuant rows are deferred.
     - The rule is unchanged: per-window bitwise equality with the paper's step 03 record, or nothing of that model is
       reused.
   - **Kept as recorded, "deferred / partial, not part of this table":**
     - the FlipQuant rows of the four new models;
     - the IF4 / Zou W+A and W-only rows and fake FourOverSix, as far as they ran;
     - the Qwen3-1.7B fake-map diagnostic and the Nemotron smoke.
     The new calibration will supersede the FlipQuant rows. `deferred.md` lists every such record without its numbers.
   - **Deliverables:**
     - the table with BF16 / NVFP4 / FourOverSix for all 7 columns, in both Mistral variants;
     - ΔNLL NVFP4 − FourOverSix ± 2 SE per model and corpus (Appendix D);
     - FourOverSix's loss recovered (NVFP4 is 0 % by definition) over the 12 pairs of each variant;
     - the deferred-rows appendix.
   - **`analyze.py`** gained `--scope baselines`, now the default. `--scope full` is the registered scope. Bold now marks
     the lower PPL of NVFP4 and FourOverSix per column; † does not apply.

## Amendment 2 (2026-10-02 ~11:50 UTC, before any of its runs): the IF4 / MixFP4 rows are restored

- **The user's update** (relayed by nvfp4-razer-c9): the FlipQuant cut of deviation 3 stays, and GPTQ is not run. The
  IF4 (Cook et al.) and MixFP4 (Zou et al.) rows are restored in both variants for all 7 columns, both Mistral
  variants:
  - (a) **W+A** (`if4`, `zou`): the rule on weights and activations, §4;
  - (b) **(W) + FO6 act** (`if4w`, `zouw`): the rule on the weights with per-token FourOverSix activations, amendment 1.
- **The appendix reference returns too:** fake FourOverSix (`fo6-fake`) in every column, so that each simulated row has
  its like-for-like reference.
- **The runs still missing:**
  - Phi-4: fo6-fake, if4, zou;
  - base Mistral-7B-v0.3: fo6-fake, if4, zou;
  - Qwen3.8-27B: fo6-fake, if4, zou, if4w, zouw.
  - Every other model already has all of these rows (or the running Nemotron `zou` completes it).
- **Order and memory:**
  - first the reuse rechecks of deviation 3;
  - then Phi-4's and base Mistral's rows, in two concurrent drivers (about 30 + 15 GB);
  - then Qwen3.8-27B alone: its fo6 reuse recheck (deviation 3), then its five rows.
- **Cross-checks:** Phi-4's and base Mistral's fo6-fake are compared bitwise with Experiment A's fake FourOverSix
  (`CROSSCHECK-A`).
- **Deliverables:**
  - the table with BF16 / NVFP4 / FourOverSix and the four simulated rows, for all 7 columns, in both Mistral variants;
  - ΔNLL ± 2 SE vs native FourOverSix, and also vs fake FourOverSix for the simulated rows;
  - loss recovered per row.
  - FlipQuant stays deferred (`deferred.md`).
- **`analyze.py`:** `--scope restored`, now the default, gives these rows plus fo6-fake as the reference; `baselines` and
  `full` are kept.
