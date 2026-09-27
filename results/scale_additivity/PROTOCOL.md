# Task 2: does scale training add to TM-OPT+TC, or cancel it? — protocol

Written 2026-09-27 on branch `tm-opt`, after Task 1 was committed and before any Task 2 run. The hash and registration
time are in `registration.json`. Deviations are appended in the last section, never edited in place. The study is
descriptive. Nothing is selected or tuned on WikiText-2, C4 or zero-shot.

**Task** (user-approved, relayed by nvfp4-razer-c9). A reviewer may argue that scale-only KL training (our arm D,
FOCUS-like) beats our method. The question is whether ours plus scale training beats either alone.

## The collaborator branch, and how this design differs

Read before the design: `origin/worktree-nvfp4-kl-scale-search`, MIXFP4_REPORT §7. It studied Llama-3.2-1B-Instruct,
and in its final panel 1B, 3B and 8B-Instruct. It used the trained-map recipe (STE, Adam lr 0.02, init −1) and the
FourOverSix evaluation convention. Findings:
- **KL scale search is a strong control.**
  - Their "scale search" is a binary per-block choice between FourOverSix's two scales (block max → 6 or → 4),
    trained with the same KL loss.
  - At a matched tile it recovers ≥ 90 % of the trained E0M3 map's gain.
  - At 1×16 (one choice per scale block, plain NVFP4) it beat every trained MixFP4 map on 1B.
- **Joint training (1×16 scale logits + E0M3 tile logits, one loss).**
  - At 8×64, E0M3 added a small, significant gain on top of the scale search (0.003–0.007 ΔNLL, 1B, 128 sequences,
    20 epochs).
  - At 256×64 it added nothing reliable.
  - Initialization mattered as much: a FourOverSix start beat an NVFP4 start.
- **More data, staging, other calibration.** With 512 sequences (overfitting after about 8 epochs), with staging
  (scale first, then tiles on top; frozen or joint), and with C4-train calibration, E0M3 on top of a trained scale
  gave no robust out-of-domain gain.
- **Model panel** (C4 + math/code calibration, early stopping on held-out C4): joint 8×64 ties scale search on
  average across 1B/3B/8B-Instruct.
- **Headroom analysis.** E0M3's gain is real per block (about 20 % error reduction), but a tile-wide type keeps
  12–18 % of it at 8×64 and about none at 256×64. Row permutations keep little more.

**Why this design is not a duplicate:**
- **Scales:** our SCALE arm is arm D, a continuous learned factor on every block's UE4M3 scale (LSQ gradient), not a
  binary FourOverSix choice.
- **Tile method:** ours is the final method TM-OPT+TC.
- **Model:** Llama-3.1-8B base, the paper's setting.
- **Evaluation:** the deployment path, NativeLinear (c).
- **Arms:** OURS→SCALE (tiles first, then scales) was not in their study. SCALE→OURS resembles their staged,
  frozen-scale variant, but with continuous scales and TM-OPT+TC.
- **JOINT is not run:** training tile logits and continuous scales in one optimizer needs a new trainer. The
  collaborator's joint results are cited instead.

## Arms

Llama-3.1-8B, the calibration fit set (128 sequences), KL to the BF16 teacher, deterministic, `--no-dev` for every
deployed arm. The unit is 16x64 (the main table's); 8x64 is added, because its cost equals 16x64's (Task 1).

1. **FourOverSix and NVFP4** (references), the Parts 2–3 artifacts.
2. **SCALE: arm D** (`run_cost_distill.py --arm scale`).
   - **Method:** weights frozen; a learned factor per 16-element block on the FourOverSix pre-rounding scale,
     scale = e4m3(clamp(f · pre)); plain NVFP4, E2M1 only.
   - **Budget and activations:** 20 epochs (TM's budget: 320 steps of 8 sequences), with per-token FourOverSix
     activations (`--act-rows`: TM-OPT+TC's training convention and the deployment convention (c); arm D used
     per-document).
   - **Learning rate, a method-development choice on the development set** (`choose_lr.py`):
     - Grid {1e-4, 3e-4, 1e-3}. Each rate is run for the full 20 epochs with the development set.
     - The rate with the lowest final development KL is chosen. The development set is the 192 math/code
       documents of the earlier campaigns (3 × 64), evaluated with per-token activations and the deployed weight.
       Ties go to the smaller rate.
     - If the lowest is at an edge of the grid, the grid is extended once in that direction (3e-3 or 3e-5), and the
       choice is the lowest of the extended grid.
     - The chosen rate is re-run with `--no-dev`. That run gives the deployed SCALE and its cost.
     - The development runs are method development, like the earlier choice of TM-OPT+TC's own settings. They are
       not counted in any arm's calibration cost; their times are reported separately.
3. **OURS:** TM-OPT+TC, the committed maps (16x64, 8x64; `results/tm_opt`), with the costs of Task 1.
4. **SCALE→OURS:** SCALE first, then TM-OPT+TC (the same settings, `--no-dev`).
   - Candidate B is SCALE's learned E2M1 (`run_train_map.py --base-scales`), packed with its learned scale bytes.
   - Candidate A (E0M3 α=1) is unchanged.
5. **OURS→SCALE:** the committed TM-OPT+TC map fixed, then learned scales for all blocks.
   - E2M1 blocks start at the FourOverSix pre-rounding scale; E0M3 blocks at block_max / 7.
   - The same budget and learning rate as SCALE, `--act-rows`, `--no-dev` (`run_cost_distill.py --arm scale --map`).
6. **JOINT:** not run (see above).

**Initialization checks, bitwise.** The learned parametrizations start exactly at their references:
- f = 1 reproduces FourOverSix;
- on a map, f = 1 reproduces the map's MixFP4 weight. `run_cost_distill.py --map` checks this on every module, in
  its own phase `map_init_check`, which is outside the calibration cost;
- the E0M3 part at f = 1 reproduces the E0M3 α=1 candidate.
- In SCALE→OURS, the packed candidate B must decode bitwise to the learned-scale E2M1 weight (the candidate store's
  check), and candidate A to the E0M3 α=1 candidate.
- The development runs of SCALE also check, after training, that the fused training quantizer equals the deployed
  weight bitwise (`fused_quantizer_bitwise_on_final_weights`).

## Deployment and evaluation

- **Export** with `export_map_artifact.py --scales` (a new option). The learned scales are valid UE4M3 values; the
  E0M3 flag is bit 7.
  - SCALE: E2M1 only, run on `auto_stock`.
  - SCALE→OURS: the map, with the learned scales on its E2M1 blocks (`--scales-apply e2m1`).
  - OURS→SCALE: the map, with learned scales on every block (`--scales-apply all`).
- **Exporter checks, per module:** weights equal the calibration record; the parametrization at f = 1 equals the
  standard candidates; the packed weight decodes to the learned fake-quant weight (value-equal). The ownership check
  (B3) runs on every learned artifact.
- **PPL:** `run_ppl_deploy.py`, NativeLinear (c), one process.
  - Policies: FourOverSix, NVFP4, OURS 16x64/8x64, SCALE, SCALE→OURS 16x64/8x64, OURS→SCALE 16x64/8x64, BF16.
  - Windows: the released protocol windows, batch 1.
  - Checks: the Parts 2–3 checks (windows, coverage). The reference policies (BF16, FourOverSix, NVFP4, OURS) must
    also reproduce the Parts 2–3 evaluation window for window, a regression check of `run_ppl_deploy.py`, which was
    refactored since (its helpers are shared with Task 3).

## Report (descriptive)

- **PPL** (WikiText-2 / C4) of every arm under NativeLinear (c).
- **Paired ΔNLL ± 2 SE:**
  - SCALE − FourOverSix;
  - OURS − SCALE (does ours beat scale-only?);
  - SCALE→OURS − SCALE and SCALE→OURS − OURS;
  - OURS→SCALE − OURS and OURS→SCALE − SCALE.
- **Additivity**, for X in {SCALE→OURS, OURS→SCALE}, per window: I = (X − FO6) − (SCALE − FO6) − (OURS − FO6), mean
  ± 2 SE. This compares the combined gain over FourOverSix with the sum of the single gains.
- **Interpretation rule, fixed in advance, per corpus.** The better single method is the one with the larger point
  gain over FourOverSix.
  - **additive:** X is significantly better than both single methods, and I is not significantly different from 0;
  - **more than additive:** X is significantly better than the better single method, and I < 0 significantly;
  - **partially overlapping:** X is significantly better than the better single method, but I > 0 significantly;
  - **cancelling:** X is not significantly better than the better single method.
  - If X is significantly better than the better single method with I not significant, but not significantly better
    than the weaker one, the label is "additive (n.s. vs the weaker single method)".
- **Also reported:**
  - the development KL of every learning rate tried, and the choice;
  - the calibration cost per arm (Task 1's definition): SCALE's run; OURS from Task 1; SCALE→OURS = SCALE + its TM
    run; OURS→SCALE = OURS + its scale run;
  - the E0M3 tile counts;
  - the share of blocks whose learned scale left its initial UE4M3 value.

## Code and runs

- **New flags; the default paths are unchanged:**
  - `run_cost_distill.py`: `--epochs`, `--act-rows`, `--map` / `--unit`;
  - `run_train_map.py`: `--base-scales`;
  - `repro_local/realquant/{native_dev,candidate_store}.py`: `base_given`, a given E2M1 base;
  - `export_map_artifact.py`: `--scales` / `--scales-apply`;
  - `quantize/learned_scale.py`, new.
  The hashes at registration are in `registration.json`.
- **Smoke test before registration**, not a result: every new path for 1 epoch, with its checks, and the three
  exports with the ownership check.
- **The queue** (`runs/queue.sh`, sequential, idle GPU), in order:
  1. the SCALE learning-rate runs;
  2. SCALE `--no-dev`;
  3. SCALE→OURS at 16x64 and 8x64;
  4. OURS→SCALE at 16x64 and 8x64;
  5. the five learned-scale artifacts, each with the ownership check;
  6. one `run_ppl_deploy.py` process, NativeLinear (c), with the policies BF16, FourOverSix, NVFP4, SCALE, and for each
     unit OURS, SCALE→OURS and OURS→SCALE. FourOverSix, NVFP4 and OURS are the Parts 2–3 artifacts.
- **Stopping:** a failed step stops the queue. Each failure is reported and handled as a deviation, never retried
  silently.

## Rules

- **Commit and push:** commit on `tm-opt` as chenjiaj109550158 <chenjiaj.cs13@nycu.edu.tw> after Task 2, and push
  (never main, no force). If a push is blocked, report it and do not retry.
- **Reporting:** report to the user, then continue with Task 3 (readiness), then stop.

## Deviations (append-only)

1. **2026-09-27: `analyze.py` corrected after registration. Reporting code only.** No run, check, rule or comparison
   changed. The registered sha256 was eb6b689ded99…; the final one is 4583aaca14b9….
   - **Before any Task 2 result existed** (22:26–22:35 UTC, during the first learning-rate run):
     - The calibration cost read only `run_train_map.py`'s resource layout; `run_cost_distill.py` stores the
       monitor summary directly, as in Task 1's analysis. It now reads both. Before, it would have raised.
     - OURS's E0M3 tile counts are read from the Parts 2–3 artifacts, where OURS's artifacts are.
     - The table of the development learning-rate runs, listed above under "Also reported", was added.
   - **After the results, reporting only:** the peak GPU and host memory per arm was added to the cost table, as
     Task 1's cost definition reports it.
   - All other code, and the queue, have their registered hashes.
