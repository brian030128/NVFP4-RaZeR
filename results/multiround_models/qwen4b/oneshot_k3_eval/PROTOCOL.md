# Qwen3-4B: N16K64 one-shot k=3 maps under the Part C evaluation protocol — addendum protocol

Written 2026-09-25, before any conversion check or evaluation of this addendum. The hash and
registration time are in `registration.json`. Deviations are appended in the last section.

## Question

The N16K64 campaign's one-shot maps for Qwen3-4B were elected with CE and KL at k = 3 and
evaluated under per-token activation scales (convention (c)). How do they compare under Part C's
evaluation protocol (convention (a): one tensor-wide FourOverSix scale per window), paired against
FourOverSix and against the multi-round DET-NATIVE and DET-FAKE maps? No map is recalibrated.

## Maps

From `/home/dev/n16k64_campaign/runs/calib_qwen4b_attempt1/maps`:
- **Primary:** `qwen4b_seed0_n8_k3.mixfp4map` (8x64, rule `max(mean_CE+3SE, mean_KL+3SE)<0`).
- **Also:** `qwen4b_seed0_n16_k3.mixfp4map` (16x64). Each 16x64 tile is two stacked 8x64 tiles,
  so it runs on the b8x64 kernel.

## Checks before any evaluation (`convert_check.py`, CPU)

1. **Provenance.** Each file is read with `campaign.mapio.read_map`, which checks:
   - the magic, the canonical header, the per-module selected counts and totals, and zero
     padding bits.
   - The file's sha256 must equal its `.provenance.json` `map_sha256`, and the mask payload hash
     its `mask_payload_sha256`.
   - The header must name `Qwen/Qwen3-4B` at the pinned revision `1cfa9a72…` (model and
     tokenizer), type block [8, 64] or [16, 64], and the 252 module names and weight shapes of
     this model's calibration record, in order.
2. **Conversion.** Each map is converted to run_multiround's map format: one bool grid per module
   at the 8x64 unit.
   - **n8:** the grid unchanged.
   - **n16:** each row of tiles repeated twice.
   - **Required:** for every module, the expanded element masks are equal before and after, and
     the 8x64 tile count equals the header's selected tiles (n8) or twice them (n16), per module
     and in total.
3. **Candidates.** The campaign's `quant.four_over_six` / `quant.e0m3` must be the functions
   `quant_nvfp4_4over6(w, 4, 16)` / `quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1',
   elect='always')` that run_multiround uses. For every one of the 252 modules and both maps, two
   comparisons are made:
   - **the weights:** the weight campaign's `Installer` rule installs, `tiles.apply_mask(four_over_six(w),
     e0m3(w), mask, type_block)` on the original map, against the weight run_multiround installs
     for the converted map, `torch.where(expand(sel), decode_alt, decode_base)`. The lean
     decoder reproduces that weight bitwise (Part A/B).
   - **the candidate functions:** the campaign's candidates against ours, called directly.

   They are compared bitwise (int16), and by value where the bit patterns differ.

   *Known difference.* `decode_alt` stores E0M3 codes in offset binary and has no negative zero,
   while `quant_mix_4_6` returns −0 for a negative value rounded to zero. A bitwise difference
   confined to the sign of zeros in E0M3 tiles is therefore expected. It is reported as counted;
   the values are equal.

   **Stop rule.** Any other difference, or any failure of checks 1 or 2, stops the addendum and is
   reported.

## Evaluation (convention (a) only; no convention (c) runs)

**Processes.** `run_multiround.py --model qwen4b --memory-mode lean --unit 8x64 --evaluate-map ...`
with the Part C data, WikiText-2 / C4 windows and per-window tensor-wide FourOverSix activations,
one window per forward. There is one process per backend: native (primary) and fake (secondary).

**Maps in each process:**
- FourOverSix;
- N16K64-n8-k3;
- N16K64-n16-k3;
- DET-NATIVE-8x64 and DET-NATIVE-256x64 (Part C);
- DET-FAKE-8x64 and DET-FAKE-256x64 (the native-decision addendum).

The 256x64 maps are expanded exactly to 8x64 inside the process.

**Validity.** FourOverSix's per-window NLL must equal Part C's evaluation bitwise, on each
backend; if not, that is reported and the comparison with Part C's numbers is dropped.

**Reported.**
- PPL.
- Paired per-window ΔNLL, mean ± 2 SE (SD with ddof = 1, over √n windows; 146 / 256 windows),
  of each one-shot map against FourOverSix and against each of the four DET maps.
- For reference only, clearly labelled as a different protocol: the campaign's own recorded PPL of
  the same maps under per-token activations (convention (c)).
  - n8_k3: local fake 11.7066 / 15.7092, native 11.6925 / 15.6958.
  - n16_k3: fake 12.1095 / 15.9714, native 12.0888 / 15.9623 (and 12.0933 / 15.9515 after the
    weights-on-A contiguity fix).

## Rules

- **Report:** `REPORT.md` here.
- **Commit:** on `repro/n16k64-rtx-pro-6000` with the usual identity, and nothing is pushed. The
  zero-shot WIP stays out.
- **No tuning:** nothing is tuned on WikiText-2 or C4.
- **Afterwards:** Part C resumes with Phi-4, then Qwen3.8-27B, unchanged.

## Deviations (append-only)

1. **2026-09-25 ~10:10 UTC: check 3 moves from CPU to the GPU.** Checks 1–2 passed on CPU for
   both maps: file and payload hashes, model and tokenizer revision, module names and shapes, type
   block, element masks after conversion, and 8,149 / 8,798 tiles.
   - **What failed:** check 3 stopped on CPU at `model.layers.4.mlp.up_proj`. There
     `quantize/packed_candidates.encode`'s E0M3 decode and `quant_mix_4_6` disagree on 113,743
     elements, whole 16-element blocks with a different FP8 scale; the E2M1 base agrees. On CPU,
     division by a Python scalar is true division, whereas CUDA multiplies by the FP32 reciprocal.
     The two functions spell the scale differently, so they diverge on CPU only.
   - **Why the GPU is the right device:** neither evaluator runs on CPU. On the GPU, every module
     of every run packs with the decode verified (all Part C reports: no dense-fallback modules),
     and the N16K64 installer also ran on the GPU.
   - **When:** check 3 therefore runs on the GPU, after DET-FAKE-8x64 and its evaluations, so its
     timing is undisturbed.
   - **Unchanged:** the evaluation's order, maps and criteria.
