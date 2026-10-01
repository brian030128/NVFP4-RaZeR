# FlipQuant maps for `~/flipquant`: delivery report

Protocol: `PROTOCOL.md` here (a1873ab, written before any GPU run). The run took place on 2026-10-01 from 15:05 to
21:08 UTC, on the RTX PRO 6000.

**18 maps are delivered**, in 6 models × 8x64 / 16x64 / 256x64:
- written as `~/flipquant/maps/<registry key>/flipquant_<unit>.pt`, in `flipquant-map/1`;
- each verified with `~/flipquant`'s `maps.load` and its `quantizable_linears`;
- rows added to `~/flipquant/maps/README.md` locally;
- nothing committed or pushed to `~/flipquant`; their existing `mixfp4_*` files are untouched.

## The maps

| model (`~/flipquant` key) | revision | modules | E0M3 tiles 8x64 / 16x64 / 256x64 | source | calibration (wall) |
|---|---|---:|---|---|---|
| phi4-14b | 2db69c1c | 160 | 1.57 / 2.16 / 6.35 % | paper map, copied | paper |
| qwen3.8-27b | 1d4bf0f2 | 496 | 1.29 / 1.58 / 4.11 % | paper map, copied | paper |
| qwen3-1.7b | 70d244cc | 196 | 4.84 / 6.18 / 15.88 % | trained | 5.7 / 4.9 / 4.7 min |
| qwen3-8b | b968826d | 252 | 2.14 / 2.81 / 7.72 % | trained | 9.4 / 9.2 / 9.9 min |
| mistral-7b (Instruct v0.3) | c170c708 | 224 | 2.67 / 3.57 / 11.20 % | trained | 7.7 / 7.7 / 7.5 min |
| nemotron-nano-9b-v2 | 6533e8de | 120 | 2.69 / 3.59 / 10.96 % | trained | 82.8 / 82.8 / 83.2 min |

**Training KL**, from the FourOverSix start to the last epoch, 8x64 / 16x64 / 256x64:

| model | start | end |
|---|---:|---|
| qwen3-1.7b | 0.169 | 0.064 / 0.068 / 0.096 |
| qwen3-8b | 0.0935 | 0.0346 / 0.0360 / 0.0460 |
| mistral-7b | 0.0573 | 0.0189 / 0.0199 / 0.0253 |
| nemotron | 0.0364 | 0.0166 / 0.0173 / 0.0209 |

The SHA-256 of every file is in `delivery.json`. The run records are in `runs/`, `data/`, `smoke/` and
`commands.log`.

## What was checked

- **Gate R (reproduction):** the paper's Mistral-7B-v0.3 8x64 map, re-trained with the paper's command in this
  environment, is bitwise equal to the committed map (sha256 8aacdd77…, 339,444 E0M3 tiles).
- **Fit sets:**
  - The builder rule (`campaign.data.builder_seed0`) reproduced the Llama, Qwen3-4B and Mistral records exactly
    before drawing each new model's 128 windows.
  - `math_code_data` then rebuilt every window and checked its token hashes.
  - Qwen3-1.7B and Qwen3-8B share their windows (same tokenizer). Mistral-Instruct's windows equal the base Mistral
    record's (same tokenizer), while all 224 of its weight hashes differ from the base model's.
- **The reuse rule:** phi4-14b's and qwen3.8-27b's registry revisions equal the paper's (phi-4 `main` = 2db69c1c on
  2026-10-01). The paper maps' module names and tile shapes equal their `quantizable_linears` on a meta-device model,
  160 and 496 modules.
- **Every trained map:**
  - its run is complete, with 20 epochs and the paper's flags;
  - its `map.pt` matches the run's report;
  - the run's recorded source hashes equal the files at its launch commit;
  - the delivered file re-loads to the same tiles.

## Deviations and incidents (disclosed)

1. **Gate R, first attempt.** It stopped at data loading because it ran with `HF_HUB_OFFLINE=1`; the paper runner
   uses 0, since the fit data is streamed. Re-run with 0, it passed. The failed attempt is kept.
2. **Downloads.** `huggingface_hub`'s Xet client stalled twice at 0 MB/s. The downloads were restarted over plain
   HTTP (`HF_HUB_DISABLE_XET=1`), at about 16 MB/s. Only the schedule was affected.
3. **Data preparation exits.** `prepare_model_data.py` crashed at interpreter exit (`PyGILState_Release` while the
   dataset stream was finalized), after writing its outputs. Each record was checked complete before use.
4. **Nemotron's batch.** No mamba_ssm or causal_conv1d is installed here, so transformers runs Mamba-2 on its
   plain-torch path. Its chunk scan materializes a (batch, chunk, l, s, heads, state) fp32 product.
   - Two 1-epoch smokes ran out of GPU memory: batch 8 (32 GiB transient) and micro-batch 4 (16 GiB).
   - Micro-batch 2 × accumulation 4 fits, at 53 GiB peak. That is Qwen3.8-27B's split; the optimizer batch stays 8,
     the paper's only allowed per-model difference.
   - The smokes are disclosed and not used. The calibrations' first epoch reproduced the passing smoke's training
     KL bit for bit.
5. **Revisions.** The registry resolves `main` for phi4-14b, qwen3-1.7b, qwen3-8b and mistral-7b. These maps are pinned
   to the commits resolved on 2026-10-01 (the table above).

## Observation (not a check)

The collaborators' own `mixfp4_8x64.pt` maps for Nemotron and Qwen3.8-27B were trained with their `train_map.py`
port, with a different fit set and a development monitor. Against ours:
- the E0M3 counts are nearly equal: 405,165 vs 404,963 tiles, and 616,331 vs 614,657;
- the selections differ: tile agreement 95.4 % and 97.6 %, E0M3 overlap (Jaccard) 0.079 and 0.031.

Different fit sets and trainers reach the same flip budget with different tiles. Neither set of maps is evaluated
here.

**Not done, by request:** no PPL, no lm-eval, no kernel work.
