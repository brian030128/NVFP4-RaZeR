# Cost comparison (calibration vs. QAT/distillation): status

**Paused by user decision on 2026-09-24, at 08:41 UTC.** The QAT-vs-ours comparison is paused so
that the native-kernel shadow verification of the development evaluation can run
(`results/native_dev_shadow/`). No cost-comparison run starts until the study is resumed; that
includes C, D, QAT, the proposed composition experiment, and the remaining B runs.

**REPORT.md is not final.** It is a draft: the conclusion and the final table wait for
B-8x64-opt.

## Complete arms

All arms ran on 1× RTX PRO 6000 Blackwell, Llama-3.1-8B, W4A4 FourOverSix activations, under
[PROTOCOL.md](PROTOCOL.md).
- ΔNLL is the paired per-window difference versus the local arm A, mean ± 2 SE (141 WikiText /
  256 C4 windows).
- GPU-hours count setup plus selection or training plus the final development evaluation; the
  WikiText/C4 evaluation is excluded. For C and D, the chosen run is given first, then the whole
  3-point LR grid.
- Peaks are the run-wide `max_memory_allocated` / `max_memory_reserved`, and `ru_maxrss` for host
  memory.

| arm | E0M3 tiles / LR | WikiText | ΔWiki | C4 | ΔC4 | GPU-h | peak GPU (GiB) | peak host (GiB) |
|---|---|---:|---|---:|---|---|---|---:|
| A: FourOverSix RTN | — | 6.8872 | — | 9.8294 | — | 0 | 42.0 / 49.9 (evaluation) | 43.6 |
| B-256-opt (ours; eval batch 16, score batch 8) | 7,625 | 6.8397 | −0.00692 ± 0.00189 | 9.7828 | −0.00475 ± 0.00204 | 0.540 | 62.8 / 68.9 | 43.6 |
| B'-256-opt (B-256-opt without the CE backward) | 4,694 | 6.8313 | −0.00815 ± 0.00182 | 9.7763 | −0.00542 ± 0.00183 | 1.040 | 60.8 / 66.9 | 43.6 |
| B-256-ref (ours; eval batch 1, score batch 1) | 7,724 | 6.8446 | −0.00620 ± 0.00187 | 9.7757 | −0.00548 ± 0.00177 | 0.935 | 27.9 / 31.2 | 41.8 |
| C1: full-weight QAT, 128 fit sequences, 1 epoch | lr 1e-6 | 6.8486 | −0.00562 ± 0.00178 | 9.7622 | −0.00686 ± 0.00212 | 0.060 / 0.221 | 85.3 / 87.9 | 71.5 |
| C2: full-weight QAT, time-matched to B-256-opt | lr 1e-6 | 6.9977 | +0.01593 ± 0.00233 | 9.8828 | +0.00542 ± 0.00297 | 0.553 / 1.702 | 85.3 / 87.9 | 71.5 |
| C3-10×: full-weight QAT, 1,280 new sequences | lr 1e-6 | 6.7813 | −0.01549 ± 0.00215 | 9.6632 | −0.01705 ± 0.00431 | 0.163 / 0.529 | 85.3 / 87.9 | 101.5 |
| C3-100×: full-weight QAT, 12,800 new sequences | lr 1e-6 | 6.7425 | −0.02123 ± 0.00238 | 9.6093 | −0.02265 ± 0.00437 | 1.194 / 3.619 | 85.3 / 87.9 | 102.0 |
| D1: scale-only, 128 fit sequences, 1 epoch | lr 1e-3 | 6.8351 | −0.00759 ± 0.00186 | 9.7587 | −0.00722 ± 0.00203 | 0.050 / 0.185 | 74.0 / 74.4 | 58.4 |
| D2: scale-only, time-matched to B-256-opt | lr 1e-4 | 6.7909 | −0.01407 ± 0.00188 | 9.6750 | −0.01584 ± 0.00372 | 0.550 / 1.682 | 74.0 / 74.4 | 58.3 |

**Arm A vs. published.** Local A is 6.887180 / 9.829412; the published H200 run is 6.875525 /
9.823733. Per window the difference is +0.0017 ± 0.0015 on WikiText and +0.0006 ± 0.0014 on C4.

**B-256-ref vs. published run 1.**
- Tiles: 7,724 locally vs. 8,405 on H200. The maps are not identical.
- Per window the two agree within noise: +0.0004 ± 0.0016 (WikiText) and +0.0002 ± 0.0020 (C4).
- Jaccard overlap of the local maps: B-256-opt vs. B' 0.44, B-256-opt vs. B-256-ref 0.14.

**Memory probes (arm C):**
- FP32 AdamW states without activation checkpointing: OOM at every micro-batch (93.4–93.9 GiB
  of 94.97 GiB).
- With checkpointing: fits at 81–86 GiB.
- Fallbacks, probed but not trained: 8-bit AdamW 46–77 GiB; CPU offload 33–64 GiB on the GPU plus
  102 GiB on the host; LoRA r16 26–65 GiB.
- A live BF16 teacher does not fit next to C. Next to D it fits, at 88.9 GiB.

**Diagnostics:**
- Evaluation equivalence of `run_cost_distill.py` with `run_multiround.py`: bitwise.
- Scoring determinism: the FlashAttention-2 backward is non-deterministic; with deterministic
  algorithms, CE-then-KL and KL-only give bitwise-identical KL gradients.
- Batch sensitivity: measured. See PROTOCOL.md §10, deviations 4–6.

## Stopped and incomplete

- **B-8x64-opt** (`--unit 8x64 --eval-batch 16 --score-batch 8`) was stopped at 08:41 UTC by user
  decision. It had finished 2 rounds, accepting 2,967 and then 584 flips. A flip can also return a
  tile to E2M1, so the map held 3,025 E0M3 tiles. Development KL went 0.10645 → 0.10470.
  - **It is incomplete, and not a result.**
  - The partial output (`report.json` with status "running", `map.pt`) is kept at
    `/home/dev/n16k64_campaign/cost_comparison/runs/B_8x64_opt/`. It is not copied into this
    directory.

## Not started

- The composition experiment (our election on QAT- or scale-trained weights) was only proposed.
  It is not registered and was not run.

## Where the records are

- `runs/`: every completed run's `report.json`, the probe reports, the diagnostics,
  `commands.log`, and the queue scripts with the exact commands. JSON only.
- `data/`: data-preparation summaries, the reconstructed calibration report, and the C3 pool
  manifest (document and token hashes).
- Raw outputs (maps, trained states, logs, the regenerated `fresh.pt` and `pool.pt`) are in
  `/home/dev/n16k64_campaign/cost_comparison/`.
