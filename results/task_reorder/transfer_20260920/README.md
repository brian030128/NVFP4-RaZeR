# Accepted-arrangement follow-up, 2026-09-20

The requested report update and tests are complete. See root `MIXFP4_REPORT.md`
for the algorithm and full tables. Qwen's accepted PPL remains unchanged.

- `llama_calibration/summary.json`: historical calibration-map identity audit;
  regenerated raw/fine counts187/3345. Fine atom scores remain on `/work`.
- `llama_search/`: three completed, fixed-setting last-MLP searches.
- `llama_compaction/`: frozen-layout provenance. Actual quantized-weight
  equality was verified in the confirmation job.
- `llama_confirmation/`: one frozen178-tile candidate,64 new documents,
  **failed** CE-primary gate versus raw256 and matched identity. No candidate
  WikiText/C4 PPL run was submitted. Archived raw/fine PPL controls are reused.
- `gb200_405441/`: full packed-code/scale input gather and full BF16 output
  restoration around the existing SM100 mixed GEMM's E2M1 path.
- `gb200_405458/`: same input gather, sparse two-pass in-place output restoration.
- `gb200_check_405476/`: distinct-output-bit correctness checks and an unchanged
  output negative control,20 combinations, all passed in8seconds.
- `permutations/`: exact compacted Qwen permutations used by the native tests.
- `kernel_snapshot/`: the sibling kernel's actual modified source, with its
  parent commit and CUTLASS revision. The full-gather job also retains the exact
  original extension header, matching its recorded hash. The sparse extension
  and correctness check are in root `native/`.
- `research_source/`: source snapshots matching the frozen confirmation and
  search provenance. `llama_transfer_initial.py` records the initial search job;
  it was stopped after completing all searches to expand document exclusions.
  No fresh data were read at that boundary. A later missing-metadata preflight
  failed before fresh data/model loading; completed layouts were reused.
- `plan.json`: frozen settings, outcomes, and complete job/usage ledger.

Native timings exclude quantization and setup and are per projection. They do
not establish fused cost, full-model latency, or native execution of the entire
212-tile mixed-format model. The initial synthetic GEMM output could contain
identical columns; the separate distinct-bit check removes that weakness from
the permutation correctness evidence without repeating timings.

All jobs used `gov113008`, at most two concurrent GPUs (limit four), and attached
completion monitoring. No jobs remain. Allocated GPU time, including CPU search
on H200 workers and failed preflights, was1683seconds =0.4675GPU-hours.

`INDEX.json` records SHA256 and sizes for the evidence bundle. Large scores,
model weights, and cached teacher tensors are not included. Frozen plans retain
original absolute artifact paths; metadata snapshots are for audit, not a claim
that those compute artifacts are present in this checkout.

Permutation text is formatted16indices per line for review; its parsed integer
sequences are unchanged from the measurement inputs. Source snapshots preserve
original bytes, including preexisting trailing whitespace, to retain their hashes.
