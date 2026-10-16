# Latency-rounds port, validation (b): GPU smoke run (2026-10-16)

The flipquant branch latency-rounds (80cd3f0) ports the multi-round prefill latency driver
(scripts/run_prefill_latency.py) and the table script (scripts/latency_table.py). Validation (a), on the CPU, reproduced
part A's and part W's tables exactly (PROTOCOL.md part Z). This is the GPU smoke test (part Z).

**The run:** run_prefill_latency.py on llama3.2-1b-instruct, fo6 and fq-16x64, 2 rounds, 1x128 and 1x2048, reps 3.
- build_V via `--kernel-build-dir`; part P's 16x64 map via `--model-maps`.
- 13:17:45–13:18:26 UTC, four processes, every one rc 0.
- Every registered check passed, including every shape captured with graph logits equal to eager, native coverage,
  the kernel family and the build's libraries.
- Records RUN/latency_rounds_smoke/ (round*.json, queue.log, driver.jsonl). latency_table.py's output is LATENCY.md
  and latency.json here.

**Against part W** (RUN/latency_llama: 5 rounds, reps 7, flipquant paper-sm120-runs 6f5ad27): median graph ms over the
rounds.

| policy | shape | smoke | part W | Δ |
|---|---|---:|---:|---:|
| fo6 | 1x128 | 2.927 | 2.933 | −0.20 % |
| fo6 | 1x2048 | 11.842 | 12.264 | −3.44 % |
| fq-16x64 | 1x128 | 2.927 | 2.938 | −0.39 % |
| fq-16x64 | 1x2048 | 11.909 | 12.293 | −3.12 % |

FlipQuant 16x64 vs FourOverSix, paired within rounds:
- smoke: −0.01 % (1x128) and +0.57 % (1x2048);
- part W: +0.18 % and +0.24 %.

**Verdict:** pass. Every check passed, and every time is within 3.5 % of part W's. The smoke run is 3 % faster at
1x2048, with 3 reps over 2 rounds against part W's 7 reps over 5. That difference is not analysed further; a smoke run is
not a result.
