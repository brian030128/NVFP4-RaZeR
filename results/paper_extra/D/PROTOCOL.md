# Experiment D: decode latency (paper §5.7) — protocol

Written 2026-09-29 on branch `tm-opt`, before the D full run; the hashes and time are in `registration.json`. A smoke test
preceded it (Llama-3.1-8B, all 7 policies, 1 round, settings 1x512 and 4x2048, into the smoke directory; every record
passed the token check). Deviations are appended at the end.

The user decided the scope (relayed by nvfp4-razer-c9): all 6 settings, 3 models, 7 policies, 5 rounds with shuffled
order. It runs after C1, C3, A (Llama) and C2, with nothing else on the GPU.

## Design

- **Models:** Llama-3.1-8B, Mistral-7B-v0.3, Phi-4. Qwen3.8-27B is out: the SM120 decode harness does not support its
  hybrid cache.
- **Policies** (the paper run's artifacts, `/home/dev/n16k64_campaign/paper/artifacts`):
  - BF16;
  - NVFP4 and FourOverSix on the stock weights-on-A set (`auto_stock`, RTX PRO 6000 tile table);
  - Ours 16x64 (`auto`) and Ours 8x64 (`n8k64_wB`);
  - NVFP4 and FourOverSix on `stock_wB`, the 8x64 same-placement references.
- **Settings:** batch {1, 4, 16} × prompt {512, 2048}, then 64 generated tokens.
- **Harness** (`experiments/paper_extra/bench_decode.py`): `sm120/bench/model.py decode_graph`, Part R's.
  - The prompt fills a StaticCache.
  - The single-token forward is captured in a CUDA graph.
  - After 32 replays for the token check, the cache is rewound, and 64 replays are timed with the host clock (one
    synchronize).
  - Recorded: ms per token and tokens per second (batch × 64 / seconds).
- **Protocol** (`experiments/paper_extra/D_decode.py`):
  - one process per (model, policy, round);
  - 5 rounds, the policy order shuffled per round (seed 20260929 + round, logged);
  - the default caching allocator;
  - the GPU must be idle.
- **Widths:** the CTA width decode uses at T = batch, from the tile table for the width-selecting sets. stock_wB and
  n8k64_wB have one build.

## Checks

- **Registered (stops the run):** in every setting, the graph's first 33 greedy tokens equal an eager StaticCache
  decode's (`decode_eager_tokens`, Part R's check); no setting may fail; coverage — every scoped Linear ran natively.
- **Recorded, not a check:** the agreement with an eager DynamicCache decode (HF's default cache). Numerical paths
  differ, so it need not be exact.

## Analysis (`experiments/paper_extra/D_analyze.py`)

- **Tokens per second:** the median over rounds.
- **The change against FourOverSix in %:** paired within rounds. For Ours 8x64, also against FourOverSix on wB.
- **The decode widths,** per batch.
- **The registered token check is re-verified from the records;** the analysis stops if any record fails it. The
  DynamicCache agreement is listed as recorded.
- **CSV:** model, policy, batch, prompt, tokens_per_s, ms_per_token, change_vs_fo6_pct, change_vs_fo6_wB_pct.

## Deviations (append-only)

(none yet)
