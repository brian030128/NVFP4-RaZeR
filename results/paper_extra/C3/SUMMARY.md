# C3 summary: GEMM latency against the E0M3 tile share

**Run:** registered protocol (`PROTOCOL.md`, 82d9af3), 2026-09-29 17:33–17:38 UTC, idle GPU. **No deviations.**
- **Shapes:** Llama's 4096x4096, 14336x4096, 4096x14336, at T ∈ {128, 512, 2048, 8192}.
- **Timing:** CUPTI GEMM time, 3 rotated rounds.
- **Records:** tables in `C3.md`, data in `C3.csv` and `C3_summary.json`; raw per-round times in `C3_raw.json`.

**The cost with no E0M3 tile** (mixed kernel vs stock):
- **At the same 128 width:** +0.8 to +3.8 %.
  - **Dispatch** (mixed@128 vs nodisp@128) is +2.0 to +2.6 % at T ≥ 2048, and −0.1 to +2.2 % at T ≤ 512.
  - **The rest is the arrangement** (nodisp@128 vs stock@128): +0.8 to +1.8 %.
- **8x64 (n8k64_wB vs stock_wB):** +6.5 to +11.5 %.
- **At T = 128 the tile table picks different widths for the two families,** so the deployment comparison
  (mixed@table vs stock@table) reaches +9.8 % and +12.5 % on the 4096-output shapes.

**The E0M3-share cost** (kernel(f) / kernel(0) − 1):
- **Random tags** (a Bernoulli per tile) cost little at small shares.
  - At f ≤ 5 %: −0.2 to +1.1 % at T ≥ 2048 for the mixed kernel (up to +3.5 % at T = 128), and −0.6 to +1.2 % for
    n8k64_wB.
  - At 100 %: +3.4 to +4.8 % (mixed, T ≥ 2048); up to +8.6 % at T = 512 and +8.9 % at T = 128 (14336x4096).
  - For n8k64_wB, 50 % mixing costs more (+3.3 to +5.5 %) than all-E0M3 (+1.0 to +4.6 %).
- **Contiguous tags** (the first f % of tiles in row-major order) behave differently at small T. At T ≤ 512 the cost
  jumps almost at once: +3.7 to +9.1 % at f = 1–2 %, then stays flat.
  - The reading: at small T the GEMM is one partial wave of CTAs, and its time is set by the slowest CTA. Packed E0M3
    tiles give one CTA a full E0M3 load; random tags spread them.
  - At T ≥ 2048 (many waves) contiguous tags cost ≤ +1.7 % at f ≤ 5 %.
- **The real FlipQuant (ours) maps** have 1.3–3.3 % E0M3 tiles (16x64: 1.6–3.3 %; 8x64: 1.3–2.5 %). At those shares
  the random-tag cost is within about ±1 % at T ≥ 2048.
  - This agrees with item #3 (`results/tm_opt/REPORT_ITEMS.md` §#3): E0M3 density was worth at most 0.7 % of the
    prefill.
  - What the curves add: a map whose E0M3 tiles cluster could cost up to about 9 % of a single-wave GEMM at small T.
    The real maps' spatial distribution is neither pattern, and was not measured here.
