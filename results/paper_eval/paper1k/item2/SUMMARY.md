**Partial report: the latency panel is complete, the memory paragraph is PENDING.**

- **Latency (Table 4, SM120 rows): done.**
  - 6 models × 7 policies × 5 rounds = 210 CUDA-graph prefill jobs in one session (2026-10-24, 15:30–18:52 UTC). Every
    record passes the registered checks: complete, graph logits equal to eager, the policy's kernel family, build_V's
    libraries.
  - FlipQuant vs FourOverSix, paired within rounds, median over the 8 shapes:
    - 16x64: +0.72 … +1.19 % per model;
    - 8x64: +3.59 … +4.84 %;
    - 256x64: +0.49 … +1.71 %.
    - BF16 / FlipQuant 16x64 at 1x2048: 1.86–2.23×.
  - The combined 1x2048 table in the `tab:latency` layout, with its LaTeX rows, is in `LATENCY.md`.
- **Step 0, the drift anchor** (Mistral, the 256-window release map, against the published session):
  - the paired FlipQuant 16x64 / FourOverSix ratio matched (+0.75 vs +0.82 %);
  - the absolute times were 0.3–3.7 % faster. The GPU ran cooler: job start temperatures were about 5–6 °C lower per model.
  - This was accepted as a recorded deviation, with an unrecorded 10-minute warm-up before the panel.
- **Against the published 256-window rows:**
  - The map-free rows (BF16, NVFP4, FourOverSix) are 1.1–3.6 % faster in this session (median −2.1 %).
  - The FlipQuant vs FourOverSix ratios at 1k are −0.2 … +0.3 points from the 256-window ones on five models.
  - Llama-3.2-1B-Instruct is the exception: +0.8 points (16x64) and +1.0 point (8x64). Its 1k maps flip the most tiles
    (E0M3 share 14.8 % at 16x64, 11.3 % at 8x64).
- **Memory (the paper's memory paragraph): PENDING.**
  - The anchor passed: installed and whole-model bytes equal the published record's, and after-load and peak
    allocations are 0.00 GiB apart.
  - Llama-3.2-1B-Instruct BF16 is done.
  - The probe stopped at the next job when the GPU stopped responding (2026-10-24 18:53 UTC). The remaining 41 jobs
    resume at Llama-3.2-1B-Instruct NVFP4 after the GPU is restored. The latency records are unaffected: the panel had
    finished 27 s earlier.
