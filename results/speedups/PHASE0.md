# Calibration speed-ups — Phase 0: profile (before implementing)

**Profile scope.**
- **Setting:** Llama-3.1-8B, `run_multiround.py --profile`, lean memory mode, `--dev-backend native`,
  deterministic, skip-CE, batch 16/8, unit 8x64, the Part B data.
- **Profiled work:** one fake development evaluation, one native development evaluation and one
  scoring pass, under `torch.profiler`.
- **Attribution:** every GPU kernel (1,335,352 in total) is counted once. It goes under the innermost
  named region open on its launching thread when it was launched (`profile_regions.breakdown_trace`,
  via each kernel's launch correlation id). Kernels outside the named regions are classified by
  kernel name.
- **Not active by default:** without `--profile` the regions are no-op contexts.
- **Record:** `phase0_profile_llama8b_8x64.json`; the 2.6 GB Kineto trace stays in the campaign
  directory.

| scoring pass (wall 60.0 s, GPU busy 56.8 s) | GPU s | share |
|---|---:|---:|
| hook: G = dyᵀx (FP32 cuBLAS matmul) | 14.67 | 25.8 % |
| hook: G⊙D, tile reduction, FP64 accumulation | 14.63 | 25.8 % |
| per-token activation quantization (`quantize_rows`, PyTorch) | 12.47 | 22.0 % |
| model GEMMs (BF16, forward + backward, lm_head) | 6.02 | 10.6 % |
| hook: candidate decode + D | 4.26 | 7.5 % |
| other kernels (norms, rotary, elementwise) | 2.39 | 4.2 % |
| teacher host-to-device | 1.11 | 2.0 % |
| lean weight decode | 0.48 | 0.8 % |
| log-softmax + CE/KL | 0.42 | 0.7 % |
| attention | 0.25 | 0.4 % |

| native development evaluation (wall 14.1 s, GPU busy 10.5 s) | GPU s | share |
|---|---:|---:|
| native activation quantization (fused; includes the 64 start-up bitwise checks) | 3.17 | 30.2 % |
| native epilogue (`(d.float() * gs).to(bf16)`, two passes) | 2.20 | 21.0 % |
| teacher host-to-device (pageable, about 25 GB) | 1.57 | 15.0 % |
| other kernels | 1.40 | 13.4 % |
| native FP4 GEMM | 1.28 | 12.2 % |
| log-softmax + CE/KL | 0.43 | 4.1 % |
| lm_head GEMM | 0.27 | 2.6 % |
| native weight build | 0.11 | 1.0 % |
| attention | 0.06 | 0.6 % |

| fake development evaluation (wall 37.4 s, GPU busy 32.5 s) | GPU s | share |
|---|---:|---:|
| per-document activation quantization (`quant_per_document`, PyTorch) | 23.88 | 73.4 % |
| model GEMMs | 3.85 | 11.8 % |
| other kernels | 2.55 | 7.9 % |
| teacher host-to-device | 1.57 | 4.8 % |
| log-softmax + CE/KL | 0.43 | 1.3 % |
| lean weight decode / attention | 0.24 | 0.8 % |

## Against the estimates

- **Item 1, fused activation quantization.** This supports the estimate.
  - Scoring spends 12.5 s (21 % of wall) in `quantize_rows`, so a fused kernel should take about
    −19 % off scoring.
  - The fake development evaluation spends 64 % of its wall time in `quant_per_document`, so fusing
    it should cut that evaluation by more than half. Most of this effect lands on fake-decided runs;
    a native-decided run uses the fake evaluator only for its initial and final evaluations.
- **Item 2, B1.** It targets 33.6 s (59 %) of the scoring pass.
  - The G⊙D and tile-reduction pass alone costs as much as the FP32 matmul, so a kernel that never
    materializes G removes most of it.
  - The open question is the math. B1 must do the matmul's 915 TFLOP (2·T·Σ(N·K)·128 sequences) in
    IEEE FP32 on CUDA cores (no TF32/BF16 tensor cores). cuBLAS SGEMM reaches about 62 TFLOPS here.
  - The −15 to −30 % estimate holds only if the Triton FP32 kernel runs near cuBLAS speed. It is
    benchmarked on real shapes before any end-to-end run.
- **Item 3, single-pass epilogue.** It moves 4 bytes per element instead of 20, so 2.2 s → about
  0.4 s, or about −12 % of a native development evaluation (the estimate was −5 to −10 %).
- **Peak memory.** The removable transients are the FP32 temporaries of the PyTorch quantizers
  (about 2 GiB at the largest input) and the hook's G, D and G⊙D (about 0.7 GiB). That fits the
  −2 to −4 GiB estimate.
- **Not in scope, recorded for later:**
  - About 3.6 s of each native development evaluation is CPU-side (wall minus GPU busy).
  - The pageable teacher copies take 1.1–1.6 s per pass.
