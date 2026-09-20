# Native full-model measurement implementation

The benchmark replaces all 224 Llama transformer projections with the sibling
SM100 kernel. Embeddings and the vocabulary head stay BF16. Weight codebooks and
format maps are the frozen quality-study artifacts; no layout search runs here.
The baseline uses FourOverSix E2M1, the raw control uses its 187-tile map, and the
arranged candidate uses 147 tiles with only the final MLP rearranged.

`native/model_runtime.cu` exposes a C ABI loaded through ctypes. Each projection
packs weights offline, swizzles its FP8 scales, and caches a GEMM plan per token
count. Runtime includes tensor amax, FourOverSix activation quantization, native
packed GEMM, and output restoration. Column permutations move intact groups of
16 during packed activation stores. Row restoration is a separate kernel in this
first full-model implementation. Plans have shared scratch buffers and are used
on one stream; this is a measurement backend, not a concurrent serving engine.

`scripts/build_native_model_runtime.py` creates a build-local mainloop override.
It selects the B format for each N256/K64 chunk, while preserving the sibling
kernel's tile256x256x256, cluster2x1x1, and full-K scheduler. It does not edit the
sibling checkout. The per-chunk counter assumes that full-K scheduler; Stream-K
is not supported. The current host wrapper requires N and K multiples of256.

## Correctness and the activation quantizer diagnosis

Small random matrix tests alone were insufficient. The first full-model attempt
passed all packed-weight audits but failed its logit check. A two-layer prefix
capture then isolated a real activation input in only12GPU-seconds. The native
GEMM agreed with decoded native operands, but activation encoding differed from
the Python quantizer at133code bytes,37scales, and478decoded values.

PyTorch's scalar division implementation uses multiplication by a rounded
reciprocal. CUDA division produced a one-bit different global scale in the saved
case:0x3b018618 versus0x3b018619. This crossed FP8 scale thresholds. The corrected
producer matches reciprocal multiplication for tensor-global scaling and the
scale-six candidate, and matches the adjacent reduction tree. One elected scale
choice is broadcast across all16values. The [PyTorch implementation](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)
documents the scalar-division operation.

Job406633 passes eight GEMM checks, over three million bitwise BF16 activation
checks, and both saved real-input checks with zero code/scale/dequant differences.
Earlier failures remain failures. They ran no latency measurements.

The full-model engineering reference decodes FP4/FP8 operands to FP32 and rounds
projection outputs to BF16. The older quality simulator instead rounds decoded
operands to BF16 before multiplication. Both comparisons are recorded; a native
latency measurement does not establish native PPL parity with the earlier quality
simulator. Quality candidates and fresh-data gates are unchanged.

See `plan.json` for thresholds, timing scope, repetitions, and the prospective
engineering-reference correction and separate prospective diagnostic latency arm.
The original full-output gates failed and remain failed. The diagnostic arm
requires all operator comparisons, exact activation encoding, and repeatable
native forwards; it does not establish native model quality. No full-model
timing is inferred from a sum of projection measurements.

## Full-model reference and timing limitations

Normalized FP4×FP8 operands are multiplied first; the rounded FP32 product of
the two tensor-global factors is applied after GEMM, before BF16 output. This
matches the native epilogue order. Baseline full-model logits match this reference
bitwise. Mixed-policy full logits do not: single-layer replay 406741 finds only
0–4 BF16 differences per projection against FP32/FP64 oracles, but later FP4
quantization amplifies them. The arranged full-logit relative gap is about10.8%.
The separate BF16 fake-quantized quality simulator also differs. Native PPL
remains unmeasured; the published fresh quality gates are unchanged.

Job406751 passed all672 operator and encoding checks and measured full requests,
but its one-step warmup left cold decode positions. Job406828 reused completed
numerical audits after library/map/source/AST checks and repeated weight audits.
It warmed two complete requests per policy/context and measured all six policy
orders with GC disabled. At128tokens, arranged request overhead was+0.531%±0.526%
(2SE). At2048tokens, all policies showed large timing variation and overhead is
inconclusive. All samples, including cold and unstable ones, are retained.
No Qwen full-model native measurement exists. Current integration is an eager
Transformers/ctypes research backend, not a serving-engine latency estimate.

The follow-up host-side reciprocal/clamp consistency edits in the standalone
producer audit and historical pipeline generator were syntax-reviewed only;
they do not retroactively change or revalidate the old microbenchmark results.
