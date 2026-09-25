# SM120 native path: frozen numeric specification

This fixes what a set of codes, scales and a format map *mean*, independently of the execution
path. `mixfp4_sm120/numerics.py` is this document as code; `tests/test_numerics.py` checks it bit
for bit against the fake quantizers the selector and the fake-quant evaluation use
(`quantize/quantizer.py`, `quantize/causal_four_over_six.py`).

## 1. Element formats and nibble encoding

A 4-bit element (nibble) is `sign << 3 | magnitude_index`. Its value depends on the format of the
16-element scale block it belongs to:

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| E2M1 magnitude | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |
| E0M3 magnitude | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |

- The sign bit negates. Nibbles 0 and 8 both decode to 0 in either format. The encoder always
  writes +0 (nibble 0), including for values that round to −0.
- Two nibbles per byte, **element 2j in the low nibble** (CUTLASS sub-byte order), K-contiguous.
- The E0M3 table is not in the PTX ISA. It was measured on SM120 hardware
  (`kernel/tests/mma_intrinsics`, and the upstream `revollllt/sm120-e0m3-mma` probe at commit
  `8b755d9`), and is re-verified on every GPU by `tests/test_gemm.py::test_decode_probe`.

## 2. Scales

- **Block scale**: one per 16 consecutive K elements of one row, stored as **UE4M3** (an FP8 E4M3
  value that is always positive). Legal values: `[2^-9, 448]`. Every quantizer clamps to this
  range before rounding to E4M3 (round-to-nearest-even), so 0, NaN and negative scales never occur
  in real blocks. An all-zero block gets scale `2^-9` and all-zero codes.
- **Format tag**: bit 7 of the scale byte (the E4M3 sign bit, always 0 for a positive scale) is
  set iff the block is E0M3. The tensor core ignores bit 7 when decoding the scale; the kernel
  reads it to choose the instruction. Stored value: `ue4m3(scale) | (is_e0m3 << 7)`.
- **Layout padding**: the kernel reads scale bytes in 128-row × 64-K atoms. Padding rows and
  blocks are written as byte 0 (scale 0, tag E2M1) and only ever multiply zero-filled operand data.
  (Byte 0x7F would be an E4M3 NaN; the fused activation quantizer writes every padding byte.)
- **Global scale**: FP32. Weights: one per tensor, `amax(|W|) / (6 · 448)`. Activations: one per
  token row, `max(amax(|x_t|) / (6 · 448), FLT_MIN)` (torch computes `/ 2688` as a multiply by the
  FP32 reciprocal; the fused quantizer does the same). E0M3 blocks use the *same* weight global
  scale as E2M1 blocks (the E0M3 candidate is computed with the NVFP4 convention `6 · 448`).

Value of element `(r, k)`: `codebook[tag(r, k/16)][nibble(r, k)] · scale(r, k/16) · global_scale(r)`.

## 3. Quantizers (rounding, clipping, saturation)

Weights (per tensor, `reshape(-1, 16)` blocks):

| name | block scale candidates | element rounding | selection |
|---|---|---|---|
| NVFP4 (`quant_nvfp4`) | `e4m3(clamp(bmax/6))` | E2M1: floor(log2) exponent, round-half-away at 1 mantissa bit, clamp ±6 | — |
| FourOverSix (`quant_nvfp4_4over6`) | `e4m3(clamp(bmax/6))`, `e4m3(clamp(bmax/4))` | E2M1 by midpoints, ties toward the lower level (`<=`) | /4 iff its block squared error is strictly smaller |
| E0M3 (`quant_mix_4_6(..., clip='a1', elect='always')`) | `e4m3(clamp(bmax · (1/7)))` | `round()` (half-to-even), clamp ±7 | — |

A **format map** (MIXFP4MAP/1, `type_block = (16, 64)`) marks tiles of 16 output channels × 64 K
as E0M3; every scale block inside a marked tile takes the E0M3 candidate, every other block the
FourOverSix candidate (`campaign.tiles.apply_mask` semantics). Saturation: a value above the
largest code after scaling clamps to ±6 (E2M1) / ±7 (E0M3); this happens only when an FP8
subnormal scale rounds down (the reason `quant_nvfp4` clamps).

Activations (per token row, scale blocks of 16 along the hidden dimension), E2M1 only:

| name | used by | definition |
|---|---|---|
| `four_over_six_rows` | FourOverSix and map policies | `quantize_rows`: per-token global scale, FourOverSix choice per block |
| `nvfp4_rows` | NVFP4 policy | per-token global scale, NVFP4 rounding |

Activations are quantized at every call from the BF16 input; there is no calibrated or static
activation range, so the global-scale range is always exactly the token's own amax.

## 4. GEMM arithmetic and output

For `y = x Wᵀ (+ b)` with weights on operand A (`n16k64_wA`):

    acc[o, t]  = Σ_k  (codeW · sW)[o, k] · (codeX · sX)[t, k]        tensor core, FP32 accumulation
    y[t, o]    = bf16( (gs_w · gs_x[t]) · acc[o, t] + b[o] )          fused epilogue, one rounding

- Each product of two decoded elements is exact; accumulation is FP32 on the tensor core (the
  summation order is the hardware's, not sequential).
- The epilogue multiplies the two FP32 global scales (one rounding), does one FP32 FMA with the
  accumulator and the bias, and rounds once to BF16 (round-to-nearest-even). Output dtype: BF16,
  row-major `[tokens, out]` (the kernel stores D = W Xᵀ column-major, which is that tensor).

## 5. Expected native vs fake-quant differences

The fake-quant path computes `bf16(x_deq) @ bf16(W_deq)ᵀ` with `x_deq = bf16((code · s) · gs_x)`
in a BF16 cuBLAS GEMM (FP32 accumulate). Both paths decode **identical** codes and scales (checked
bit for bit at export and, in `eval/layerwise.py`, on real activations). They differ only in:

1. **Operand rounding**: fake quant rounds every dequantized operand to BF16 (8 significant bits);
   `(code · s) · gs` generally needs more, so fake quant perturbs each operand by up to 2^-9 relative.
   Native multiplies the exact decoded values.
2. **Where the global scales are applied**: native applies `gs_w · gs_x[t]` once to the FP32
   accumulator; fake quant folds them into the operands before rounding.
3. **Accumulation order** (both FP32, different hardware reduction trees).

Consequently the native result is at least as close to the exact product as fake quant
(measured per layer on Qwen3-4B: mean relative error 0.17% native vs 0.25% fake quant, native
closer in 252/252 layers). Through a W4A4 network these small differences are amplified by
activation re-quantization (a value on a rounding boundary flips a code), so logits of the two
paths are not bitwise equal and per-window NLLs differ with zero mean; perplexity must be compared
as a paired, window-level difference (`eval/ppl.py`), not expected to match exactly.

The repro branch's harness applied the per-token scale after a BF16 GEMM output (two roundings);
this path fuses it (one rounding), so its numbers are not bit-identical to that harness's.
