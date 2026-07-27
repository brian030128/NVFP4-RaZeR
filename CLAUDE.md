# CLAUDE.md

Guidance for working in this repository.

## Repository overview

RaZeR is a research codebase for **simulated (fake) 4-bit LLM quantization**. Nothing here needs
FP4 hardware: every format is emulated in FP32/BF16 and the result is written back into the model's
BF16 weights or activations. All quantizers therefore run on the CPU.

- `quantize/quantizer.py` — every fake quantizer (`quant_mxfp4`, `quant_nvfp4`, `quant_nvif4`,
  `quant_mixfp4`, the RaZeR variants, ...) plus the two dispatch functions `quant_weight` (walks the
  model and rewrites `nn.Linear.weight` in place) and `quant_act` (called inline from the quantized
  model modules).
- `quantize/quant_config.py` — `QuantConfig`, the single object threaded through the model.
- `quantize/utils.py` — scale-quantization and type-block helpers.
- `utils.py` — CLI argument definitions, `QuantConfig` construction, model loading, result-file naming.
- `models/qmodule_*.py` — HuggingFace model copies with `quant_act` calls inserted at the
  activation, KV-cache and attention-output boundaries.
- `run_ppl.py`, `run_zeroshot.py`, `run_llama_cot.py` — evaluation entry points.
- `run_mixfp4_sim.py` — CPU-only MixFP4 error sweep (no model download required).
- `tests/test_mixfp4.py` — CPU tests for MixFP4.
- `inference/` — the published CUDA/kernel artifact. Unrelated to the simulation path above; do not
  change it when adding a simulated format.

### Adding a new simulated format

1. Write `quant_<name>(w_fp, n_bits, groupsize, ...)` in `quantize/quantizer.py`. It takes a tensor
   of any shape, quantizes along the last (reduction) dimension, and returns a **dequantized**
   tensor of the same shape in `torch.bfloat16`.
2. Register the name in the `if/elif` chains of both `quant_weight` and `quant_act`.
3. If the format needs extra parameters, add them to `QuantConfig` and to `add_quant_args` in
   `utils.py`, and make sure they end up in the result file name (`get_output_file_tag`) so that
   sweeps do not overwrite each other.

---

## MixFP4

MixFP4 is NVFP4 plus a second, coarser block granularity that selects the FP4 **element data type**.
Everything else — the FP32 per-tensor global scale, the E4M3 block scale, the 16-element scale block
— is inherited from NVFP4 unchanged.

### There are TWO kinds of block

| | **Scale block** | **Type block** |
|---|---|---|
| Size | **Always 16 elements** along K (the NVFP4 block) | **Configurable `<M>x<K>`**, e.g. `1x16`, `16x16`, `256x16`, `32x64`, `32x128` |
| Shape | 1-D, `1 x 16` along the reduction dimension | 2-D tile: `M` rows x `K` columns |
| What it owns | one **E4M3 block scale** | one **element data type**: either **E2M1** or **E0M3** |
| Relationship | — | **A type block contains multiple scale blocks** |

A type block of shape `M x K` contains `M * (K / 16)` scale blocks. Consequently:

- **The K dimension of a type block is always a multiple of 16.** `32x24` is invalid; `32x64` holds
  `32 * 4 = 128` scale blocks. This is asserted in `parse_type_block`.
- `1x16` is the degenerate case where the type block **is** the scale block, i.e. the data type is
  chosen per NVFP4 block. In that configuration MixFP4 is numerically identical to the existing
  `nvif4` quantizer (there is a test asserting exactly this).
- Every scale block inside a type block uses the **same** element data type. Each of them still has
  its **own** E4M3 scale.

```
type block 32x64  (M = 32 rows, K = 64 columns) -> ONE data type for the whole tile
+---------------------------------------------------------------+
| row 0  [ scale blk ][ scale blk ][ scale blk ][ scale blk ]    |  each [ scale blk ] = 16 elements
| row 1  [ scale blk ][ scale blk ][ scale blk ][ scale blk ]    |  with its own E4M3 scale
|  ...                                                          |
| row 31 [ scale blk ][ scale blk ][ scale blk ][ scale blk ]    |
+---------------------------------------------------------------+
   = 32 * 4 = 128 scale blocks, all E2M1 or all E0M3
```

### The two element data types

- **E2M1** — the standard FP4 grid `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`, max magnitude 6. This is
  what plain NVFP4 always uses.
- **E0M3** — the *evenly spaced* signed 4-bit grid `{0, ±1, ..., ±7}`, max magnitude 7. Numerically
  equivalent to signed INT4. The block scale is `block_max / 7`.

Both grids encode 15 distinct values in 16 codes — the redundant zero that RaZeR exploits — and both
use the same ue4m3 block scale. Only the spacing differs: E2M1 has finer resolution near zero and
coarser resolution near the block maximum, E0M3 is uniform. Which one wins depends on the
distribution inside the tile, which is why the choice is data driven.

### Hardware grounding and the minimum type-block size

MixFP4 is not a paper format: the public NVFP4 path already issues

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

and the same instruction can read **A, B, or both** as E0M3 instead of E2M1. That fixes three things:

- **`scale_vec::4X` + `ue4m3`** is exactly the NVFP4 scale block: four ue4m3 scales across `k64`,
  i.e. one scale per 16 elements. The scale block is 16 for both element types, and the E0M3 branch
  reuses the same ue4m3 scale — which is what `quant_mixfp4` does.
- **The data type is selected per operand, not per element.** Weights and activations choose
  independently, hence the separate `--w_type_block` and `--a_type_block`.
- **A single instruction cannot subdivide its operand tile**, so the *smallest hardware-realizable*
  type block is one MMA operand tile. For the usual `Y = X · Wᵀ` mapping (`X` = A, `Wᵀ` = B):

  | operand | tile | minimum type block |
  |---|---|---|
  | A — activations | `m16 x k64` | **16x64** (16 tokens x 64 K) |
  | B — weights | `n8 x k64` | **8x64** (8 output channels x 64 K) |

  Confirm this mapping against the actual kernel before quoting it — a kernel that puts weights in
  A swaps the two rows.

Anything coarser (`32x64`, `32x128`, `256x64`, ...) is a union of whole operand tiles and is
realizable. Anything with **K < 64** — including `1x16`, `16x16` and `256x16` — is *not* expressible
with this instruction, because one MMA consumes 64 contiguous K elements under a single declared
element type. Those configurations are still worth sweeping as **accuracy upper bounds**: `1x16` is
the finest possible selection and bounds what any coarser scheme can achieve. Just do not report
them as deployable without a different instruction or a 4x-cost K-splitting trick.

`quant_mixfp4` deliberately does not enforce the K >= 64 rule, so that these reference points stay
measurable.

### Quantization procedure

For a tensor reshaped to `(M, K)` (rows = output channels for weights, tokens for activations):

1. **Global scale** (per tensor, FP32), the NVFP4 convention: `amax / (6 * 448)`.
2. Tile into type blocks, then into 16-element scale blocks.
3. For **each scale block**, compute both candidate E4M3 block scales: `block_max / 6` for E2M1 and
   `block_max / 7` for E0M3, each clamped to `[2^-9, 448]` and rounded to `float8_e4m3fn`.
4. Quantize every scale block **both ways**.
5. For **each type block**, sum the squared error over all of its scale blocks for each data type,
   and keep the data type with the smaller total. Ties go to E2M1.

### Configuration

```
--w_dtype mixfp4 --w_groupsize 16 --w_type_block 32x128
--a_dtype mixfp4 --a_groupsize 16 --a_type_block 1x16
```

- `--w_type_block` / `--a_type_block` accept `"<M>x<K>"` (default `1x16`) and are ignored by every
  other data type.
- `--w_groupsize` / `--a_groupsize` **must be 16** for MixFP4 — the scale block is the NVFP4 block
  and is not configurable.
- The type-block shape is appended to the result file name (`..._mixfp4-32x128__...`) so a sweep
  over type-block shapes does not collide.

### Implementation notes / gotchas

- `quant_mixfp4` reshapes to `(-1, last_dim)`, so 3-D activations `(batch, seq, hidden)` and 4-D
  KV tensors `(batch, heads, seq, head_dim)` are all handled; `M` spans everything except the last
  dimension.
- **Outer-dimension padding**: if `M` is not divisible by the type block's `M`, the tensor is
  zero-padded at the bottom. The padded rows form their own all-zero type blocks and never
  influence the data type chosen for real rows.
- **Narrow reduction dimension**: if `K` is smaller than the type block's `K` (e.g. a 64-wide head
  dimension with a `32x128` type block), the type block shrinks to the full row instead of raising,
  so that a sweep keeps running. `K` must still be a multiple of 16.
- Finer type blocks can never have a higher total squared error than coarser ones, because `1x16`
  divides every other configuration and the selection is error-minimizing. Use this as a sanity
  check when changing the selection rule.

### Testing (CPU only)

```
python tests/test_mixfp4.py        # correctness: tiling, per-type-block uniformity, padding, nvif4 equivalence
python run_mixfp4_sim.py           # NMSE / SQNR sweep over type-block shapes on synthetic tensors
python run_mixfp4_sim.py --model_name llama-2-7b --max_layers 4   # ... on real weights
```

`tests/test_mixfp4.py` includes a negative control: the per-type-block uniformity check must *fail*
on `nvif4` output. Keep that control if you touch the check, otherwise it can silently pass on
anything.

### Measured accuracy (Llama-2-7B, Llama-3.1-8B)

`results/mixfp4_sweep/REPORT.md` holds the full perplexity sweep. The headline, which should shape
any further work on this format:

- **MixFP4 only helps at `1x16`**, where it is bit-identical to `nvif4` and beats NVFP4 by
  0.009 (Llama-2-7B) / 0.064 (Llama-3.1-8B) wikitext ppl.
- **Every type block coarser than one scale block is worse than plain NVFP4**, including all
  hardware-realizable shapes (`8x64` and up): +0.03 wikitext ppl on Llama-2-7B W4A16, +0.014 on
  Llama-3.1-8B. Almost the entire loss happens in the first coarsening step (`1x16` -> `16x16`);
  `16x16` through `32x128` are within ~0.005 ppl of each other.
- RaZeR remains the strongest format in every setting measured.

The cause is visible in the E0M3 selection rates: at `1x16` the choice is genuinely mixed within a
tensor (41% E0M3 in `q_proj`, 60% in `v_proj`), but a large tile has to elect a single winner, so
`q_proj` collapses to ~all E2M1 (5.5% E0M3 at `32x128`, i.e. back to NVFP4) while `v_proj` collapses
to ~all E0M3 (99.6%). The mixing that produced the gain is exactly what the coarse granularity
averages away.

Note also that the selection minimizes MSE, and lower MSE does not always mean lower perplexity --
`nvif4`/`mixfp4_1x16` has clearly lower NMSE than `nvfp4_4over6` yet slightly higher wikitext ppl on
Llama-2-7B. Treat the sim NMSE as a fast proxy only, and confirm with `run_ppl_sweep.py`.
