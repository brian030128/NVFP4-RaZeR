# Permutation fusion experiments on GB200

The measured maps are the exact compacted **Qwen** up/down maps in
`results/task_reorder/transfer_20260920/permutations`. These native experiments
measure permutation cost. They do not execute the complete mixed-format model.
All compilation and execution run through the `gov113008` GB200 Slurm partition.

## Keep the GEMM epilogue; fuse with adjacent operations

The direct BF16 scatter epilogue (`build_fused_permutation.py`) was bitwise
correct but much slower. Its scalar stores displaced the efficient TMA output
path. It is retained as a failed experiment, not the recommended implementation.

The consumer approach leaves the sibling SM100 GEMM unchanged:

- Let `row[new] = old` and `inverse[old] = new` describe up's output channels.
  Gate's compact map is identity. The SiLU/multiply consumer reads
  `gate[q[j]]` and `up[inverse[q[j]]]`, then writes output channel `j`.
  Here `q` is down's column permutation, so the consumer also emits the input
  order needed by down. SiLU is rounded to BF16 before the BF16 multiply,
  matching the tested operation sequence.
- Down's consumer reads `down[inverse[j]]` and `residual[j]`, and writes their
  BF16 sum at `j`. No standalone output restoration is necessary.
- The vector version handles two BF16 values per thread. Contiguous pairs use
  a 32-bit load; noncontiguous permuted pairs fall back to scalar loads.
  Both permutations and a negative control are checked on every measured shape.

When connecting these projection prototypes into a full MLP, apply each column
permutation exactly once: if SiLU already emits down-column order, down
quantization uses identity destinations. Alternatively, SiLU emits natural order
and down quantization applies the column map. The reported per-projection tests
are independent and are not summed into an end-to-end MLP latency.

`consumer_permutation*.cu` measures the consumer alone.
`build_consumer*_pipeline.py` measures GEMM plus consumer, including a separate
packed-input gather. These are different timing scopes and their baselines must
not be mixed.

## Fuse input permutation into FourOverSix quantization

`quantize_permutation.cu` assigns 16 lanes to each original 16-value scale group.
It computes the E4M3 scales corresponding to maxima 4 and 6, chooses the lower
reconstruction error, and packs E2M1 codes. The fused variant changes only the
output group address: it stores the packed 64-bit code word and scale byte at
`inverse_column_group[old_group]`. Scales use CUTLASS's actual SM100 SFA layout.
No arithmetic inside a scale group changes.

Whole-group permutation preserves tensor-wide absolute maximum and every
within-group calculation. The same global scale is shared by all benchmark
variants. Computing that common maximum is excluded from timing.

A separate H200 oracle compares native dequantized results with
`quant_nvfp4_4over6`. The first oracle caught discarded negative-zero signs;
the corrected implementation preserves them and passes all 22,528 sampled
BF16 values bitwise. Fused/unfused code and scale bytes are independently checked.
This is a prototype quantizer, not an assertion that its absolute throughput is
optimal or that every input distribution has been exhaustively tested.

`build_full_permutation_pipeline.py` combines the producer, unchanged GEMM, and
consumer. It reapplies the common global activation scale through GEMM's alpha.
The baseline uses the same quantizer, GEMM, and consumer without permutations;
the separate variant uses explicit gather/restoration passes; the fused variant
uses the producer/consumer indexing above. The tests still use fixed E2M1 GEMM
format selection and synthetic weights. They do not validate the 212-tile model,
full MLP execution, model latency, or native Llama throughput.

Source-generating scripts write build directories under `/work`; they do not
modify `../mixfp4`. See the corresponding `slurm/gb200_*` scripts for bounded
worker-only builds and `MIXFP4_REPORT.md` for measured results.
