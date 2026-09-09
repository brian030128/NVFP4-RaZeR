# Baseline discrepancy audit

The user identified a large gap from RaZeR Table 3, including its W4A4
FourOverSix row. This audit changes no calibration maps and selects no method.
Use the pinned Qwen3-4B and base Llama-3.1-8B checkpoints from the completed
math/code study, native BF16 with eager attention, and unquantized KV tensors.

Compare BF16, FourOverSix W4A4 with per-token activation factors, and
FourOverSix W4A4 with the released tensor-wide factor at 512 and 2048 tokens.
Additionally evaluate standard NVFP4 W4A4 with tensor-wide factors at 2048.
All policies quantize the same nonhead linear modules; only their quantization
changes. Check native-model hooks against the released quantized wrappers
on the first 2048-token Wiki window for BF16 and tensor-wide FourOverSix.

Wiki: identical full 2048-token spans of the concatenated raw test split;
split these into four independent windows for 512-context evaluation. This
omits at most three 512-token windows relative to the historical full-512
Wiki evaluation. Additional boundary tokens are unscored at 512; this is an
evaluation-protocol contrast rather than a strictly identical-target ablation.

C4: reproduce the released evaluator's random.Random(0), shard 00000,
256 random document/crop selections with 2048-token contexts. Pin the dataset
revision and record documents, offsets, and token hashes. Split each parent
crop into four 512-token windows to compare context on the same documents.
Also evaluate historical shard-00001 C4 windows at 512 and check every old
baseline loss against its replay (1e-6 tolerance). Check the shared Wiki prefix
against historical losses too.

The released evaluator's default attention backend and checkpoint versions
may differ from these pinned runs. Wrapper equality here holds at matched
eager attention and weights. A residual paper gap must be reported rather
than ascribed to a single factor without evidence. The local standard NVFP4
implementation includes the previously documented saturation fix. No seed
replication, tuning, calibration on Wiki/C4, or E0M3 selection occurs.

Reference: https://arxiv.org/html/2501.04052v2#S4.T3 and the local
run_ppl.py, models/qmodule_llama.py, models/qmodule_qwen3.py, and
quantize/quantizer.py implementations.

## Follow-up after the matched-context measurements

The full W4A4 Qwen3-4B tensor-scale Wiki result still exceeded Table 3 after
matching context. Git history then identified repository commit
abab3c65de95e199c6ab977dd2ac7f43238bf369 (2026-06-17), which changed Qwen's
attention output projection from o_proj(attn_output) to
o_proj(attn_output_quant). The earlier implementation computed quantized
activations but discarded them at that projection. Llama did not have this
specific change. The commit postdates the January 2026 paper version.

Run a labeled historical-behavior reproduction for Qwen3-4B at 2048, with
the same FourOverSix and standard-NVFP4 weights and tensor-wide scaling,
but leave only the attention output projection inputs unquantized. This is
a diagnostic of the residual discrepancy, not a proposed quantization
method or a replacement for the actual W4A4 baseline. Keep both outcomes.
