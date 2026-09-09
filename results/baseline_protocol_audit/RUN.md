# Completed baseline discrepancy audit

All jobs completed with exit code 0:

| Job | Work | Elapsed |
|---|---|---:|
| 335043_0 | Qwen3-4B, 512 tokens | 13m52s |
| 335043_1 | Llama-3.1-8B, 512 tokens | 13m27s |
| 335043_2 | Qwen3-4B, 2048 tokens | 13m40s |
| 335043_3 | Llama-3.1-8B, 2048 tokens | 13m15s |
| 335061 | Initial validation and aggregation | 5s |
| 335163_2 | Qwen historical projection-input behavior | 7m32s |
| 335167 | Final validation, aggregation, and MIXFP4 report update | 4s |

The initial matrix covers BF16 and two activation-scale conventions for
FourOverSix at both contexts, plus standard NVFP4 at 2048. Every shared
historical baseline window replayed within 1e-6 NLL. Source/checkpoint,
quantizer, framework, and data identities match across context comparisons.
Native hooks and released wrappers had exactly equal first-window logits for
BF16 and tensor-wide FourOverSix under eager attention. PPL reconstruction
and scored-token counts passed for every completed cell.

The BF16 2048-token Wiki/C4 results match the paper's displayed precision for
both models. Context length explains most of the large baseline discrepancy.
Llama W4A4 measurements are within 0.011 PPL of the paper. Qwen retains a
smaller gap, especially for FourOverSix. The targeted historical Qwen bug
reproduction reduces this gap only partially; it does not establish an exact
reproduction of Table 3. No quantizer was tuned to match the published values.

The historical diagnostic was specified after inspecting the remaining gap
and repository commit abab3c6. It leaves only Qwen attention-output projection
inputs unquantized; weights and all other quantized inputs stay unchanged.
The original matrix and this follow-up are separate, labeled measurements.
Source hashes are retained in each report. The wrapper parity check explicitly
copies nonpersistent rotary buffers when constructing a meta-device wrapper.

No calibration, E0M3 map selection, or calibration-seed replication occurred.
Existing sparse-map performance numbers remain measured at 512 tokens; the
audit does not establish their gains at 2048.

See [REPORT.md](REPORT.md), [PROTOCOL.md](PROTOCOL.md), and the recorded
[Qwen fix](qwen_o_proj_fix.patch). All raw per-window losses and input hashes
are retained in the job directories.
