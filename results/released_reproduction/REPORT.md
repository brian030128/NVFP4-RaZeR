# Released-code reproduction outcome

Full exact Table 3 reproduction remains incomplete: both complete attempts
match 15/28 published WikiText-2/C4 cells to two decimals. All cases ran the
February released evaluator directly, with no calibration or parameter search.

Llama-3.1-8B reproduces every weight-only cell and both RaZeR W4A4 cells.
Qwen3-4B retains material FourOverSix/RaZeR differences. Qwen NVFP4
weight-only WikiText reproduces Table 1's 13.63, while Table 3 lists 13.83.
The report retains Table 3 as the primary target and discloses this inconsistency.

Changing from Python 3.11/Torch 2.9 to Python 3.10.18 and the release's core
package pins, with inferred Torch 2.7.1/CUDA 12.6, changes PPL by at most
0.000004. All evaluation token windows and weight hashes match across the
two environments. This tested environment change does not explain the gaps.

* [All per-dataset results, signed residuals, and environment comparison](job_335302/REPORT.md).
* [First complete attempt](job_335297/REPORT.md).
* [Source-history findings and reproduction limits](CODE_HISTORY.md).
* [Fixed protocol](PROTOCOL.md).
* [Jobs and launch command](RUN.md).

Jobs 335297 and 335302 used gov113008/taide_h200, eight H200s per allocation
and up to eight independent one-GPU Slurm steps. All 14 cases per job
completed in 6m38s and 7m06s respectively, including setup and validation.

The archived Qwen wrapper omits o_proj input quantization. Its W4A4 rows
must retain that disclosure. The authors' optional original C4 token cache
and a complete environment/table-to-commit mapping are unavailable; the
remaining residuals cannot be assigned a proven cause from these runs.
The existing MixFP4 gains remain separate 512-token results.
