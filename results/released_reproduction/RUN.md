# Execution record

* `335287`: eight-H100 request on gov113008/taide. Cancelled while pending:
  scheduler estimated 2026-09-11, two days after submission.
* `335294`: eight-H200 replacement on gov113008/taide_h200, hgpn45.
  All cases stopped before loading models because `loguru` was missing.
  Failed case reports are preserved; they contain no PPL measurements.
* `335297`: same eight-H200 allocation shape, with the release-pinned
  `loguru==0.7.3` installed into worker-local scratch and checked before
  parallel cases start. The launcher validates and aggregates all 14 cases
  after successful evaluation, then updates MIXFP4_REPORT.md.
  Completed successfully in 6m38s; all 14 cases validated, 15/28 displayed
  Table 3 matches.
* `335302`: dependent eight-H200 run, repeating all cases with Python
  3.10.18 and the release's core package pins. Torch 2.7.1/CUDA 12.6 is an
  explicitly inferred environment hypothesis. Both attempts are retained;
  the summary also checks exact input hashes and compares weight hashes
  and every PPL cell across environments.
  Completed successfully in 7m06s; all 14 cases validated, the same 15/28
  displayed matches. Every token window and weight-matrix hash matched
  between environments; maximum absolute PPL change was 0.000004.

Submission command for the H200 run:

```bash
sbatch --parsable --partition=taide_h200 --gres=gpu:H200:8 \
  --export=ALL,GPU_TYPE=H200 slurm/released_reproduction.sbatch
```

The default script requests eight H100s on taide. Each srun step explicitly
requests one GPU, four CPUs, and 40 GB host memory. All 14 steps share the
allocation and wait for step resources when all eight slots are busy.
