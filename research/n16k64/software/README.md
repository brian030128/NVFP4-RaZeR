# Historical N16K64 campaign software

These directories preserve the code that generated the five N16K64 campaigns.
They are separate from the current repository implementation because historical
outcomes must remain tied to their actual code/config provenance.

| Snapshot | Campaigns represented | Contents |
|---|---|---|
| [`primary/`](primary/) | primary full validation | 61-file campaign package plus the historical `run_kse_paper.py`, `run_math_code_calibration.py`, `build_mixfp4_report.py`, and quantizer support files |
| [`ppl_extension/`](ppl_extension/) | post-hoc PPL-improvement extension | 146-file extension package and its support files |
| [`boundary/`](boundary/) | mechanism, follow-up, boundary/corruption | 84-file superset package and its support files |

Do not import more than one `campaign` package in a process. A historical CPU
entry point can be inspected with a snapshot-first path such as:

```bash
PYTHONPATH=research/n16k64/software/boundary/support:research/n16k64/software/boundary:$PWD \
  python3 -m campaign.boundary_protocol --help
```

Actual derivation/evaluation commands need a campaign root and the immutable
inputs named by that campaign's protocol. No GPU job should be launched merely
to test this review branch.

## Public-copy integration changes

The archived source hashes and public-copy hashes are recorded separately in
[`../SNAPSHOT_SOURCE_INDEX.json`](../SNAPSHOT_SOURCE_INDEX.json). Changes are
restricted to path plumbing and test identities:

- private absolute workspace/archive defaults became environment-configurable
  paths (`MIXFP4_WORKSPACE_ROOT`, `MIXFP4_PRIMARY_CAMPAIGN`,
  `MIXFP4_MECHANISM_CAMPAIGN`, `MIXFP4_FOLLOWUP_CAMPAIGN`, and handoff variables);
- the archived-results availability probe uses `NVFP4_ARCHIVED_RESULTS_ROOT`;
- historical stage roots use `NVFP4_RAZER_STAGE_ROOT` or an explicit CLI value;
- collection-host prose is generic;
- real device identifiers in GPU-policy unit-test fixtures became synthetic
  identifiers.

No format, quantization, selector, score, threshold, seed, dataset, aggregation,
or metric behavior was intentionally changed. Historical outcomes were not
rerun from these portable copies; they remain results of the original hashes.

## Entry points, dependencies and practical scope

The daily checkout contains current main plus this historical research package.
It is an archive with executable CPU/evaluation components and external input
requirements. No new method is wired into main's existing CLI, and a clean clone
alone cannot replay all outcomes. Main's `quantize/quantizer.py` and generic
`--w_dtype mixfp4 --w_type_block 16x64` do not by themselves reproduce the frozen
gradient conjunction: use the exact map-aware historical evaluator.

| Task | Historical entry point (inside one selected snapshot) |
|---|---|
| CE/KL directional scoring and initial election | `campaign.calibrate`; `campaign.tiles.Moments`, `upper_bound`, `elect` |
| Candidate quantization and causal activation | `campaign.quant`, `support/quantize/quantizer.py`, `support/quantize/causal_four_over_six.py` |
| Map decoding/installation | `campaign.mapio`, `campaign.policies` |
| PPL and downstream evaluation | `campaign.evaluate_ppl`, `campaign.evaluate_lmeval` |
| Primary analysis/reporting | `campaign.analyze_ppl`, `campaign.analyze_accuracy`, `campaign.final_reports` |
| Extension selectors and analysis | `campaign.extension_maps`, `extension_selector_stats`, `extension_*_analysis` in `ppl_extension` |
| Mechanism maps/statistics | `campaign.mechanism_maps`, `mechanism_analyze`; standalone `campaigns/mechanism/analysis/*.py` |
| Follow-up maps/statistics | `campaign.followup_maps`, `followup_analyze`, `followup_render` |
| Boundary maps/statistics | `campaign.boundary_prepare`, `boundary_maps`, `boundary_analyze_postfreeze`, `boundary_render` |

Select both the versioned `support` and `campaign` paths before current main:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/research/n16k64/software/boundary/support:$PWD/research/n16k64/software/boundary:$PWD"
CUDA_VISIBLE_DEVICES='' python3 -m campaign.evaluate_ppl --help
```

The original support modules' static local imports have now been copied rather
than accidentally resolved to a potentially changed current-main helper.
Standalone ops/scripts and generated plans containing `<USER_HOME>` or
`<ARCHIVE_ROOT>` are archival provenance, not portable launch defaults. Use a
new plan with explicit resolved paths and GPU-policy wrapper. Some legacy
support-driver defaults also remain archival placeholders; the supported import
and help paths are distinguished from a certified full campaign replay.

Original calibration reports needed by `data.archived_manifest` and historical
baseline/teacher files are still external even after supplying model/data caches.
Exact maps, moments, source manifests, original protocol seal and token windows
must be supplied under the original logical campaign layout or through reviewed
path adapters. A redacted protocol's public hash is not its original launch hash.
Six boundary paired-cluster NPZ files and definitions now allow independent CPU
statistics; the supplied historical full finalizer additionally expects original
run reports, so it is not a standalone public-array CLI.

Source Git identity: campaign source directories are file snapshots without
their own Git repository. A `git rev-parse` inside a nested primary source
would misleadingly return the enclosing workspace's HEAD. Exact provenance is
the per-file original SHA in `SNAPSHOT_SOURCE_INDEX.json`, protocol code hashes,
run source-manifest hashes and archive seals, not that parent HEAD. Later
protocols reference the f692459 archive lineage plus custom campaign code.
The current integrated HEAD did not generate the historical outcomes.

Extension v1 differs from v2 in eight files; the historical differences are in
`campaigns/ppl_improvement/source_v1_differences/`. Treat them as provenance for
earlier attempts, not active code to merge into v2. Mechanism's 66 and follow-up's
73 campaign Python files match the corresponding boundary source files exactly;
one superset preserves these implementations without separate maintained copies.

## Quality and native execution paths

The [root research homepage](../../../README.md) distinguishes these paths.
This is a source/record audit, not a new kernel run or performance validation.

### Frozen quality path

In the primary snapshot, `campaign.evaluate_ppl.main` loads a BF16 model through
`campaign.models`, then `campaign.policies.Installer` reads/verifies the format
map and copies selected dequantized weights into ordinary Linear parameters.
`campaign.quant.ActivationQuant._pre` applies causal per-token FourOverSix
quantize–dequantize before each targeted Linear. The forward remains
Transformers → `torch.nn.Linear` → `torch.nn.functional.linear`, consuming
floating-point operands. `campaign.evaluate_lmeval` installs the same policy
before wrapping the model in HFLM.

Concrete source: [PPL](primary/campaign/evaluate_ppl.py),
[loading](primary/campaign/models.py), [installation](primary/campaign/policies.py),
[quantization/hooks](primary/campaign/quant.py),
[activation producer](primary/support/quantize/causal_four_over_six.py),
[accuracy](primary/campaign/evaluate_lmeval.py).
Weights keep shape `[out_features, in_features]` and BF16 storage; candidate
computation uses FP32 intermediates. There is no packed-FP4 GEMM dispatch here.
The [map codec](primary/campaign/mapio.py) packs Boolean ownership, not numerical
nibbles/scales. [Tile expansion](primary/campaign/tiles.py) repeats each N16K64
decision over 16 rows and 64 columns, containing 64 independent 1×16 scales.
No primary reorder is installed. A policy name containing “native” does not
alter this call chain; moving these commands to Blackwell does not enable a
different backend. Existing A6000/Ada runs establish simulated quality only.

### Separate SM100 / GB200 prototype

[NativeLinear](../../../scripts/native_model_runtime.py) packs two weight codes
per uint8: weights `[N,K/2]`, flat scale bytes `[N,K/16]`, plus a type map.
Its forward calls `mf_run` in
[model_runtime.cu](../../../native/model_runtime.cu): tensor-global activation
amax → native activation encoding → CUTLASS GEMM → BF16 output. The
[build-local override](../../../scripts/build_native_model_runtime.py) selects
B's format per **N256K64**, not N16K64; native accumulation is FP32. It requires
N/K multiples of 256, a full-K scheduler (not Stream-K), single-stream scratch,
external sibling kernel/CUTLASS inputs and a reviewed build environment. The
current build script still assumes a machine-specific sibling source location;
it is not a portable clone-and-build command.

The frozen [kernel snapshot](../../../results/task_reorder/transfer_20260920/kernel_snapshot/)
and [implementation record](../../../results/task_reorder/full_model_20260920/IMPLEMENTATION.md)
preserve the engineering context. Tensor-global activation scaling and rounding
order differ from the primary causal-per-token simulation. Operator and exact
activation-code checks in
[406633](../../../results/task_reorder/full_model_20260920/kernel_406633/report.json)
passed as recorded. The later
[406828 report](../../../results/task_reorder/full_model_20260920/llama_406828/report.json)
still has `full_output_equivalence_gate_passed=false` and
`native_accuracy_established=false`; native PPL is unmeasured.
Diagnostic timing is neither native N16 quality validation nor a serving result.
Historical failed checks and cold/unstable timings are retained, not superseded
into passes.

The override writes descriptor B-format 0 for E0M3 and 1 for E2M1.
[NVIDIA PTX ISA 8.8, CUDA 12.9, Table 45](https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html#tcgen05-instruction-descriptor)
documents E2M1=1 and UE4M3 scale support for the relevant instruction family;
it does not define E0M3=0. This source was rechecked on 2026-09-21. Experimental
GB200 behavior is not an official encoding guarantee or evidence of portability
to SM120. Missing public definition also does not prove physical impossibility.

### Other repository backends

The [upstream inference artifact](../../../inference/README.md) is separate.
Its weight-only Marlin path decodes low-bit values before FP16 MMA; the
`razer_cuda` path has dequantization/FP floating-point fallbacks.
[Blackwell W4A4](../../../inference/w4a4/) includes SM120 NVFP4 and other
low-precision implementations; RaZeR decomposition is not the frozen
E2M1/E0M3 map rule. Kernel presence, packed storage and architecture flags alone
do not establish primary-campaign dispatch.

Before any native N16 claim, a future effort must implement exact N16 map
ownership and primary activation/scale/rounding semantics, resolve documented
versus experimental encodings, validate full outputs and quality, and only then
measure end-to-end performance against matched baselines. None of those missing
validations was run during this documentation refresh.
