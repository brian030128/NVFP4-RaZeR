# Validation commands and scope

## Daily-workspace integration follow-up

The current repository is the daily `mixfp4` checkout. Earlier commands/results
below remain historical evidence. The integration reran the primary and
extension commands after restoring missing support imports: **105 passed / 1
deselected** (13.75 s) and **209 passed / 1 deselected** (14.58 s). The primary
run emitted the same two expected constant-input correlation warnings. No main
or GPU campaign suite was rerun because its code did not change.

The previously omitted campaign-specific tests were run from this checkout:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
MIXFP4_WORKSPACE_ROOT=<REPO> CAMPAIGN_ROOT=<PRIMARY> \
PYTHONPATH=<REPO>/research/n16k64/software/boundary/support:<REPO>/research/n16k64/software/boundary:<REPO> \
<PYTHON> -m pytest -q -p no:cacheprovider \
  research/n16k64/software/boundary/support/tests/test_boundary_campaign.py \
  research/n16k64/software/boundary/support/tests/test_followup_campaign.py
```

Result: **7 passed in 2.17 s**. Covers remainder rotation, deterministic priorities,
rounding/nesting, dose-bin accounting, objective quotas and basic Holm/fit behavior.
The same boundary path/environment successfully ran `python -m campaign.evaluate_ppl
--help`, importing its parser/dependencies without loading a model.

Fresh lightweight audit commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 research/n16k64/tools/audit_current_evidence.py --workspace "$PWD"
PYTHONDONTWRITEBYTECODE=1 python3 research/n16k64/tools/reverse_source_inventory.py --workspace "$PWD"
python3 research/n16k64/tools/verify_public_snapshot.py
git diff --check
```

The evidence audit verifies 604 source/public pairs, 171 map hashes/headers,
45 natural maps and 15 actual calibration manifests, six paired-array policy
and cluster counts, p=1 means and retained-gain arithmetic. Reverse inventory
checks candidates before inclusion/exclusion, not only copied files. No new
bootstrap, GPU experiment or historical outcome is generated. An initial audit
development assertion used `full_conjunction`; inspecting the actual array
schema corrected it to `corruption_p000_anchor` before successful verification.
Initial sealing also found new-file permission mismatches under host umask;
the review-tree-only modes were normalized and the verifier rerun.
The prospective-list checker initially mistook `git diff --no-index` exit 1
(ordinary difference) for an error; it now checks diagnostics/error codes and
uses tracked-diff scope for root README so pre-existing main whitespace is not
misclassified as new. The final explicit submission check stages nothing.

The same three absent-module main failures remain established by exact-base
evidence. Their files are still absent in main; the research package does not
use those reorder entry points. Deselections below remain unchanged and explicit.

## Earlier isolated-checkout verification

Validation was performed CPU-only with Python 3.11.11, PyTorch 2.9.0+cu128,
NumPy 2.4.4, SciPy 1.17.1, and pytest 9.1.1 from the frozen main campaign
environment. `CUDA_VISIBLE_DEVICES` was empty. No model was downloaded and no
campaign evaluation was rerun.

Paths below use placeholders so the public record does not publish private
machine locations:

- `<REPO>`: this isolated checkout;
- `<PRIMARY>`: immutable primary campaign root;
- `<EXTENSION>`: immutable PPL-extension campaign root;
- `<BOUNDARY>`: immutable boundary/corruption campaign root;
- `<PYTHON>`: frozen main-environment Python.

## Historical snapshot tests

Primary snapshot:

```bash
CAMPAIGN_ROOT=<PRIMARY> CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=<REPO>/research/n16k64/software/primary/support:<REPO>/research/n16k64/software/primary:<REPO> \
<PYTHON> -m pytest -q -p no:cacheprovider \
  research/n16k64/software/primary/campaign/tests \
  -k 'not audited_tile_totals_match_known_results'
```

Result: **105 passed, 1 deselected**, with two expected SciPy constant-input
warnings. The deselected audit test uses a source-tree-relative handoff fixture
that is not present in the namespaced public layout; the authoritative tile
totals were separately checked from the frozen protocol/maps and later anchor
audits.

PPL-extension snapshot:

```bash
CAMPAIGN_ROOT=<EXTENSION> CAMPAIGN_PARENT_ROOT=<PRIMARY> \
CAMPAIGN_REPO_ROOT=<WORKSPACE> CAMPAIGN_SOURCE_ROOT=<REPO>/research/n16k64/software/ppl_extension/support \
CAMPAIGN_STORAGE_ROOT=<IMMUTABLE_EXTENSION_STORAGE> CUDA_VISIBLE_DEVICES='' \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=<REPO>/research/n16k64/software/ppl_extension/support:<REPO>/research/n16k64/software/ppl_extension:<REPO> \
<PYTHON> -m pytest -q -p no:cacheprovider \
  research/n16k64/software/ppl_extension/campaign/tests \
  --ignore=research/n16k64/software/ppl_extension/campaign/tests/test_fidelity_noise.py \
  --ignore=research/n16k64/software/ppl_extension/campaign/tests/test_final_reports_accuracy.py \
  -k 'not audited_tile_totals_match_known_results'
```

Result: **209 passed, 1 deselected**. The two ignored files are inherited
primary tests that expect `PROTOCOL_FREEZE.json` at `CAMPAIGN_ROOT`; the
extension root correctly contains `PROTOCOL_EXTENSION.json` instead. Those
tests passed in the primary snapshot above. All extension-specific tests were
included.

Mechanism/follow-up/boundary superset snapshot:

```bash
CAMPAIGN_ROOT=<PRIMARY> MIXFP4_WORKSPACE_ROOT=<WORKSPACE> \
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=<REPO>/research/n16k64/software/boundary/support:<REPO>/research/n16k64/software/boundary:<REPO> \
<PYTHON> -m pytest -q -p no:cacheprovider \
  research/n16k64/software/boundary/campaign/tests \
  -k 'not audited_tile_totals_match_known_results'
```

Result: **105 passed, 1 deselected**, with the same two expected constant-input
warnings. A separate import smoke loaded 20 mechanism/follow-up/boundary modules
using explicit campaign environment variables. AST parsing succeeded for all
383 Python files present at that point in the snapshot.

The first unconfigured invocations were retained in the review notes: primary
reported 93 passes and 13 failures from missing campaign-root/schema fixtures;
extension first stopped at environment-variable collection errors, then reached
216 passes plus 11 failures from applying inherited primary report tests to the
extension root. No numerical assertion failed. The configured commands above
match the historical campaign layout and close those packaging diagnostics.

## Current-main compatibility tests

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<REPO> \
<PYTHON> -m pytest -q -p no:cacheprovider \
  tests/test_mixfp4.py tests/test_reorder.py tests/test_task_reorder.py
```

Result: **86 passed, 3 failed**. Each failure is an import of a module absent
from the recorded `main` tree: `run_fine_row_validate`,
`merge_reorder_evaluations`, or `run_reorder_map_refine`. `git cat-file` confirms
that all three files are absent at base commit
`2d3e8f397d57009ba843c83b2a75a176262b026c`; this branch changes only README
text and the namespaced `research/n16k64/` subtree, so the failures pre-exist and
are unrelated to the delivery. They are reported as a current-main limitation,
not relabeled as passes.

## Snapshot integrity and inexpensive recomputation

```bash
python3 research/n16k64/tools/verify_public_snapshot.py
git diff --check
```

The verifier checks strict JSON/JSONL parsing, CSV shape, SVG/XML and PNG
signatures, authored-document links, source-index/public hashes, private
identifier and credential patterns, authoritative campaign totals, frozen
classifications, four-pool uniqueness, and every available
`exp(delta_NLL)-1` pair. Exact final counts and hashes are recorded in
`LOCAL_VALIDATION_RESULTS.json` and `ARTIFACT_MANIFEST.sha256`. The manifest
excludes itself and the generated local-validation report to avoid a circular
hash dependency; the report records the final manifest hash.

## Not run

- No GPU smoke, model load, calibration, evaluation, or expensive campaign.
- No native CUDA build or timing benchmark; the delivery does not modify native
  code.
- No full raw bootstrap from excluded paired arrays. Existing immutable audits
  were checked, and compact delta-NLL conversions were independently recomputed.
- No CI, because no commit or remote branch exists after GitHub denied the
  configured SSH identity's scoped push dry-run.
