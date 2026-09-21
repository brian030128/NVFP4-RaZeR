# Compatibility, validation, and release gates

## Current integration addendum — 2026-09-21

The daily `mixfp4` workspace now contains the research branch based on the same
recorded main SHA. The former delivery checkout remains a temporary backup.
[INTEGRATION_AUDIT.md](INTEGRATION_AUDIT.md) supersedes the historical workspace
and completeness descriptions below: it records the authorized branch change,
recovery checkpoint, real producer-code omissions restored, corrected 77.98%
gain-ratio interpretation, stricter timing qualification, and current evidence checks.
The earlier statement that the research tree was complete is qualified by this
reverse audit; its omitted producers and two campaign-specific test files have
now been included. Seven focused tests passed from the daily workspace.

At the release-readiness recheck, authenticated API and SSH both identify
Xrelifen; GraphQL permission is WRITE and REST `push:true`. Repository rulesets,
effective candidate-branch rules, and classic branch-protection rules are empty.
G6 now passes. The earlier READ/denial evidence below is historical, not a
current blocker. The final pre-publication gate table and delivery-removal
qualification are in [INTEGRATION_AUDIT.md](INTEGRATION_AUDIT.md).
The historical G1–G8 table below is not a substitute for current validation.
The earlier 512-file safety audit remains a record of its original cutoff, while
`CURRENT_EVIDENCE_AUDIT.json` records the expanded source scope.

## Earlier isolated-checkout review

This report records the 2026-09-21 review of the N16K64 research snapshot. It
distinguishes checks rerun for this delivery from conclusions copied from sealed
campaign audits. The working branch remains local and uncommitted because the
same GitHub SSH transport identity used for fetch was denied push permission.

## Snapshot and workspace protection

- Target repository: `https://github.com/brian030128/NVFP4-RaZeR`.
- Recorded remote `main`: `2d3e8f397d57009ba843c83b2a75a176262b026c`.
- Historical shared-workspace HEAD: `0d39f7d7f70560ba0e61b842ed02e3b3d7bc379f`.
- Merge base: `0d39f7d7f70560ba0e61b842ed02e3b3d7bc379f`; it is an ancestor of the
  recorded remote `main`.
- Snapshot cutoff: `2026-09-21T05:24:15Z`.
- Local candidate branch: `research/consolidated-results-20260921`, created in
  an isolated checkout directly from the recorded remote `main`.

The shared checkout was not switched, stashed, reset, cleaned, staged, or
otherwise rewritten. It contained pre-existing tracked and untracked work. A
process, open-file, timestamp, and campaign-state inspection found no visible
process writing the selected N16K64 inputs at the cutoff; unrelated same-user
sessions were left untouched. Full file-descriptor visibility for other users
is restricted on this host, and many source artifacts are world-writable, so
stability does not rely on an unsupported global-process claim: all 512 source
hashes matched both the copy-time index and a later independent recheck. The
copied scope was then fixed. Newer outputs, if any, belong to a future snapshot
rather than being chased by this branch. See
[`validation/SNAPSHOT_SAFETY_AUDIT.json`](validation/SNAPSHOT_SAFETY_AUDIT.json).

Each copied artifact has a logical source ID, original SHA-256, public SHA-256,
size, and transformation class in
[`SNAPSHOT_SOURCE_INDEX.json`](SNAPSHOT_SOURCE_INDEX.json). The immutable source
campaigns were not edited. Text-only path/account/device redaction and portable
path plumbing are enumerated in [`REDACTION_LOG.json`](REDACTION_LOG.json); no
scientific value transform is declared. Large evidence remains local and is
listed with full checksums in [`ARTIFACT_INVENTORY.json`](ARTIFACT_INVENTORY.json).
CRLF CSVs and legacy end-of-line whitespace in copied campaign/source evidence
are preserved and locally marked `-whitespace`; they are not silently
reformatted. Newly authored review files keep normal Git whitespace checks.
Candidate file modes were normalized without changing bytes: only four
shebang-bearing scripts are executable; JSON, CSV, Markdown, figures, locks,
and non-entry-point modules are not.

## Latest-main integration

The candidate branch starts at the exact remote-main commit above, rather than
copying an old workspace tree over current code. The historical workspace HEAD
is 1,555 changed paths behind that recorded `main`. Existing current-main work
in `quantize/`, `native/`, `results/`, `scripts/`, `slurm/`, and the public
inference artifact is preserved unchanged. This delivery changes only the root
README and adds the namespaced `research/n16k64/` review tree.

Historical outcomes were produced by campaign-specific source snapshots, not by
the current branch HEAD. Those sources are retained under [`software/`](software/)
and must not be mixed across snapshots. Machine-specific defaults were replaced
with environment variables and synthetic device IDs only. No quantization,
selector, calibration, tokenization, model-loading, evaluation, or metric
semantics were changed. Therefore the old outcomes remain explicitly attributed
to their historical source/config hashes; this report does not claim they were
regenerated by current `main`.

No package import or CLI path in current `main` is replaced by the research
tree. JSON/CSV schemas are preserved as evidence snapshots. The public README
uses relative paths and all executable machine roots are placeholders or
environment variables rather than private absolute defaults.

## Research-rigor review

The review independently rechecked campaign seals, primary totals, decision
labels, matrix coverage, boundary coverage, four-pool uniqueness, final
classifications, model revisions, method/data definitions, failures and invalid
attempts, and available delta-NLL-to-relative-PPL conversions. Detailed claim
lineage is in [`CLAIM_EVIDENCE_MATRIX.md`](CLAIM_EVIDENCE_MATRIX.md).

Key findings are:

- The primary k=3 CE-and-KL N16K64 result is a strong pass for
  software/fake-quantized **quality only**. Its six confirmatory PPL endpoints
  pass the frozen Holm-adjusted non-inferiority gates, while N8 has the better
  point estimate in all six cells.
- The k=2 extension is post-hoc robustness evidence. Replacement-selector,
  held-out-generalization, downstream, SOTA, and portability claims are not
  promoted.
- Aggregate group ranking is supported, but individual-tile causal sign,
  magnitude, and finite-effect calibration remain unsupported. CE and KL are
  dependent, and k=3 is not a simultaneous-confidence guarantee.
- Boundary and corruption pattern gates pass, but both final classifications
  remain `power_limited_support`. Qwen power status is endpoint-specific;
  Mistral remains one-shot analysis validation. Reported confidence intervals
  and Holm decisions are kept separate.
- Corruption `p` is an imposed stress level. Full replacement versus the frozen
  map is not the frozen selector's improvement over FourOverSix. Standardized
  pooled sensitivity is not raw NLL effect size, accuracy, or explained
  variance.
- The boundary campaign completed all required evaluations but missed both the
  hour-20 launch cutoff and 24-hour target. The approximately 61.4-hour wall
  time and fail-closed co-tenancy retries are disclosed.
- The N16K64 evidence is BF16-dequantized fake quantization. It supports no
  native Tensor Core, latency, throughput, speedup, overhead, area, power, or
  Blackwell-performance claim.

No unexplained cross-report discrepancy was found in the compact authoritative
outputs. Raw maps, moments, and paired arrays are not in Git, so their sealed
campaign audits were hash-checked but a full raw bootstrap was not rerun.

## Software validation

All commands used `CUDA_VISIBLE_DEVICES=''`; no model was downloaded and no GPU
campaign was launched. Exact reproducible commands and environment versions are
in [`validation/VALIDATION_COMMANDS.md`](validation/VALIDATION_COMMANDS.md).

| Check | Result | Scope |
|---|---|---|
| Primary historical campaign tests | PASS: 105 passed, 1 deselected | Protocol, maps, statistics, artifact/report helpers; the deselected source-layout fixture is covered by frozen anchor audits. |
| PPL-extension campaign tests | PASS: 209 passed, 1 deselected | All extension-specific tests; two inapplicable inherited primary report files were excluded and pass in the primary suite. |
| Mechanism/follow-up/boundary superset tests | PASS: 105 passed, 1 deselected | Later CPU derivation, map, statistics, and reporting code. |
| Historical module import smoke | PASS: 20 modules | Explicit versioned paths and campaign variables. |
| Python syntax parse | PASS: 383 files | Every Python file in the public snapshot. |
| Public snapshot verifier | PASS after final sealing | Formats, links, source/public hashes, privacy patterns, campaign invariants, conversions, and manifest membership. |
| Current-main targeted tests | BASELINE LIMITATION: 86 passed, 3 failed | Three tests import files absent at the unmodified base commit: `run_fine_row_validate`, `merge_reorder_evaluations`, and `run_reorder_map_refine`. The research diff does not touch these interfaces. |
| GPU/model/campaign rerun | SKIPPED | Not needed for a documentation, compact-evidence, and namespaced historical-code delivery; prohibited expensive work was not restarted. |
| Native build/timing | SKIPPED | Native code is unchanged and no native-performance claim is introduced. |
| Full raw paired bootstrap | SKIPPED | Required large arrays are deliberately excluded from Git; existing sealed audits and compact calculations were checked. |

The three current-main failures were reproduced at the exact base commit, and
`git cat-file` confirms the imported files do not exist there. They are not
called passes, but they do not block change-specific validation because this
delivery adds no runtime module to current-main import paths.

## GitHub identity, permission, and branch-rule evidence

The configured remote resolves exactly to the target repository, and Git SSH
authentication identifies an account accepted by GitHub. Read-only
`ls-remote` agrees with the REST-visible `main` SHA. The existing Git author
name/email configuration was retained and no identity was invented. Commit
signing is not configured.

The final access check is recorded in
[`validation/GITHUB_ACCESS_AUDIT.json`](validation/GITHUB_ACCESS_AUDIT.json):

- GitHub CLI has no authenticated session, so it cannot report the operator's
  actual repository role or push permission.
- The unauthenticated repository API identifies a public repository but returns
  no caller permission object. Authenticated permission and classic-protection
  queries returned HTTP 401.
- Public queries reported zero repository rulesets, zero effective rules on
  `main`, and `main` as unprotected. Those observations cannot prove that every
  rule applicable to a not-yet-created branch has been disclosed.
- After all other local gates, a supplemental `git push --dry-run` for exactly
  `research/consolidated-results-20260921` was attempted through the configured
  SSH remote. GitHub rejected it with `Permission to ... denied`; the remote
  branch count remained zero.

The dry-run rejection is direct evidence that the current transport identity
does **not** have usable push permission. Applicable new-branch rules also
remain incompletely verified. No test branch was created, and no permission was
modified or bypassed. Per the task's commit-before-push hard gates, no commit or
actual push is permitted.

## Hard gates

| Gate | Status | Evidence and qualification |
|---|---|---|
| G1 — stable, traceable snapshot | PASS | Isolated cutoff, 512-entry before/after source-hash match, no visible writer, and explicit host-visibility/mode limitations; source campaigns unchanged during the checked interval. |
| G2 — claims and numbers reviewed | PASS | Claim matrix, campaign seals, compact independent checks, negative results, uncertainty, power, and deviations retained. |
| G3 — latest-main compatibility | PASS | Branch is based directly on recorded remote `main`; changes are README plus namespaced research, with no unresolved conflict or changed runtime interface. |
| G4 — risk-proportionate validation | PASS | Historical suites and snapshot checks pass. Three current-main failures are reproduced at the exact unchanged base and documented above. |
| G5 — publication/sensitive-data/history review | PASS | Explicit inventory, redaction/source hashes, no oversized Git artifacts, privacy/credential scans, and no new commits or hidden history. |
| G6 — identity, push permission, branch rules | **FAIL — BLOCKER** | The configured SSH identity authenticated, but GitHub denied the scoped push dry-run; authenticated rule/role queries are also unavailable. |
| G7 — important research covered | PASS | All five N16K64 campaigns and relevant current-main work are indexed; large/excluded/incomplete classes and reproduction effects are explicit. |
| G8 — sealed campaigns and other work untouched | PASS | Work occurred in an isolated checkout; source campaigns and unrelated processes were not modified. |

Because G6 failed, the permitted terminal state is a complete local,
reviewable working tree before commit. No CI exists for this unpushed branch.
Resolution requires granting the existing SSH identity write access to this
repository and providing authenticated read access (or maintainer evidence) for
all rules applicable to the candidate branch. Once both are confirmed, the
remaining release procedure is to refetch `main`, recheck the scoped
diff/manifest, explicitly stage the listed files, commit with the existing
author identity, push only this branch, and compare the remote branch SHA with
the local commit.
