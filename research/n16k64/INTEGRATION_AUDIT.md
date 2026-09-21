# Current evidence, completeness and workspace integration audit

This supplements the earlier delivery audit. It records work done in the daily
repository on 2026-09-21; the earlier isolated checkout is retained as a backup.
No GPU experiment, model download, selector development or large bootstrap was
performed. Sealed campaign files were read only.

## Workspace and recovery

The original and delivery are independent clones, not linked worktrees. Each
`git worktree list --porcelain` lists only itself; each has its own `.git`
common directory/object store and no alternates. Their origin URLs both resolve
to `git@github.com:brian030128/NVFP4-RaZeR.git`.

| Directory (relative to the operator's parent workspace) | Before this integration | After |
|---|---|---|
| `mixfp4` | main, 0d39f7d7f70560ba0e61b842ed02e3b3d7bc379f | research/consolidated-results-20260921, 2d3e8f397d57009ba843c83b2a75a176262b026c plus uncommitted delivery |
| `nvfp4_razer_delivery_20260921TJ3sEm3` | research/consolidated-results-20260921, 2d3e8f397d57009ba843c83b2a75a176262b026c | Same HEAD/branch; retained temporary backup |

Same branch names in separate clones do not cause a linked-worktree occupancy
conflict. The original local main ref remains at its original commit. Fetch
advanced origin/main; main itself was not pushed or merged. No delivery commit
exists. No research upstream is set while its remote branch does not exist.

Private recovery checkpoint: sibling `mixfp4_recovery_20260921_integration/`.
It contains a verified complete Git bundle, original tracked binary diff,
recovery tar with 19,067 source/document/config files from both directories,
and before/after per-file hashes. Original README SHA-256:
`d038023fa596fee6f876e43e3e8fff67370a38c878a37f69593138453de90a55`.
The checkpoint preserves this 454-insertion/8-deletion draft; it was not used
to overwrite newer main. To recover the draft, extract
`original/README.md` from `recovery.tar.gz` into a separate directory. To inspect
old history, clone the bundle elsewhere. Do not restore over active work blindly.
The checkpoint excludes raw binaries, environments and symlink archive targets,
which remain untouched at their existing locations; it is not a full evidence backup.

Before authorization to integrate, reflog/HEAD/index and 512 selected-source
hash rechecks supported no branch switch or source changes by the delivery
workflow. There was no pre-task filesystem-wide seal: absence of every transient
change to every unselected artifact cannot be independently proved. This turn's
authorized changes are explicit: fetch, research-branch checkout, README revision,
and the new `research/n16k64` tree. No clean, hard reset, stash, artifact deletion,
process termination, or campaign rewrite was performed. Visible sessions were
left alone; process visibility is incomplete, as the earlier safety audit records.

## Reverse completeness and source-to-final mapping

Before checkout the original HEAD had no local-only committed changes relative
to fetched main; it was 86 commits/1,555 changed paths behind. The only tracked
working change was README. Git enumerated 19,224 untracked and 79,820 ignored
entries; three campaign symlinks required separately examining their targets.
Ignoring environments/caches/bytecode/logs leaves four legacy `.pt` importance
files, not an omitted ignored custom implementation. Environment locks are
included; machine-specific shell environment setup is intentionally not a portable
default. All maps, caches, archives and run trees remain local.

| Original source | Final daily-workspace destination | Decision/reason |
|---|---|---|
| Original tracked tree and newer remote main | Existing root/modules/results/scripts | Main history retained at recorded SHA; no old code copied over newer interfaces |
| Uncommitted original README | Private recovery tar; reviewed claims in root/research README | Preserve draft; supersede stale pending-boundary status and overbroad interpretations |
| Delivery `research/n16k64` | `mixfp4/research/n16k64` | Imported review layer, then corrected and supplemented here |
| Primary `source/.../campaign`, extension v2, boundary source | `software/{primary,ppl_extension,boundary}` | Historical executable namespaces; no registration in current main CLI |
| Mechanism/follow-up `source/.../campaign` | `software/boundary/campaign` | All 66/73 earlier Python files are byte-identical to their archived boundary counterparts before public transforms |
| Extension original source versus v2 | `campaigns/ppl_improvement/source_v1_differences` | Eight differing files retained as historical overlays, not silently relabeled v2 |
| Mechanism `analysis/*.py` | `campaigns/mechanism/analysis` | Three omitted producer/promotion/render scripts restored |
| Follow-up `freeze_history` | `campaigns/followup/freeze_history` | Superseded analysis/render code and amendments retained |
| Boundary `ops/*.py` | `campaigns/boundary/ops` | Both watchdog implementations restored; host paths are archival placeholders |
| Mechanism/follow-up/boundary `job_specs` | Corresponding campaign `job_specs` | Exact planned policy definitions retained with text path redaction |
| Source root follow-up/boundary tests | `software/boundary/support/tests` | Previously omitted actual campaign-specific tests restored |
| Historical local Python imports | Versioned `support` directories | Static import closure restored; historical helpers no longer accidentally rely on different current-main files |
| Primary 15 actual calibration manifests | `campaigns/primary/calibration_manifests` | Sample/token identities retained; maps resolved by evaluated SHA, including valid retries |
| Boundary 6 compact paired-cluster NPZ | `campaigns/boundary/arrays` | Byte-identical small arrays enable independent mean/statistical recomputation |
| Boundary definitions and all later map manifests | Corresponding campaign directories | Exact tile lists, quotas, hashes and exclusions are reviewable |
| Weights, datasets, moments, binary maps, per-token arrays, full logs/registries, handoff ZIPs | Existing local `research_runs`, `research_artifacts`, `research_handoffs` | Not published; inventory/archive hashes and failure summaries preserve traceability |
| Unrelated environment/vendor caches and duplicate handoff source trees | Local only | No research implementation substituted from a cache |

The fixed evidence cutoff stays 2026-09-21T05:24:15Z. New documentation and
audits are dated review products; later experimental changes require a future
snapshot. [Completeness additions](validation/COMPLETENESS_ADDITIONS.json)
and the updated source index give each restored file's original/public SHA.
[Current evidence audit](validation/CURRENT_EVIDENCE_AUDIT.json) distinguishes
fresh hash/header/array checks from historical campaign claims.

The [reverse inventory](validation/REVERSE_SOURCE_INVENTORY.json) additionally
enumerates 3,930 code/config candidates, including generated JSON plans:
2,525 inherit matching main content, 796 resolve to preserved snapshot content,
and 609 are excluded generated per-run launch/config records. No custom candidate
is unclassified or newer than the fixed cutoff. This includes inspection of
symlink campaign targets; it is not just a checksum check of selected copies.

## Boundary semantics verified against code and artifacts

Selection/intervention unit is a **16×64 weight format block**, 1,024 weights,
containing 64 scale blocks of 1×16. `tiles.grid_shape` maps weight `(O,K)` to
`(O/16,K/64)`. Flat index `r*(K/64)+c` selects rows `[16r,16(r+1))`, columns
`[64c,64(c+1))`. `mapio` packs one row-major little-bit-order boolean per
format block. All 171 actual map headers and hashes were rechecked.

Scores are directional derivatives toward E0M3 minus FourOverSix, so negative
CE mean predicts lower NLL to first order. SE is across paired sequences, not
tokens or tiles. `U_L=mean_L+3*SE_L`, conjunction `max(U_CE,U_KL)<0`.
`k=3` is a fixed multiplier. `kappa_L=-mean_L/SE_L`, joint kappa is their minimum;
higher kappa is more favorable. The follow-up margin `-max(U_CE,U_KL)` is a
different quantity. KL is not a direct NLL-effect estimate.
Primary `calibrate.py` forms N16 per-sequence directional scores by summing two
vertical N8 children before computing float64 moments/SE. This is the aligned
N16 score construction, not OR/AND of N8 masks or combining child SEs, and it
does not import the historical tensor-wide activation ablation.

The zero-SE rule assigns +infinity for negative mean and -infinity otherwise;
all three frozen summaries report zero such tiles. Calibration rejects non-finite
scores and `tiles.elect` rejects non-finite upper statistics. Caution: the
historical `boundary_common.objective_kappa` helper itself does not separately
reject arbitrary NaNs (its `~(SE>0)` branch assumes validated inputs). Do not use
it as a general validated public API on new moments; add an explicit finite-input
guard in any future adapter. This review does not alter the frozen helper or
claim to have reread the entire ~2 GiB moment payload. Anchor equality is a
hash-verified historical audit; map header verification was rerun here.

B=4 means four selected and four rejected bands **per construction partition**.
Four partitions rotate outcome-blind remainder exclusions, yielding 32 policies
per model. Quotas are floor(selected_count/4) per exact layer/module, identical
for selected and rejected. Strata with fewer than four selected tiles contribute
zero; retained selected counts are 1480/1781 Llama, 3724/4077 Qwen, 3912/4179
Mistral (83.10%,91.34%,93.61%). Per-band totals are 370/931/978. B=6 and B=8
fail Llama's coverage requirement. These construction variants are dependent.

All boundary maps are group-only against FourOverSix. Selected-minus-rejected
is the equally averaged selected-band effect minus the rejected-band effect,
over bands/partitions, not an intervention on their union. Weakest-minus-nearest
subtracts rejected band01 from selected band04, averaging partitions. Negative
favors selected. A common local sharp separation is unsupported.

Corruption uses **all** original selected units as denominator: coverage is 1
and shortage lists are empty for every model. For each module with K selected,
the closest 4K rejected by descending kappa are rank-interleaved into four
disjoint K-sized, score-balanced pools. These are not four increasing-distance
reservoirs. At p=1 all selected units are replaced. Each intermediate map swaps
floor(p*K+.5) units using nested score-octile/hash-priority prefixes, preserving
each module's count. Rounding means achieved p can differ slightly from target.
Four p=.5 random controls use all rejected candidates with the same removal
sets/quotas; overlap with the near reservoir is allowed and recorded.

Accounting per model: 32 boundary +20 nonzero corruption +4 random =56 new
policies, plus one p=0 map =57 maps. Add the no-map FourOverSix baseline to get
58 policies per corpus. Total: 171 maps, 168 new policy evaluations, each on two
corpora (336 new model/corpus cells). Four p=1 payloads are distinct per model.
The six paired arrays contain exactly 58 policies and 231/53,231/53,235/57
C4/Wiki clusters for Llama,Qwen,Mistral. Recomputed mean p=1 effects match stored
figures to floating-point precision; no bootstrap was rerun.

## Formal classifications and timing

Both stored pattern gates pass, both terminal labels remain
`power_limited_support`. Pre-outcome endpoint labels are preserved in
`ENDPOINT_PROMOTION.json`: Llama slopes/selected-minus-rejected are limited
inference, p=.5/p=1 and boundary-neighbor contrast descriptive; Mistral Wiki
endpoints are limited inference, C4 p=.5/p=1 descriptive; Qwen C4 corruption
slope and boundary-neighbor contrast are limited inference, its other endpoints
and all Wiki endpoints descriptive. No observed CI upgrades those labels.
Inference uses 10,000 shared paired cluster bootstrap draws, plus-one p-values,
Holm families of 12 development and 6 validation contrasts per experiment.
Mistral remains one-shot analysis validation. Standardized all-model/leave-Qwen
sensitivities are not raw NLL, accuracy or explained variance.

Original handoff `GOAL_PROMPT.md`, lines 146–148, says required retries in hours
17–20, **“At hour 20, stop launching new GPU jobs.”**, then analysis/packaging
in hours 20–24. Frozen `PROTOCOL.json/time_gates` resolves the cutoff to
2026-09-19T04:03:03Z and target to 08:03:03Z. The former is mandatory launch
discipline; the latter is the stated completion target. Valid Llama attempt5
and Qwen attempt4 launched on September 20. `PROTOCOL_DEVIATIONS.json` TD01/TD02
discloses this and leaves strict admissibility unresolved. No prospective text
was found that authorizes an after-cutoff exception. Therefore keep those
outcomes as protocol-deviating supportive/sensitivity evidence, do not describe
the campaign as time-conformant, and do not rewrite the archived terminal label.
Whether a submission can treat late evidence as fully conformant remains
unresolved; this limitation need not prevent publishing a candid archive.

## Corrections to the prior review layer

| Prior wording/omission | Corrected interpretation | Effect |
|---|---|---|
| 0.77984 described as selected-tile fraction | Ratio of summed six confirmatory delta-NLL gains, N16/N8 | Documentation correction; frozen numerical result unchanged |
| Timing described only as operational | Mandatory launch-stop violation and unresolved strict admissibility | Restricts evidence status; no retrospective waiver |
| 105 superset CPU tests implied coverage of later campaign logic | Actual follow-up/boundary root tests were omitted | Restored seven focused tests; earlier test count is not their coverage |
| Standalone mechanism producers/watchdogs/job plans omitted | Restored with source/public hashes | Closes real producer-code omissions |
| All paired arrays excluded | Six small boundary arrays now included byte-identically | Enables public recomputation of those panel means |
| Broad “all favorable”/“accuracy preserved” shorthand | 30/30 five-draw PPL point estimates only; Mistral accuracy has negative draws | Prevents endpoint/model/task expansion |
| Primary and extension downstream confused | Primary GSM8K/PG19 done; extension promotion stopped | Completed studies removed from future run list |

The three existing-main failures import absent modules `run_fine_row_validate`,
`merge_reorder_evaluations`, `run_reorder_map_refine`; they concern separate
reorder utilities, not the historical N16 flow. They were reproduced at exact
main, and those interfaces remain unchanged. Do not claim those utilities work.
The `audited_tile_totals_match_known_results` deselection needs a missing
source-layout fixture; totals were separately audited. Two inherited primary
report-test files excluded from extension require primary `PROTOCOL_FREEZE`,
while extension has `PROTOCOL_EXTENSION`; they pass in the primary suite.
These limitations do not block compact review/analysis, but full end-to-end GPU
replay was not performed and is not certified by the CPU tests.

## Earlier publication attempt — superseded permission status

Authenticated GraphQL identifies Xrelifen with viewerPermission **READ**;
authenticated REST reports `push:false`; SSH transport authenticates the same
identity. Repository rulesets and effective candidate-branch rules returned
empty lists. The decisive blocker is write permission. Existing author/email
configuration is retained. No staging, commit, push or CI was initiated.
The minimum permission fix is repository write access for the existing identity,
then repeat authenticated role/rules checks and fetch before publication.
No fork, test-branch push, permission modification or bypass is used.

Current local gates: G1–G5,G7,G8 are supported by the updated audits and targeted
validation; G6 fails. Scientific limitations and future-study implementation/
input/budget blockers are documented, not hidden as unfinished old campaigns.
The original daily workspace is integrated even while publication is blocked.
Machine-readable final workspace/gate state is in
[WORKSPACE_INTEGRATION.json](validation/WORKSPACE_INTEGRATION.json); the exact
prospective submission list is [SUBMISSION_FILES.txt](validation/SUBMISSION_FILES.txt).
It excludes local campaign trees and contains no staged files or new commits.

## Final release readiness and delivery-removal audit — 2026-09-21

This section supersedes the earlier permission blocker, not the historical
research limitations. RELEASE_READINESS: **PASS at the pre-commit gate**.
No GPU work, model download or new bootstrap campaign was performed.

| Gate | Current result | Evidence and qualification |
|---|---|---|
| G1 | PASS | Fixed source cutoff; 604 source/public pairs, five campaign seals and original-work checkpoint; current release also captured in a separately verified recovery archive. |
| G2 | PASS | README, claim matrix and archived statistical report retain power-limited support and timing deviation. Late Llama/Qwen evidence is not strictly time-conformant confirmatory evidence. Scientific limitations remain disclosed, not waived. |
| G3 | PASS | Refetched main remains `2d3e8f397d57009ba843c83b2a75a176262b026c`; only root README and namespaced research files change. |
| G4 | PASS | Existing affected CPU/CLI tests retained; current source/map/array audit, manifest, links, privacy and scope checks repeated. No unrelated GPU replay or bootstrap rerun. |
| G5 | PASS | Explicit 637-path list; no prospective file over 10 MiB; zero pre-existing new commits beyond main. Stage only this list and compare the resulting new commit tree before push. |
| G6 | PASS | Fresh API user and SSH identity Xrelifen; GraphQL WRITE, REST push:true. Rulesets, effective branch rules and classic protection rules all empty; classic-rule pagination exhausted. SSH origin is the target repository. Existing author configuration retained. |
| G7 | PASS | Reverse inventory covers source candidates; full delivery walk includes tracked/untracked/ignored files and Git metadata, independently of the submission manifest. Excluded raw evidence remains explicitly non-public. |
| G8 | PASS within checked scope | Original research hashes unchanged; delivery read only; no process terminated and no evidence removed. Global process visibility is restricted, as qualified below. |

### Reverse delivery inventory and recovery

The full delivery capture contains **4,053 regular files**: 3,117 tracked,
532 untracked, 404 Git-metadata files; no ignored files or symlinks were found.
All 3,649 non-metadata files were compared, not only manifest-listed files:
3,637 are byte-identical in the daily workspace. The remaining 12 are older
versions of the following files, all retained in the verified backup:

| Original delivery path | Final daily path | Disposition / reason |
|---|---|---|
| `README.md` | `README.md` | Updated research entry while preserving latest-main usage; old wording is historical. |
| `research/n16k64/README.md`, `CLAIM_EVIDENCE_MATRIX.md` | Same paths | Corrected gain-ratio, timing and endpoint qualifications. |
| `research/n16k64/ARTIFACT_INVENTORY.md`, `ARTIFACT_INVENTORY.json`, `SNAPSHOT_SOURCE_INDEX.json` | Same paths | Expanded source coverage and explicit exclusions. |
| `research/n16k64/ARTIFACT_MANIFEST.sha256` | Same path | Resealed expanded, corrected snapshot; old seal preserved. |
| `research/n16k64/COMPATIBILITY_AND_VALIDATION.md`, `validation/VALIDATION_COMMANDS.md`, `validation/LOCAL_VALIDATION_RESULTS.json` | Same paths | Added actual daily-workspace validation and current permissions. |
| `research/n16k64/software/README.md`, `tools/verify_public_snapshot.py` | Same paths | Clarified runnable entries/dependencies; expanded verification. |
| Delivery `.git/` including refs, reflogs and unreachable objects | Private complete-delivery recovery archive | Independent clone metadata retained intact; do not overwrite daily Git metadata. |

No absent producer or necessary result was found among these differences.
The delivery has 259 reachable/reflog commits, all present in daily Git's object
store, no stashes or tags, and no unreachable commits reported by `git fsck`.
Unreachable blobs and the exact reflogs/config/index are nevertheless preserved
by backing up the entire `.git` directory, not merely creating a branch bundle.

Private recovery location: the existing sibling recovery directory, retained
outside Git. `DELIVERY_REVERSE_INVENTORY.json` records every file, Git class,
hash and A/B disposition; `DELIVERY_GIT_HISTORY.json` records refs/stashes/reflogs
and fsck. `delivery_complete_20260921.tar.gz` is 67,884,396 bytes, SHA-256
`4076d12a9da2e9c7fafe6cf144c026edcc9e87aa1389558c5aeba1545dc283ab`.
Every archive member was read and compared to source hashes; source inventory
was unchanged before/after capture. A separate `integrated_release_*.tar.gz`
captures the final README and complete research tree; its exact member hashes,
archive name and checksum are in private `LATEST_INTEGRATED_BACKUP.json`.
These private archives may contain local metadata and must not be published.

This is local recoverability, not disaster-recovery replication: recovery and
workspace share storage. Excluded large raw artifacts under the original
campaign locations were not copied into these archives or pushed. No independent
backup of all such raw data is asserted; treat unverified originals as single
copies. Delivery contained no unique large raw-evidence payload to rescue.

### Runtime independence and removal decision

The daily tracked/new runtime code contains no hardcoded delivery path; matches
are inventory/workspace/history documentation. The checkpoint tool accepts an
explicit source argument only when performing a backup, not in research runtime.
No delivery symlinks were found. A separate daily-tree check examined 1,368
symlinks without following directory links; none targets delivery. The active
audit environment also contains no delivery-valued environment variable.
The daily public package remains a historical
research archive with executable components and documented external inputs, not
a self-contained GPU replay package.

Visible process cwd/executable/file-descriptor/memory-map checks found no delivery
dependency (82 successful cwd/executable reads). There were **6,088 permission
denials** across process checks. Thus a host-wide claim that no process uses the
directory cannot be independently certified. No process was signalled.

DELIVERY_REMOVAL_READINESS: **UNRESOLVED**, solely for incomplete live-process
visibility; content/history recoverability checks pass. Removing delivery would
remove a redundant old checkout, old review-layer versions and its Git metadata,
all recoverable from the verified archive, but could still disrupt an inaccessible
process. Before any future removal, obtain an authorized complete process/open-file
check and recheck for changes since this capture. This does not block publishing
the stable daily snapshot. Delivery is retained untouched in this task.

The permission check used `gh api user`, repository REST/GraphQL permissions,
repository rulesets, effective candidate-branch rules, all classic branch rules,
and `ssh -T git@github.com`. Fetch succeeded through the same configured SSH
origin; fetch alone was not treated as proof of write permission. No test branch,
permission change, force push, main push or pull request is authorized here.
